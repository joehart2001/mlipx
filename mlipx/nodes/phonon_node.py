import pathlib
import typing as t
import json

import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import zntrack
from ase import Atoms, units
from ase.build import bulk
from ase.phonons import Phonons
from ase.dft.kpoints import bandpath
from ase.optimize import LBFGS
from dataclasses import field


from scipy.stats import gaussian_kde

from tqdm import tqdm

from mlipx.abc import ComparisonResults, NodeWithCalculator



class PhononSpectrum(zntrack.Node):
    """Performs structure relaxation and Computes the phonon spectrum of a given structure.

    Paramerters
    -----------
    data: list[ase.Atoms]
        List of ASE Atoms objects.
    model: NodeWithCalculator
        Model node with calculator for phonon spectrum calculation.
    """
    
    data: list[ase.Atoms] = zntrack.deps()
    model: NodeWithCalculator = zntrack.deps()
    #model: t.Any = zntrack.deps()

    special_points: dict[str, list[float]] = zntrack.params({'Γ': [0., 0., 0.],
                                                             'H': [0.5, -0.5, 0.5],
                                                             'N': [0., 0., 0.5],
                                                             'P': [0.25, 0.25, 0.25]})
    special_points: dict[str, list[float]] = zntrack.params(
        default_factory=lambda: {'Γ': [0., 0., 0.],
                                    'H': [0.5, -0.5, 0.5],
                                    'N': [0., 0., 0.5],
                                    'P': [0.25, 0.25, 0.25]})
    path_segments: list[str] = zntrack.params(default_factory=lambda: ['Γ', 'H', 'N', 'Γ', 'P', 'H', 'P', 'N'])
    path_labels: list[str] = zntrack.params(default_factory=lambda: ['Γ', 'H', 'N', 'Γ', 'P', 'H', 'P', 'N'])

    npoints: int = zntrack.params(100) # Number of k-points sampled along the path in the Brillouin zone.
    supercell: tuple[int, int, int] = zntrack.params((3, 3, 3))
    delta: float = zntrack.params(0.05) # Displacement distance in Angstroms for finite difference calculation.
    fmax: float = zntrack.params(0.01)
    
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")
    phonon_cache: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_cache")
    phonon_spectrum_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_spectrum.csv")
    phonon_plot_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_spectrum.png")
    results: pd.DataFrame = zntrack.plots(y="Energy [eV]", x="Wave Vector")
    phonon_plot: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_spectrum.png")
    
    
    system: t.Literal["bulk", "molecule", "surface", "other"] = zntrack.params("bulk")
    #calc_type: t.Literal["phonons", "relax", "static"] = zntrack.params("phonons")

    def run(self):
        calc = self.model.get_calculator()
        atoms = self.data
        results = []
        
        self.frames_path.parent.mkdir(exist_ok=True)
        
        
        #for current_frame, atoms in tqdm(enumerate(self.data)): not looping over frames as only one structure is passed
            # type/molecule checkes copied from vibrational_analysis.py
        
        if self.system is None:
            self.system = "bulk" # Default to bulk if not specified
        if self.system not in ["bulk", "molecule", "surface", "other"]:
            raise ValueError(f"Invalid system type: {self.system}")
        print(f"System type: {self.system}")
        
        if self.calc_type is None:
            if "calc_type" in atoms.info:
                self.calc_type = "phonons"
        if self.calc_type not in ["phonons", "relax", "static"]:
            raise ValueError(f"Invalid calculation type: {self.calc_type}")
        print(f"Calculation type: {self.calc_type}")

        atoms.calc = calc
        
        # relax the structure
        optimizer = LBFGS(atoms)
        optimizer.run(fmax = self.fmax)
        # Save relaxed structure to cache
        relaxed_structure_path = self.phonon_cache / "relaxed_structure.xyz"
        ase.io.write(relaxed_structure_path, atoms, format="extxyz")  # Fixed
        print(f"Relaxed structure saved to: {relaxed_structure_path}")
        
        
        # Phonon spectrum calculation
        
        ph = Phonons(atoms, calc, supercell=self.supercell, delta=self.delta)
        ph.clean()  # Clean previous results to avoid caching
        ph.run() # run phonon displcaement calculation
        ph.read(acoustic=True) # read vibrational modes
        
        
        # Define the path through the Brillouin zone
        path = bandpath(self.path_segments, atoms.cell, npoints=self.npoints, special_points=self.special_points)
        bands = ph.get_band_structure(path)
        frequencies = bands.energies.T
    
        
        if not self.phonon_spectrum_path.parent.exists():
            self.phonon_spectrum_path.parent.mkdir(parents=True, exist_ok=True)

        # save to cache
        band_data = {"frequencies": frequencies.tolist(), "path": path.kpts.tolist()}
        
        with self.phonon_spectrum_path.open("w") as f:
            json.dump(band_data, f)
        print(f"Phonon spectrum saved to: {self.phonon_spectrum_path}")
        
        ase.io.write(self.frames_path, atoms, append=True)
        #ase.io.write(self.frames_path, atoms, format="extxyz")
        print(f"Frames saved at {self.frames_path}")
        
        for k_idx, k_point in enumerate(path.kpts):
            for mode_idx, frequency in enumerate(frequencies[:, k_idx]):  
                results.append({
                    "Wave Vector": k_idx,
                    "k-point": k_point,
                    "Mode": mode_idx,
                    "Frequency (eV)": frequency
                }) # and "Frame": current_frame,  # If iterating over multiple structures



        self.results = pd.DataFrame(results)
        self.results.to_csv(self.phonon_spectrum_path, index=False)  # Save as CSV
        print(f"Results saved to: {self.phonon_spectrum_path}")





    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))
        
    @property
    def phonon_data(self) -> dict:
        """Load the phonon spectrum data from the saved JSON file."""
        if not self.phonon_spectrum_path.exists():
            raise FileNotFoundError(f"Phonon spectrum data not found at {self.phonon_spectrum_path}")
        
        with self.phonon_spectrum_path.open("r") as f:
            return json.load(f)
        
    @property
    def figures(self) -> dict[str, go.Figure]:
        phonon_data = self.phonon_data
        fig = go.Figure()
        for freq in phonon_data["frequencies"]:
            fig.add_trace(go.Scatter(y=freq, mode="lines", name="Phonon Mode"))
        fig.update_layout(
            title="Phonon Spectrum",
            xaxis_title="Wave Vector",
            yaxis_title="Energy (eV)",
        )
        return {"Phonon Spectrum": fig}

    @staticmethod
    def compare(*nodes: "PhononSpectrum") -> ComparisonResults:
        frames = sum([node.frames for node in nodes], [])
        fig = go.Figure()
        print(node.phonon_data["frequencies"])
        for i, node in enumerate(nodes):
            for freq in node.phonon_data["frequencies"]:
                fig.add_trace(go.Scatter(
                    x=node.phonon_data["path"][0],
                    y=freq,
                    mode="lines",
                    name=node.name.replace(f"_{node.__class__.__name__}", ""), # remove class name from node name
                ))
        fig.update_layout(
            title="Phonon Spectrum Comparison",
            xaxis_title="Wave Vector",
            yaxis_title="Energy (eV)"
        )
        return ComparisonResults(frames=frames, figures={"Phonon Comparison": fig})
