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
    path_segments: list[str] = zntrack.params(['Γ', 'H', 'N', 'Γ', 'P', 'H', 'P', 'N'])
    path_labels: list[str] = zntrack.params(['Γ', 'H', 'N', 'Γ', 'P', 'H', 'P', 'N'])
    npoints: int = zntrack.params(100) # Number of k-points sampled along the path in the Brillouin zone.
    supercell: tuple[int, int, int] = zntrack.params((3, 3, 3))
    delta: float = zntrack.params(0.05) # Displacement distance in Angstroms for finite difference calculation.
    fmax: float = zntrack.params(0.01)
    
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

        
        # relax the structure
        optimizer = LBFGS(atoms)
        optimizer.run(fmax = self.fmax)
        # Save relaxed structure to cache
        relaxed_structure_path = self.phonon_cache / "relaxed_structure.xyz"
        self.atoms.write(relaxed_structure_path)
        print(f"Relaxed structure saved to: {relaxed_structure_path}")
        
        
        # Phonon spectrum calculation
        atoms.calc = calc
        ph = Phonons(atoms, calc, supercell=self.supercell, delta=self.delta)
        ph.clean()  # Clean previous results to avoid caching
        ph.run() # run phonon displcaement calculation
        ph.read(acoustic=True) # read vibrational modes
        
        # Define the path through the Brillouin zone
        path = bandpath(self.path_segments, atoms.cell, npoints=self.npoints, special_points=self.special_points)
        bands = ph.get_band_structure(path)
        frequencies = bands.energies.T
        
        # save to cache
        band_data = {"frequencies": frequencies.tolist(), "path": path.kpts.tolist()}
        spectrum_path = self.phonon_spectrum_path
        with spectrum_path.open("w") as f:
            json.dump(band_data, f)
        print(f"Phonon spectrum saved to: {spectrum_path}")

        # # Step 8: Generate and save plot
        # print("Generating and saving phonon spectrum plot...")
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.set_title("Phonon Spectrum")
        # ax.set_ylabel("Frequency (eV)")
        # ax.set_xlabel("Wave Vector")
        # ax.axhline(0, color="black", linestyle="--", alpha=0.5)

        # # Plot the frequencies
        # for freq in frequencies:
        #     ax.plot(range(self.npoints), freq, color="blue", lw=1)

        # # Add high-symmetry points to the plot
        # x_positions = [0]
        # for i in range(len(self.path_segments) - 1):
        #     k1 = self.special_points[self.path_segments[i]]
        #     k2 = self.special_points[self.path_segments[i + 1]]
        #     x_positions.append(x_positions[-1] + np.linalg.norm(np.array(k1) - np.array(k2)))
        # x_positions = np.array(x_positions) * (self.npoints - 1) / max(x_positions)

        # ax.set_xticks(x_positions)
        # ax.set_xticklabels(self.path_segments)
        # plt.savefig(self.phonon_plot_path, dpi=300)
        # plt.close()
        # print(f"Phonon spectrum plot saved to: {self.phonon_plot_path}")
        
        
    @property
    def figures(self) -> dict[str, go.Figure]:
        fig = go.Figure()
        for freq in self.phonon_data["frequencies"]:
            fig.add_trace(go.Scatter(y=freq, mode="lines", name="Phonon Mode"))
        fig.update_layout(
            title="Phonon Spectrum",
            xaxis_title="Wave Vector",
            yaxis_title="Energy (eV)",
        )
        return {"Phonon Spectrum": fig}

    @staticmethod
    def compare(*nodes: "PhononSpectrum") -> ComparisonResults:
        fig = go.Figure()
        for i, node in enumerate(nodes):
            for freq in node.phonon_data["frequencies"]:
                fig.add_trace(go.Scatter(
                    y=freq,
                    mode="lines",
                    name=f"{node.name} - Mode {i}"
                ))
        fig.update_layout(
            title="Phonon Spectrum Comparison",
            xaxis_title="Wave Vector",
            yaxis_title="Energy (eV)"
        )
        return ComparisonResults(figures={"Phonon Comparison": fig})
