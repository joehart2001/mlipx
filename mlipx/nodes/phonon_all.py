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
from ase.optimize import LBFGS, FIRE
from dataclasses import field
import pickle

import warnings
from pathlib import Path
from typing import Any, Callable
from ase.calculators.calculator import Calculator
from tqdm import tqdm
from phonopy.api_phonopy import Phonopy
import yaml
from typing import Union
from ase.filters import FrechetCellFilter

from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator
from ase.constraints import FixSymmetry


from mlipx.phonons_utils import *
from phonopy import load as load_phonopy

from joblib import Parallel, delayed

from mlipx import PhononDispersion

from mlipx.phonons_utils import get_fc2_and_freqs, init_phonopy, load_phonopy, get_chemical_formula
from phonopy.structure.atoms import PhonopyAtoms
from seekpath import get_path
import zntrack.node
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")



class PhononAllBatch(zntrack.Node):
    """Batch phonon calculations (FC2 + thermal props) for multiple mp-ids."""

    mp_ids: list[str] = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    phonopy_yaml_dir: str = zntrack.params()

    N_q_mesh: int = zntrack.params(2)
    supercell: int = zntrack.params(3)
    fmax: float = zntrack.params(0.0001)
    thermal_properties_temperatures: list[float] = zntrack.params(
        default_factory=lambda: [0, 75, 150, 300, 600]
    )

    phonon_band_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_band_paths.json")
    phonon_dos_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_dos_paths.json")
    #phonon_qpoints_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_qpaths_paths.json")
    #phonon_labels_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_labels_paths.json")
    #phonon_connections_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_connections_paths.json")
    thermal_properties_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "thermal_properties_paths.json")
    get_chemical_formula_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mp_ids_and_formulas.json")
    
    def run(self):
        #calc = self.model.get_calculator()
        
        yaml_dir = Path(self.phonopy_yaml_dir)
        nwd = Path(self.nwd)
        fmax = self.fmax
        
        q_mesh = self.N_q_mesh
        q_mesh_thermal = 6
        temperatures = self.thermal_properties_temperatures
        

        def process_mp_id(mp_id: str, model, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures):
            try:
                print(f"\nProcessing {mp_id}...")
                calc = model.get_calculator()
                yaml_path = yaml_dir/ f"{mp_id}.yaml"
                                
                      
                phonons_pred = load_phonopy(str(yaml_path))
                
                displacement_dataset = phonons_pred.dataset
                atoms = phonopy2aseatoms(phonons_pred)
                
                atoms_sym = atoms.copy()
                atoms_sym.calc = calc
                atoms_sym.set_constraint(FixSymmetry(atoms_sym))
                opt = LBFGS(atoms_sym)
                opt.run(fmax=fmax, steps=1000)

                # primitive matrix not always available in reference data e.g. mp-30056
                if "primitive_matrix" in atoms.info.keys():
                    primitive_matrix = atoms.info["primitive_matrix"]
                else:
                    primitive_matrix = "auto"
                    print("Primitive matrix not found in atoms.info. Using 'auto' for primitive matrix.")
            
                phonons_pred = init_phonopy_from_ref(
                    atoms=atoms_sym,
                    fc2_supercell=atoms.info["fc2_supercell"],
                    primitive_matrix=primitive_matrix,
                    displacement_dataset=displacement_dataset,
                    symprec=1e-5,
                )

                phonons_pred, _, _ = get_fc2_and_freqs(
                    phonons=phonons_pred,
                    calculator=calc,
                    q_mesh=np.array([q_mesh] * 3),
                    symmetrize_fc2=True
                )

                #phonon_obj_path = nwd / f"phonon_obj/{mp_id}_phonon_obj.yaml"
                #Path(phonon_obj_path).parent.mkdir(parents=True, exist_ok=True)
                #phonons_pred.save(filename=str(phonon_obj_path)) #, settings={"force_constants": True})
                
                phonons_pred.auto_band_structure()
                band_structure_pred = phonons_pred.get_band_structure_dict()
                phonons_pred.auto_total_dos()
                dos_pred = phonons_pred.get_total_dos_dict()
                
                phonon_pred_path = nwd / f"phonon_pred_data/"
                phonon_pred_path.mkdir(parents=True, exist_ok=True)
                
                phonon_pred_band_structure_path = phonon_pred_path / f"{mp_id}_band_structure.npz"
                phonon_pred_dos_path = phonon_pred_path / f"{mp_id}_dos.npz"
                thermal_path = phonon_pred_path / f"{mp_id}_thermal_properties.json"
                
                with open(phonon_pred_band_structure_path, "wb") as f:
                    pickle.dump(band_structure_pred, f)
                with open(phonon_pred_dos_path, "wb") as f:
                    pickle.dump(dos_pred, f)

                
                chemical_formula = get_chemical_formula(phonons_pred, empirical=True)

                    
                # with open(phonon_pred_path / f"{mp_id}_qpoints.npz", "wb") as f:
                #     pickle.dump(qpoints, f)
                # with open(phonon_pred_path / f"{mp_id}_labels.json", "w") as f:
                #     json.dump(labels, f)
                # with open(phonon_pred_path / f"{mp_id}_connections.json", "w") as f:
                #     json.dump(connections, f)
                                  
                phonons_pred.run_mesh([q_mesh_thermal] * 3) #TODO 20x20x20
                phonons_pred.run_thermal_properties(
                    temperatures=temperatures,
                    cutoff_frequency=0.05
                )

                thermal_dict = phonons_pred.get_thermal_properties_dict()
                thermal_dict_safe = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in thermal_dict.items()
                }
                with open(thermal_path, "w") as f:
                    json.dump(thermal_dict_safe, f, indent=4)

                return {
                    "mp_id": mp_id,
                    "phonon_band_path_dict": str(phonon_pred_band_structure_path),
                    "phonon_dos_dict": phonon_pred_dos_path,
                    "thermal_properties_dict": str(thermal_path),
                    "formula": chemical_formula,
                    # "phonon_qpoints_dict": str(phonon_pred_path / f"{mp_id}_qpoints.npz"),
                    # "phonon_labels_dict": str(phonon_pred_path / f"{mp_id}_labels.json"),
                    # "phonon_connections_dict": str(phonon_pred_path / f"{mp_id}_connections.json"),
                }
            except Exception as e:
                print(f"Skipping {mp_id} due to error: {e}")
                return None

        # Run jobs in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_mp_id)(mp_id, self.model, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures)
            for mp_id in self.mp_ids
        )

        # Optional: collect paths if needed later

        
        phonon_band_path_dict = {
            res["mp_id"]: str(res["phonon_band_path_dict"])
            for res in results if res is not None
        }
        phonon_dos_path_dict = {
            res["mp_id"]: str(res["phonon_dos_dict"])
            for res in results if res is not None
        }
        thermal_properties_path_dict = {
            res["mp_id"]: str(res["thermal_properties_dict"])
            for res in results if res is not None
        }
        chemical_formula_dict = {
            res["mp_id"]: res["formula"]
            for res in results if res is not None
        }
        print(chemical_formula_dict)

        
        # Save paths to JSON files
        with open(self.phonon_band_paths, "w") as f:
            json.dump(phonon_band_path_dict, f, indent=4)
        with open(self.phonon_dos_paths, "w") as f:
            json.dump(phonon_dos_path_dict, f, indent=4)
        with open(self.thermal_properties_paths, "w") as f:
            json.dump(thermal_properties_path_dict, f, indent=4)
        with open(self.get_chemical_formula_path, "w") as f:
            json.dump(chemical_formula_dict, f, indent=4)

    
    @property
    def get_phonon_band_paths(self) -> dict[str, str]:
        """Returns a dictionary of mp_id to phonon band structure paths."""
        with open(self.phonon_band_paths, "r") as f:
            return json.load(f)
    @property
    def get_phonon_dos_paths(self) -> dict[str, str]:
        """Returns a dictionary of mp_id to phonon DOS paths."""
        with open(self.phonon_dos_paths, "r") as f:
            return json.load(f)
    @property
    def get_thermal_properties_paths(self) -> dict[str, str]:
        """Returns a dictionary of mp_id to thermal properties paths."""
        with open(self.thermal_properties_paths, "r") as f:
            return json.load(f)
    
    @property
    def get_chemical_formulas_dict(self) -> dict[str, str]:
        with open(self.get_chemical_formula_path, "r") as f:
            return json.load(f)

        

    @property
    def get_phonon_ref_data(self) -> dict[str, dict[str, t.Any]]:
        """Returns a dictionary mapping mp_id to loaded phonon reference data."""
        band_paths = self.get_phonon_band_paths
        dos_paths = self.get_phonon_dos_paths
        thermal_paths = self.get_thermal_properties_paths
        chemical_formulas = self.get_chemical_formulas_dict

        def load_pickle(path: str):
            with open(path, "rb") as f:
                return pickle.load(f)

        def load_json(path: str):
            with open(path, "r") as f:
                return json.load(f)

        def load_npz(path: str):
            return dict(np.load(path, allow_pickle=True))

        result = {}
        for mp_id in sorted(self.mp_ids):
            try:
                result[mp_id] = {
                    "band_structure": load_pickle(band_paths[mp_id]),
                    "dos": load_pickle(dos_paths[mp_id]),
                    "thermal_properties": load_json(thermal_paths[mp_id]),
                    "formula": chemical_formulas[mp_id],
                }
            except Exception as e:
                print(f"Skipping {mp_id} due to loading error: {e}")
                continue

        return result
        
        
        
        
        




    @staticmethod
    def compare_reference(node_pred, node_ref, correlation_plot_mode=False, model_name=None):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        band_structure_pred = node_pred.band_structure
        distances_pred = band_structure_pred["distances"]
        frequencies_pred = band_structure_pred["frequencies"]
        dos_freqs_pred, dos_values_pred = node_pred.dos
        dos_freqs_pred, dos_values_pred = dos_freqs_pred[dos_values_pred > 0], dos_values_pred[dos_values_pred > 0]

        band_structure_ref = node_ref.band_structure
        distances_ref = band_structure_ref["distances"]
        frequencies_ref = band_structure_ref["frequencies"]
        dos_freqs_ref, dos_values_ref = node_ref.dos
        dos_freqs_ref, dos_values_ref = dos_freqs_ref[dos_values_ref > 0], dos_values_ref[dos_values_ref > 0]

        labels = node_ref.labels
        connections = node_ref.connections
        connections = [True] + connections

        # Start plotting
        fig = plt.figure(figsize=(9, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])
        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])

        for dist_segment, freq_segment in zip(distances_pred, frequencies_pred):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='--', label=model_name, color='red')

        for dist_segment, freq_segment in zip(distances_ref, frequencies_ref):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='-', label="Reference", color='blue')

        ax2.plot(dos_values_pred, dos_freqs_pred, lw=1.2, color="red", linestyle='--')
        ax2.plot(dos_values_ref, dos_freqs_ref, lw=1.2, color="blue")

        # Ticks
        xticks, xticklabels = PhononDispersion._build_xticks(distances_ref, labels, connections)
        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)
        ax1.set_xticks(xticks, xticklabels)

        # Axis settings
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)")
        ax1.set_xlabel("Wave Vector")

        freqs_all = np.concatenate(frequencies_pred + frequencies_ref)
        ax1.set_ylim(freqs_all.min() - 0.4, freqs_all.max() + 0.4)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("DOS")
        plt.setp(ax2.get_yticklabels(), visible=False)

        # Legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.8, 1.02), frameon=False, ncol=2)

        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)

        # Save or show
        mp_id = node_ref.name.split("_")[-1]
        plt.suptitle(f"{mp_id} — {model_name}", x=0.4)

        if correlation_plot_mode:
            out_dir = f"benchmark_stats/bulk_crystal_benchmark/phonons/{model_name}/phonon_plots"
            os.makedirs(out_dir, exist_ok=True)
            plot_path = f"{out_dir}/dispersion_{model_name}_{mp_id}.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            return plot_path
        else:
            plt.show()
            
            
            
            




    @staticmethod
    def benchmark_interactive(
        pred_node_dict: dict[str, "PhononAllBatch"],
        ref_phonon_node,
        ui=None,
        run_interactive=True,
        report=True,
        normalise_to_model: t.Optional[str] = None,
    ):
        """
        Benchmarking with multiple models (each one is a PhononAllBatch node).
        """
        class PhononDataWrapper:
            def __init__(self, mp_id: str, data: dict[str, t.Any]):
                self.name = f"phonon_{mp_id}"
                self._data = data

            @property
            def band_structure(self):
                return self._data["band_structure"]

            @property
            def dos(self):
                dos_dict = self._data["dos"]
                return dos_dict["frequency_points"], dos_dict["total_dos"]

            @property
            def get_thermal_properties(self):
                return self._data["thermal_properties"]

            @property
            def labels(self):
                return self._data["labels"]

            @property
            def connections(self):
                return self._data["connections"]
            
            @property
            def formula(self):
                return self._data["formula"] if "formula" in self._data else None
            
        
        def convert_batch_to_node_dict(
            batch_node: PhononAllBatch, model_name: t.Optional[str] = None
        ) -> dict[str, t.Any]:
            raw_data = batch_node.get_phonon_ref_data
            if model_name is None:
                return {mp_id: PhononDataWrapper(mp_id, data) for mp_id, data in raw_data.items()}
            else:
                return {mp_id: {model_name: PhononDataWrapper(mp_id, data)} for mp_id, data in raw_data.items()}
        
        ref_node_dict = convert_batch_to_node_dict(ref_phonon_node)

        pred_node_dict_new = {}
        for model_name, batch_node in pred_node_dict.items():
            model_data = convert_batch_to_node_dict(batch_node, model_name)
            for mp_id, model_wrapper_dict in model_data.items():
                if mp_id not in pred_node_dict_new:
                    pred_node_dict_new[mp_id] = {}
                pred_node_dict_new[mp_id].update(model_wrapper_dict)
                
        
        
    
        PhononDispersion.benchmark_interactive(
            pred_node_dict=pred_node_dict_new,
            ref_node_dict=ref_node_dict,
            ui=ui,
            run_interactive=run_interactive,
            report=report,
            normalise_to_model=normalise_to_model,
        )
        
        

