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

from joblib import Parallel, delayed, parallel_backend
import traceback
from joblib import parallel_backend
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
#import torch
#torch._dynamo.config.suppress_errors = True

import ray




# Batched Ray remote function for processing multiple mp_ids at once
@ray.remote(num_gpus=1)
def process_mp_ids_batch_ray(mp_ids, model, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures, check_completed, n_jobs):
    from joblib import Parallel, delayed

    # Make model accessible without pickling
    global global_model
    global_model = model

    def handle_mp_id(mp_id):
        try:
            return PhononAllBatch._process_mp_id_static(
                mp_id, global_model, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures, check_completed
            )
        except Exception as e:
            print(f"Skipping {mp_id} due to error: {e}")
            return None

    #results = Parallel(n_jobs=n_jobs)(delayed(handle_mp_id)(mp_id) for mp_id in mp_ids)
    with parallel_backend("threading", n_jobs=n_jobs):
        results = Parallel()(delayed(handle_mp_id)(mp_id) for mp_id in mp_ids)
    return [res for res in results if res is not None]



class PhononAllBatch(zntrack.Node):
    """Batch phonon calculations (FC2 + thermal props) for multiple mp-ids."""

    mp_ids: list[str] = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    phonopy_yaml_dir: str = zntrack.params()
    n_jobs: int = zntrack.params(-1)
    check_completed: bool = zntrack.params(False)
    #threading: bool = zntrack.params(False)
    cpu: bool = zntrack.params(False)

    N_q_mesh: int = zntrack.params(6)
    supercell: int = zntrack.params(3)
    #fmax: float = zntrack.params(0.0001)
    fmax: float = zntrack.params(0.005)
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
    
    @staticmethod
    def _process_mp_id_static(mp_id: str, model, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures, check_completed):
        import traceback
        try:
            yaml_path = yaml_dir / f"{mp_id}.yaml"
            calc = model.get_calculator()
            phonons_pred = load_phonopy(str(yaml_path))
            chemical_formula = get_chemical_formula(phonons_pred, empirical=True)
            phonon_pred_path = nwd / f"phonon_pred_data/"
            phonon_pred_path.mkdir(parents=True, exist_ok=True)
            phonon_pred_band_structure_path = phonon_pred_path / f"{mp_id}_band_structure.npz"
            phonon_pred_dos_path = phonon_pred_path / f"{mp_id}_dos.npz"
            thermal_path = phonon_pred_path / f"{mp_id}_thermal_properties.json"
            
            if check_completed and phonon_pred_band_structure_path.exists() and phonon_pred_dos_path.exists() and thermal_path.exists():
                print(f"Skipping {mp_id} as results already exist.")
                return {
                    "mp_id": mp_id,
                    "phonon_band_path_dict": str(phonon_pred_band_structure_path),
                    "phonon_dos_dict": str(phonon_pred_dos_path),
                    "thermal_properties_dict": str(thermal_path),
                    "formula": chemical_formula,
                }
            print(f"\nProcessing {mp_id}...")
            displacement_dataset = phonons_pred.dataset
            atoms = phonopy2aseatoms(phonons_pred)
            atoms_sym = atoms.copy()
            atoms_sym.calc = calc
            atoms_sym.set_constraint(FixSymmetry(atoms_sym))
            opt = FIRE(atoms_sym)
            opt.run(fmax=fmax, steps=1000)
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
            phonons_pred, fc2, freqs = get_fc2_and_freqs(
                phonons=phonons_pred,
                calculator=calc,
                q_mesh=np.array([q_mesh] * 3),
                symmetrize_fc2=True
            )
            phonons_pred.auto_band_structure()
            band_structure_pred = phonons_pred.get_band_structure_dict()
            phonons_pred.auto_total_dos()
            dos_pred = phonons_pred.get_total_dos_dict()
            with open(phonon_pred_band_structure_path, "wb") as f:
                pickle.dump(band_structure_pred, f)
            with open(phonon_pred_dos_path, "wb") as f:
                pickle.dump(dos_pred, f)
            phonons_pred.run_mesh([q_mesh_thermal] * 3)
            phonons_pred.run_thermal_properties(
                temperatures=temperatures,
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
            }
        except Exception as e:
            print(f"Skipping {mp_id} due to error: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"\nError while processing {mp_id}:\n")
                traceback.print_exc(file=f)


    def run(self):
        from pathlib import Path
        import ray

        yaml_dir = Path(self.phonopy_yaml_dir)
        nwd = Path(self.nwd)
        fmax = self.fmax
        q_mesh = self.N_q_mesh
        q_mesh_thermal = 20
        temperatures = self.thermal_properties_temperatures

        ray.init(ignore_reinit_error=True, num_gpus=1)

        calc_model = self.model  # Materialize to avoid lazy ZnTrack object

        # Split mp_ids into chunks (1 mp_id per job since we have 1 GPU)
        futures = [
            process_mp_ids_batch_ray.remote(
                [mp_id], calc_model, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures, self.check_completed, self.n_jobs
            )
            for mp_id in self.mp_ids
        ]

        results_nested = ray.get(futures)
        ray.shutdown()

        results = [res for sublist in results_nested for res in sublist if res is not None]
        print(f"\nFinished with {len(results)} successful results.")

        phonon_band_path_dict = {
            res["mp_id"]: str(res["phonon_band_path_dict"]) for res in results
        }
        phonon_dos_path_dict = {
            res["mp_id"]: str(res["phonon_dos_dict"]) for res in results
        }
        thermal_properties_path_dict = {
            res["mp_id"]: str(res["thermal_properties_dict"]) for res in results
        }
        chemical_formula_dict = {
            res["mp_id"]: res["formula"] for res in results
        }

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
    def benchmark_interactive(
        pred_node_dict,
        ref_phonon_node,
        ui=None,
        run_interactive=True,
        report=False,
        normalise_to_model: t.Optional[str] = None,
        no_plots: bool = False,
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
            batch_node, model_name: t.Optional[str] = None
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
        
        output = PhononDispersion.benchmark_interactive(
            pred_node_dict=pred_node_dict_new,
            ref_node_dict=ref_node_dict,
            ui=ui,
            run_interactive=run_interactive,
            report=report,
            normalise_to_model=normalise_to_model,
            no_plots=no_plots,
        )
        return output







    @staticmethod
    def benchmark_precompute(
        pred_node_dict,
        ref_phonon_node,
        cache_dir: str = "app_cache/bulk_crystal_benchmark/phonons_cache",
        ui=None,
        run_interactive: bool = False,
        report: bool = False,
        normalise_to_model: t.Optional[str] = None,
        no_plots: bool = False,
    ):
        """
        Precompute all data for benchmarking and save to cache_dir.
        """
        import os
        os.makedirs(cache_dir, exist_ok=True)
        import pickle
        import pandas as pd

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
            batch_node, model_name: t.Optional[str] = None
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

        app, mae_summary_df, scatter_to_dispersion_map, model_benchmarks_dict, _ = PhononDispersion.benchmark_interactive(
            pred_node_dict=pred_node_dict_new,
            ref_node_dict=ref_node_dict,
            ui=ui,
            run_interactive=run_interactive,
            report=report,
            normalise_to_model=normalise_to_model,
            no_plots=no_plots,
        )
        
        # create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        mae_summary_df.to_pickle(f"{cache_dir}/mae_summary.pkl")
        with open(f"{cache_dir}/scatter_to_dispersion_map.pkl", "wb") as f:
            pickle.dump(scatter_to_dispersion_map, f)
        with open(f"{cache_dir}/model_benchmarks_dict.pkl", "wb") as f:
            pickle.dump(model_benchmarks_dict, f)

        return


    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/bulk_crystal_benchmark/phonons_cache", 
        ui=None
        ):
        
        import pickle
        import pandas as pd
        from mlipx.dash_utils import run_app
        import dash

        with open(f"{cache_dir}/scatter_to_dispersion_map.pkl", "rb") as f:
            scatter_to_dispersion_map = pickle.load(f)
        with open(f"{cache_dir}/model_benchmarks_dict.pkl", "rb") as f:
            model_benchmarks_dict = pickle.load(f)
        mae_summary_df = pd.read_pickle(f"{cache_dir}/mae_summary.pkl")
        
        app = dash.Dash(__name__)
        
        app.layout = PhononDispersion.build_layout(mae_summary_df)

        PhononDispersion.register_callbacks(
            app=app,
            mae_df=mae_summary_df,
            scatter_to_dispersion_map=scatter_to_dispersion_map,
            model_benchmarks_dict=model_benchmarks_dict,
        )

        return run_app(app, ui=ui)