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
from ase.build.supercells import make_supercell

from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator
from ase.constraints import FixSymmetry
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points


from mlipx.phonons_utils import *
from phonopy import load as load_phonopy

from joblib import Parallel, delayed
import traceback
from joblib import parallel_backend
from mlipx import PhononDispersion

from mlipx.phonons_utils import get_fc2_and_freqs, init_phonopy, load_phonopy, get_chemical_formula
from phonopy.structure.atoms import PhonopyAtoms
from seekpath import get_path
import zntrack.node
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
import spglib

# For multiprocessing mode: no cache, use on-demand disk loading
def _process_single_mp_id_meta(mp_id, model, nwd, yaml_dir, dft_ref_path, fmax, q_mesh, q_mesh_thermal, temperatures, check_completed):
    """
    Wrapper for multiprocessing that avoids cache serialization.
    Uses on-demand disk loading (the fallback path in process_mp_id).
    """
    try:
        # Pass None for cache to trigger on-demand disk loading
        return PhononAllBatchMeta.process_mp_id(
            mp_id, model, nwd, yaml_dir, dft_ref_path, fmax, q_mesh, q_mesh_thermal, temperatures, check_completed, dft_ref_cache=None
        )
    except Exception as e:
        print(f"Skipping {mp_id} due to error: {e}")
        return None

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
warnings.filterwarnings("ignore", category=FutureWarning)

# import torch
# torch._dynamo.config.suppress_errors = True
import warnings
warnings.filterwarnings("ignore", module="torch.fx.experimental.symbolic_shapes")
#torch.fx._symbolic_trace.TRACED_MODULES.clear()  # optional
#from mace.calculators import atomic_energies_fn
#torch.fx.wrap('atomic_energies_fn')

# import torch
# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False
#     print("TF32 is disabled for CUDA operations.")



class PhononAllBatchMeta(zntrack.Node):
    """Batch phonon calculations (FC2 + thermal props) for multiple mp-ids."""

    mp_ids: list[str] = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    phonopy_yaml_dir: str = zntrack.params()
    dft_ref_path: str = zntrack.params(None)  # Path to DFT reference data
    n_jobs: int = zntrack.params(-1)
    check_completed: bool = zntrack.params(False)
    threading: bool = zntrack.params(False)

    # Job array support
    array_chunk_size: int = zntrack.params(None)  # Materials per array job (None = process all)
    array_task_id: int = zntrack.params(None)     # SLURM_ARRAY_TASK_ID (None = process all)

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
    def process_mp_id(mp_id: str, model, nwd, yaml_dir, dft_ref_path, fmax, q_mesh, q_mesh_thermal, temperatures, check_completed, dft_ref_cache=None):
        try:


            #calc = model.get_calculator()
            yaml_path = yaml_dir/ f"{mp_id}.yaml"
            
            
                    
            phonons_pred = load_phonopy(str(yaml_path))
            chemical_formula = get_chemical_formula(phonons_pred, empirical=True)
            
            
            # save paths for later
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
            
            calc = model.get_calculator()
            
            atoms = phonopy2aseatoms(phonons_pred) # primtive = True?
            atoms_sym = atoms.copy()
        
            atoms_sym.calc = calc
            #atoms_sym.set_constraint(FixSymmetry(atoms_sym))
            opt = FIRE(FrechetCellFilter(atoms_sym))
            opt.run(fmax=fmax, steps=1000)


            

            # primitive matrix not always available in reference data e.g. mp-30056
            try:
                if "primitive_matrix" in atoms_sym.info.keys():
                    primitive_matrix = atoms_sym.info["primitive_matrix"]
                else:
                    # calculate primitive matrix from unitcell and primitive cell
                    unitcell = phonons_pred.unitcell
                    primitive_cell = phonons_pred.primitive
                    primitive_matrix = np.linalg.inv(np.array(unitcell.get_cell())) @ np.array(primitive_cell.get_cell())
                    print("Primitive matrix not found in atoms.info. Calculated primitive matrix: ", primitive_matrix)
                    print("unit cell lattice: ", unitcell.get_cell())
                    print("primitive cell lattice: ", primitive_cell.get_cell())

                
                phonons_pred = init_phonopy_from_ref(
                    atoms=atoms_sym,
                    fc2_supercell=atoms_sym.info["fc2_supercell"],
                    primitive_matrix=primitive_matrix,
                    displacement_dataset=None,
                    displacement_distance=0.01,
                    symprec=1e-5,
                )
            # except error: Remapping of atoms by TrimmedCell failed.
            except Exception as e2:
                print(f"Warning: Failed to initialize phonopy with primitive matrix for {mp_id} as symmetry was broken during relaxation. error: {e2}")
                print("Generating new primitive cell")

                # Reset lattice representation
                # avoids trimmed cell issues for a small number of materials
                #cell_std, _ = atoms_sym.cell.standard_form()
                #atoms_sym.set_cell(cell_std, scale_atoms=True)

                # spglib standardised cell, no ideal, is primitive = false
                #atoms_sym = PhononAllBatchMeta.spglib_standardize_ase(atoms_sym, to_primitive=False, no_idealize=True)


                # primitive_matrix_new = PhononAllBatchMeta.primitive_matrix_from_relaxed_atoms(
                #     atoms_sym,
                #     symprec=1e-5,
                # )
                # print("primitive_matrix_new:\n", primitive_matrix_new)

                phonons_pred = init_phonopy_from_ref(
                    atoms=atoms_sym,
                    fc2_supercell=atoms_sym.info["fc2_supercell"],
                    primitive_matrix=None,
                    displacement_dataset=None,
                    displacement_distance=0.01,
                    symprec=1e-5,
                )

            phonons_pred, fc2, freqs = get_fc2_and_freqs(
                phonons=phonons_pred,
                calculator=calc,
                q_mesh=np.array([q_mesh] * 3),
                symmetrize_fc2=False
            )
            
            # qpoints = get_commensurate_points(phonons_pred.supercell_matrix)
            # frequencies = np.stack([phonons_pred.get_frequencies(q) for q in qpoints])

            #phonon_obj_path = nwd / f"phonon_obj/{mp_id}_phonon_obj.yaml"
            #Path(phonon_obj_path).parent.mkdir(parents=True, exist_ok=True)
            #phonons_pred.save(filename=str(phonon_obj_path)) #, settings={"force_constants": True})

            # Load DFT reference q-points to ensure consistent band structure paths
            if dft_ref_cache and mp_id in dft_ref_cache:
                # Use pre-loaded cached data (faster)
                qpoints = dft_ref_cache[mp_id]["qpoints"]
                labels = dft_ref_cache[mp_id]["labels"]
                connections = dft_ref_cache[mp_id]["connections"]

                phonons_pred.run_band_structure(
                    paths=qpoints,
                    labels=labels,
                    path_connections=connections,
                )
            elif dft_ref_path:
                # Fallback: load from disk
                dft_ref_base = Path(dft_ref_path)
                ref_qpoints_path = dft_ref_base / f"{mp_id}_qpoints.npz"
                ref_labels_path = dft_ref_base / f"{mp_id}_labels.json"
                ref_connections_path = dft_ref_base / f"{mp_id}_connections.json"

                if ref_qpoints_path.exists() and ref_labels_path.exists() and ref_connections_path.exists():
                    # Load saved DFT q-points
                    with open(ref_qpoints_path, "rb") as f:
                        qpoints = pickle.load(f)
                    with open(ref_labels_path, "r") as f:
                        labels = json.load(f)
                    with open(ref_connections_path, "r") as f:
                        connections = json.load(f)

                    # Use DFT's q-points to ensure consistent comparison
                    phonons_pred.run_band_structure(
                        paths=qpoints,
                        labels=labels,
                        path_connections=connections,
                    )
                else:
                    print(f"Warning: DFT reference q-points not found for {mp_id}, using auto_band_structure()")
                    phonons_pred.auto_band_structure()
            else:
                # No DFT reference data provided
                phonons_pred.auto_band_structure()

            band_structure_pred = phonons_pred.get_band_structure_dict()
            phonons_pred.auto_total_dos()
            dos_pred = phonons_pred.get_total_dos_dict()
            

            
            with open(phonon_pred_band_structure_path, "wb") as f:
                pickle.dump(band_structure_pred, f)
            with open(phonon_pred_dos_path, "wb") as f:
                pickle.dump(dos_pred, f)

            
            

                
            # with open(phonon_pred_path / f"{mp_id}_qpoints.npz", "wb") as f:
            #     pickle.dump(qpoints, f)
            # with open(phonon_pred_path / f"{mp_id}_labels.json", "w") as f:
            #     json.dump(labels, f)
            # with open(phonon_pred_path / f"{mp_id}_connections.json", "w") as f:
            #     json.dump(connections, f)
                                
            phonons_pred.run_mesh([q_mesh_thermal] * 3)
            phonons_pred.run_thermal_properties(
                temperatures=temperatures,
                #cutoff_frequency=0.05
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
            with open("error_log.txt", "a") as f:
                f.write(f"\nError while processing {mp_id}:\n")
                traceback.print_exc(file=f)

    @staticmethod
    def primitive_matrix_from_relaxed_atoms(atoms, symprec=1e-5):
        """
        Build a primitive_matrix consistent with the *current* (relaxed) atoms
        using spglib's primitive_lattice.
        """
        lattice = np.array(atoms.cell, dtype=float)
        positions = atoms.get_scaled_positions(wrap=True)  # fractional
        numbers = atoms.numbers

        cell = (lattice, positions, numbers)
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            raise RuntimeError("spglib.get_symmetry_dataset returned None (try larger symprec).")

        prim_lat = np.array(dataset["primitive_lattice"], dtype=float)

        # Phonopy-style primitive_matrix: unitcell -> primitive axes
        prim_mat = np.linalg.inv(lattice) @ prim_lat
        return prim_mat

    import numpy as np
    import spglib
    @staticmethod
    def spglib_standardize_ase(atoms, symprec=1e-3, to_primitive=False, no_idealize=True):
        """
        Standardize an ASE Atoms using spglib. Keeps *this* structure but returns
        a symmetry-consistent conventional (or primitive) representation.
        """
        lattice = np.array(atoms.cell, dtype=float)
        positions = atoms.get_scaled_positions(wrap=True)  # fractional
        numbers = atoms.numbers

        cell = (lattice, positions, numbers)

        std = spglib.standardize_cell(
            cell,
            to_primitive=to_primitive,   # False -> conventional, True -> primitive
            no_idealize=no_idealize,           # allow slight idealization (often helps mapping)
            symprec=symprec,
        )

        if std is None:
            raise RuntimeError("spglib.standardize_cell returned None (symmetry not found / too strict symprec).")

        lat_std, pos_std, nums_std = std

        atoms_std = atoms.copy()
        atoms_std.set_cell(lat_std, scale_atoms=False)
        atoms_std.set_scaled_positions(pos_std)
        atoms_std.numbers = np.array(nums_std, dtype=int)
        atoms_std.wrap()

        return atoms_std


    def run(self):
        #calc = self.model.get_calculator()

        yaml_dir = Path(self.phonopy_yaml_dir)
        nwd = Path(self.nwd)
        dft_ref_path = self.dft_ref_path
        fmax = self.fmax

        q_mesh = self.N_q_mesh
        q_mesh_thermal = 20
        temperatures = self.thermal_properties_temperatures
        calc_model = self.model  # Materialize model before parallel loops

        # Handle job array chunking
        mp_ids_to_process = self.mp_ids
        if self.array_chunk_size is not None and self.array_task_id is not None:
            # Split mp_ids into chunks for job array
            start_idx = self.array_task_id * self.array_chunk_size
            end_idx = min(start_idx + self.array_chunk_size, len(self.mp_ids))
            mp_ids_to_process = self.mp_ids[start_idx:end_idx]
            print(f"Job array mode: Task {self.array_task_id} processing materials {start_idx} to {end_idx-1} ({len(mp_ids_to_process)} materials)")
        else:
            print(f"Processing all {len(mp_ids_to_process)} materials")

        # Pre-load DFT reference data only when threading (multiprocessing uses on-demand loading)
        dft_ref_cache = {}
        if self.threading and dft_ref_path:
            print("Pre-loading DFT reference data...")
            dft_ref_base = Path(dft_ref_path)
            for mp_id in mp_ids_to_process:
                ref_qpoints_path = dft_ref_base / f"{mp_id}_qpoints.npz"
                ref_labels_path = dft_ref_base / f"{mp_id}_labels.json"
                ref_connections_path = dft_ref_base / f"{mp_id}_connections.json"

                if ref_qpoints_path.exists() and ref_labels_path.exists() and ref_connections_path.exists():
                    with open(ref_qpoints_path, "rb") as f:
                        qpoints = pickle.load(f)
                    with open(ref_labels_path, "r") as f:
                        labels = json.load(f)
                    with open(ref_connections_path, "r") as f:
                        connections = json.load(f)

                    dft_ref_cache[mp_id] = {
                        "qpoints": qpoints,
                        "labels": labels,
                        "connections": connections
                    }
            print(f"Loaded DFT reference data for {len(dft_ref_cache)} materials.")
        elif not self.threading and dft_ref_path:
            print("Multiprocessing mode: using on-demand disk loading (no cache pre-load).")
        
        

        # # Run jobs in parallel
        # successful_results = []
        # failed_hard = []

        # try:
        #     if self.threading:
        #         parallel_backend_mode = "threading"
        #         print("Using threading for parallel processing.")
        #     else:
        #         parallel_backend_mode = "multiprocessing"
        #         print("Using multiprocessing for parallel processing.")
                
        #     with parallel_backend(parallel_backend_mode):
        #         raw_results = Parallel(n_jobs=self.n_jobs)(
        #             delayed(process_mp_id)(mp_id, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures)
        #             for mp_id in self.mp_ids
        #         )

        #     # Wrap result with (mp_id, result) so we know which ones succeeded
        #     successful_results = [(res["mp_id"], res) for res in raw_results if res is not None]

        # except Exception as e:
        #     if "terminated" in str(e).lower():
        #         print("Detected possible worker termination (e.g. OOM). Retrying serially to identify faulty materials.")
        #     else:
        #         raise  # unknown error, re-raise

        #     # Get mp-ids that already succeeded
        #     processed_mp_ids = {mp_id for mp_id, _ in successful_results}

        #     # Retry only unprocessed ones
        #     for mp_id in self.mp_ids:
        #         if mp_id in processed_mp_ids:
        #             continue
        #         try:
        #             res = process_mp_id(mp_id, self.model, nwd, yaml_dir, fmax, q_mesh, q_mesh_thermal, temperatures)
        #             if res is not None:
        #                 successful_results.append((res["mp_id"], res))
        #         except Exception as err:
        #             print(f"Serial run failed for {mp_id}: {err}")
        #             traceback.print_exc()
        #             failed_hard.append(mp_id)

        # # Unpack the final result
        # results = [res for _, res in successful_results]

        # print(f"\nFinished with {len(results)} successful results.")
        # if failed_hard:
        #     print(f"{len(failed_hard)} materials failed due to memory or hard crashes.")
        #     print("Failed mp-ids:", failed_hard)





        if self.threading:
            # threading can share calc instead of having multiple instances, prevent OOM
            # Threading also shares the cache naturally via closure
            def handle(mp_id):
                return PhononAllBatchMeta.process_mp_id(
                    mp_id, calc_model, nwd, yaml_dir, dft_ref_path, fmax,
                    q_mesh, q_mesh_thermal, temperatures, self.check_completed, dft_ref_cache
                )
            with parallel_backend("threading", n_jobs=self.n_jobs):
                results = Parallel()(delayed(handle)(mp_id) for mp_id in mp_ids_to_process)
        else:
            # Multiprocessing: use wrapper to avoid cache serialization
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_process_single_mp_id_meta)(
                    mp_id, calc_model, nwd, yaml_dir, dft_ref_path, fmax,
                    q_mesh, q_mesh_thermal, temperatures, self.check_completed
                ) for mp_id in mp_ids_to_process
            )

        results = [res for res in results if res is not None]

        print(f"\nFinished with {len(results)} successful results.")

        phonon_band_path_dict = {res["mp_id"]: str(res["phonon_band_path_dict"]) for res in results}
        phonon_dos_path_dict = {res["mp_id"]: str(res["phonon_dos_dict"]) for res in results}
        thermal_properties_path_dict = {res["mp_id"]: str(res["thermal_properties_dict"]) for res in results}
        chemical_formula_dict = {res["mp_id"]: res["formula"] for res in results}

        

        
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
    def benchmark_interactive(
        pred_node_dict: dict[str, "PhononAllBatch"],
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
            no_plots=no_plots,
        )
        
        

