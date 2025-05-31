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

    #force_constants_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "force_constants_paths.json")
    phonon_obj_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_obj_paths.json")
    thermal_properties_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "thermal_properties_paths.json")

    # def run(self):
    #     calc = self.model.get_calculator()
    #     force_constants_path_dict = {}
    #     thermal_properties_path_dict = {}

    #     for mp_id in tqdm(self.mp_ids, desc="PhononAllBatch"):
    #         try:
    #             print(f"\nProcessing {mp_id}...")

    #             yaml_path = Path(self.phonopy_yaml_dir )/ f"{mp_id}.yaml"
    #             phonons = load_phonopy(str(yaml_path))
    #             displacement_dataset = phonons.dataset
    #             atoms = phonopy2aseatoms(phonons)
                
    #             atoms_sym = atoms.copy()
    #             atoms_sym.calc = calc
    #             atoms_sym.set_constraint(FixSymmetry(atoms_sym))
    #             opt = LBFGS(atoms_sym)
    #             opt.run(fmax=0.005, steps=1000)
                
    #             # primitive matrix not always available in reference data e.g. mp-30056
    #             if "primitive_matrix" in atoms.info.keys():
    #                 primitive_matrix = atoms.info["primitive_matrix"]
    #             else:
    #                 primitive_matrix = "auto"
    #                 print("Primitive matrix not found in atoms.info. Using 'auto' for primitive matrix.")
            

    #             phonons = init_phonopy_from_ref(
    #                 atoms=atoms_sym,
    #                 fc2_supercell=atoms.info["fc2_supercell"],
    #                 primitive_matrix=primitive_matrix,
    #                 displacement_dataset=displacement_dataset,
    #                 symprec=1e-5,
    #             )

    #             phonons, _, _ = get_fc2_and_freqs(
    #                 phonons=phonons,
    #                 calculator=calc,
    #                 q_mesh=np.array([self.N_q_mesh] * 3),
    #                 symmetrize_fc2=True
    #             )

    #             force_constants_path = self.nwd / f"{mp_id}_force_constants.yaml"
    #             thermal_props_path = self.nwd / f"{mp_id}_thermal_properties.json"
    #             phonons.save(filename=force_constants_path, settings={"force_constants": True})
    #             force_constants_path_dict[mp_id] = str(force_constants_path)
                
    #             #phonons.save(filename=self.phonon_obj_path, settings={"force_constants": True})

    #             phonons.run_mesh([self.N_q_mesh] * 3)
    #             phonons.run_thermal_properties(
    #                 temperatures=self.thermal_properties_temperatures,
    #                 cutoff_frequency=0.05
    #             )

    #             thermal_properties_dict = phonons.get_thermal_properties_dict()
    #             thermal_properties_dict_safe = {
    #                 key: value.tolist() if isinstance(value, np.ndarray) else value
    #                 for key, value in thermal_properties_dict.items()
    #             }

    #             # with open(thermal_props_path, "w") as f:
    #             #     json.dump(thermal_props_clean, f)
    #             # thermal_properties_path_dict[mp_id] = thermal_props_path

    #         except Exception as e:
    #             print(f"Skipping {mp_id} due to error: {e}")
                
        
    #     # with open(self.force_constants_paths, "w") as f:
    #     #     json.dump(force_constants_path_dict, f, indent=4)
    #     # with open(self.thermal_properties_paths, "w") as f:
    #     #     json.dump(thermal_properties_path_dict, f, indent=4)


    def run(self):
        calc = self.model.get_calculator()
        
        phonon_obj_paths_dict = {}
        thermal_properties_path_dict = {}
        
        yaml_dir = Path(self.phonopy_yaml_dir)
        nwd = Path(self.nwd)
        fmax = self.fmax
        
        q_mesh = self.N_q_mesh
        q_mesh_thermal = 20
        temperatures = self.thermal_properties_temperatures  # ✅ resolve this too
        

        def process_mp_id(mp_id: str, calc, nwd, yaml_dir, fmax, q_mesh, temperatures):
            try:
                print(f"\nProcessing {mp_id}...")
                
                yaml_path = yaml_dir/ f"{mp_id}.yaml"
                phonons = load_phonopy(str(yaml_path))
                displacement_dataset = phonons.dataset
                atoms = phonopy2aseatoms(phonons)
                
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
            
                phonons = init_phonopy_from_ref(
                    atoms=atoms_sym,
                    fc2_supercell=atoms.info["fc2_supercell"],
                    primitive_matrix=primitive_matrix,
                    displacement_dataset=displacement_dataset,
                    symprec=1e-5,
                )

                phonons, _, _ = get_fc2_and_freqs(
                    phonons=phonons,
                    calculator=calc,
                    q_mesh=np.array([q_mesh] * 3),
                    symmetrize_fc2=True
                )

                phonon_obj_path = nwd / f"phonon_obj/{mp_id}_phonon_obj.yaml"
                Path(phonon_obj_path).parent.mkdir(parents=True, exist_ok=True)
                #phonon_obj_paths_dict[mp_id] = str(phonon_obj_path)
                #force_constants_path = nwd / f"{mp_id}_force_constants.yaml"
                #force_constants_path = self.nwd / f"{mp_id}_force_constants.yaml"
                phonons.save(filename=str(phonon_obj_path)) #, settings={"force_constants": True})

                phonons.run_mesh([q_mesh] * 3) #TODO 20x20x20
                phonons.run_thermal_properties(
                    temperatures=temperatures,
                    cutoff_frequency=0.05
                )

                thermal_dict = phonons.get_thermal_properties_dict()
                thermal_dict_safe = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in thermal_dict.items()
                }
                thermal_path = nwd / f"thermal_properties/{mp_id}_thermal_properties.json"
                Path(thermal_path).parent.mkdir(parents=True, exist_ok=True)
                with open(thermal_path, "w") as f:
                    json.dump(thermal_dict_safe, f, indent=4)

                return {
                    "mp_id": mp_id,
                    "phonon_obj_path_dict": str(phonon_obj_path),
                    "thermal_properties_dict": thermal_dict_safe
                }

            except Exception as e:
                print(f"Skipping {mp_id} due to error: {e}")
                return None

        # Run jobs in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_mp_id)(mp_id, calc, nwd, yaml_dir, fmax, q_mesh, temperatures)
            for mp_id in self.mp_ids
        )

        # Optional: collect paths if needed later
        phonon_obj_path_dict = {
            res["mp_id"]: res["phonon_obj_path_dict"]
            for res in results if res is not None
        }
        thermal_properties_path_dict = {
            res["mp_id"]: res["thermal_properties_dict"]
            for res in results if res is not None
        }
        
        # Save paths to JSON files
        with open(self.phonon_obj_paths, "w") as f:
            json.dump(phonon_obj_path_dict, f, indent=4)
        with open(self.thermal_properties_paths, "w") as f:
            json.dump(thermal_properties_path_dict, f, indent=4)
            
    
    @property
    def phonon_obj_path_dict(self) -> dict[str, str]:
        """Returns a dictionary of mp_id to phonon object paths."""
        with open(self.phonon_obj_paths, "r") as f:
            return json.load(f)
    
    @property
    def thermal_properties_path_dict(self) -> dict[str, dict[str, Any]]:
        """Returns a dictionary of mp_id to thermal properties."""
        with open(self.thermal_properties_paths, "r") as f:
            return json.load(f)