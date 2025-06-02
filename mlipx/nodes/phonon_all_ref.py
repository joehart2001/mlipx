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



class PhononAllRef(zntrack.Node):
    """Batch phonon calculations (FC2 + thermal props) for multiple mp-ids."""

    mp_ids: list[str] = zntrack.params()
    phonopy_yaml_dir: str = zntrack.params()
    n_jobs: int = zntrack.params(-1)


    phonon_band_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_band_paths.json")
    phonon_dos_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_dos_paths.json")
    phonon_qpoints_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_qpaths_paths.json")
    phonon_labels_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_labels_paths.json")
    phonon_connections_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_connections_paths.json")
    thermal_properties_paths: pathlib.Path = zntrack.outs_path(zntrack.nwd / "thermal_properties_paths.json")
    get_chemical_formula_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mp_ids_and_formulas.json")
    
    def run(self):
        #calc = self.model.get_calculator()
        
        yaml_dir = Path(self.phonopy_yaml_dir)
        nwd = Path(self.nwd)
        

        def process_mp_id(mp_id, nwd, yaml_dir):
            try:
                print(f"\nProcessing ref {mp_id}...")
                yaml_path = yaml_dir/ f"{mp_id}.yaml"
                
                
                # REF
                phonons_ref = load_phonopy(str(yaml_path))
                
                qpoints, labels, connections = get_band_qpoints_by_seekpath(
                    phonons_ref.primitive, npoints=101, is_const_interval=True
                )
                phonons_ref.run_band_structure(
                    paths=qpoints,
                    labels=labels,
                    path_connections=connections,
                )
                #phonons_ref.auto_band_structure()
                band_structure_ref = phonons_ref.get_band_structure_dict()
                phonons_ref.auto_total_dos()
                dos_ref = phonons_ref.get_total_dos_dict()
                
                
                with open(yaml_path) as f:
                    thermal_properties = yaml.safe_load(f)
            
                thermal_properties_dict = {
                    "temperatures": [],
                    "free_energy": [],
                    "entropy": [],
                    "heat_capacity": []
                }
                
                for temp, free_energy, entropy, heat_capacity in zip(
                    thermal_properties["temperatures"],
                    thermal_properties["free_e"],
                    thermal_properties["entropy"],
                    thermal_properties["heat_capacity"]
                ):
                    thermal_properties_dict["temperatures"].append(temp)
                    thermal_properties_dict["free_energy"].append(free_energy)
                    thermal_properties_dict["entropy"].append(entropy)
                    thermal_properties_dict["heat_capacity"].append(heat_capacity)
                    
                    
            
            
                
                phonon_ref_path = nwd / f"phonon_ref_data/"
                phonon_ref_path.mkdir(parents=True, exist_ok=True)
                
                phonon_ref_band_structure_path = phonon_ref_path / f"{mp_id}_band_structure.npz"
                phonon_ref_dos_path = phonon_ref_path / f"{mp_id}_dos.npz"
                thermal_path = phonon_ref_path / f"{mp_id}_thermal_properties.json"
                
                # chemical formula from yaml for plotting later
                chemical_formula = get_chemical_formula(phonons_ref, empirical=True)
                # with open(phonon_ref_path / f"{mp_id}_chemical_formula.txt", "w") as f:
                #     f.write(chemical_formula)
                
                with open(phonon_ref_band_structure_path, "wb") as f:
                    pickle.dump(band_structure_ref, f)
                with open(phonon_ref_dos_path, "wb") as f:
                    pickle.dump(dos_ref, f)
                with open(thermal_path, "w") as f:
                    json.dump(thermal_properties_dict, f)
                    
                with open(phonon_ref_path / f"{mp_id}_qpoints.npz", "wb") as f:
                    pickle.dump(qpoints, f)
                with open(phonon_ref_path / f"{mp_id}_labels.json", "w") as f:
                    json.dump(labels, f)
                with open(phonon_ref_path / f"{mp_id}_connections.json", "w") as f:
                    json.dump(connections, f)
                    
                return {
                    "mp_id": mp_id,
                    "phonon_band_path_dict": str(phonon_ref_band_structure_path),
                    "phonon_dos_dict": phonon_ref_dos_path,
                    "thermal_properties_dict": str(thermal_path),
                    "phonon_qpoints_dict": str(phonon_ref_path / f"{mp_id}_qpoints.npz"),
                    "phonon_labels_dict": str(phonon_ref_path / f"{mp_id}_labels.json"),
                    "phonon_connections_dict": str(phonon_ref_path / f"{mp_id}_connections.json"),
                    "formula": chemical_formula,
                }

            except Exception as e:
                print(f"Skipping {mp_id} due to error: {e}")
                return None

        # Run jobs in parallel
        # results = Parallel(n_jobs=n_jobs)(
        #     delayed(process_mp_id)(mp_id, nwd, yaml_dir)
        #     for mp_id in self.mp_ids
        # )

        from math import ceil

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        batch_size = 1000  # or adjust based on your available memory
        all_results = []

        for i, mp_batch in enumerate(chunks(self.mp_ids, batch_size)):
            print(f"\nProcessing batch {i+1}/{ceil(len(self.mp_ids)/batch_size)}...")
            results = Parallel(n_jobs=self.n_jobs)(  
            delayed(process_mp_id)(mp_id, nwd, yaml_dir)
            for mp_id in tqdm(mp_batch, desc="Processing mp-ids")
                for mp_id in mp_batch
            )
            all_results.extend(results)
            
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
        phonon_qpoints_path_dict = {
            res["mp_id"]: str(res["phonon_qpoints_dict"])
            for res in results if res is not None
        }
        phonon_labels_path_dict = {
            res["mp_id"]: str(res["phonon_labels_dict"])
            for res in results if res is not None
        }
        phonon_connections_path_dict = {
            res["mp_id"]: str(res["phonon_connections_dict"])
            for res in results if res is not None
        }
        chemical_formula_dict = {
            res["mp_id"]: str(res["formula"])
            for res in results if res is not None
        }

        
        # Save paths to JSON files
        with open(self.phonon_band_paths, "w") as f:
            json.dump(phonon_band_path_dict, f, indent=4)
        with open(self.phonon_dos_paths, "w") as f:
            json.dump(phonon_dos_path_dict, f, indent=4)
        with open(self.thermal_properties_paths, "w") as f:
            json.dump(thermal_properties_path_dict, f, indent=4)
        with open(self.phonon_qpoints_paths, "w") as f:
            json.dump(phonon_qpoints_path_dict, f, indent=4)
        with open(self.phonon_labels_paths, "w") as f:
            json.dump(phonon_labels_path_dict, f, indent=4)
        with open(self.phonon_connections_paths, "w") as f:
            json.dump(phonon_connections_path_dict, f, indent=4)

        # Write all mp_ids and their chemical formulas
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
    def get_phonon_qpoints_paths(self) -> dict[str, str]:
        """Returns a dictionary of mp_id to phonon qpoints paths."""
        with open(self.phonon_qpoints_paths, "r") as f:
            return json.load(f)
    @property
    def get_phonon_labels_paths(self) -> dict[str, str]:
        """Returns a dictionary of mp_id to phonon labels paths."""
        with open(self.phonon_labels_paths, "r") as f:
            return json.load(f)
    @property
    def get_phonon_connections_paths(self) -> dict[str, str]:
        """Returns a dictionary of mp_id to phonon connections paths."""
        with open(self.phonon_connections_paths, "r") as f:
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
        qpoints_paths = self.get_phonon_qpoints_paths
        labels_paths = self.get_phonon_labels_paths
        connections_paths = self.get_phonon_connections_paths
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
                    "qpoints": load_pickle(qpoints_paths[mp_id]),
                    "labels": load_json(labels_paths[mp_id]),
                    "connections": load_json(connections_paths[mp_id]),
                    "formula": chemical_formulas[mp_id],
                }
            except Exception as e:
                print(f"Skipping {mp_id} due to loading error: {e}")
                continue

        return result

