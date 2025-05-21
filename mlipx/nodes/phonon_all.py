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


from mlipx.phonons_utils import *
from phonopy import load as load_phonopy

class PhononAllBatch(zntrack.Node):
    """Batch phonon calculations (FC2 + thermal props) for multiple mp-ids."""

    mp_ids: list[str] = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    phonopy_yaml_dir: str = zntrack.params()

    N_q_mesh: int = zntrack.params(20)
    supercell: int = zntrack.params(3)
    fmax: float = zntrack.params(0.0001)
    thermal_properties_temperatures: list[float] = zntrack.params(
        default_factory=lambda: [0, 75, 150, 300, 600]
    )

    force_constants_paths: dict = zntrack.outs()
    thermal_properties_paths: dict = zntrack.outs()

    def run(self):
        calc = self.model.get_calculator()
        self.force_constants_paths = {}
        self.thermal_properties_paths = {}

        for mp_id in tqdm(self.mp_ids, desc="PhononAllBatch"):
            try:
                print(f"\nProcessing {mp_id}...")

                yaml_path = Path(self.phonopy_yaml_dir )/ f"{mp_id}.yaml"
                phonons = load_phonopy(str(yaml_path))
                displacement_dataset = phonons.dataset
                atoms = phonopy2aseatoms(phonons)
                atoms.calc = calc

                #ecf = FrechetCellFilter(atoms)
                #opt = FIRE(ecf)
                #opt.run(fmax=self.fmax, steps=1000)

                primitive_matrix = atoms.info.get("primitive_matrix", "auto")
                if primitive_matrix == "auto":
                    print("Primitive matrix not found. Using 'auto'.")

                phonons = init_phonopy_from_ref(
                    atoms=atoms,
                    fc2_supercell=atoms.info["fc2_supercell"],
                    primitive_matrix=primitive_matrix,
                    displacement_dataset=displacement_dataset,
                    symprec=1e-5,
                )

                phonons, _, _ = get_fc2_and_freqs(
                    phonons=phonons,
                    calculator=calc,
                    q_mesh=np.array([self.N_q_mesh] * 3),
                    symmetrize_fc2=True
                )

                force_constants_path = self.nwd / f"{mp_id}_force_constants.yaml"
                thermal_props_path = self.nwd / f"{mp_id}_thermal_properties.json"
                phonons.save(filename=force_constants_path, settings={"force_constants": True})
                self.force_constants_paths[mp_id] = force_constants_path

                phonons.run_mesh([self.N_q_mesh] * 3)
                phonons.run_thermal_properties(
                    temperatures=self.thermal_properties_temperatures,
                    cutoff_frequency=0.05
                )

                thermal_props = phonons.get_thermal_properties_dict()
                thermal_props_clean = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in thermal_props.items()
                }

                with open(thermal_props_path, "w") as f:
                    json.dump(thermal_props_clean, f)
                self.thermal_properties_paths[mp_id] = thermal_props_path

            except Exception as e:
                print(f"Skipping {mp_id} due to error: {e}")