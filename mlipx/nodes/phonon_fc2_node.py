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


class PhononForceConstants(zntrack.Node):
    """Performs structure relaxation and Computes the phonon force constants of a given structure.
    
    Parameters
    ----------
    data: list[ase.Atoms]
        List of ASE Atoms objects.
    model: NodeWithCalculator
        Model node with calculator for phonon force constants calculation.
    """
    
    # inputs
    #data: ase.Atoms = zntrack.deps() # input dependency
    #data: list[ase.Atoms] = zntrack.deps()
    
    # can either be a phonopy.yaml file or a list of (geometry optimised) atoms
    data: Union[pathlib.Path, list[Atoms]] = zntrack.deps()
    model: NodeWithCalculator = zntrack.deps()
    
    # parameters
    material_idx: int = zntrack.params(0)
    N_q_mesh: int = zntrack.params(6)
    supercell: int = zntrack.params(3)
    delta: float = zntrack.params(0.05) # displacement for finite difference calculation
    fmax: float = zntrack.params(0.0001) # max force on atoms for relaxation
    thermal_properties_temperatures: t.Optional[t.List[float]] = zntrack.params(
        default_factory=lambda: [0, 75, 150, 300, 600]
    )
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files
    force_constants_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "force_constants.yaml")
    thermal_properties_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "thermal_properties.json")



    def run(self):
        
        calc = self.model.get_calculator()
        
        # provided data is a list of atoms
        if isinstance(self.data, list):
            atoms = self.data[self.material_idx]
                    
            atoms.calc = calc
            
            # relax the structure
            ecf = FrechetCellFilter(atoms)
            opt = FIRE(ecf)
            opt.run(fmax=0.005, steps=1000)
            
            # initialize Phonopy with displacements
            phonons = init_phonopy(
                atoms=atoms,
                fc2_supercell=np.diag([self.supercell] * 3),
                primitive_matrix="auto",
                displacement_distance=self.delta,
                symprec=1e-5
            )
            print("Phonopy initialized")
            
            
            
        # provided data is a reference phonopy.yaml file
        elif isinstance(self.data, pathlib.Path):
            phonons = load_phonopy(str(self.data))
            displacement_dataset = phonons.dataset
            atoms = phonopy2aseatoms(phonons)
            atoms.calc = calc
            
            atoms_sym = atoms.copy()
            atoms_sym.set_constraint(FixSymmetry(atoms_sym))
            opt = FIRE(atoms_sym)
            opt.run(fmax=0.005, steps=1000)
            
            # primitive matrix not always available in reference data e.g. mp-30056
            if "primitive_matrix" in atoms.info.keys():
                primitive_matrix = atoms.info["primitive_matrix"]
            else:
                primitive_matrix = "auto"
                print("Primitive matrix not found in atoms.info. Using 'auto' for primitive matrix.")
        
            phonons = init_phonopy_from_ref(
                atoms=atoms,
                fc2_supercell=atoms.info['fc2_supercell'],
                primitive_matrix=primitive_matrix,
                displacement_dataset=displacement_dataset,
                symprec=1e-5
            )
            
            print("Phonopy initialized from reference")

        else:
            raise TypeError(
                f"Unsupported `data` format: {type(self.data)}. Expected list[Atoms] or phonopy.yaml path."
            )


        # compute FC2 (2nd order force constants, 2nd dervative of energy) and phonon frequencies on mesh
        try:
            phonons, _, _ = get_fc2_and_freqs(
                phonons=phonons,
                calculator=calc,
                q_mesh=np.array([self.N_q_mesh] * 3),
                symmetrize_fc2=True
            )
        except ValueError as e:
            # Gracefully skip unsupported element e.g. for maceoff
            if "not in list" in str(e):
                print(f"Skipping material index {self.material_idx}: {e}")
                return
            else:
                raise
        print("Force constants computed")
    
    
        phonons.save(filename=self.force_constants_path, settings={"force_constants": True})
        print(f"Force constants saved to: {self.force_constants_path}")
        
        # thermal properties
        phonons.run_mesh([20, 20, 20])
        phonons.run_thermal_properties(
            temperatures=self.thermal_properties_temperatures,
            cutoff_frequency = 0.05 # cuttoff for negative frequencies
            )
        thermal_properties_dict = phonons.get_thermal_properties_dict()
        
        thermal_properties_dict_safe = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in thermal_properties_dict.items()
        }

        with open(self.thermal_properties_path, "w") as f:
            json.dump(thermal_properties_dict_safe, f)
        print(f"Thermal properties saved to: {self.thermal_properties_path}")
        
        #print("Forces shape:", atoms.get_forces().shape)
        #print("Number of atoms:", len(atoms))

        #ase.io.write(self.frames_path, atoms, append=True) use this for a list of atoms
        #ase.io.write(self.frames_path, atoms)
        #print(f"Relaxed structure saved to: {self.frames_path}")




    @property
    def phonons_obj_dict(self) -> dict:
        with self.force_constants_path.open("r") as f:
            return yaml.safe_load(f)
    
    @property
    def phonons_obj(self) -> np.ndarray:
        phonons = load_phonopy(self.force_constants_path)
        return phonons
    
    @property
    def fc2_path(self) -> str:
        return self.force_constants_path
    
    @property
    def get_thermal_properties_path(self) -> str:
        return self.thermal_properties_path