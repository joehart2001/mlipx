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

import warnings
from pathlib import Path
from typing import Any, Callable
from ase.calculators.calculator import Calculator
from tqdm import tqdm
from phonopy.api_phonopy import Phonopy
import yaml


from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator


from mlipx.phonons_utils import get_fc2_and_freqs, init_phonopy
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
    data: list[ase.Atoms] = zntrack.deps()
    model: NodeWithCalculator = zntrack.deps()
    
    # parameters
    material_idx: int = zntrack.params(0)
    data_type: str = zntrack.params("MP")
    N_q_mesh: int = zntrack.params(20)
    supercell: int = zntrack.params(3)
    delta: float = zntrack.params(0.05) # displacement for finite difference calculation
    fmax: float = zntrack.params(0.0001) # max force on atoms for relaxation
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files
    force_constants_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "force_constants.yaml")
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")



    def run(self):
        atoms = self.data[0]

        calc = self.model.get_calculator()
        print(atoms)
        frames = []
        
        self.frames_path.parent.mkdir(exist_ok=True)
    
        atoms.calc = calc
        
        # initialize Phonopy with displacements
        phonons = init_phonopy(
            atoms=atoms,
            fc2_supercell=np.diag([self.supercell] * 3),
            primitive_matrix="auto",
            displacement_distance=self.delta,
            symprec=1e-5
        )
        print("Phonopy initialized")

        # compute FC2 (2nd order force constants, 2nd dervative of energy) and phonon frequencies on mesh
        phonons, _, _ = get_fc2_and_freqs(
            phonons=phonons,
            calculator=calc,
            q_mesh=np.array([self.N_q_mesh] * 3),
            symmetrize_fc2=True
        )
        print("Force constants computed")
    
    
        phonons.save(filename=self.force_constants_path, settings={"force_constants": True})
        print(f"Force constants saved to: {self.force_constants_path}")
        
        print("Forces shape:", atoms.get_forces().shape)
        print("Number of atoms:", len(atoms))

        #ase.io.write(self.frames_path, atoms, append=True) use this for a list of atoms
        ase.io.write(self.frames_path, atoms)
        print(f"Relaxed structure saved to: {self.frames_path}")



    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))

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
    

