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



class BuildASEcrystal(zntrack.Node):
    """Generate a bulk material structure.

    Parameters
    ----------
    element: str
        The chemical symbol of the element.
    lattice_type: str
        e.g., "bcc", "fcc", "hcp"
    supercell: tuple[int, int, int]
        The supercell size.
    a: float
        The lattice constant.
    c: float, optional
        The c/a ratio for hexagonal structures.

    Example
    -------
    >>> Crystal(element="W", lattice_type="bcc", a=3.16)

    """

    element: str = zntrack.params()
    lattice_type: str = zntrack.params()
    supercell: tuple[int, int, int] = zntrack.params(None)
    a: float = zntrack.params()
    c: float = zntrack.params(default=None)  # Only needed for hcp
    
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")
    
    def run(self):
        if self.lattice_type == "hcp":
            if self.c is None:
                raise ValueError("hcp structure requires a c/a ratio (c).")
            atoms = bulk(self.element, self.lattice_type, a=self.a, c=self.c)
        
        if self.lattice_type == "2H-SiC":
            cell = [[self.a, 0, 0], [-self.a/2, self.a * 3**0.5 / 2, 0], [0, 0, self.c]]
            scaled_positions = [(1/3, 2/3, 0.0),(2/3, 1/3, 0.5),(1/3, 2/3, 0.25),(2/3, 1/3, 0.75),]
            symbols = ['C', 'C', 'Si', 'Si']
            atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
        else:
            atoms = bulk(self.element, self.lattice_type, a=self.a, c=self.c)

        atoms.info["lattice_type"] = self.lattice_type
        ase.io.write(self.frames_path, atoms)
    

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))


