import pathlib
import typing as t
import json

import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative
import zntrack
from ase import Atoms, units
from ase.build import bulk
import subprocess
from mlipx.abc import ComparisonResults, NodeWithCalculator

import warnings
from pathlib import Path
from typing import Any, Callable
from ase.calculators.calculator import Calculator
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import glob
import re
import pandas as pd
from dash.exceptions import PreventUpdate
from dash import dash_table
import socket
import time
from typing import List, Dict, Any, Optional


import mlipx


import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class AtomisationEnergy(zntrack.Node):
    """ Node to compute atomisation energy of a bulk solid
    """
    # inputs
    # either input optimised structure or an element + type
    structure: List[mlipx.StructureOptimization] = zntrack.deps()
    
    formula: str = zntrack.params(None)
    lattice_type: str = zntrack.params(None)
    lattice_const: float = zntrack.params(None)

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files
    atomisation_e_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_const.json")

    
    
    def run(self):
        calc = self.model.get_calculator()
        
        if self.structure:
            from ase.formula import Formula

            structure = self.structure[-1]
            formula = structure.get_chemical_formula()
            formula_obj = Formula(formula)
            atom_counts = formula_obj.count() # e.g. {'Al': 2, 'O': 3}
            
            atom_tot_e = 0
            for element, count in atom_counts.items():
                atom = Atoms(element, pbc=False)
                atom.calc = calc
                atom_tot_e += atom.get_potential_energy() * count
            
            
            structure.calc = calc
            atomisation_e = (atom_tot_e - structure.get_potential_energy()) / len(structure)
            

        elif self.formula and self.lattice_type and self.lattice_const:
            from ase.formula import Formula

            if isinstance(self.lattice_const, dict):
                a = self.lattice_const["a"]
                c = self.lattice_const.get("c")
            elif isinstance(self.lattice_const, (list, tuple)):
                a = self.lattice_const[0]
                c = self.lattice_const[1] if len(self.lattice_const) > 1 else None
            else:
                a = self.lattice_const
                c = None

            bulk_structure = bulk(self.formula, self.lattice_type, a=a, c=c) if c else bulk(self.formula, self.lattice_type, a=a)
            bulk_structure.calc = calc

            formula_obj = Formula(self.formula)
            atom_counts = formula_obj.composition

            atom_tot_e = 0
            for element, count in atom_counts.items():
                atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=False)
                atom.calc = calc
                atom_tot_e += count * atom.get_potential_energy()

            bulk_e = bulk_structure.get_potential_energy()
            atomisation_e = atom_tot_e - bulk_e
            atomisation_e /= len(bulk_structure)  # Optional normalization

        else:
            raise ValueError("Either structure or formula, lattice_type and lattice_const must be provided.")

        with open(self.atomisation_e_output, 'w') as f:
            json.dump(atomisation_e, f)
            print(f"Saved atomisation energy to {self.atomisation_e_output}")
                
    
    @property
    def get_atomisation_e(self):
        with open(self.atomisation_e_output, 'r') as f:
            atomisation_e = json.load(f)
        return atomisation_e
    
    



    @staticmethod
    def mae_plot_interactive(
        node_dict,
        ref_node,
        ui: str | None = None,
        run_interactive: bool = True,
    ):
        pass
        
        