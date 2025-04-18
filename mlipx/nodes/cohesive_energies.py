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
import cctk
from ase.io.trajectory import Trajectory
from plotly.io import write_image
from ase.io import read

from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator


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
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class CohesiveEnergies(zntrack.Node):
    """Benchmark model against X23, S66 and DMC-ICE13
    """
    # inputs
    lattice_energy_dir: pathlib.Path = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    abs_error_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "abs_error.csv")
    lattice_e_ouptut: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_e.json")
    mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae.json")
    



    def run(self):
        
        calc = self.model.get_calculator()
        
        with open(self.lattice_energy_dir + "/list", "r") as f:
            systems = f.read().splitlines()
        
        
        error_data = pd.DataFrame(columns=['System'] + [self.model_name])
        ev_to_kjmol = 96.485
        
        
        lattice_e_dict = {}

        for system in systems:
            # Read molecule and solid structures
            mol = read(f"{self.lattice_energy_dir}/{system}/POSCAR_molecule", '0')
            sol = read(f"{self.lattice_energy_dir}/{system}/POSCAR_solid", '0')
            # Reference lattice energy
            ref = np.loadtxt(f"{self.lattice_energy_dir}/{system}/lattice_energy_DMC")
            nmol = np.loadtxt(f"{self.lattice_energy_dir}/{system}/nmol")
            
            lattice_e_dict[system] = {}
            lattice_e_dict[system]['ref'] = ref[0]
            
            
            def get_lattice_energy(model, sol, mol, nmol):
                # Assign calculator to structures
                # sol.info["head"] = "mp_pbe"
                # mol.info["head"] = "mp_pbe"
                sol.calc = model
                mol.calc = model
                # Compute energies
                energy_solid = sol.get_potential_energy()
                energy_molecule = mol.get_potential_energy()
                return energy_solid / nmol - energy_molecule
            
            
            system_errors = {'System': system}

            lat_energy = get_lattice_energy(calc, sol, mol, nmol)
            lat_energy_kjmol = lat_energy * ev_to_kjmol  # Convert to kJ/mol
            # absolute error
            error = abs(lat_energy_kjmol - ref[0])
            system_errors[self.model_name] = error
            
            # lattice energy
            lattice_e_dict[system] = lat_energy_kjmol
            
            # absolute error for each system
            error_data = pd.concat([error_data, pd.DataFrame([system_errors])], ignore_index=True)
            
            # mae for the model model
            mae = error_data[self.model_name].mean()
            
            
        with open(self.abs_error_output, "w") as f:
            error_data.to_csv(f, index=False)
        with open(self.lattice_e_ouptut, "w") as f:
            json.dump(lattice_e_dict, f)
        with open(self.mae_output, "w") as f:
            json.dump(mae, f)
            
        
    
    @property
    def abs_error(self):
        """Absolute error"""
        return pd.read_csv(self.abs_error_output)
    @property
    def lattice_e(self):
        """Lattice energy"""
        with open(self.lattice_e_ouptut, "r") as f:
            return json.load(f)
    @property
    def mae(self):
        """Mean absolute error"""
        with open(self.mae_output, "r") as f:
            return json.load(f)