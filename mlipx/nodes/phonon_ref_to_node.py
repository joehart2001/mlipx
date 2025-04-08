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
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import glob


from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator


from mlipx.phonons_utils import get_fc2_and_freqs, init_phonopy, load_phonopy, get_chemical_formula
from phonopy.structure.atoms import PhonopyAtoms
from seekpath import get_path
import zntrack.node

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import base64

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class PhononRefToNode(zntrack.Node):
    """Compute the phonon dispersion from a phonopy object
    """
    # inputs
    phonopy_yaml_path: pathlib.Path = zntrack.params()
    
    # outputs
    force_constants_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "force_constants.yaml")



    def run(self):        
        
        phonons = load_phonopy(str(self.phonopy_yaml_path))
        phonons.save(filename=self.force_constants_path, settings={"force_constants": True})
        print(f"Force constants saved to: {self.force_constants_path}")