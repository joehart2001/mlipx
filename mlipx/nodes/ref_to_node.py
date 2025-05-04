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


from mlipx.phonons_utils import get_fc2_and_freqs, init_phonopy, load_phonopy, get_chemical_formula

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import base64
from mlipx.benchmark_download_utils import get_benchmark_data

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class RefToNode(zntrack.Node):
    """Adds reference data into the DAG so it can be accessed by other nodes.
    

    """
    # inputs
    ref_path: str = zntrack.params()
    
    # outputs
    output_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "ref.json")



    def run(self):
        
        ref_path = Path(self.ref_path)
        
        if ref_path.suffix == ".json":
            with open(ref_path, "r") as f:
                data = json.load(f)
            with open(self.output_path, "w") as f:
                json.dump(data, f)
        
        else:
            raise ValueError(f"Unsupported file format: {ref_path.suffix}")
        # also add csv etc
        
    @property
    def get_ref(self):
        if ".json" in str(self.output_path):
            with open(self.output_path, "r") as f:
                data = json.load(f)
            return data
        else:
            raise ValueError(f"Unsupported file type for reading: {self.output_path}")