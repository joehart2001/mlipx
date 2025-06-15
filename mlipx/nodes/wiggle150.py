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
from mlipx.dash_utils import dash_table_interactive
import socket
import time
from typing import List, Dict, Any, Optional
import cctk
from ase.io.trajectory import Trajectory
from plotly.io import write_image
from ase.io import read

from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mlipx.benchmark_download_utils import get_benchmark_data



class DMCICE13Benchmark(zntrack.Node):
    """Benchmark model against DMC-ICE13
    """
    
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    
    lattice_e_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_e_ice.csv")
    mae_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae_ice.json")
    ref_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "ref_ice.csv")
    

    def run(self):
        calc = self.model.get_calculator()
        ev_to_kcal_per_mol = 23.0605
        
        # download wiggle150 dataset
        dmc_ice_dir = get_benchmark_data("dmc-ice13-main.zip") / "dmc-ice13-main/INPUT/VASP"
        
        
        with open(dmc_ice_dir / "../../ice_polymorph_ref_PBE_D3.json", "r") as f:
            ice_ref = json.load(f)
    

        polymorphs = [p.name for p in dmc_ice_dir.iterdir() if p.is_dir() and p.name != "water"]
        

            
        with open(self.lattice_e_ice_output, "w") as f:
            ice_lattice_e_df.to_csv(f, index=True)
        with open(self.mae_ice_output, "w") as f:
            json.dump(mae_ice, f)
        with open(self.ref_ice_output, "w") as f:
            ref_df.to_csv(f, index=True)
        
            
        
    @property
    def get_lattice_e(self):
        """Lattice energy ICE"""
        return pd.read_csv(self.lattice_e_ice_output).set_index("Polymorph")
    @property
    def get_mae(self):
        """Mean absolute error ICE"""
        with open(self.mae_ice_output, "r") as f:
            return json.load(f)
    @property
    def get_ref(self):
        """Reference ICE lattice energy"""
        return pd.read_csv(self.ref_ice_output).set_index("Polymorph")
    
    