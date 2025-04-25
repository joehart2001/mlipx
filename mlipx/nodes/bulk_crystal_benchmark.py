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

import mlipx
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




class BulkCrystalBenchmark(zntrack.Node):
    """
    """
    # inputs
    phonon_dict_ref: t.Dict[str, mlipx.PhononDispersion] = zntrack.deps()
    phonon_dict_pred: t.Dict[str, t.Dict[str, mlipx.PhononDispersion]] = zntrack.deps()
    elasticity_dict: t.Dict[str, mlipx.Elasticity] = zntrack.deps()
    

    

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    #abs_error_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "abs_error.csv")

    

    def run(self):
        mlipx.PhononDispersion.benchmark_interactive(
            pred_node_dict=self.phonon_dict_pred,
            ref_node_dict=self.phonon_dict_ref,
            ui="browser"
        )
        
        mlipx.Elasticity.benchmark_interactive(
            node_dict=self.elasticity_dict,
            ui="browser"
        )