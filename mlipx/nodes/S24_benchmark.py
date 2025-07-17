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
from tqdm import tqdm

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




# OC157 benchmark node
class S24Benchmark(zntrack.Node):
    """Benchmark model for s24 dataset.
    
    
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    oc_rel_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_rel_energies.csv")
    ref_rel_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_ref_rel_energies.csv")
    oc_mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_mae.json")
    oc_ranks_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_rank_accuracy.json")
    

    def run(self):
        from ase.io import read
        import copy

        calc = self.model.get_calculator()
        base_dir = get_benchmark_data("OC_Dataset.zip") / "OC_Dataset"

    
    
    
    