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
from matcalc.benchmark import ElasticityBenchmark



class Elasticity(zntrack.Node):
    """Bulk and shear moduli benchmark model against all available MP data.
    """
    # inputs
    #dataset_path: pathlib.Path = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    
    norm_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.1, -0.05, 0.05, 0.1))
    shear_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.02, -0.01, 0.01, 0.02))
    relax_structure: bool = zntrack.params(True)
    n_samples: int = zntrack.params(10)
    fmax: float = zntrack.params(0.05)

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    results_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "moduli_results.csv")

    



    def run(self):
        
        calc = self.model.get_calculator()
        
        # with open(self.dataset_path, "r") as f:
        #     ref_data = json.load(f)
        
        
        # from matcalc
        benchmark = ElasticityBenchmark(n_samples=self.n_samples, seed=2025, 
                                        fmax=self.fmax, 
                                        relax_structure=self.relax_structure,
                                        norm_strains = self.norm_strains,
                                        shear_strains = self.shear_strains,
        )
        
        
        print(self.model_name)
        results = benchmark.run(calc, self.model_name)
        
        results.to_csv(self.results_path, index=False)
        
        
    @property
    def results(self) -> pd.DataFrame:
        """Load the results from the benchmark
        """
        results = pd.read_csv(self.results_path)
        return results
    
    
    @staticmethod
    def mae_plot_interactive(node_dict, ui = None):
        """Interactive MAE table -> scatter plot for bulk and shear moduli for each model 
        """
        
        
        benchmarks = [
            'K_vrh', 
            'G_vrh',
        ]
        benchmark_units = {
            'K_vrh': '[GPa]', 
            'G_vrh': '[GPa]',
        }
        benchmark_labels = {
            'K_vrh': 'K_bulk',
            'G_vrh': 'K_shear',
        }
        
        label_to_key = {v: k for k, v in benchmark_labels.items()}
        
        mae_df = pd.DataFrame() # rows are models, cols are K and G
        
        for model in tqdm(node_dict.keys(), desc="Processing models"):

            results_df = node_dict[model].results
            
            mae_K = np.abs(results_df[f'K_vrh_{model}'].values - results_df['K_vrh_DFT'].values).mean()
            mae_G = np.abs(results_df[f'G_vrh_{model}'].values - results_df['G_vrh_DFT'].values).mean()
            mae_df.loc[model, 'K_vrh (MAE)'] = mae_K
            mae_df.loc[model, 'G_vrh (MAE)'] = mae_G
            
            
        #mae_df = mae_df.rename(index=label_to_key)

        
            

                
            