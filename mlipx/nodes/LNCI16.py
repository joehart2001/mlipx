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
from ase.build import bulk, surface
from ase.io import write, read
import subprocess

import warnings
from pathlib import Path
from typing import Any, Callable
from ase.calculators.calculator import Calculator
from tqdm import tqdm
from phonopy.api_phonopy import Phonopy
from mlipx.abc import ComparisonResults, NodeWithCalculator
from mlipx.dash_utils import dash_table_interactive
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
from mlipx import OC157Benchmark, S24Benchmark
import os
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
from copy import deepcopy



class LNCI16Benchmark(zntrack.Node):
    """ 
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    #slab_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "slab_energy_extensivity.csv")
    
    
    def run(self):

        calc = self.model.get_calculator()
        
        


        # # Prepare dataframe and write to CSV
        # df = pd.DataFrame([{
        #     "Model": self.model_name,
        #     "E1 (eV)": e1,
        #     "E2 (eV)": e2,
        #     "E12 (eV)": e12,
        #     "Delta (meV)": delta_meV
        # }])
        # df.to_csv(self.slab_energy_output, index=False)



    # @property
    # def get_results(self):
    #     """Load results from CSV file."""
    #     return pd.read_csv(self.slab_energy_output)



    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "LNCI16Benchmark"],
        cache_dir: str = "app_cache/supramolecular_complexes/LNCI16_cache/",
        normalise_to_model: Optional[str] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        df_list = []
        for model_name, node in node_dict.items():
            df = node.get_results.copy()
            df["Model"] = model_name  # ensure model name column
            df_list.append(df)

        full_df = pd.concat(df_list, ignore_index=True)

        full_df["Score"] = full_df["Delta (meV)"].abs()

        if normalise_to_model:
            norm_value = full_df.loc[full_df["Model"] == normalise_to_model, "Score"].values[0]
            full_df["Score"] /= norm_value

        full_df["Rank"] = full_df["Score"].rank(ascending=True, method="min")

        full_df.to_pickle(os.path.join(cache_dir, "results_df.pkl"))



    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/supramolecular_complexes/LNCI16_cache/",
        app: dash.Dash | None = None,
        ui=None,
    ):
        from mlipx.dash_utils import run_app

        results_df = pd.read_pickle(os.path.join(cache_dir, "results_df.pkl"))

        layout = LNCI16Benchmark.build_layout(results_df)

        if app is None:
            app = dash.Dash(__name__)
            app.layout = layout
            return run_app(app, ui=ui)
        else:
            return layout, lambda _: None

    @staticmethod
    def build_layout(results_df):
        return html.Div([
            dash_table_interactive(
                df=results_df.round(3),
                id="LNCI16-table",
                benchmark_info="",
                title="LNCI16 Benchmark",
                tooltip_header={
                    "Model": "Name of the MLIP model",
                    "Score": "Absolute value of Delta (meV); normalized if specified",
                    "Rank": "Ranking of model by Score (lower is better)"
                }
            )
        ])
        
        
    @staticmethod
    def register_callbacks(app, results_df):
        pass