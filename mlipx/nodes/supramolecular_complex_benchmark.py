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
import subprocess

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


import mlipx
from mlipx import LNCI16Benchmark, S30LBenchmark



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class SupramolecularComplexBenchmark(zntrack.Node):
    """ Node to combine all molecular benchmarks
    """
    # inputs
    S30L_list: List[S30LBenchmark] = zntrack.deps()
    LNCI16_list: List[LNCI16Benchmark] = zntrack.deps()


    def run(self):
        pass
        



        
    @staticmethod
    def benchmark_precompute(
        S30L_data: List[S30LBenchmark] | Dict[str, S30LBenchmark],
        LNCI16_data: List[LNCI16Benchmark] | Dict[str, LNCI16Benchmark],
        cache_dir: str = "app_cache/supramolecular_complexes/",
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all surface benchmarks
        """
        
        from mlipx.dash_utils import process_data
        # list -> dict or dict -> dict
        # S30L
        S30L_dict = process_data(
            S30L_data,
            key_extractor=lambda node: node.name.split("_S30LBenchmark")[0],
            value_extractor=lambda node: node
        )
        # LNCI16
        LNCI16_dict = process_data(
            LNCI16_data,
            key_extractor=lambda node: node.name.split("_LNCI16Benchmark")[0],
            value_extractor=lambda node: node
        )
        
        os.makedirs(cache_dir, exist_ok=True)
        S30LBenchmark.benchmark_precompute(
            node_dict=S30L_dict,
            normalise_to_model=normalise_to_model,
        )
        LNCI16Benchmark.benchmark_precompute(
            node_dict=LNCI16_dict,
            normalise_to_model=normalise_to_model,
        )
        
        
        # ------- Load precomputed data -------
        # S30L
        S30L_mae_df = pd.read_pickle(os.path.join(cache_dir, "S30L_cache/mae_df.pkl"))
        S30L_pred_df = pd.read_pickle(os.path.join(cache_dir, "S30L_cache/predictions_df.pkl"))
        # LNCI16
        LNCI16_mae_df = pd.read_pickle(os.path.join(cache_dir, "LNCI16_cache/mae_df.pkl"))
        LNCI16_results_df = pd.read_pickle(os.path.join(cache_dir, "LNCI16_cache/results_df.pkl"))        
        
        
        from mlipx.data_utils import category_weighted_benchmark_score

        benchmark_score_df = category_weighted_benchmark_score(
            S30L=S30L_mae_df,
            LNCI16=LNCI16_mae_df,
            normalise_to_model=normalise_to_model,
            weights={"S30L": 1.0, "LNCI16": 1.0},
        )
        
        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
        
        callback_fn = SupramolecularComplexBenchmark.callback_fn_from_cache(

            
            normalise_to_model=normalise_to_model,
        )
        
        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(SupramolecularComplexBenchmark.callback_fn_from_cache, f)
        
        return 
    
    
    
    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/physicality_benchmark/",
        ui=None,
        full_benchmark: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        import pandas as pd
        import pickle
        import dash
        from mlipx.dash_utils import run_app, combine_apps
        from mlipx import GhostAtomBenchmark, SlabExtensivityBenchmark

        # ------- Load precomputed data -------
        # S30L
        S30L_mae_df = pd.read_pickle(os.path.join(cache_dir, "S30L_cache/mae_df.pkl"))
        S30L_pred_df = pd.read_pickle(os.path.join(cache_dir, "S30L_cache/predictions_df.pkl"))
        # LNCI16
        LNCI16_mae_df = pd.read_pickle(os.path.join(cache_dir, "LNCI16_cache/mae_df.pkl"))
        LNCI16_results_df = pd.read_pickle(os.path.join(cache_dir, "LNCI16_cache/results_df.pkl"))
        
        # supramolecular benchmark
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")

        callback_fn = SupramolecularComplexBenchmark.callback_fn_from_cache(
            S30L_mae_df=S30L_mae_df,
            LNCI16_mae_df=LNCI16_mae_df,
            normalise_to_model=normalise_to_model,
        )

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Physicality Benchmark",
            apps_or_layouts_list=[
                S30LBenchmark.build_layout(S30L_mae_df),
                LNCI16Benchmark.build_layout(LNCI16_results_df),
            ],
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
            id="physicality-benchmark-score-table",
            static_coloured_table=True,
        )
        
        if full_benchmark:
            return layout, callback_fn

        app.layout = layout
        callback_fn(app)
        
        
        
        return run_app(app, ui=ui)
    
    
    
    @staticmethod
    def callback_fn_from_cache(
        S30L_mae_df,
        LNCI16_mae_df,
        normalise_to_model=None,
    ):
        from mlipx import S30LBenchmark, LNCI16Benchmark

        def callback_fn(app):
            S30LBenchmark.register_callbacks(app, S30L_mae_df)
            LNCI16Benchmark.register_callbacks(app, LNCI16_mae_df)
        return callback_fn