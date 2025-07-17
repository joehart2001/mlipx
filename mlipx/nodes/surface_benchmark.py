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
from mlipx import OC157Benchmark, S24Benchmark
import os
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings



class SurfaceBenchmark(zntrack.Node):
    """ Node to combine all surface benchmarks
    """
    # inputs
    OC157_list: List[OC157Benchmark] = zntrack.deps()
    S24_list: List[S24Benchmark] = zntrack.deps()


    def run(self):
        pass
        


    
    

    @staticmethod
    def benchmark_precompute(
        OC157_data: List[OC157Benchmark] | Dict[str, OC157Benchmark],
        S24_data: List[S24Benchmark] | Dict[str, S24Benchmark],
        cache_dir: str = "app_cache/surface_benchmark/",
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all surface benchmarks
        """
        
        from mlipx.dash_utils import process_data
        # list -> dict or dict -> dict
        # OC157
        OC157_dict = process_data(
            OC157_data,
            key_extractor=lambda node: node.name.split("_oc157")[0],
            value_extractor=lambda node: node
        )
        # S24
        S24_dict = process_data(
            S24_data,
            key_extractor=lambda node: node.name.split("_s24")[0],
            value_extractor=lambda node: node
        )
        
        os.makedirs(cache_dir, exist_ok=True)
        OC157Benchmark.benchmark_precompute(
            node_dict=OC157_dict,
            normalise_to_model=normalise_to_model,
        )
        S24Benchmark.benchmark_precompute(
            node_dict=S24_dict,
            normalise_to_model=normalise_to_model,
        )
        
        # ------- Load precomputed data -------
        # OC157
        OC157_mae_df = pd.read_pickle(os.path.join(cache_dir, "oc157_cache/mae_df.pkl"))
        OC157_rel_df_all = pd.read_pickle(os.path.join(cache_dir, "oc157_cache/rel_energy_df.pkl"))
        OC157_dft_df = pd.read_pickle(os.path.join(cache_dir, "oc157_cache/dft_df.pkl"))
        # S24
        S24_mae_df = pd.read_pickle(os.path.join(cache_dir, "s24_cache/mae_df.pkl"))
        S24_pred_df = pd.read_pickle(os.path.join(cache_dir, "s24_cache/pred_df.pkl"))
        S24_ref_df = pd.read_pickle(os.path.join(cache_dir, "s24_cache/ref_df.pkl"))
        
        
        
        from mlipx.data_utils import category_weighted_benchmark_score

        benchmark_score_df = category_weighted_benchmark_score(
            OC157=OC157_mae_df,
            S24=S24_mae_df,
            normalise_to_model=normalise_to_model,
            weights={"OC157": 1.0, "S24": 1.0},
        )
        
        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
        
        callback_fn = SurfaceBenchmark.callback_fn_from_cache(
            OC157_rel_df_all=OC157_rel_df_all,
            OC157_dft_df=OC157_dft_df,
            S24_pred_df=S24_pred_df,
            S24_ref_df=S24_ref_df,
            normalise_to_model=normalise_to_model,
        )
        
        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(SurfaceBenchmark.callback_fn_from_cache, f)
        
        return 
    
    
    
    
    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/surface_benchmark/",
        ui=None,
        full_benchmark: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        import pandas as pd
        import pickle
        import dash
        from mlipx.dash_utils import run_app, combine_apps
        from mlipx import GMTKN55Benchmark, HomonuclearDiatomics

        # ------- Load precomputed data -------
        # OC157
        OC157_mae_df = pd.read_pickle(os.path.join(cache_dir, "oc157_cache/mae_df.pkl"))
        OC157_rel_df_all = pd.read_pickle(os.path.join(cache_dir, "oc157_cache/rel_energy_df.pkl"))
        OC157_dft_df = pd.read_pickle(os.path.join(cache_dir, "oc157_cache/dft_df.pkl"))
        # S24
        S24_mae_df = pd.read_pickle(os.path.join(cache_dir, "s24_cache/mae_df.pkl"))
        S24_pred_df = pd.read_pickle(os.path.join(cache_dir, "s24_cache/pred_df.pkl"))
        S24_ref_df = pd.read_pickle(os.path.join(cache_dir, "s24_cache/ref_df.pkl"))
        # surface benchmark
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")

        callback_fn = SurfaceBenchmark.callback_fn_from_cache(
            OC157_rel_df_all=OC157_rel_df_all,
            OC157_dft_df=OC157_dft_df,
            S24_pred_df=S24_pred_df,
            S24_ref_df=S24_ref_df,
            normalise_to_model=normalise_to_model,
        )

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Surface Benchmark",
            apps_or_layouts_list=[
                OC157Benchmark.build_layout(OC157_mae_df),
                S24Benchmark.build_layout(S24_mae_df),
            ],
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
            id="surface-benchmark-score-table",
            static_coloured_table=True,
        )
        
        if full_benchmark:
            return layout, callback_fn

        app.layout = layout
        callback_fn(app)
        
        
        
        return run_app(app, ui=ui)
    
    
    
    
    

    # ------------- helper functions -------------


    @staticmethod
    def callback_fn_from_cache(
        OC157_rel_df_all,
        OC157_dft_df,
        S24_pred_df,
        S24_ref_df,
        normalise_to_model=None,
    ):
        from mlipx import OC157Benchmark, S24Benchmark

        def callback_fn(app):
            OC157Benchmark.register_callbacks(app, OC157_rel_df_all, OC157_dft_df)
            S24Benchmark.register_callbacks(app, S24_pred_df, S24_ref_df)
        return callback_fn