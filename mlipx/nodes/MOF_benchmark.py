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
from mlipx import QMOFBenchmark



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class MOFBenchmark(zntrack.Node):
    """ Node to combine all molecular benchmarks
    """
    # inputs
    QMOF_list: List[QMOFBenchmark] = zntrack.deps()

    def run(self):
        pass
    
    
    






    @staticmethod
    def benchmark_precompute(
        QMOF_data: List[QMOFBenchmark] | Dict[str, QMOFBenchmark],
        cache_dir: str = "app_cache/MOF/",
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all surface benchmarks
        """
        
        from mlipx.dash_utils import process_data
        # list -> dict or dict -> dict
        
        QMOF_dict = process_data(
            QMOF_data,
            key_extractor=lambda node: node.name.split("_QMOFBenchmark")[0],
            value_extractor=lambda node: node
        )
        
        os.makedirs(cache_dir, exist_ok=True)
        QMOFBenchmark.benchmark_precompute(
            node_dict=QMOF_dict,
            normalise_to_model=normalise_to_model,
        )
        
        
        # ------- Load precomputed data -------
        # QMOF
        QMOF_mae_df = pd.read_pickle(os.path.join(cache_dir, "QMOF_cache/mae_df.pkl"))
        QMOF_results_df = pd.read_pickle(os.path.join(cache_dir, "QMOF_cache/results_df.pkl"))
                
        from mlipx.data_utils import category_weighted_benchmark_score

        benchmark_score_df = category_weighted_benchmark_score(
            QMOF=QMOF_mae_df,
            normalise_to_model=normalise_to_model,
            weights={"QMOF": 1.0},
        )
        
        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
        
        callback_fn = MOFBenchmark.callback_fn_from_cache(
            QMOF_results_df=QMOF_results_df,
            normalise_to_model=normalise_to_model,
        )
        
        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(MOFBenchmark.callback_fn_from_cache, f)
        
        return 
    
    
    
    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/MOF/",
        ui=None,
        full_benchmark: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        import pandas as pd
        import pickle
        import dash
        from mlipx.dash_utils import run_app, combine_apps

        # ------- Load precomputed data -------
        # QMOF
        QMOF_mae_df = pd.read_pickle(os.path.join(cache_dir, "QMOF_cache/mae_df.pkl"))
        QMOF_results_df = pd.read_pickle(os.path.join(cache_dir, "QMOF_cache/results_df.pkl"))
        # MOF benchmark
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")

        callback_fn = MOFBenchmark.callback_fn_from_cache(
            QMOF_results_df=QMOF_results_df,
            normalise_to_model=normalise_to_model,
        )

        assets_dir = os.path.abspath("assets")
        print("Serving assets from:", assets_dir)
        app = dash.Dash(__name__, assets_folder=assets_dir)
        
        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="MOF Benchmark",
            apps_or_layouts_list=[
                QMOFBenchmark.build_layout(QMOF_mae_df),
            ],
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
            id="MOF-complex-benchmark-score-table",
            static_coloured_table=True,
        )
        
        if full_benchmark:
            return layout, callback_fn

        app.layout = layout
        callback_fn(app)
        
        
        
        return run_app(app, ui=ui)
    
    
    
    @staticmethod
    def callback_fn_from_cache(
        QMOF_results_df,
        normalise_to_model=None,
    ):
        from mlipx import QMOFBenchmark

        def callback_fn(app):
            QMOFBenchmark.register_callbacks(app, QMOF_results_df)
            
        return callback_fn