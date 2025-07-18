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
        ghost_atom_data: List[GhostAtomBenchmark] | Dict[str, GhostAtomBenchmark],
        slab_extensivity_data: List[SlabExtensivityBenchmark] | Dict[str, SlabExtensivityBenchmark],
        cache_dir: str = "app_cache/physicality_benchmark/",
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all surface benchmarks
        """
        
        from mlipx.dash_utils import process_data
        # list -> dict or dict -> dict
        # OC157
        ghost_atom_dict = process_data(
            ghost_atom_data,
            key_extractor=lambda node: node.name.split("_ghost-atom")[0],
            value_extractor=lambda node: node
        )
        # S24
        slab_dict = process_data(
            slab_extensivity_data,
            key_extractor=lambda node: node.name.split("_slab-extensivity")[0],
            value_extractor=lambda node: node
        )
        
        os.makedirs(cache_dir, exist_ok=True)
        GhostAtomBenchmark.benchmark_precompute(
            node_dict=ghost_atom_dict,
            normalise_to_model=normalise_to_model,
        )
        SlabExtensivityBenchmark.benchmark_precompute(
            node_dict=slab_dict,
            normalise_to_model=normalise_to_model,
        )
        
        
        # ------- Load precomputed data -------
        # ghost atom
        ghost_atom_results_df = pd.read_pickle(os.path.join(cache_dir, "ghost_atom_cache/results_df.pkl"))
        # slab
        slab_results_df = pd.read_pickle(os.path.join(cache_dir, "slab_extensivity_cache/results_df.pkl"))
        
        
        
        from mlipx.data_utils import category_weighted_benchmark_score

        benchmark_score_df = category_weighted_benchmark_score(
            ghost_atom=ghost_atom_results_df,
            slab=slab_results_df,
            normalise_to_model=normalise_to_model,
            weights={"ghost_atom": 1.0, "slab": 1.0},
        )
        
        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
        
        callback_fn = PhysicalityBenchmark.callback_fn_from_cache(
            ghost_atom_results_df=ghost_atom_results_df,
            slab_results_df=slab_results_df,
            normalise_to_model=normalise_to_model,
        )
        
        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(PhysicalityBenchmark.callback_fn_from_cache, f)
        
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
        # ghost atom
        ghost_atom_results_df = pd.read_pickle(os.path.join(cache_dir, "ghost_atom_cache/results_df.pkl"))
        # slab
        slab_results_df = pd.read_pickle(os.path.join(cache_dir, "slab_extensivity_cache/results_df.pkl"))
        
        # physicality benchmark
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")

        callback_fn = PhysicalityBenchmark.callback_fn_from_cache(
            ghost_atom_results_df=ghost_atom_results_df,
            slab_results_df=slab_results_df,
            normalise_to_model=normalise_to_model,
        )

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Physicality Benchmark",
            apps_or_layouts_list=[
                GhostAtomBenchmark.build_layout(ghost_atom_results_df),
                SlabExtensivityBenchmark.build_layout(slab_results_df),
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
        ghost_atom_results_df,
        slab_results_df,
        normalise_to_model=None,
    ):
        from mlipx import GhostAtomBenchmark, SlabExtensivityBenchmark

        def callback_fn(app):
            GhostAtomBenchmark.register_callbacks(app, ghost_atom_results_df)
            SlabExtensivityBenchmark.register_callbacks(app, slab_results_df)
        return callback_fn