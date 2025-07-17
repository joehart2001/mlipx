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
from mlipx import MolecularDynamics
from mlipx.dash_utils import combine_apps


import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class FurtherApplications(zntrack.Node):
    """ Node to combine all molecular benchmarks
    """
    # inputs
    MD_list: List[MolecularDynamics] = zntrack.deps()


    def run(self):
        pass
        


    


    
    
    


    @staticmethod
    def benchmark_precompute(
        MD_data: List[MolecularDynamics] | Dict[str, MolecularDynamics],
        cache_dir: str = "app_cache/further_applications_benchmark",
        ui: str = "browser",
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        from mlipx.dash_utils import process_data
        os.makedirs(cache_dir, exist_ok=True)

        MD_dict = process_data(
            MD_data,
            key_extractor=lambda node: node.name.split("_config-0_MolecularDynamics")[0],
            value_extractor=lambda node: node
        )
        
        os.makedirs(cache_dir, exist_ok=True)
        MolecularDynamics.benchmark_precompute(
            node_dict=MD_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )

        
        import pickle
        import pandas as pd
        import dash
        # Load groups structure
        with open(f"{cache_dir}/molecular_dynamics_cache/groups.pkl", "rb") as f:
            groups = pickle.load(f)
        # Load group mae tables
        group_mae_tables = {}
        for group_name in groups:
            group_mae_tables[group_name] = pd.read_pickle(f"{cache_dir}/molecular_dynamics_cache/mae_df_{group_name}.pkl")
        with open(f"{cache_dir}/molecular_dynamics_cache/rdf_data.pkl", "rb") as f:
            properties_dict = pickle.load(f)
        with open(f"{cache_dir}/molecular_dynamics_cache/msd_data.pkl", "rb") as f:
            msd_dict = pickle.load(f)
        properties_dict["msd_O"] = msd_dict
        
        

        benchmark_score_df = FurtherApplications.benchmark_score(groups=groups, group_mae_tables=group_mae_tables, normalise_to_model=normalise_to_model).round(3)
        benchmark_score_df = benchmark_score_df.sort_values(by='Avg MAE \u2193', ascending=True).reset_index(drop=True)
        benchmark_score_df['Rank'] = benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)

        benchmark_score_df.to_csv(f"{cache_dir}/benchmark_score.csv", index=False)




    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/further_applications_benchmark",
        ui: str = "browser",
        full_benchmark: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        import pandas as pd
        import json
        from mlipx.dash_utils import run_app
        import dash

        benchmark_score_df = pd.read_csv(f"{cache_dir}/benchmark_score.csv")
        with open(f"{cache_dir}/molecular_dynamics_cache/groups.pkl", "rb") as f:
            groups = pickle.load(f)
        # Load group mae tables
        group_mae_tables = {}
        for group_name in groups:
            group_mae_tables[group_name] = pd.read_pickle(f"{cache_dir}/molecular_dynamics_cache/mae_df_{group_name}.pkl")
        with open(f"{cache_dir}/molecular_dynamics_cache/rdf_data.pkl", "rb") as f:
            properties_dict = pickle.load(f)
        with open(f"{cache_dir}/molecular_dynamics_cache/msd_data.pkl", "rb") as f:
            msd_dict = pickle.load(f)
        properties_dict["msd_O"] = msd_dict

        callback_fn = FurtherApplications.callback_fn_from_cache(
            groups=groups,
            group_mae_tables=group_mae_tables,
            properties_dict=properties_dict,
        )

        app = dash.Dash(__name__)

        layout = FurtherApplications.build_layout(
            groups=groups,
            group_mae_tables=group_mae_tables,
            normalise_to_model=normalise_to_model,
        )
        
        
        
        if full_benchmark:
            return layout, callback_fn

        app.layout = layout
        callback_fn(app)

        return run_app(app, ui=ui)
    





    @staticmethod
    def callback_fn_from_cache(
        #cache_dir,
        #mae_df_list,
        groups,
        group_mae_tables,
        properties_dict,
        normalise_to_model=None
    ):
        """
        Return a function that registers MD callbacks using MolecularDynamics.register_callbacks,
        passing the proper group MAE tables and reference data from cache.
        """
        import pickle
        import pandas as pd
        def callback_fn(app):
            # # Load group MAE tables and groups from cache
            # # Assume cache_dir points to e.g. "app_cache/further_applications_benchmark"
            # md_cache = f"{cache_dir}/molecular_dynamics_cache"
            # with open(f"{cache_dir}/groups.pkl", "rb") as f:
            #     groups = pickle.load(f)
            # # Load group mae tables
            # group_mae_tables = {}
            # for group_name in groups:
            #     group_mae_tables[group_name] = pd.read_pickle(f"{cache_dir}/mae_df_{group_name}.pkl")
            # with open(f"{cache_dir}/molecular_dynamics_cache/rdf_data.pkl", "rb") as f:
            #     properties_dict = pickle.load(f)
            # with open(f"{cache_dir}/molecular_dynamics_cache/msd_data.pkl", "rb") as f:
            #     msd_dict = pickle.load(f)
            # properties_dict["msd_O"] = msd_dict
            
            # Use loaded properties_dict, not the one passed in
            MolecularDynamics.register_callbacks(
                app,
                groups=groups,
                group_mae_tables=group_mae_tables,
                properties_dict=properties_dict,
            )
        return callback_fn
    
    
    
    
    
    @staticmethod
    def build_layout(groups, group_mae_tables, normalise_to_model=None):

        score_df = FurtherApplications.benchmark_score(groups, group_mae_tables, normalise_to_model=normalise_to_model).round(3)
        score_df["Rank"] = score_df["Avg MAE \u2193"].rank(ascending=True).astype(int)
        score_df = score_df.sort_values(by="Avg MAE \u2193", ascending=True).reset_index(drop=True)

        layout = combine_apps(
            benchmark_score_df=score_df,
            benchmark_title="Water MD",
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
            apps_or_layouts_list=[
                MolecularDynamics.build_layout(groups, group_mae_tables)
            ],
            id="rdf-benchmark-score-table",
            static_coloured_table=True,
        )

        return layout
    
    
    
    
    @staticmethod
    def register_callbacks(app, groups, group_mae_tables, properties_dict):
        from mlipx import MolecularDynamics
        MolecularDynamics.register_callbacks(
            app,
            groups=groups,
            group_mae_tables=group_mae_tables,
            properties_dict=properties_dict
        )
        





    
    @staticmethod
    def benchmark_score(
        groups,
        group_mae_tables,
        normalise_to_model: Optional[str] = None,
        weights: Dict[str, float] = None,
    ):
        """Average MAE over all property groups using specified weights."""

        if weights is None:
            weights = {group_name: 1.0 for group_name in groups}

        models = group_mae_tables[next(iter(groups))]["Model"].tolist()
        scores = {}

        for model in models:
            model_scores = {}
            weighted_sum = 0.0
            total_weight = 0.0

            for group_name, group_df in group_mae_tables.items():
                try:
                    score = group_df.loc[group_df["Model"] == model, "Score ↓"].values[0]
                except IndexError:
                    continue
                model_scores[f"{group_name} Score ↓"] = score
                group_weight = weights.get(group_name, 1.0)
                weighted_sum += group_weight * score
                total_weight += group_weight

            model_scores["Avg MAE \u2193"] = weighted_sum / total_weight if total_weight > 0 else None
            scores[model] = model_scores

        df = pd.DataFrame.from_dict(scores, orient="index").reset_index().rename(columns={"index": "Model"})

        if normalise_to_model:
            norm_val = df.loc[df["Model"] == normalise_to_model, "Avg MAE \u2193"].values[0]
            df["Avg MAE \u2193"] = df["Avg MAE \u2193"] / norm_val

        return df
