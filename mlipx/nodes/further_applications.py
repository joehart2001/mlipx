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
    """ Node to combine all molecular benchmarks and aggregate scores. """
    # inputs
    MD_list: List[MolecularDynamics] = zntrack.deps()

    def run(self):
        pass

    @staticmethod
    def aggregate_scores(group_mae_tables: dict, groups: dict, normalise_to_model: Optional[str] = None):
        """
        Aggregate static (RDF) and dynamic (MSD, VACF, VDOS) scores for each model.
        Returns a DataFrame with static_score, dynamic_score, overall_score per model.
        """
        import numpy as np
        import pandas as pd

        # Identify static and dynamic groups
        static_group_keys = [k for k in groups if "rdf" in k.lower()]
        dynamic_group_keys = [k for k in groups if any(p in k.lower() for p in ["dynamic", "msd", "vacf", "vdos"])]
        # Fallback: if not found, use all groups with "rdf" for static, the rest for dynamic
        if not static_group_keys:
            static_group_keys = [k for k in groups if "rdf" in k.lower()]
        if not dynamic_group_keys:
            dynamic_group_keys = [k for k in groups if k not in static_group_keys]

        # For each group, get the score column ("Score ↓")
        static_scores = []
        dynamic_scores = []
        # To get all unique models
        model_names = set()
        for group_key in static_group_keys:
            df = group_mae_tables[group_key]
            if "Model" in df.columns and "Score ↓" in df.columns:
                static_scores.append(df[["Model", "Score ↓"]].set_index("Model"))
                model_names.update(df["Model"])
        for group_key in dynamic_group_keys:
            df = group_mae_tables[group_key]
            if "Model" in df.columns and "Score ↓" in df.columns:
                dynamic_scores.append(df[["Model", "Score ↓"]].set_index("Model"))
                model_names.update(df["Model"])
        model_names = sorted(model_names)

        # For each model, compute mean static and dynamic scores
        rows = []
        for model in model_names:
            static_group_scores = []
            for df in static_scores:
                if model in df.index:
                    static_group_scores.append(df.loc[model, "Score ↓"])
            dynamic_group_scores = []
            for df in dynamic_scores:
                if model in df.index:
                    dynamic_group_scores.append(df.loc[model, "Score ↓"])
            static_score = float(np.mean(static_group_scores)) if static_group_scores else np.nan
            dynamic_score = float(np.mean(dynamic_group_scores)) if dynamic_group_scores else np.nan
            overall_score = float(np.mean([static_score, dynamic_score]))
            rows.append({
                "Model": model,
                "Static Score ↓": static_score,
                "Dynamic Score ↓": dynamic_score,
                "Overall Score ↓": overall_score,
            })
        df_score = pd.DataFrame(rows)
        df_score["Rank"] = df_score["Overall Score ↓"].rank(method="min").astype(int)
        df_score = df_score.sort_values("Overall Score ↓", ascending=True).reset_index(drop=True)
        if normalise_to_model is not None and normalise_to_model in df_score["Model"].values:
            norm_val = df_score.loc[df_score["Model"] == normalise_to_model, "Overall Score ↓"].values[0]
            df_score["Overall Score ↓"] = df_score["Overall Score ↓"] / norm_val
        return df_score



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
        # Precompute and save group MAE tables, properties, groups
        MolecularDynamics.benchmark_precompute(
            node_dict=MD_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )
        import pickle
        import pandas as pd
        # Load groups and group_mae_tables
        with open(f"{cache_dir}/molecular_dynamics_cache/groups.pkl", "rb") as f:
            groups = pickle.load(f)
        group_mae_tables = {}
        for group_name in groups:
            group_mae_tables[group_name] = pd.read_pickle(f"{cache_dir}/molecular_dynamics_cache/mae_df_{group_name}.pkl")
        # Aggregate scores
        score_df = FurtherApplications.aggregate_scores(group_mae_tables, groups, normalise_to_model=normalise_to_model).round(3)
        score_df.to_csv(f"{cache_dir}/molecular_dynamics_cache/benchmark_score.csv", index=False)



    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/further_applications_benchmark",
        ui: str = "browser",
        full_benchmark: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        import pandas as pd
        import pickle
        from mlipx.dash_utils import run_app
        import dash
        # Load group mae tables, groups, and score table
        with open(f"{cache_dir}/molecular_dynamics_cache/groups.pkl", "rb") as f:
            groups = pickle.load(f)
        group_mae_tables = {}
        for group_name in groups:
            group_mae_tables[group_name] = pd.read_pickle(f"{cache_dir}/molecular_dynamics_cache/mae_df_{group_name}.pkl")
        with open(f"{cache_dir}/molecular_dynamics_cache/rdf_data.pkl", "rb") as f:
            properties_dict = pickle.load(f)
        with open(f"{cache_dir}/molecular_dynamics_cache/msd_data.pkl", "rb") as f:
            msd_dict = pickle.load(f)
        with open(f"{cache_dir}/molecular_dynamics_cache/vdos_data.pkl", "rb") as f:
            vdos_dict = pickle.load(f)
        properties_dict["msd_O"] = msd_dict
        properties_dict["vdos"] = vdos_dict
        benchmark_score_df = pd.read_csv(f"{cache_dir}/molecular_dynamics_cache/benchmark_score.csv")

        app = dash.Dash(__name__)
        # Use new MD build_layout and callback registration
        layout = MolecularDynamics.build_layout(groups=groups, group_mae_tables=group_mae_tables, score_df=benchmark_score_df)
        app.layout = layout
        MolecularDynamics.register_callbacks(
            app,
            groups=groups,
            group_mae_tables=group_mae_tables,
            properties_dict=properties_dict
        )
        if full_benchmark:
            return layout, app
        return run_app(app, ui=ui)