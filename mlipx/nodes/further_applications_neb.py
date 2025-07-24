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
from mlipx import NEB2
from mlipx.dash_utils import combine_apps


import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class NEBFurtherApplications(zntrack.Node):
    """ Node to combine all molecular benchmarks
    """
    # inputs
    NEB_list: List[NEB2] = zntrack.deps()


    def run(self):
        pass
        


    
    
    

    @staticmethod
    def benchmark_precompute(
        neb_data,
        cache_dir: str = "app_cache/nebs_further_apps/",
        ui = None,
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        
        """
        Precompute and cache all sub-benchmarks for bulk crystal benchmark.
        """

        from mlipx.dash_utils import process_data
        from mlipx.phonons_utils import convert_batch_to_node_dict

        # ---- convert all data to dicts if not already----

        
        os.makedirs(cache_dir, exist_ok=True)

        from mlipx.dash_utils import process_data

        # NEB
        neb_dict = process_data(
            neb_data,
            key_extractor=lambda node: node.name.split("_neb")[0],
            value_extractor=lambda node: node
        )
        
        # Run precomputes for all sub-benchmarks
        mlipx.NEB2.benchmark_precompute(
            node_dict=neb_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )

                    
        with open(f"{cache_dir}/nebs_cache/all_group_data.pkl", "rb") as f:
            all_group_data = pickle.load(f)

        #for group_name, (mae_df, neb_data_dict, assets_dir) in all_group_data.items():
            #print(f"\nGroup: {group_name}")
            #print(mae_df)


        benchmark_score_df = NEBFurtherApplications.benchmark_score(
            all_group_data=all_group_data,
            normalise_to_model=normalise_to_model,
        ).round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)
        benchmark_score_df['Rank'] = benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)
        
        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
        
        callback_fn = NEBFurtherApplications.callback_fn_from_cache(
            all_group_data=all_group_data,
        )
        
        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(NEBFurtherApplications.callback_fn_from_cache, f)

        return



    @staticmethod
    def callback_fn_from_cache(
        all_group_data,
    ):
        from mlipx import NEB2

        def callback_fn(app):
            NEB2.register_callbacks(app, all_group_data)

        return callback_fn
    
    
    

    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/nebs_further_apps",
        ui=None,
        full_benchmark: bool = False,
        normalise_to_model: str | None = None,
    ):
        import pandas as pd
        from mlipx.dash_utils import run_app, combine_apps


        with open(f"{cache_dir}/nebs_cache/all_group_data.pkl", "rb") as f:
            all_group_data = pickle.load(f)
            
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")


        callback_fn = NEBFurtherApplications.callback_fn_from_cache(
            all_group_data=all_group_data,
        )

        assets_dir = next(iter(all_group_data.values()))[2]
        app = dash.Dash(__name__, assets_folder=assets_dir)
        

        layout = NEBFurtherApplications.build_layout(
            benchmark_score_df=benchmark_score_df,
            apps_or_layouts_list=[
                NEB2.build_layout(all_group_data),
            ],
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
        )

        if full_benchmark:
            return layout, callback_fn

        app.layout = layout
        callback_fn(app)
        
        return run_app(app, ui=ui)
    
    
    @staticmethod
    def build_layout(
        benchmark_score_df: pd.DataFrame,
        apps_or_layouts_list: t.List[t.Union[dash.Dash, t.Callable, html.Div]],
        benchmark_table_info: str = "",
        id: str = "furtherapp-neb-benchmark-score-table",
    ):
        """ 
        """
        from mlipx.dash_utils import combine_apps
        
        return combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="NEB Benchmark",
            apps_or_layouts_list=apps_or_layouts_list,
            benchmark_table_info=benchmark_table_info,
            id=id,
            static_coloured_table=True,
        )
        
        
    @staticmethod
    def benchmark_score(
        all_group_data: dict, 
        normalise_to_model: Optional[str] = None,
    ) -> pd.DataFrame:
        import pandas as pd

        score_table = None

        for group_name, (mae_df, _, _) in all_group_data.items():
            #print(mae_df)
            group_score_col = f"{group_name} Score \u2193"
            if group_score_col not in mae_df.columns:
                continue
            df = mae_df[["Model", group_score_col]].copy()
            if score_table is None:
                score_table = df
            else:
                score_table = pd.merge(score_table, df, on="Model", how="outer")

        if score_table is not None:
            score_cols = [col for col in score_table.columns if col.endswith("Score \u2193")]

            if normalise_to_model:
                normalised_scores = score_table[score_cols].loc[score_table["Model"] == normalise_to_model].values.flatten()
                score_table[score_cols] = score_table[score_cols].div(normalised_scores, axis=1)

            score_table["Avg MAE \u2193"] = score_table[score_cols].mean(axis=1)

        return score_table