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
from mlipx import MolecularCrystalBenchmark, BulkCrystalBenchmark, PhononDispersion, Elasticity, LatticeConstant, X23Benchmark, DMCICE13Benchmark, GMTKN55Benchmark, MolecularBenchmark, HomonuclearDiatomics
from mlipx import PhononAllRef, PhononAllBatch



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class FullBenchmark(zntrack.Node):
    """ Node to combine all bulk crystal benchmarks
    """
    # inputs
    bulk_crystal_benchmark: List[BulkCrystalBenchmark] = zntrack.deps()
    mol_crystal_benchmark: List[MolecularCrystalBenchmark] = zntrack.deps()
    mol_benchmark: List[MolecularBenchmark] = zntrack.deps()
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files

    
    def run(self):
        pass
        


    
    

    @staticmethod
    def benchmark_interactive(
        elasticity_data: List[Elasticity] | Dict[str, Elasticity],
        lattice_const_data: List[LatticeConstant] | Dict[str, Dict[str, LatticeConstant]],
        lattice_const_ref_node_dict: LatticeConstant,
        phonon_ref_data: List[PhononDispersion] | Dict[str, PhononDispersion] | PhononAllRef,
        phonon_pred_data: List[PhononDispersion] | Dict[str, Dict[str, PhononDispersion]] | List[PhononAllBatch] | Dict[str, PhononAllBatch],
        
        X23_data: List[X23Benchmark] | Dict[str, X23Benchmark],
        DMC_ICE_data: List[DMCICE13Benchmark] | Dict[str, DMCICE13Benchmark],
        
        GMTKN55_data: List[GMTKN55Benchmark] | Dict[str, GMTKN55Benchmark],
        HD_data: List[HomonuclearDiatomics] | Dict[str, HomonuclearDiatomics],
        
        ui: str = "browser",
        
        return_app: bool = False,
        report: bool = True,
        normalise_to_model: t.Optional[str] = None,
    ):
        
        # extract apps
        print("Building bulk crystal benchmark 1/3...")
        bulk_benchmark_app, bulk_benchmark_score_df, bulk_register_callbacks = BulkCrystalBenchmark.benchmark_interactive(
            elasticity_data=elasticity_data,
            lattice_const_data=lattice_const_data,
            lattice_const_ref_node_dict=lattice_const_ref_node_dict,
            phonon_ref_data=phonon_ref_data,
            phonon_pred_data=phonon_pred_data,
            full_benchmark=True,
            report=report,
            normalise_to_model=normalise_to_model,
        )
        
        print("Building molecular crystal benchmark 2/3...")
        mol_crystal_benchmark_app, mol_crystal_benchmark_score_df, mol_crystal_register_callbacks = MolecularCrystalBenchmark.benchmark_interactive(
            X23_data=X23_data,
            DMC_ICE_data=DMC_ICE_data,
            full_benchmark=True,
            normalise_to_model=normalise_to_model,
        )
        
        print("Building molecular benchmark 3/3...")
        mol_benchmark_app, mol_benchmark_score_df, mol_register_callbacks = MolecularBenchmark.benchmark_interactive(
            GMTKN55_data=GMTKN55_data,
            HD_data=HD_data,
            full_benchmark=True,
            normalise_to_model=normalise_to_model,
        )

        # df with score for each benchmark and model
        scores_all_df = FullBenchmark.get_overall_score_df(
            (bulk_benchmark_score_df, "Bulk Crystal"),
            (mol_crystal_benchmark_score_df, "Molecular Crystal"),
            (mol_benchmark_score_df, "Molecular"),
        )
        
        if not os.path.exists("benchmark_stats/"):
            os.makedirs("benchmark_stats/")
        scores_all_df.to_csv("benchmark_stats/overall_benchmark.csv", index=False)
        
        
        from mlipx.dash_utils import colour_table
        style_data_conditional = colour_table(scores_all_df, all_cols=True)
        
        
        summary_layout = html.Div([
            html.H1("Overall Benchmark Scores (avg MAE)"),
            html.P("Scores are avg MAEs normalised to: " + normalise_to_model if normalise_to_model else "Scores are avg MAEs"),
            html.Div([
                dash_table.DataTable(
                    id="summary-table",
                    columns=[{"name": i, "id": i} for i in scores_all_df.columns],
                    data=scores_all_df.to_dict("records"),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'fontSize': '14px'},
                    style_header={'fontWeight': 'bold'},
                    style_data_conditional=style_data_conditional,

                )
            ])
        ])
        
        
        app_summary = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        # combine apps into one with tabs for each benchmark
        tab_layouts = {
            "Overall Benchmark": summary_layout,
            "Bulk Crystal Score": bulk_benchmark_app.layout,
            "Molecular Crystal Score": mol_crystal_benchmark_app.layout,
            "Molecular Score": mol_benchmark_app.layout,
        }
        
        # Register callbacks for each app
        bulk_register_callbacks(app_summary)
        mol_crystal_register_callbacks(app_summary)
        mol_register_callbacks(app_summary)
        
        app_summary.layout = html.Div([
            dcc.Tabs(
                id="tabs",
                value="Overall Benchmark",
                children=[
                    dcc.Tab(label=tab, value=tab) for tab in tab_layouts
                ]
            ),
            html.Div(id="tab-content")
        ],
        style={
        "backgroundColor": "white",
        "padding": "20px",
        "border": "2px solid black",}
        )

        # Callback to switch tabs
        @app_summary.callback(
            dash.Output("tab-content", "children"),
            dash.Input("tabs", "value"),
        )
        def render_tab(tab_name):
            return tab_layouts[tab_name]


        if return_app:
            return app_summary
        
        from mlipx.dash_utils import run_app
        return run_app(app_summary, ui=ui)
    
    
    
    
    # ------------- helper functions -------------

    @staticmethod
    def get_overall_score_df(
        *dfs_with_names: t.Tuple[pd.DataFrame, str]
    ) -> pd.DataFrame:
        """Combine multiple benchmark DataFrames into an overall score DataFrame.
        """
        merged_df = None
        
        for df, name in dfs_with_names:
            df_renamed = df[["Model", "Avg MAE \u2193"]].rename(columns={"Avg MAE \u2193": name})
            if merged_df is None:
                merged_df = df_renamed
            else:
                merged_df = pd.merge(merged_df, df_renamed, on="Model", how="outer")

        # Compute average MAE across all benchmarks
        benchmark_cols = [name for _, name in dfs_with_names]
        merged_df["Overall Score \u2193"] = merged_df[benchmark_cols].mean(axis=1)

        # Sort and rank
        merged_df = merged_df.sort_values(by="Overall Score \u2193", ascending=True)
        merged_df = merged_df.reset_index(drop=True)
        merged_df['Rank'] = merged_df['Overall Score \u2193'].rank(ascending=True)

        return merged_df.round(3)