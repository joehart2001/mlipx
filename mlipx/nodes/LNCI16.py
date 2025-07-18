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
        mae_dict = {}
        pred_dfs = []

        ref_df = list(node_dict.values())[0].get_ref

        for model_name, node in node_dict.items():
            mae_dict[model_name] = node.get_mae
            pred_df = node.get_pred.rename(columns={f"E_{model_name} (eV)": "E_model (eV)"})
            merged = pd.merge(pred_df, ref_df, on="Index")
            merged["Model"] = model_name
            merged["Error (eV)"] = merged["E_model (eV)"] - merged["E_ref (eV)"]
            merged["Error (kcal/mol)"] = merged["Error (eV)"] / 0.04336414
            pred_dfs.append(merged)

        mae_df = pd.DataFrame.from_dict(mae_dict, orient="index", columns=["MAE (kcal/mol)"]).reset_index()
        mae_df = mae_df.rename(columns={"index": "Model"})

        pred_full_df = pd.concat(pred_dfs, ignore_index=True)

        mae_df["Score"] = mae_df["MAE (kcal/mol)"]

        if normalise_to_model:
            norm_value = mae_df.loc[mae_df["Model"] == normalise_to_model, "Score"].values[0]
            mae_df["Score"] /= norm_value

        mae_df["Rank"] = mae_df["Score"].rank(ascending=True, method="min")

        mae_df.to_pickle(os.path.join(cache_dir, "results_df.pkl"))
        pred_full_df.to_pickle(os.path.join(cache_dir, "predictions_df.pkl"))
        


    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/supramolecular_complexes/LNCI16_cache/",
        app: dash.Dash | None = None,
        ui=None,
    ):
        from mlipx.dash_utils import run_app

        results_df = pd.read_pickle(os.path.join(cache_dir, "results_df.pkl"))
        pred_df = pd.read_pickle(os.path.join(cache_dir, "predictions_df.pkl"))

        layout = LNCI16Benchmark.build_layout(results_df)

        def callback_fn(app_instance):
            LNCI16Benchmark.register_callbacks(app_instance, pred_df)

        if app is None:
            app = dash.Dash(__name__)
            app.layout = layout
            callback_fn(app)
            return run_app(app, ui=ui)
        else:
            return layout, callback_fn


    @staticmethod
    def build_layout(results_df):
        return html.Div([
            dash_table_interactive(
                df=results_df.round(3),
                id="LNCI16-table",
                benchmark_info="Benchmark info:",
                title="LNCI16 Benchmark",
                tooltip_header={
                    "Model": "Name of the MLIP model",
                    "Score": "Absolute value of Delta (meV); normalized if specified",
                    "Rank": "Ranking of model by Score (lower is better)"
                },
                extra_components=[
                    html.Div(id="LNCI16-plot"),
                ]
            )
        ])
        
        
    @staticmethod
    def register_callbacks(
        app, 
        pred_df
    ):
        
        @app.callback(
            Output("LNCI16-plot", "children"),
            Input("LNCI16-table", "active_cell"),
            State("LNCI16-table", "data"),
        )
        def update_LNCI16_plot(active_cell, table_data):
            if not active_cell:
                raise PreventUpdate

            row = active_cell["row"]
            clicked_model = table_data[row]["Model"]
            col = active_cell["column_id"]

            if col == "Model":
                return None

            df = pred_df.copy()
            df = df[df["Model"] == clicked_model]
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["E_ref (eV)"],
                    y=df["E_model (eV)"],
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    text=[
                        f"Index: {i}<br>DFT: {e_ref:.3f} eV<br>{clicked_model}: {e_model:.3f} eV"
                        for i, e_ref, e_model in zip(df["Index"], df["E_ref (eV)"], df["E_model (eV)"])
                    ],
                    hoverinfo="text",
                    name=clicked_model
                )
            )
            fig.add_trace(go.Scatter(
                x=[-10, 10], y=[-10, 10],
                mode="lines",
                line=dict(dash="dash", color="black", width=1),
                showlegend=False
            ))

            mae = df["Error (kcal/mol)"].abs().mean()

            fig.update_layout(
                title=f"{clicked_model} vs DFT Interaction Energies",
                xaxis_title="DFT Energy [eV]",
                yaxis_title=f"{clicked_model} Energy [eV]",
                annotations=[
                    dict(
                        text=f"N = {len(df)}<br>MAE = {mae:.2f} kcal/mol",
                        xref="paper", yref="paper",
                        x=0.01, y=0.99, showarrow=False,
                        align="left", bgcolor="white", font=dict(size=10)
                    )
                ]
            )

            return dcc.Graph(figure=fig)