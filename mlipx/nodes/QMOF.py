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
from mlipx.benchmark_download_utils import get_benchmark_data



class QMOFBenchmark(zntrack.Node):
    """ 
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    n_mofs: int = zntrack.params(None)
    
    #qmof_ref_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "qmof_ref.csv")
    qmof_results_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "qmof_results.csv")
    qmof_mae_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "qmof_mae.json")
    qmof_atoms_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "qmof_atoms.xyz")
    
    
    def run(self):

        calc = self.model.get_calculator()
        base_dir = get_benchmark_data("QMOF.zip") / "QMOF"
        structures = read(base_dir / "qmof_valid_structures.xyz", index=":")


        def calc_mof_energies(structures, calc, model_name, n_mofs = None):
            energies_mlip = {}
            energies_ref = {}
            
            if n_mofs is not None:
                structures = structures[:n_mofs]

            for mof in tqdm(structures, desc=f"Calculating energies for {model_name}"):
                energies_ref[mof.info['qmof_id']] = mof.info['dft_energy'] - mof.info['dft_energy_vdw']
                
                mof.calc = calc
                e_pred = mof.get_potential_energy()
                energies_mlip[mof.info['qmof_id']] = e_pred
                
            num_atoms = {mof.info['qmof_id']: len(mof) for mof in structures}
                
            return energies_mlip, energies_ref, structures, num_atoms

        mlip_e, ref_e, mofs, num_atoms = calc_mof_energies(structures, calc, self.model_name, n_mofs=self.n_mofs)
        
        # Save complex structures for visualization
        write(self.qmof_atoms_path, mofs)
        
        

        all_results_df = pd.DataFrame({
            "qmof_id": list(mlip_e.keys()),
            "ref energy (eV)": list(ref_e.values()),
            "ref energy (eV/atom)": [ref_e[k] / num_atoms[k] for k in ref_e.keys()],
            f"{self.model_name} energy (eV)": list(mlip_e.values()),
            f"{self.model_name} energy (eV/atom)": [mlip_e[k] / num_atoms[k] for k in mlip_e.keys()],
        })
        
        mae = (all_results_df[f"{self.model_name} energy"] - all_results_df["ref energy"]).abs().mean()

        all_results_df.to_csv(self.qmof_results_path, index=False)

        with open(self.qmof_mae_path, "w") as f:
            json.dump(mae, f)



    @property
    def get_results(self) -> pd.DataFrame:
        """ Returns the results DataFrame """
        return pd.read_csv(self.qmof_results_path)
    @property
    def get_mae(self) -> float:
        """ Returns the mean absolute error """
        with open(self.qmof_mae_path, "r") as f:
            return json.load(f)
    @property
    def get_mofs(self) -> Atoms:
        """ Returns the MOFs as ASE Atoms object """
        return read(self.qmof_atoms_path, index=":")



    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "QMOFBenchmark"],
        cache_dir: str = "app_cache/MOF/QMOF_cache/",
        normalise_to_model: Optional[str] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        mae_dict = {}
        pred_dfs = []

        # save images for WEAS viewer
        mofs = list(node_dict.values())[0].get_mofs
        save_dir = os.path.abspath(f"assets/QMOF/")
        os.makedirs(save_dir, exist_ok=True)
        write(os.path.join(save_dir, "mofs.xyz"), mofs)
        
        ref_df = list(node_dict.values())[0].get_ref

        for model_name, node in node_dict.items():
            results_df = node.get_results
            mae = node.get_mae
            mae_dict[model_name] = mae

            # Save predictions DataFrame
            results_df["Model"] = model_name
            pred_dfs.append(results_df)



        mae_df = pd.DataFrame.from_dict(mae_dict, orient="index", columns=["MAE (kcal/mol)"]).reset_index()
        mae_df = mae_df.rename(columns={"index": "Model"})

        pred_full_df = pd.concat(pred_dfs, ignore_index=True)

        mae_df["Score"] = mae_df["MAE (kcal/mol)"]

        if normalise_to_model:
            norm_value = mae_df.loc[mae_df["Model"] == normalise_to_model, "Score"].values[0]
            mae_df["Score"] /= norm_value

        mae_df["Rank"] = mae_df["Score"].rank(ascending=True, method="min")

        mae_df.to_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        pred_full_df.to_pickle(os.path.join(cache_dir, "predictions_df.pkl"))
        


    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/supramolecular_complexes/S30L_cache/",
        app: dash.Dash | None = None,
        ui=None,
    ):
        from mlipx.dash_utils import run_app

        mae_df = pd.read_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        pred_df = pd.read_pickle(os.path.join(cache_dir, "predictions_df.pkl"))

        layout = S30LBenchmark.build_layout(mae_df)

        def callback_fn(app_instance):
            S30LBenchmark.register_callbacks(app_instance, pred_df)

        if app is None:
            assets_dir = os.path.abspath("assets")
            print("Serving assets from:", assets_dir)
            app = dash.Dash(__name__, assets_folder=assets_dir)
            
            app.layout = layout
            callback_fn(app)
            return run_app(app, ui=ui)
        else:
            return layout, callback_fn


    @staticmethod
    def build_layout(mae_df):
        return html.Div([
            dash_table_interactive(
                df=mae_df.round(3),
                id="S30L-table",
                benchmark_info="Benchmark info: Interaction energies for host-guest complexes in S30L.",
                title="S30L Benchmark",
                tooltip_header={
                    "Model": "Name of the MLIP model",
                    "Score": "Absolute value of Delta (meV); normalized if specified",
                    "Rank": "Ranking of model by Score (lower is better)"
                },
                extra_components=[
                    html.Div(
                        children=[
                            html.Div(
                                "Click on the points to see the structure!",
                                style={
                                    "color": "red",
                                    "fontWeight": "bold",
                                    "marginBottom": "10px"
                                }
                            ),
                            dcc.Graph(id="S30L-plot")
                        ],
                        id="S30L-plot-container",
                        style={"display": "none"},
                    ),
                    html.Div(id="weas-viewer-S30L", style={'marginTop': '20px'}),
                ]
            )
        ])
        
        
    @staticmethod
    def register_callbacks(
        app,
        pred_df
    ):
        from mlipx.dash_utils import weas_viewer_callback

        @app.callback(
            Output("S30L-plot", "figure"),
            Output("S30L-plot-container", "style"),
            Input("S30L-table", "active_cell"),
            State("S30L-table", "data"),
        )
        def update_s30l_plot(active_cell, table_data):
            if not active_cell:
                raise PreventUpdate

            row = active_cell["row"]
            clicked_model = table_data[row]["Model"]
            col = active_cell["column_id"]

            if col == "Model":
                return dash.no_update, {"display": "none"}

            df = pred_df.copy()
            df = df[df["Model"] == clicked_model]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["E_ref (eV)"],
                    y=df["E_model (eV)"],
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    customdata=df["Index"],  # ← clean value
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

            return fig, {"display": "block"}

        @app.callback(
            Output("weas-viewer-S30L", "children"),
            Output("weas-viewer-S30L", "style"),
            Input("S30L-plot", "clickData"),
        )
        def update_weas_viewer(clickData):
            if not clickData:
                raise PreventUpdate
            return weas_viewer_callback(
                clickData,
                "assets/S30L/complex_atoms.xyz",
                mode="index",
                index_key="pointIndex"
            )