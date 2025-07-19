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



class S30LBenchmark(zntrack.Node):
    """ 
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    
    s30l_ref_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s30l_ref.csv")
    s30l_pred_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s30l_pred.csv")
    s30l_mae_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s30l_mae.json")
    s30l_complex_atoms_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s30l_complex_atoms.xyz")
    
    
    def run(self):

        def _read_charge(folder: Path) -> float:
            for f in folder.iterdir():
                if f.name.upper() == ".CHRG":
                    try:
                        return float(f.read_text().strip())
                    except ValueError:
                        warnings.warn(f"Invalid charge in {f} – assuming neutral.")
            return 0.0

        def _read_atoms(folder: Path, ident: str) -> Atoms:
            coord = next((p for p in folder.iterdir() if p.name.lower().startswith("coord")), None)
            if coord is None:
                raise FileNotFoundError(f"No coord file in {folder}")
            atoms = read(coord, format="turbomole")
            atoms.info.update({"identifier": ident, "charge": int(_read_charge(folder))})
            return atoms

        def load_complex(index: int, root: Path) -> Dict[str, Atoms]:
            base = root / f"{index}"
            if not base.exists():
                raise FileNotFoundError(base)
            return {
                "host":    _read_atoms(base / "A",  f"{index}_host"),
                "guest":   _read_atoms(base / "B",  f"{index}_guest"),
                "complex": _read_atoms(base / "AB", f"{index}_complex"),
            }

        def interaction_energy(frags: Dict[str, Atoms], calc: Calculator) -> float:
            frags["complex"].calc = calc
            e_complex = frags["complex"].get_potential_energy()
            frags["host"].calc = calc
            e_host = frags["host"].get_potential_energy()
            frags["guest"].calc = calc
            e_guest = frags["guest"].get_potential_energy()
            return e_complex - e_host - e_guest

        def parse_references(path: Path) -> Dict[int, float]:
            KCAL_TO_EV = 0.04336414
            refs: Dict[int, float] = {}
            for idx, ln in enumerate(path.read_text().splitlines()):
                ln = ln.strip()
                if not ln:
                    continue
                kcal = float(ln.split()[0])
                refs[idx+1] = kcal * KCAL_TO_EV
            return refs

        calc = self.model.get_calculator()
        base_dir = get_benchmark_data("S30L.zip") / "S30L/s30l_test_set"
        ref_file = base_dir / "references_s30.txt"
        refs = parse_references(ref_file)

        rows = []
        complex_atoms_list = []
        for idx in tqdm(range(1, 31), desc=f"Benchmarking {self.model_name}"):
            fragments = load_complex(idx, base_dir)
            e_model = interaction_energy(fragments, calc)
            e_ref = refs[idx]
            rows.append(
                {
                    "Index": idx,
                    "E_ref (eV)": e_ref,
                    f"E_{self.model_name} (eV)": e_model,
                    "Error (eV)": e_model - e_ref,
                    "Error (kcal/mol)": (e_model - e_ref) / 0.04336414,
                    "n_atoms": len(fragments["complex"]),
                }
            )
            fragments["complex"].info["Index"] = idx
            complex_atoms_list.append(fragments["complex"])
            

        df = pd.DataFrame(rows)

        ref_df = df[["Index", "E_ref (eV)"]]
        model_df = df[["Index", f"E_{self.model_name} (eV)"]].copy()
        mae = df["Error (kcal/mol)"].abs().mean()

        # Save complex structures for visualization
        write(self.s30l_complex_atoms_path, complex_atoms_list)

        ref_df.to_csv(self.s30l_ref_path, index=False)
        model_df.to_csv(self.s30l_pred_path, index=False)
        with open(self.s30l_mae_path, "w") as f:
            json.dump(mae, f)


    @property
    def get_ref(self) -> pd.DataFrame:
        """Load reference data from CSV file."""
        return pd.read_csv(self.s30l_ref_path)
    
    @property
    def get_pred(self) -> pd.DataFrame:
        """Load predicted data from CSV file."""
        return pd.read_csv(self.s30l_pred_path)

    @property
    def get_mae(self):
        """Load MAE from JSON file."""
        with open(self.s30l_mae_path, "r") as f:
            data = json.load(f)
        return data

    @property
    def get_complex_atoms(self) -> List[Atoms]:
        """Load complex atoms from xyz file."""
        return read(self.s30l_complex_atoms_path, index=":")



    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "S30LBenchmark"],
        cache_dir: str = "app_cache/supramolecular_complexes/S30L_cache/",
        normalise_to_model: Optional[str] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        mae_dict = {}
        pred_dfs = []

        # save images for WEAS viewer
        complex_atoms = list(node_dict.values())[0].get_complex_atoms
        save_dir = os.path.abspath(f"assets/S30L/")
        os.makedirs(save_dir, exist_ok=True)
        write(os.path.join(save_dir, "complex_atoms.xyz"), complex_atoms)
        
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