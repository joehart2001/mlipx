import pathlib
import typing as t
import json

import ase
import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative
import zntrack

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
from mlipx.dash_utils import dash_table_interactive
import socket
import time
from typing import List, Dict, Any, Optional
import cctk
from ase.io.trajectory import Trajectory
from plotly.io import write_image
from ase.io import read
from tqdm import tqdm
from copy import deepcopy
from ase.io import read, write
from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mlipx.benchmark_download_utils import get_benchmark_data




class S24Benchmark(zntrack.Node):
    """Benchmark model for s24 dataset.
    
    
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    pred_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s24_pred_energies.csv")
    ref_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s24_ref_energies.csv")
    s24_mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s24_mae.json")
    s24_structures_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "s24_mol_surface_atoms.xyz")
    

    def run(self):
        from ase.io import read
        import copy

        calc = self.model.get_calculator()
        data = get_benchmark_data("s24.zip") / "s24/s24_data.extxyz"
        atoms_list = read(data, ":")

        def compute_absorption_energy(surface_e, mol_surf_e, molecule_e, adsorbate_count: int = 1) -> float:
            print("adsorbate_count:", adsorbate_count)
            return mol_surf_e - (surface_e + adsorbate_count * molecule_e)

        def get_adsorbate_count(surface_atoms: ase.Atoms, mol_surface_atoms: ase.Atoms, molecule_atoms: ase.Atoms) -> int:
            """Infer how many copies of the molecule are present in the adsorbed structure."""
            molecule_atom_count = len(molecule_atoms)
            if molecule_atom_count == 0:
                return 1

            extra_atoms = len(mol_surface_atoms) - len(surface_atoms)
            if extra_atoms <= 0:
                return 1

            adsorbate_count, remainder = divmod(extra_atoms, molecule_atom_count)
            if remainder:
                # Fall back to rounding if the structure deviates slightly from an exact multiple.
                adsorbate_count = max(1, round(extra_atoms / molecule_atom_count))

            return max(1, adsorbate_count)

        ref = {}
        pred = {}
        mol_surface_list = []
        

        for i in tqdm(range(0, len(atoms_list), 3), desc=f"Processing Triplets for model: {self.model_name}"):
            surface = atoms_list[i]
            mol_surface = atoms_list[i + 1]
            molecule = atoms_list[i + 2]
            surface_formula = surface.get_chemical_formula()
            molecule_formula = molecule.get_chemical_formula()

            adsorbate_count = get_adsorbate_count(surface, mol_surface, molecule)
            system_name = f"{surface_formula}-{molecule_formula}"
            if adsorbate_count > 1:
                system_name = f"{surface_formula}-{adsorbate_count}x{molecule_formula}"

            mol_surface.info["name"] = system_name
            mol_surface.info["adsorbate_count"] = adsorbate_count


            dft_surface_energy = surface.get_potential_energy()
            dft_mol_surface_energy = mol_surface.get_potential_energy()
            dft_molecule_energy = molecule.get_potential_energy()
            dft_abs_energy = compute_absorption_energy(
                dft_surface_energy, dft_mol_surface_energy, dft_molecule_energy, adsorbate_count
            )

            surface.calc = deepcopy(calc)
            mol_surface.calc = deepcopy(calc)
            molecule.calc = deepcopy(calc)

            surface_energy = surface.get_potential_energy()
            mol_surface_energy = mol_surface.get_potential_energy()
            molecule_energy = molecule.get_potential_energy()

            calc_abs_energy = compute_absorption_energy(
                surface_energy, mol_surface_energy, molecule_energy, adsorbate_count
            )

            pred[system_name] = calc_abs_energy
            ref[system_name] = dft_abs_energy
            
            mol_surface_list.append(mol_surface)

        mae = np.mean([abs(pred[key] - ref[key]) for key in pred.keys() if key in ref])

        write(self.s24_structures_output, mol_surface_list)

        # Save pred and ref as DataFrames with a system col, as CSV
        self.pred_energy_output.parent.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame.from_dict(pred, orient="index", columns=["Predicted Adsorption Energy (eV)"])
        pred_df.index.name = "System"
        ref_df = pd.DataFrame.from_dict(ref, orient="index", columns=["Reference Adsorption Energy (eV)"])
        ref_df.index.name = "System"
        pred_df.to_csv(self.pred_energy_output, index=True)
        ref_df.to_csv(self.ref_energy_output, index=True)
        with open(self.s24_mae_output, "w") as f:
            json.dump(mae, f)
            
            
            
    @property
    def pred_energy(self) -> Dict[str, float]:
        """Predicted adsorption energies."""
        df = pd.read_csv(self.pred_energy_output, index_col="System")
        return dict(zip(df.index, df["Predicted Adsorption Energy (eV)"]))

    @property
    def ref_energy(self) -> Dict[str, float]:
        """Reference adsorption energies."""
        df = pd.read_csv(self.ref_energy_output, index_col="System")
        return dict(zip(df.index, df["Reference Adsorption Energy (eV)"]))
    @property
    def get_mae(self) -> Dict[str, float]:
        """Mean Absolute Error for S24 benchmark."""
        with open(self.s24_mae_output, "r") as f:
            return json.load(f)
        
    @property
    def get_structures(self) -> List[ase.Atoms]:
        """Get the structures from the S24 benchmark."""
        return read(self.s24_structures_output, ":")
        
    
        
        
        
    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "S24Benchmark"],
        cache_dir: str = "app_cache/surface_benchmark/s24_cache/",
        normalise_to_model: t.Optional[str] = None,
    ):
        from scipy.stats import pearsonr
        
        mol_surface_atoms = node_dict[list(node_dict.keys())[0]].get_structures
        save_dir = os.path.abspath(f"assets/S24/")
        os.makedirs(save_dir, exist_ok=True)
        write(os.path.join(save_dir, "mol_surface_atoms.xyz"), mol_surface_atoms)
        
        
        mae_dict = {}
        
        pred_dict = {}
        ref_dict = {}
        mae_dict = {}


        for model_name, node in node_dict.items():
            pred_dict[model_name] = node.pred_energy
            ref_dict[model_name] = node.ref_energy
            mae_dict[model_name] = node.get_mae
        
        pred_df = pd.DataFrame(pred_dict).T
        ref_df = pd.DataFrame(ref_dict).T

        mae_df = pd.DataFrame.from_dict(mae_dict, orient="index", columns=["MAE (meV)"])
        mae_df.index.name = "Model"
        mae_df.reset_index(inplace=True)
                    
        mae_df["Score \u2193"] = mae_df["MAE (meV)"]
        
        if normalise_to_model is not None:
            mae_df["Score \u2193"] = mae_df["Score \u2193"] / mae_df[mae_df["Model"] == normalise_to_model]["Score \u2193"].values[0]

        mae_df['Rank'] = mae_df['Score \u2193'].rank(ascending=True, method="min")
        
        mae_df = mae_df.round(3)


        os.makedirs(cache_dir, exist_ok=True)
        mae_df.to_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        pred_df.to_pickle(os.path.join(cache_dir, "pred_df.pkl"))
        ref_df.to_pickle(os.path.join(cache_dir, "ref_df.pkl"))






    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/surface_benchmark/s24_cache",
        app: dash.Dash | None = None,
        ui=None,
    ):
        """Launch the S24 dashboard or register it into an existing Dash app."""
        
        from mlipx.dash_utils import run_app
        mae_df = pd.read_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        pred_df = pd.read_pickle(os.path.join(cache_dir, "pred_df.pkl"))
        ref_df = pd.read_pickle(os.path.join(cache_dir, "ref_df.pkl"))
        
        
        layout = S24Benchmark.build_layout(mae_df)

        def callback_fn(app_instance):
            S24Benchmark.register_callbacks(
                app_instance,
                pred_df=pred_df,
                ref_df=ref_df
            )

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
                id="s24-mae-table",
                benchmark_info="Benchmark info:",
                title="S24 Dataset",
                tooltip_header={
                    "Model": "Name of the MLIP",
                    "MAE (meV)": "Mean Absolute Error (meV)",
                    "Score \u2193": "MAE (normalised)",
                    "Rank": "Model rank based on score (lower is better)"
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
                            dcc.Graph(id="s24-plot")
                        ],
                        id="s24-plot-container",
                        style={"display": "none"},
                    ),
                    html.Div(id="weas-viewer-s24", style={'marginTop': '20px'}),
                ]
            )
        ])
        
        
        
        
            
            
            
    @staticmethod
    def register_callbacks(
        app, 
        pred_df: pd.DataFrame,
        ref_df: pd.DataFrame
    ): 
        from mlipx.dash_utils import weas_viewer_callback
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr

        @app.callback(
            Output("s24-plot", "figure"),
            Output("s24-plot-container", "style"),
            Input("s24-mae-table", "active_cell"),
            State("s24-mae-table", "data"),
        )
        def update_s24_plot(active_cell, table_data):

            if not active_cell:
                raise PreventUpdate

            row = active_cell["row"]
            clicked_model = table_data[row]["Model"]
            col = active_cell["column_id"]
            
            if col == "Model":
                return None

            if clicked_model not in pred_df.index or clicked_model not in ref_df.index:
                raise PreventUpdate

            # Get prediction and reference dicts
            pred = pred_df.loc[clicked_model].dropna()
            ref = ref_df.loc[clicked_model].dropna()

            # Align keys
            common_keys = list(set(pred.index) & set(ref.index))
            pred_vals = [pred[k] for k in common_keys]
            ref_vals = [ref[k] for k in common_keys]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ref_vals,
                    y=pred_vals,
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    text=[f"System: {k}<br>DFT: {ref[k]:.3f} eV<br>Pred: {pred[k]:.3f} eV" for k in common_keys],
                    hoverinfo="text",
                    name=clicked_model
                )
            )
            fig.add_trace(go.Scatter(
                x=[min(ref_vals), max(ref_vals)], y=[min(ref_vals), max(ref_vals)],
                mode="lines",
                line=dict(dash="dash", color="black", width=1),
                showlegend=False
            ))

            mae = mean_absolute_error(ref_vals, pred_vals)

            fig.update_layout(
                title=f"{clicked_model} vs DFT Adsorption Energies",
                xaxis_title="DFT Adsorption Energy [eV]",
                yaxis_title=f"{clicked_model} Prediction [eV]",
                annotations=[
                    dict(
                        text=f"N = {len(ref_vals)}<br>MAE = {mae:.3f} eV",
                        xref="paper", yref="paper",
                        x=0.01, y=0.99, showarrow=False,
                        align="left", bgcolor="white", font=dict(size=10)
                    )
                ]
            )

            return fig, {"display": "block"}

        @app.callback(
            Output("weas-viewer-s24", "children"),
            Output("weas-viewer-s24", "style"),
            Input("s24-plot", "clickData"),
        )
        def update_weas_viewer(clickData):
            if not clickData:
                raise PreventUpdate
            return weas_viewer_callback(
                clickData,
                "assets/S24/mol_surface_atoms.xyz",
                mode="info",
                info_key="name",
            )
