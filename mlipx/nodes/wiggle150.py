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



class Wiggle150(zntrack.Node):
    """Benchmark model against DMC-ICE13
    """
    
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    relative_energies_df_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "relative_energies.csv")
    mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae.txt")

    def run(self):
        calc = self.model.get_calculator()

        
        # download wiggle150 dataset
        dir = get_benchmark_data("wiggle150-structures.zip") / "wiggle150-structures"
        
        
        structures = read(dir / "ct5c00015_si_003.xyz", ":")


        # categorize structures into the three molecules
        ado_structures = []
        bpn_structures = []
        efa_structures = []

        ado_gs_structure = None
        bpn_gs_structure = None
        efa_gs_structure = None

        for structure in structures:
            
            name = list(structure.info.keys())[0]
            
            if "ado" in name:
                if "ado_00" in name:
                    ado_gs_structure = structure
                else:
                    ado_structures.append(structure)
            elif "bpn" in name:
                if "bpn_00" in name:
                    bpn_gs_structure = structure
                else:
                    bpn_structures.append(structure)
            elif "efa" in name:
                if "efa_00" in name:
                    efa_gs_structure = structure
                else:
                    efa_structures.append(structure)
                    

        def relative_energies(structures, gs_structure, calc, model_name):
    
            gs_structure.calc = calc
            gs_e_pred = gs_structure.get_potential_energy()
            gs_e_ref = float(list(gs_structure.info.keys())[1])

            rel_energy_pred = []
            rel_energy_ref = []

            for structure in tqdm(structures, desc=f"Evaluating relative energies for {model_name}"):
                structure.calc = calc
                e_pred = structure.get_potential_energy()
                e_ref = float(list(structure.info.keys())[1])
                
                rel_e_pred = e_pred - gs_e_pred
                rel_e_ref = e_ref - gs_e_ref
                
                rel_energy_pred.append(rel_e_pred)
                rel_energy_ref.append(rel_e_ref)
                
            rel_energy_pred = np.array(rel_energy_pred)
            rel_energy_ref = np.array(rel_energy_ref)
            
            return rel_energy_pred, rel_energy_ref
        
        
        ado_rel_energy_pred, ado_rel_energy_ref = relative_energies(ado_structures, ado_gs_structure, calc, model_name=self.model_name)
        bpn_rel_energy_pred, bpn_rel_energy_ref = relative_energies(bpn_structures, bpn_gs_structure, calc, model_name=self.model_name)
        efa_rel_energy_pred, efa_rel_energy_ref = relative_energies(efa_structures, efa_gs_structure, calc, model_name=self.model_name)
                
        wiggle_150_pred = np.concatenate([ado_rel_energy_pred, bpn_rel_energy_pred, efa_rel_energy_pred])
        wiggle_150_pred = wiggle_150_pred * 23.0605 # convert from eV to kcal/mol
        wiggle_150_ref = np.concatenate([ado_rel_energy_ref, bpn_rel_energy_ref, efa_rel_energy_ref])

        mae = np.mean(np.abs(wiggle_150_pred - wiggle_150_ref))
        with open(self.mae_output, "w") as f:
            f.write(f"MAE: {mae:.4f} kcal/mol\n")
            f.write(f"Number of structures: {len(wiggle_150_pred)}\n")
            f.write(f"Model: {self.model_name}\n")
        
        def build_rel_energy_df(structures, rel_pred, rel_ref, model_name):
            data = []
            for struct, pred, ref in zip(structures, rel_pred, rel_ref):
                name = list(struct.info.keys())[0]
                data.append({
                    "structure": name,
                    "ref": ref,
                    model_name: pred * 23.0605,  # convert from eV to kcal/mol
                })
            return pd.DataFrame(data)

        ado_df = build_rel_energy_df(ado_structures, ado_rel_energy_pred, ado_rel_energy_ref, self.model_name)
        bpn_df = build_rel_energy_df(bpn_structures, bpn_rel_energy_pred, bpn_rel_energy_ref, self.model_name)
        efa_df = build_rel_energy_df(efa_structures, efa_rel_energy_pred, efa_rel_energy_ref, self.model_name)

        df = pd.concat([ado_df, bpn_df, efa_df], axis=0)
        df.to_csv(self.relative_energies_df_output, index=False)



    @property
    def get_rel_energy_df(self):
        return pd.read_csv(self.relative_energies_df_output)
    
    
    


    @staticmethod
    def mae_plot_interactive(
        node_dict: dict,
        ui: Optional[str] = None,
        run_interactive: bool = True,
    ):
        # Merge all rel_energy_dfs
        rel_energy_df = pd.concat([
            node.get_rel_energy_df for node in node_dict.values()
        ], axis=0).reset_index(drop=True)

        functional_data = [
            {"Method": "ωB97M-D3BJ", "Relative Energy MAE [kcal/mol]": 1.18, "RMSE [kcal/mol]": 1.59, "Type": "Functional"},
            {"Method": "PBE-D3BJ", "Relative Energy MAE [kcal/mol]": 4.91, "RMSE [kcal/mol]": 5.68, "Type": "Functional"},
        ]

        # MAE and RMSE table for MLIPs
        mae_data = []
        model_names = [col for col in rel_energy_df.columns if col not in ["structure", "ref"]]
        for model in model_names:
            mae = np.mean(np.abs(rel_energy_df["ref"] - rel_energy_df[model]))
            rmse = np.sqrt(np.mean((rel_energy_df["ref"] - rel_energy_df[model]) ** 2))
            mae_data.append({
                "Method": model,
                "Relative Energy MAE [kcal/mol]": round(mae, 3),
                "RMSE [kcal/mol]": round(rmse, 3),
                "Type": "MLIP"
            })
        mae_df = pd.DataFrame(functional_data + mae_data)

        app = dash.Dash(__name__)
        from mlipx.dash_utils import dash_table_interactive

        app.layout = dash_table_interactive(
            df=mae_df,
            id="mae-score-table",
            title="Wiggle150 Relative Energy MAE Summary Table",
            extra_components=[
                html.Div(id="mae-model-plot"),
                dcc.Store(id="mae-table-last-clicked", data=None),
            ],
        )

        Wiggle150.register_callbacks(app, mae_df, rel_energy_df)

        from mlipx.dash_utils import run_app
        if not run_interactive:
            return app, mae_df, rel_energy_df
        return run_app(app, ui=ui)
    
    
    
    
    @staticmethod
    def register_callbacks(app, mae_df, rel_energy_df):
        @app.callback(
            Output("mae-model-plot", "children"),
            Output("mae-table-last-clicked", "data"),
            Input("mae-score-table", "active_cell"),
            State("mae-table-last-clicked", "data"),
        )
        def update_mae_scatter_plot(active_cell, last_clicked):
            if active_cell is None:
                raise PreventUpdate

            row = active_cell["row"]
            col = active_cell["column_id"]
            model_name = mae_df.loc[row, "Method"]

            if col not in mae_df.columns or col == "Method":
                return None, active_cell

            if last_clicked is not None and (
                active_cell["row"] == last_clicked.get("row") and
                active_cell["column_id"] == last_clicked.get("column_id")
            ):
                return None, None

            df_model = rel_energy_df[["structure", "ref", model_name]].dropna()

            # Fit line
            x = df_model["ref"].values
            y = df_model[model_name].values
            coeffs = np.polyfit(x, y, 1)
            fit_line = np.poly1d(coeffs)

            mae_val = np.mean(np.abs(y - x))
            rmse_val = np.sqrt(np.mean((y - x)**2))
            line_eq = f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"

            fig = px.scatter(
                df_model,
                x="ref",
                y=model_name,
                hover_name="structure",
                hover_data={"structure": True, "ref": True, model_name: True},
                labels={"ref": "Reference Energy", model_name: f"{model_name} Prediction"},
                title=f"Predicted vs Reference Energies: {model_name}"
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=fit_line(x),
                    mode="lines",
                    name=f"{line_eq}<br>MAE={mae_val:.3f}, RMSE={rmse_val:.3f}",
                    line=dict(color="black", dash="dot"),
                )
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="lightgrey"),
                yaxis=dict(showgrid=True, gridcolor="lightgrey")
            )

            return dcc.Graph(figure=fig), active_cell