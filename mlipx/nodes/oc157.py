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
from tqdm import tqdm

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




# OC157 benchmark node
class OC157Benchmark(zntrack.Node):
    """Benchmark model for OC157 dataset
    
    Prediction of the most stable structures for a molecule-surface system
    - relative energies between 3 structures and 157 molecule surface combinations
    - identification of the most stable structure

    
    
    
    reference: MPRelaxSet DFT (Becke-Johnson damped D3 dispersion correction)
    - surfaces taken from the Open Catalyst Challenge 2023
    - 200 refs but excludes those with Hubbard U -> 157
    - 3 structures per system (triplet)
    
    
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    oc_rel_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_rel_energies.csv")
    ref_rel_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_ref_rel_energies.csv")
    oc_mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_mae.json")
    oc_ranks_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "oc157_rank_accuracy.json")
    

    def run(self):
        from ase.io import read
        import copy

        calc = self.model.get_calculator()
        base_dir = get_benchmark_data("OC_Dataset.zip") / "OC_Dataset"
        n_systems = 200
        skip_hubbard_U = True

        # ---- Helper functions ----
        def incar_has_hubbard_u(incar_path):
            if not incar_path.is_file():
                return False
            return "LDAU" in incar_path.read_text()

        def find_energy(outcar, key="energy  without entropy="):
            with open(outcar, encoding="ISO-8859-1") as fh:
                hits = [line for line in fh if key in line]
            if not hits:
                raise RuntimeError(f"No energy found in {outcar}")
            return float(hits[-1].split()[-1])

        def read_structure(folder):
            for fname in ("CONTCAR", "POSCAR"):
                fpath = folder / fname
                if fpath.is_file():
                    return read(fpath, format="vasp")
            raise FileNotFoundError(f"No CONTCAR/POSCAR in {folder}")

        def relative_energies(E1, E2, E3):
            return [E2 - E1, E3 - E1, E3 - E2]

        def evaluate_model(triplet):
            energies = []
            for atoms in triplet:
                atoms_c = copy.deepcopy(atoms)
                atoms_c.calc = calc
                energies.append(atoms_c.get_potential_energy())
            return energies

        triplets = []
        dft_E = []
        system_ids = []
        system_compositions = []

        for idx in tqdm(range(1, n_systems + 1), desc="Loading OC157 systems"):
            sys_id = f"{idx:03d}"
            sys_dir = base_dir / sys_id

            if skip_hubbard_U and incar_has_hubbard_u(sys_dir / "1" / "INCAR"):
                continue

            poscar = (sys_dir / "1" / "POSCAR").read_text().splitlines()[0].strip()
            system_ids.append(sys_id)
            system_compositions.append(poscar)

            trio_atoms, trio_E = [], []
            for member in (1, 2, 3):
                subdir = sys_dir / str(member)
                atoms = read_structure(subdir)
                energy = find_energy(subdir / "OUTCAR")
                trio_atoms.append(atoms)
                trio_E.append(energy)

            triplets.append(trio_atoms)
            dft_E.extend(trio_E)

        rel_dft = []
        rank_dft = []
        rel_pred = []
        rank_pred = []

        dft_E = np.array(dft_E)
        
        for trio in tqdm(triplets, desc="Evaluating model on triplets"):
            Eref = dft_E[:3]
            dft_E = dft_E[3:]
            rel_dft.append(relative_energies(*Eref))
            rank_dft.append(int(np.argmin(Eref)))

            pred_E = evaluate_model(trio)
            rel_pred.append(relative_energies(*pred_E))
            rank_pred.append(int(np.argmin(pred_E)))

        rel_dft = np.array(rel_dft)
        rel_pred = np.array(rel_pred)
        rank_dft = np.array(rank_dft)
        rank_pred = np.array(rank_pred)

        mae = float(np.mean(np.abs(rel_dft - rel_pred)))
        rank_acc = float(np.mean(rank_dft == rank_pred))

        df_rel = pd.DataFrame({
            "system_id": system_ids,
            "composition": system_compositions,
            "dE_2-1": rel_pred[:, 0],
            "dE_3-1": rel_pred[:, 1],
            "dE_3-2": rel_pred[:, 2]
        })

        df_rel.to_csv(self.oc_rel_energy_output, index=False)
        df_ref = pd.DataFrame({
            "system_id": system_ids,
            "composition": system_compositions,
            "dE_2-1": rel_dft[:, 0],
            "dE_3-1": rel_dft[:, 1],
            "dE_3-2": rel_dft[:, 2]
        })
        df_ref.to_csv(self.ref_rel_energy_output, index=False)
        with open(self.oc_mae_output, "w") as f:
            json.dump(mae, f)
        with open(self.oc_ranks_output, "w") as f:
            json.dump(rank_acc, f)

    @property
    def get_mae(self):
        with open(self.oc_mae_output, "r") as f:
            return json.load(f)

    @property
    def get_rank_acc(self):
        with open(self.oc_ranks_output, "r") as f:
            return json.load(f)

    @property
    def get_relative_energies(self):
        return pd.read_csv(self.oc_rel_energy_output)
    
    @property
    def get_ref(self):
        return pd.read_csv(self.ref_rel_energy_output)
    
    
    
    
    
    
    
    
    
    
    
    
    # @staticmethod
    # def benchmark_interactive(
    #     node_dict: dict[str, "OC157Benchmark"],
    #     ui: str | None = None,
    #     run_interactive: bool = True,
    #     normalise_to_model: t.Optional[str] = None,
    # ):
    #     from mlipx.dash_utils import run_app
    #     from scipy.stats import pearsonr

    #     mae_dict = {}
    #     rmsd_dict = {}
    #     pearsons_dict = {}
    #     rank_dict = {}
    #     rel_df_all = []

    #     dft_df = list(node_dict.values())[0].get_ref.copy()
    #     dft_df = dft_df[["dE_2-1", "dE_3-1", "dE_3-2"]]
    #     dft_values = dft_df.values.flatten()

    #     for model_name, node in node_dict.items():
    #         rel_df = node.get_relative_energies.copy()
    #         rel_df["Model"] = model_name
    #         pred_values = rel_df[["dE_2-1", "dE_3-1", "dE_3-2"]].values.flatten()
    #         mae = np.mean(np.abs(pred_values - dft_values))
    #         # Insert RMSD and Pearson r calculation
    #         rmsd = np.sqrt(np.mean((pred_values - dft_values) ** 2))
    #         r, _ = pearsonr(dft_values, pred_values)
    #         rmsd_dict[model_name] = rmsd
    #         pearsons_dict[model_name] = r
    #         mae_dict[model_name] = mae
    #         rank_dict[model_name] = node.get_rank_acc
    #         rel_df_all.append(rel_df)

    #     rel_all_df = pd.concat(rel_df_all, axis=0)

    #     mae_df = (
    #         pd.DataFrame(mae_dict.items(), columns=["Model", "MAE (meV)"])
    #         .merge(pd.DataFrame(rank_dict.items(), columns=["Model", "Ranking Accuracy"]), on="Model")
    #         .merge(pd.DataFrame(rmsd_dict.items(), columns=["Model", "RMSD (meV)"]), on="Model")
    #         .merge(pd.DataFrame(pearsons_dict.items(), columns=["Model", "Pearson r"]), on="Model")
    #     )

    #     mae_df["Score"] = mae_df[["MAE (meV)", "Ranking Accuracy"]].apply(
    #         lambda row: row["MAE (meV)"] / (row["Ranking Accuracy"] + 1e-8), axis=1
    #     )

    #     if normalise_to_model is not None:
    #         mae_df["Score"] = mae_df["Score"] / mae_df[mae_df["Model"] == normalise_to_model]["Score"].values[0]

    #     mae_df['Rank'] = mae_df['Score'].rank(ascending=True)

    #     save_path = Path("benchmark_stats/further_applications/oc157")
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     mae_df.to_csv(save_path / "mae.csv", index=False)
    #     rel_all_df.to_csv(save_path / "relative_energies.csv", index=False)

    #     if ui is None and run_interactive:
    #         return mae_df, rel_all_df

    #     # Create a grid of subplots, one for each model
    #     from plotly.subplots import make_subplots
    #     from sklearn.metrics import mean_absolute_error, mean_squared_error
    #     from scipy.stats import pearsonr

    #     n_models = mae_df["Model"].nunique()
    #     n_cols = min(n_models, 3)
    #     n_rows = int(np.ceil(n_models / n_cols))

    #     fig_rel = make_subplots(
    #         rows=n_rows, cols=n_cols,
    #         subplot_titles=mae_df["Model"].tolist(),
    #         shared_xaxes=True, shared_yaxes=True,
    #         horizontal_spacing=0.05, vertical_spacing=0.1
    #     )

    #     for i, model in enumerate(mae_df["Model"]):
    #         full_df = rel_df_all[i].copy()
    #         system_labels = full_df[["system_id", "composition"]].apply(
    #             lambda row: f"{row.system_id}: {row.composition}", axis=1
    #         ).tolist()
    #         energy_types = ["dE_2-1", "dE_3-1", "dE_3-2"]
    #         pred_df = full_df.drop(columns=["Model", "system_id", "composition"]).reset_index(drop=True)
    #         pred_values = pred_df[["dE_2-1", "dE_3-1", "dE_3-2"]].values.flatten()
    #         n_pred = len(pred_values)
    #         dft_slice = dft_values[:n_pred]
    #         energy_labels = []
    #         hover_labels = []
    #         for system_label in system_labels:
    #             for energy_type in energy_types:
    #                 hover_labels.append(system_label)
    #                 energy_labels.append(energy_type)
    #         merged_df = pd.DataFrame({
    #             "DFT Relative Energy": dft_slice.astype(float),
    #             "Predicted Relative Energy": pred_values.astype(float)
    #         })
    #         row = i // n_cols + 1
    #         col = i % n_cols + 1
    #         fig_rel.add_trace(
    #             go.Scatter(
    #                 x=merged_df["DFT Relative Energy"],
    #                 y=merged_df["Predicted Relative Energy"],
    #                 mode="markers",
    #                 name=model,
    #                 marker=dict(size=6, opacity=0.6),
    #                 text=[
    #                     f"{label}<br>{etype}:<br>DFT: {dft:.3f} eV<br>Pred: {pred:.3f} eV"
    #                     for label, etype, dft, pred in zip(
    #                         hover_labels,
    #                         energy_labels,
    #                         merged_df["DFT Relative Energy"],
    #                         merged_df["Predicted Relative Energy"],
    #                     )
    #                 ],
    #                 hoverinfo="text",
    #             ),
    #             row=row, col=col
    #         )
    #         fig_rel.add_trace(
    #             go.Scatter(
    #                 x=[-1, 5], y=[-1, 5],
    #                 mode="lines",
    #                 line=dict(dash="dash", color="black", width=1),
    #                 showlegend=False
    #             ),
    #             row=row, col=col
    #         )
    #         mae = mean_absolute_error(merged_df["DFT Relative Energy"], merged_df["Predicted Relative Energy"])
    #         mse = mean_squared_error(merged_df["DFT Relative Energy"], merged_df["Predicted Relative Energy"])
    #         rmsd = np.sqrt(mse)
    #         r, _ = pearsonr(merged_df["DFT Relative Energy"], merged_df["Predicted Relative Energy"])
    #         n_points = len(merged_df)
    #         fig_rel.add_annotation(
    #             text=f"N = {n_points} energies, {n_points/3} systems<br>MAE = {mae:.2f} eV<br>RMSD = {rmsd:.2f} eV<br>Pearson r = {r:.2f}",
    #             xref="paper",
    #             yref="paper",
    #             x=0.01, y=0.99,
    #             xanchor="left", yanchor="top",
    #             showarrow=False,
    #             font=dict(size=10),
    #             align="left",
    #             bgcolor="white"
    #         )
    #         fig_rel.update_xaxes(title_text="Relative energy DFT [eV]", range=[-1, 5], row=row, col=col)
    #         fig_rel.update_yaxes(title_text=f"Relative energy {model} [eV]", range=[-1, 5], row=row, col=col)

    #     fig_rel.update_layout(
    #         #height=300 * n_rows,
    #         #width=400 * n_cols,
    #         title_text="Predicted vs DFT Relative Energies (Per Model)",
    #         showlegend=False
    #     )

    #     app = dash.Dash(__name__)
    #     app.layout = html.Div([
    #         dash_table_interactive(
    #             df=mae_df.round(3),
    #             id="oc157-mae-table",
    #             benchmark_info="Benchmark info: Performance in predicting relative energies between 3 structures for 157 molecule-surface combinations.",
    #             title="OC157 Dataset: MAE and Ranking Accuracy",
    #         ),
    #         html.Div(id="oc157-model-plot"),
    #     ], style={"backgroundColor": "white", "padding": "20px"})

    #     register_oc157_callbacks(app, rel_df_all=rel_df_all, dft_df=dft_df)

    #     if not run_interactive:
    #         return app, mae_df, rel_all_df

    #     return run_app(app, ui=ui)



    
    
    
    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "OC157Benchmark"],
        cache_dir: str = "app_cache/surface_benchmark/oc157_cache/",
        normalise_to_model: t.Optional[str] = None,
    ):
        from scipy.stats import pearsonr

        mae_dict = {}
        rmsd_dict = {}
        pearsons_dict = {}
        rank_dict = {}
        rel_df_all = []

        dft_df = list(node_dict.values())[0].get_ref.copy()
        dft_df = dft_df[["dE_2-1", "dE_3-1", "dE_3-2"]]
        dft_values = dft_df.values.flatten()

        for model_name, node in node_dict.items():
            rel_df = node.get_relative_energies.copy()
            rel_df["Model"] = model_name
            pred_values = rel_df[["dE_2-1", "dE_3-1", "dE_3-2"]].values.flatten()
            mae = np.mean(np.abs(pred_values - dft_values))
            rmsd = np.sqrt(np.mean((pred_values - dft_values) ** 2))
            r, _ = pearsonr(dft_values, pred_values)
            rmsd_dict[model_name] = rmsd
            pearsons_dict[model_name] = r
            mae_dict[model_name] = mae
            rank_dict[model_name] = node.get_rank_acc
            rel_df_all.append(rel_df)

        rel_all_df = pd.concat(rel_df_all, axis=0)

        mae_df = (
            pd.DataFrame(mae_dict.items(), columns=["Model", "MAE (meV)"])
            .merge(pd.DataFrame(rank_dict.items(), columns=["Model", "Ranking Accuracy"]), on="Model")
            .merge(pd.DataFrame(rmsd_dict.items(), columns=["Model", "RMSD (meV)"]), on="Model")
            .merge(pd.DataFrame(pearsons_dict.items(), columns=["Model", "Pearson r"]), on="Model")
        )
        
        mae_df["Score"] = mae_df["MAE (meV)"] + (1 - mae_df["Ranking Accuracy"]) / 2

        # mae_df["Score"] = mae_df[["MAE (meV)", "Ranking Accuracy"]].apply(
        #     lambda row: row["MAE (meV)"] / (row["Ranking Accuracy"] + 1e-8), axis=1
        # )

        if normalise_to_model is not None:
            mae_df["Score"] = mae_df["Score"] / mae_df[mae_df["Model"] == normalise_to_model]["Score"].values[0]

        mae_df['Rank'] = mae_df['Score'].rank(ascending=True)

        os.makedirs(cache_dir, exist_ok=True)
        mae_df.to_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        rel_all_df.to_pickle(os.path.join(cache_dir, "rel_energy_df.pkl"))
        dft_df.to_pickle(os.path.join(cache_dir, "dft_df.pkl"))
        
        return



    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/surface_benchmark/oc157_cache",
        app: dash.Dash | None = None,
        ui=None,
    ):
        """Launch the OC157 dashboard or register it into an existing Dash app."""
        
        from mlipx.dash_utils import run_app
        mae_df = pd.read_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        rel_df_all = pd.read_pickle(os.path.join(cache_dir, "rel_energy_df.pkl"))
        dft_df = pd.read_pickle(os.path.join(cache_dir, "dft_df.pkl"))

        layout = OC157Benchmark.build_layout(mae_df)

        def callback_fn(app_instance):
            OC157Benchmark.register_oc157_callbacks(
                app_instance,
                rel_df_all=rel_df_all,
                dft_df=dft_df,
            )

        if app is None:
            app = dash.Dash(__name__)
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
                id="oc157-mae-table",
                benchmark_info="Benchmark info: Performance in predicting relative energies between 3 structures for 157 molecule-surface combinations.",
                title="OC157 Dataset: MAE and Ranking Accuracy",
                extra_components=[
                    html.Div(id="oc157-model-plot"),
                ],
                tooltip_header={
                    "Model": "Name of the model",
                    "MAE (meV)": "Mean Absolute Error (meV)",
                    "RMSD (meV)": "Root Mean Square Deviation (meV)",
                    "Ranking Accuracy": "Accuracy in ranking stability across triplets",
                    "Pearson r": "Pearson correlation coefficient",
                    "Score": "Avg of MAE and 1 - Ranking Accuracy (lower is better)",
                    "Rank": "Model rank based on score (lower is better)"
                }
            )
        ])
        
        
        
        
            
            
            
    # ---- OC157 interactive callbacks ----
    @staticmethod
    def register_oc157_callbacks(
        app, 
        rel_df_all, 
        dft_df
    ):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr
        model_df_dict = {model: group for model, group in rel_df_all.groupby("Model")}
        #model_df_dict = {df["Model"].iloc[0]: df for df in rel_df_all}

        @app.callback(
            Output("oc157-model-plot", "children"),
            Input("oc157-mae-table", "active_cell"),
            State("oc157-mae-table", "data"),
        )
        def update_oc157_plot(active_cell, table_data):

            if not active_cell:
                raise PreventUpdate

            row = active_cell["row"]
            clicked_model = table_data[row]["Model"]
            col = active_cell["column_id"]
            
            if col == "Model":
                return None

            if clicked_model not in model_df_dict:
                raise PreventUpdate

            df = model_df_dict[clicked_model].copy()
            system_labels = df[["system_id", "composition"]].apply(
                lambda row: f"{row.system_id}: {row.composition}", axis=1
            ).tolist()

            energy_types = ["dE_2-1", "dE_3-1", "dE_3-2"]
            pred_df = df.drop(columns=["Model", "system_id", "composition"]).reset_index(drop=True)
            pred_values = pred_df[energy_types].values.flatten()
            dft_values = dft_df[energy_types].values.flatten()
            n_points = len(pred_values)

            hover_labels = []
            energy_labels = []
            for label in system_labels:
                for etype in energy_types:
                    hover_labels.append(label)
                    energy_labels.append(etype)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dft_values[:n_points],
                    y=pred_values,
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    text=[
                        f"{label}<br>{etype}:<br>DFT: {dft:.3f} eV<br>Pred: {pred:.3f} eV"
                        for label, etype, dft, pred in zip(hover_labels, energy_labels, dft_values[:n_points], pred_values)
                    ],
                    hoverinfo="text",
                    name=clicked_model
                )
            )
            fig.add_trace(go.Scatter(
                x=[-1, 5], y=[-1, 5],
                mode="lines",
                line=dict(dash="dash", color="black", width=1),
                showlegend=False
            ))

            mae = mean_absolute_error(dft_values[:n_points], pred_values)
            rmsd = np.sqrt(mean_squared_error(dft_values[:n_points], pred_values))
            r, _ = pearsonr(dft_values[:n_points], pred_values)

            fig.update_layout(
                title=f"{clicked_model} vs DFT Relative Energies",
                xaxis_title="DFT Relative Energy [eV]",
                yaxis_title=f"{clicked_model} Relative Energy [eV]",
                annotations=[
                    dict(
                        text=f"N = {n_points}<br>MAE = {mae:.2f} eV<br>RMSD = {rmsd:.2f} eV<br>Pearson r = {r:.2f}",
                        xref="paper", yref="paper",
                        x=0.01, y=0.99, showarrow=False,
                        align="left", bgcolor="white", font=dict(size=10)
                    )
                ]
            )

            return dcc.Graph(figure=fig)