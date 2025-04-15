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
from ase.build import bulk
from ase.phonons import Phonons
from ase.dft.kpoints import bandpath
from ase.optimize import LBFGS
from dataclasses import field

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
import cctk
from ase.io.trajectory import Trajectory

from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator


from mlipx.phonons_utils import get_fc2_and_freqs, init_phonopy, load_phonopy, get_chemical_formula
from phonopy.structure.atoms import PhonopyAtoms
from seekpath import get_path
import zntrack.node
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class GMTKN55_benchmark(zntrack.Node):
    """Benchmark model against GMTKN55
    """
    # inputs
    GMTKN55_yaml: pathlib.Path = zntrack.params()
    subsets_csv: pathlib.Path = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    
    subsets: Optional[List[str]] = zntrack.params(None)
    skip_subsets: Optional[List[str]] = zntrack.params(None)
    allowed_multiplicity: Optional[List[int]] = zntrack.params(None)
    allowed_charge: Optional[List[int]] = zntrack.params(None)
    allowed_elements: Optional[List[int]] = zntrack.params(None)


    # outputs
    # nwd: ZnTrack's node working directory for saving files
    model_benchmark_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "benchmark.csv")
    reference_values_ouptut: pathlib.Path = zntrack.outs_path(zntrack.nwd / "reference_values.json")
    predicted_values_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "predicted_values.json")
    system_atoms_traj: pathlib.Path = zntrack.outs_path(zntrack.nwd / "system_atoms.traj")

    



    def run(self):
        
        calc = self.model.get_calculator()
        
        with open(self.GMTKN55_yaml, "r") as file:
            structure_dict = yaml.safe_load(file)
            
        ref_values = {}
        pred_values = {}
            
        results_summary = []
        
        print(f"\nEvaluating with model: {self.model_name}")
        overall_errors = []
        overall_weights = []
        traj = Trajectory(str(self.system_atoms_traj), mode="w")
        
        with open(self.model_benchmark_output, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Subset", "MAE", "Completed"])

            for subset_name, subset in structure_dict.items():
                if self.subsets and subset_name not in self.subsets:
                    continue
                if self.skip_subsets and subset_name in self.skip_subsets:
                    continue
                
                subset_name = subset_name.lower()
                

                
                subset_errors = []
                weights = []
                
                ref_values[subset_name] = []
                pred_values[subset_name] = []

                for system_name, system in subset.items():
                    

                    
                    ref_value = system["Energy"]
                    weight = system["Weight"]
                    
                    def _should_run_system(
                    system: Dict[str, Any],
                    allowed_elements: Optional[List[int]],
                    allowed_multiplicity: Optional[List[int]],
                    allowed_charge: Optional[List[int]],
                    ) -> bool:
                        for species in system["Species"].values():
                            elements = [cctk.helper_functions.get_number(e) for e in species["Elements"]]
                            multiplicity = species["UHF"] + 1
                            charge = species["Charge"]

                            if allowed_elements and any(el not in allowed_elements for el in elements):
                                return False
                            if allowed_multiplicity and multiplicity not in allowed_multiplicity:
                                return False
                            if allowed_charge and charge not in allowed_charge:
                                return False

                        return True

                    if not _should_run_system(
                        system, self.allowed_elements, self.allowed_multiplicity, self.allowed_charge
                    ):
                        continue

                    try:
                        comp_value = 0
                        for species_name, species in system["Species"].items():
                            atoms = ase.Atoms(
                                species["Elements"],
                                positions=np.array(species["Positions"])
                            )
                            atoms.info['head'] = 'mp_pbe'
                            atoms.cell = None
                            atoms.pbc = False

                            atoms.calc = calc
                            result = atoms.get_potential_energy()
                            comp_value += result * species["Count"] * 23.0609  # eV to kcal/mol
                            
                            atoms.info["subset_name"] = subset_name
                            atoms.info["system_name"] = system_name
                            atoms.info["model_name"] = self.model_name
                            traj.write(atoms)


                        error = ref_value - comp_value
                        ref_values[subset_name].append(ref_value)
                        pred_values[subset_name].append(comp_value)
                        weights.append(weight)
                        subset_errors.append(error)

                    except Exception as e:
                        print(f"Error in system {system_name}, skipping. Exception: {e}")

                mae = np.mean(np.abs(subset_errors)) if subset_errors else None
                completed = len(subset_errors) == len(subset.items())
                csv_writer.writerow([subset_name, mae, completed])
                overall_errors.extend(subset_errors)
                overall_weights.extend(weights)

        wtmad = np.average(np.abs(overall_errors), weights=overall_weights)
        results_summary.append((self.model_name, wtmad, len(overall_errors)))

        # Print formatted summary table
        print("\nSummary of WTMAD for each model:")
        print(f"{'Model':<25} {'WTMAD (kcal/mol)':<20} {'Systems':<10}")
        print("-" * 55)
        #for model_name, wtmad, count in results_summary:
        print(f"{self.model_name:<25} {wtmad:<20.3f} {len(overall_errors):<10}")
        print("-" * 55)
        
        
        with open(self.reference_values_ouptut, "w") as f:
            json.dump(ref_values, f)
        with open(self.predicted_values_output, "w") as f:
            json.dump(pred_values, f)
            
            
            
    @property
    def reference_dict(self):
        """Get the reference values dictionary."""
        with open(self.reference_values_ouptut, "r") as f:
            ref_values = json.load(f)
        return ref_values
    
    @property
    def predicted_dict(self):
        """Get the predicted values dictionary."""
        with open(self.predicted_values_output, "r") as f:
            pred_values = json.load(f)
        return pred_values
    
    @property
    def benchmark_results(self):
        """Get the benchmark results."""
        benchmark_results = pd.read_csv(self.model_benchmark_output)
        return benchmark_results
    

            
            
            
            
            
            
    # @staticmethod
    # def mae_plot_interactive(benchmark_node_dict, subsets_path, ui = None):
        
        
    #     subsets_df = pd.read_csv(subsets_path)
    #     subsets_df.columns = subsets_df.columns.str.lower()
    #     subsets_df["subset"] = subsets_df["subset"].str.lower()
    #     subsets_df["excluded"] = subsets_df["excluded"].astype(str)
        
    
    #     mae_data = []
        
    #     #for file in benchmark_files:
    #     for model_name, node in benchmark_node_dict.items():
            
    #         df = node.benchmark_results.copy()
    #         df.columns = df.columns.str.lower()
    #         df = df[df["completed"].astype(str).str.lower().str.lower() == "true"]
    #         df["subset"] = df["subset"].str.lower()  
            
    #         # merge descriptions
    #         df = df.merge(subsets_df[["subset", "description"]], on="subset", how="left")

    #         for _, row in df.iterrows():
    #             mae_data.append({
    #                 "model": model_name,
    #                 "subset": row["subset"],
    #                 "mae": row["mae"],
    #                 "description": row["description"]
    #             })

    #     mae_df = pd.DataFrame(mae_data)
    #     print(mae_df["model"].value_counts())
        
    #     if ui is None:
    #         return

    #     # --- Dash app ---
    #     app = dash.Dash(__name__)
    #     app.title = "GMTKN55 Dashboard"

    #     # Main MAE plot (customdata holds [model, subset])
    #     fig = px.scatter(
    #         mae_df,
    #         x="subset",
    #         y="mae",
    #         color="model",
    #         hover_data={"description": True},
    #         custom_data=["model", "subset", "description"],
    #         title="Per-subset MAE by Model",
    #         labels={
    #             "subset": "Subset",
    #             "mae": "MAE",
    #             "model": "Model",
    #         }
    #     )
    #     fig.update_traces(
    #         hovertemplate="<br>".join([
    #             "Model: %{customdata[0]}",
    #             "Subset: %{customdata[1]}",
    #             "Description: %{customdata[2]}",
    #             "MAE: %{y:.3f} kcal/mol",
    #             "<extra></extra>"
    #         ])
    #     )
    #     fig.update_layout(
    #         paper_bgcolor='white',
    #         font_color='black',
    #         title_font=dict(size=20),
    #         margin=dict(t=50, r=30, b=50, l=50),
    #         xaxis=dict(showgrid=True, gridcolor='lightgray'),
    #         yaxis=dict(showgrid=True, gridcolor='lightgray')
    #     )

    #     app.layout = html.Div([
    #         html.H1("GMTKN55 Benchmarking Dashboard", style={"color": "black"}),

    #         dcc.Graph(id="mae-plot", figure=fig),

    #         html.H2("Predicted vs Reference Energies", style={"color": "black"}),
    #         dcc.Graph(id="pred-vs-ref-plot")
    #     ], style={"backgroundColor": "white", "padding": "20px"})

    #     @app.callback(
    #         Output("pred-vs-ref-plot", "figure"),
    #         Input("mae-plot", "clickData"),
    #     )
    #     def update_scatter(click_data):
    #         if click_data is None:
    #             raise dash.exceptions.PreventUpdate

    #         model_name, subset_name, *_ = click_data["points"][0]["customdata"]
    #         subset_name = subset_name.lower()
    #         print(f"Selected: {model_name} | {subset_name}")

    #         try:
    #             preds = benchmark_node_dict[model_name].predicted_dict[subset_name]
    #             refs = benchmark_node_dict[model_name].reference_dict[subset_name]
    #         except KeyError:
    #             print(f"Model {model_name} or subset {subset_name} not found in data.")
    #             return go.Figure()

    #         # Compute error metrics
    #         preds = np.array(preds)
    #         refs = np.array(refs)
    #         errors = preds - refs
    #         mae = np.mean(np.abs(errors))

    #         # Axis ranges based on max/min of both predicted and reference
    #         min_val = min(refs.min(), preds.min(), 0)
    #         max_val = max(refs.max(), preds.max())
    #         pad = 0.05 * (max_val - min_val)
    #         x_range = [min_val - pad, max_val + pad]
    #         y_range = x_range

    #         fig = px.scatter(
    #             x=refs,
    #             y=preds,
    #             labels={"x": "Reference Energy (kcal/mol)", "y": "Predicted Energy (kcal/mol)"},
    #             title=f"{model_name} — {subset_name}: Predicted vs Reference",
    #         )

    #         fig.add_trace(go.Scatter(
    #             x=np.linspace(min_val - 100, max_val + 100, 100), y=np.linspace(min_val - 100, max_val + 100, 100),
    #             mode="lines", name="y = x",
    #             line=dict(dash="dot", color="gray")
    #         ))

    #         fig.add_annotation(
    #             xref="paper", yref="paper",
    #             x=0.02, y=0.98,
    #             text=f"MAE: {mae:.3f}",
    #             showarrow=False,
    #             align="left",
    #             font=dict(size=12, color="black"),
    #             bordercolor="black",
    #             borderwidth=1,
    #             borderpad=4,
    #             bgcolor="white",
    #             opacity=0.8
    #         )

    #         fig.update_layout(
    #             height=500,
    #             plot_bgcolor='white',
    #             paper_bgcolor='white',
    #             font_color='black',
    #             margin=dict(t=50, r=30, b=50, l=50),
    #             xaxis=dict(range=x_range, showgrid=True, gridcolor='lightgray', scaleanchor='y', scaleratio=1),
    #             yaxis=dict(range=y_range, showgrid=True, gridcolor='lightgray')
    #         )

    #         return fig



    #     def get_free_port():
    #         """Find an unused local port."""
    #         s = socket.socket()
    #         s.bind(('', 0))  # let OS pick a free port
    #         port = s.getsockname()[1]
    #         s.close()
    #         return port


    #     def run_app(app, ui):
    #         port = get_free_port()
    #         url = f"http://localhost:{port}"

    #         def _run_server():
    #             app.run(debug=True, use_reloader=False, port=port)
                
    #         if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
    #             import threading
    #             print(f"\n Detected SSH session — skipping browser launch.")
    #             #threading.Thread(target=_run_server, daemon=True).start()
    #             return

    #         if ui == "popup":
    #             import threading
    #             import webview
    #             # Start Dash app in background
    #             threading.Thread(target=_run_server, daemon=True).start()
    #             time.sleep(1.5)  # Give it time to start

    #             # Open popup window with pywebview
    #             webview.create_window("Phonon Benchmark Viewer", url)
    #             webview.start()
    #         elif ui == "browser":
    #             import webbrowser
    #             import threading
    #             threading.Thread(target=_run_server, daemon=True).start()
    #             time.sleep(1.5)
    #             #webbrowser.open(url)
    #         elif ui == "notebook":
    #             _run_server()
            
    #         else:
    #             print(f"Unknown UI option: {ui}. Please use 'popup', 'browser', or 'notebook'.")
    #             return

    #         print(f"Dash app running at {url}")
            
    #     return run_app(app, ui=ui)
    
    