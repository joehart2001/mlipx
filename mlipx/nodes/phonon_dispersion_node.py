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
import signal
from datetime import datetime

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from concurrent.futures import ProcessPoolExecutor


class PhononDispersion(zntrack.Node):
    """Compute the phonon dispersion from a phonopy object
    """
    # inputs
    phonopy_yaml_path: pathlib.Path = zntrack.deps()
    thermal_properties_path: pathlib.Path = zntrack.deps()
    
    node_idx: int = zntrack.params(None)
    total_no_nodes: int = zntrack.params(None)
    

    #qpoints_input_path: t.Optional[pathlib.Path] = zntrack.deps(None)
    labels_input_path: t.Optional[pathlib.Path] = zntrack.deps(None)
    connections_input_path: t.Optional[pathlib.Path] = zntrack.deps(None)
    


    # outputs
    # nwd: ZnTrack's node working directory for saving files
    band_structure_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "band_structure.npz")
    dos_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "dos.npz")
    phonon_obj_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_obj.yaml")
    qpoints_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "qpoints.pkl")
    labels_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "labels.json")
    connections_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "connections.json")
    thermal_properties_path_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "thermal_properties.json")
    
    



    def run(self):        
        
        phonons = load_phonopy(str(self.phonopy_yaml_path))

        
        if self.labels_input_path:
            # calculate phonon structure along the reference path to ensure a valid comparison
            # with open(self.qpoints_input_path, "rb") as f:
            #     qpoints = pickle.load(f)
            with open(self.labels_input_path, "r") as f:
                labels = json.load(f)
            with open(self.connections_input_path, "r") as f:
                connections = json.load(f)
            

            
            #phonons.run_band_structure(paths=qpoints, labels=labels, path_connections=connections)
            #print(len(phonons.get_band_structure_dict()["distances"][0]))

            phonons.auto_band_structure()
            

            # save, zntrack requires each output declared to be an output
            # with open(self.qpoints_path, "wb") as f:
            #     pickle.dump(qpoints, f)
            with open(self.labels_path, "w") as f:
                json.dump(labels, f)
            with open(self.connections_path, "w") as f:
                json.dump(connections, f)

                
                            
        else:
            #phonons.auto_band_structure() # uses seekpath
            
            qpoints, labels, connections = get_band_qpoints_by_seekpath(
                phonons.primitive, npoints=101, is_const_interval=True
            )
            
            phonons.run_band_structure(
                paths=qpoints,
                labels=labels,
                path_connections=connections,
            )
            
            # with open(self.qpoints_path, "wb") as f:
            #     pickle.dump(qpoints, f)
            with open(self.labels_path, "w") as f:
                json.dump(labels, f)
            with open(self.connections_path, "w") as f:
                json.dump(connections, f)

                
            

                
        band_structure = phonons.get_band_structure_dict()
        phonons.auto_total_dos()
        dos = phonons.get_total_dos_dict()
        
        
        with self.band_structure_path.open("wb") as f:
            pickle.dump(band_structure, f)        
        print(f"Band structure saved to: {self.band_structure_path}")
        with self.dos_path.open("wb") as f:
            pickle.dump(dos, f)
        
        phonons.save(filename=self.phonon_obj_path, settings={"force_constants": True})
        
        # open thermal properties and resave it 
        with open(self.thermal_properties_path, "r") as f:
            thermal_properties = json.load(f)
        with open(self.thermal_properties_path_output, "w") as f:
            json.dump(thermal_properties, f)
            
        if self.node_idx is not None and self.total_no_nodes is not None:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Phonons {self.node_idx}/{self.total_no_nodes} predicted.")        

    # @property
    # def qpoints(self):
    #     with open(self.qpoints_path, "rb") as f:
    #         return pickle.load(f)

    @property
    def labels(self):
        with open(self.labels_path, "r") as f:
            return json.load(f)

    @property
    def connections(self):
        with open(self.connections_path, "r") as f:
            return json.load(f)
    
    @property
    def get_thermal_properties(self):
        with open(self.thermal_properties_path_output, "r") as f:
            return json.load(f)
        
    @property
    def band_structure(self):
        with self.band_structure_path.open("rb") as f:
            return pickle.load(f)
        
    @property
    def dos(self):
        with self.dos_path.open("rb") as f:
            dos = pickle.load(f)
        return dos["frequency_points"], dos["total_dos"]
    
    @property
    def band_width(self):
        with self.band_structure_path.open("rb") as f:
            band_structure = pickle.load(f)
        
        freqs = np.concatenate(band_structure["frequencies"])
        return np.max(freqs) - np.min(freqs)
    
    @property
    def max_freq(self):
        with self.band_structure_path.open("rb") as f:
            band_structure = pickle.load(f)
        
        freqs = np.concatenate(band_structure["frequencies"])
        return np.max(freqs)
    

        
    @property
    def plot_auto_band_structure(self):
        phonons = load_phonopy(self.phonon_obj_path)
        phonons.auto_band_structure(plot=True)
        return
    
    def _load_band_and_dos(self, phonon_path):
        phonons = load_phonopy(phonon_path)
        phonons.auto_band_structure()
        phonons.auto_total_dos()
        return phonons.get_band_structure_dict(), phonons.get_total_dos_dict(), phonons

    @staticmethod
    def _build_xticks(distances, labels, connections):
        # begins with _ as this is a private method only for internal use
        xticks, xticklabels = [], []
        cumulative_dist, i = 0.0, 0
        connections = [True] + connections
        for seg_dist, connected in zip(distances, connections):
            start, end = labels[i], labels[i+1]
            pos_start = cumulative_dist
            pos_end = cumulative_dist + (seg_dist[-1] - seg_dist[0])
            xticks.append(pos_start)
            xticklabels.append(f"{start}|{end}" if not connected else start)
            i += 2 if not connected else 1
            cumulative_dist = pos_end
        xticks.append(cumulative_dist)
        xticklabels.append(labels[-1])
        return xticks, xticklabels
    
    
    @property
    def plot_band_structure(self):
        band_structure, dos, phonons = self._load_band_and_dos(self.phonon_obj_path)


        dos_freqs = dos["frequency_points"]
        dos_values = dos["total_dos"]


        # plotting
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])
        
        for dist_segment, freq_segment in zip(band_structure['distances'], band_structure['frequencies']):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, color='red', lw=1)

        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])  # DOS plot on the right
        ax2.plot(dos_values, dos_freqs, color="red", lw=1)
        

        # sorting out xticks
        labels = phonons.band_structure.labels
        connections = phonons.band_structure.path_connections
        distances = band_structure["distances"]
        print(labels)

        
        xticks, xticklabels = self._build_xticks(distances, labels, connections)

        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)
        
        ax1.set_xticks(xticks, xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)")
        ax1.set_xlabel("Wave Vector")
        
        max_freq, min_freq = np.max(np.concatenate(band_structure["frequencies"])), np.min(np.concatenate(band_structure["frequencies"]))
        ax1.set_ylim(min_freq - 0.4, max_freq + 0.4)
        
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("DOS")
        
        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        return phonons





    
    
    @staticmethod
    def prettify_chemical_formula(formula: str) -> str:
        """
        Converts a chemical formula into a prettier format for matplotlib titles.

        Examples:
            "Al2O3"   -> "Al$_2$O$_3$"
            "Fe12O19" -> "Fe$_{12}$O$_{19}$"
            "NaCl"    -> "NaCl"
        """
        
        parts = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
        pretty = ""
        for element, count in parts:
            if count == "":
                pretty += element
            else:
                pretty += f"{element}$_{{{count}}}$"  # use curly braces for multi-digit numbers
        return pretty

    
    
    
    
    


    @staticmethod
    def compare_reference(node_pred, node_ref, correlation_plot_mode = False, model_name = None):
        
        # ----------setup ticks using node------------
        fig = plt.figure(figsize=(9, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])  # band structure
        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])  # DOS

        phonons_pred = load_phonopy(node_pred.phonon_obj_path)
        phonons_ref = load_phonopy(node_ref.phonon_obj_path)

        band_structure_pred = node_pred.band_structure
        distances_pred = band_structure_pred["distances"]
        frequencies_pred = band_structure_pred["frequencies"]
        dos_freqs_pred, dos_values_pred = node_pred.dos
        

        
        band_structure_ref = node_ref.band_structure
        distances_ref = band_structure_ref["distances"]
        frequencies_ref = band_structure_ref["frequencies"]
        dos_freqs_ref, dos_values_ref = node_ref.dos
        # remove points of zero density to avoid artifacts in the plot
        dos_freqs_pred, dos_values_pred = dos_freqs_pred[dos_values_pred > 0], dos_values_pred[dos_values_pred > 0]


        
        
        labels = node_ref.labels
        connections = node_ref.connections
        connections = [True] + connections
        
        xticks, xticklabels = PhononDispersion._build_xticks(distances_ref, labels, connections)
        

        #-----------------------plotting-----------------------
        # pred band structure
        for dist_segment, freq_segment in zip(distances_pred, frequencies_pred):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='--', label=model_name, color='red')

        ax2.plot(dos_values_pred, dos_freqs_pred, lw=1.2, color="red", linestyle='--')
        
        # reference band structure
        for dist_segment, freq_segment in zip(distances_ref, frequencies_ref):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='-', label="PBE", color='blue')
            
        ax2.plot(dos_values_ref, dos_freqs_ref, lw=1.2, color="blue")
            
        

        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)
        
        ax1.set_xticks(xticks, xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)")
        ax1.set_xlabel("Wave Vector")
        
        pred_freqs_flat = np.concatenate(frequencies_pred).flatten()
        ref_freqs_flat = np.concatenate(frequencies_ref).flatten()
        all_freqs = np.concatenate([pred_freqs_flat, ref_freqs_flat])

        
        ax1.set_ylim(all_freqs.min() - 0.4, all_freqs.max() + 0.4)
        ax2.set_ylim(ax1.get_ylim())
                
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("DOS")
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc='upper center',
                    bbox_to_anchor=(0.8, 1.02),  # Right of center title
                    frameon=False,
                    ncol=2,
                    fontsize=14,    
                    )
        
        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        
        chemical_formula = get_chemical_formula(phonons_ref)
        chemical_formula = PhononDispersion.prettify_chemical_formula(chemical_formula)
        mp_id = node_ref.name.split("_")[-1]
        plt.suptitle(f"{chemical_formula} ({mp_id})", x=0.4, fontsize = 14)
            
            
        
        if correlation_plot_mode:
            
            if not os.path.exists(f"benchmark_stats/bulk_crystal_benchmark/phonons/{model_name}/phonon_plots"):
                os.makedirs(f"benchmark_stats/bulk_crystal_benchmark/phonons/{model_name}/phonon_plots")
            
            phonon_plot_path = f"benchmark_stats/bulk_crystal_benchmark/phonons/{model_name}/phonon_plots/dispersion_{model_name}_{mp_id}.png"
            fig.savefig(phonon_plot_path, bbox_inches='tight')
            plt.close(fig)
            return phonon_plot_path
        
        else:
            plt.show()
            
        return
    

        
















    @staticmethod
    def benchmark_interactive(
        pred_node_dict, 
        ref_node_dict, 
        ui = None, 
        run_interactive = True,
        report = True,
        normalise_to_model: t.Optional[str] = None,
    ):
        """
        Main benchmarking function that coordinates the benchmarking process.
        
        Args:
            pred_node_dict: Dictionary of prediction nodes
            ref_node_dict: Dictionary of reference nodes
            ui: Optional UI object for interactive display
            run_interactive: Boolean flag to run the interactive UI or not
            
        Returns:
            Various benchmarking data and plots
        """

        ref_band_data_dict = {}
        ref_benchmarks_dict = {}
        pred_benchmarks_dict = {}
        scatter_to_dispersion_map = {}     # model_name -> {property -> {ref, pred}, hover, point_map, img_paths}
        phonon_plot_paths = {}
        point_index_to_id = []
        
        benchmarks = [
            'max_freq',
            'min_freq',
            'S',
            'F',
            'C_V',
        ]
        benchmark_units = {
            'max_freq': 'THz', 
            'min_freq': 'THz',
            'S': '[J/K/mol]',
            'F': '[kJ/mol]',
            'C_V': '[J/K/mol]',
        }
        
        pretty_benchmark_labels = {
            'max_freq': 'ω_max [THz]',
            'min_freq': 'ω_min [THz]',
            'S': 'S [J/mol·K]',
            'F': 'F [kJ/mol]',
            'C_V': 'C_V [J/mol·K]'
        }
        label_to_key = {v: k for k, v in pretty_benchmark_labels.items()}

        model_benchmarks_dict = {} # max_freq, min_freq ...
        plot_stats_dict = {}
        
        
        #-------------
            
            
        # Process each structure
        for mp_id in tqdm(pred_node_dict.keys(), desc="Processing structures"):
            if not PhononDispersion.process_reference_data(ref_node_dict, mp_id, ref_band_data_dict, 
                                        ref_benchmarks_dict, pred_benchmarks_dict):
                continue
            
            # Initialize phonon plot paths for this mp_id
            phonon_plot_paths[mp_id] = {}
            
            # Process prediction data for each model
            PhononDispersion.process_prediction_data(
                pred_node_dict[mp_id], 
                ref_node_dict[mp_id], 
                mp_id,
                ref_benchmarks_dict,
                pred_benchmarks_dict,
                model_benchmarks_dict,
                plot_stats_dict,
                scatter_to_dispersion_map,
                phonon_plot_paths,
                point_index_to_id,
                benchmarks
            )
        
        # Calculate summary statistics and generate plots
        mae_summary_df = PhononDispersion.calculate_summary_statistics(
            plot_stats_dict, 
            pretty_benchmark_labels, 
            benchmarks,
        )
        # Add stability classification column using static method
        mae_summary_df = PhononDispersion.add_stability_classification_column(mae_summary_df, model_benchmarks_dict)
        mae_summary_df = PhononDispersion.add_band_mae_column(mae_summary_df, scatter_to_dispersion_map)
        
        # recalculate phonon score (avg) to include band MAE
        mae_cols = [col for col in mae_summary_df.columns if col not in ['Model', 'Phonon Score \u2193', 'Rank']]

        model_list = list(pred_node_dict[list(pred_node_dict.keys())[0]].keys())
        
        if normalise_to_model is not None:
            # Normalise the MAE values to the specified model
            for model in model_list:
                score = 0
                for col in mae_cols:
                    score += mae_summary_df.loc[mae_summary_df['Model'] == model, col].values[0] / mae_summary_df.loc[mae_summary_df['Model'] == normalise_to_model, col].values[0]
                mae_summary_df.loc[mae_summary_df['Model'] == model, 'Phonon Score \u2193'] = score / len(mae_cols)
            
        else:
            # recalc score with band mae included
            mae_summary_df['Phonon Score \u2193'] = mae_summary_df[mae_cols].mean(axis=1).round(3)
        
        mae_summary_df = mae_summary_df.round(3)
        mae_summary_df['Rank'] = mae_summary_df['Phonon Score \u2193'].rank(method='min').astype(int)
        
        PhononDispersion.generate_and_save_plots(
            scatter_to_dispersion_map,
            model_benchmarks_dict,
            plot_stats_dict,
            pretty_benchmark_labels,
            benchmarks
        )
        
        model_list = list(pred_node_dict[list(pred_node_dict.keys())[0]].keys())
        
        if report:
            md_path = PhononDispersion.generate_phonon_report(
                mae_summary_df=mae_summary_df,
                models_list=model_list,
            )
        else:
            md_path = None
        
        if ui is None and run_interactive:
            return

        # --------------------------------- Dash app ---------------------------------

        from mlipx.dash_utils import dash_table_interactive
        app = dash.Dash(__name__)

        # Separate the "Stability Classification (F1)" column for the stability table
        summary_columns = [col for col in mae_summary_df.columns if col != "Stability Classification (F1)"]
        stability_columns = ["Model", "Stability Classification (F1)"] if "Stability Classification (F1)" in mae_summary_df.columns else []

        # Use shared dash_table_interactive for summary table
        summary_table_layout = dash_table_interactive(
            df=mae_summary_df[summary_columns],
            id='phonon-mae-summary-table',
            title="Phonon dispersion MAE Summary Table (300 K)",
            extra_components=[
                dcc.Store(id="phonon-summary-table-last-clicked"),
                html.Div(id="phonon-summary-plot-container"),
            ],
        )

        # Use shared dash_table_interactive for stability table
        stability_table_layout = dash_table_interactive(
            df=mae_summary_df[stability_columns],
            id='phonon-stability-table',
            title=html.Span([
                "Stability Classification: imaginary modes (threshold |ω",
                html.Sub("imag"),
                "| < 0.05 THz)"
            ]),
            extra_components=[
                dcc.Store(id="phonon-stability-table-last-clicked"),
                html.Div(id="phonon-stability-plot-container"),
            ],
        )

        app.layout = html.Div(
            [
                summary_table_layout,
                html.Br(),
                stability_table_layout,
            ],
            style={"backgroundColor": "white", "padding": "20px"}
        )

        # Register callbacks using the static method
        PhononDispersion.register_callbacks(
            app=app,
            mae_df=mae_summary_df,
            scatter_to_dispersion_map=scatter_to_dispersion_map,
            model_benchmarks_dict=model_benchmarks_dict,
        )

        from mlipx.dash_utils import run_app

        if not run_interactive:
            return app, mae_summary_df, scatter_to_dispersion_map, model_benchmarks_dict , md_path

        return run_app(app, ui=ui)
                








    # --------------------------------- Helper Functions ---------------------------------



    @staticmethod
    def register_callbacks(
        app,
        mae_df,
        scatter_to_dispersion_map,
        model_benchmarks_dict,
    ):
        from dash import callback_context
        import numpy as np
        # Callback for summary table plot
        @app.callback(
            Output("phonon-summary-plot-container", "children"),
            Output("phonon-summary-table-last-clicked", "data"),
            Input("phonon-mae-summary-table", "active_cell"),
            State("phonon-summary-table-last-clicked", "data"),
            prevent_initial_call=True
        )
        def update_summary_plot(summary_active_cell, summary_last_clicked):
            ctx = callback_context
            triggered_id = ctx.triggered_id
            pretty_benchmark_labels = {
                'max_freq': 'ω_max [THz]',
                'min_freq': 'ω_min [THz]',
                'S': 'S [J/mol·K]',
                'F': 'F [kJ/mol]',
                'C_V': 'C_V [J/mol·K]'
            }
            label_to_key = {v: k for k, v in pretty_benchmark_labels.items()}
            if summary_active_cell is None:
                raise PreventUpdate
            row = summary_active_cell["row"]
            col = summary_active_cell["column_id"]
            model_name = mae_df.loc[row, "Model"]
            # Only allow property columns (not "Model" or "Stability Classification (F1)")
            if col not in mae_df.columns or col == "Model":
                return None, summary_active_cell
            if summary_last_clicked is not None and (
                summary_active_cell["row"] == summary_last_clicked.get("row") and
                summary_active_cell["column_id"] == summary_last_clicked.get("column_id")
            ):
                return None, None
            if col == "Stability Classification (F1)":
                return None, summary_active_cell
            if col == "Avg BZ MAE":
                band_error_dict = scatter_to_dispersion_map[model_name].get("band_errors", {})
                all_errors = np.concatenate(list(band_error_dict.values())) if band_error_dict else np.array([])
                import plotly.express as px
                fig = px.violin(
                    y=all_errors,
                    box=True,
                    points="outliers",
                    title=f"{model_name} - BZ MAE Distribution",
                    labels={"y": "Absolute Error (THz)"}
                )
                return html.Div([dcc.Graph(figure=fig)]), summary_active_cell
            selected_property = label_to_key.get(col)
            if selected_property is None:
                return html.Div("Invalid property selected."), summary_active_cell
            data = model_benchmarks_dict[model_name]
            ref_vals = data[selected_property]['ref']
            pred_vals = data[selected_property]['pred']
            pretty_label = pretty_benchmark_labels[selected_property]
            mae = mae_df.loc[mae_df["Model"] == model_name, pretty_label].values[0]
            scatter_fig = PhononDispersion.create_scatter_plot(
                ref_vals,
                pred_vals,
                model_name,
                selected_property,
                mae,
                pretty_benchmark_labels
            )
            return html.Div([
                #html.H4(f"{model_name} - {selected_property.replace('_', ' ').title()}", style={"color": "black"}),
                html.P("Info: Click on a point to view its phonon dispersion plot.", style={"fontSize": "14px", "color": "#555"}),
                html.Div([
                    dcc.Graph(
                        id={'type': 'mae-summary-pair-plot', 'index': model_name},
                        figure=scatter_fig,
                        style={"width": "45vw", "height": "60vh"}
                    ),
                    html.Div(
                        id={'type': 'mae-summary-plot-display', 'index': model_name},
                        style={
                            "width": "50vw",
                            "height": "60vh",
                            "marginLeft": "2vw",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "border": "1px solid #ccc",
                            "padding": "10px",
                        }
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "40px"
                })
            ]), summary_active_cell

        # Callback for stability table plot
        @app.callback(
            Output("phonon-stability-plot-container", "children"),
            Output("phonon-stability-table-last-clicked", "data"),
            Input("phonon-stability-table", "active_cell"),
            State("phonon-stability-table-last-clicked", "data"),
            prevent_initial_call=True
        )
        def update_stability_plot(stability_active_cell, stability_last_clicked):
            ctx = callback_context
            if stability_active_cell is None:
                raise PreventUpdate
            row = stability_active_cell["row"]
            col = stability_active_cell["column_id"]
            model_name = mae_df.loc[row, "Model"]
            # Only allow clicking the F1 for stability
            if col != "Stability Classification (F1)":
                return None, stability_active_cell
            if stability_last_clicked is not None and (
                stability_active_cell["row"] == stability_last_clicked.get("row") and
                stability_active_cell["column_id"] == stability_last_clicked.get("column_id")
            ):
                return None, None
            threshold = -0.05
            data = model_benchmarks_dict[model_name]
            ref_vals = np.array(data["min_freq"]['ref'])
            pred_vals = np.array(data["min_freq"]['pred'])
            labels = []
            for r, p in zip(ref_vals, pred_vals):
                ref_stable = r > threshold
                pred_stable = p > threshold
                if ref_stable and pred_stable:
                    labels.append("TN")
                elif not ref_stable and not pred_stable:
                    labels.append("TP")
                elif not ref_stable and pred_stable:
                    labels.append("FN")
                else:
                    labels.append("FP")
            scatter_fig = PhononDispersion.create_stability_scatter_plot(ref_vals, pred_vals, labels)
            tn = sum(l == "TN" for l in labels)
            fp = sum(l == "FP" for l in labels)
            fn = sum(l == "FN" for l in labels)
            tp = sum(l == "TP" for l in labels)
            conf_matrix = np.array([[tn, fp], [fn, tp]])
            confusion_fig = PhononDispersion.create_confusion_matrix_figure(conf_matrix, labels=["Stable", "Not Stable"], model_name=model_name)
            visuals = html.Div([
                html.P("Info: Click on a point to view its phonon dispersion plot.", style={"fontSize": "14px", "color": "#555"}),
                html.Div([
                    dcc.Graph(
                        id={'type': 'stability-pair-plot', 'index': model_name},
                        figure=scatter_fig,
                        style={"width": "45vw", "height": "60vh"}
                    ),
                    html.Div(
                        id={'type': 'stability-plot-display', 'index': model_name},
                        style={
                            "width": "50vw",
                            "height": "60vh",
                            "marginLeft": "2vw",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "border": "1px solid #ccc",
                            "padding": "10px",
                        }
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "20px"
                }),
                dcc.Graph(
                    id={'type': 'confusion-matrix', 'index': model_name},
                    figure=confusion_fig,
                    style={"width": "90vw", "height": "45vh", "marginTop": "10px"}
                ),
            ])
            return visuals, stability_active_cell

        # Callback for clicking on summary pair plot (distinct ID namespace)
        @app.callback(
            Output({'type': 'mae-summary-plot-display', 'index': MATCH}, 'children'),
            Input({'type': 'mae-summary-pair-plot', 'index': MATCH}, 'clickData'),
            State({'type': 'mae-summary-pair-plot', 'index': MATCH}, 'id'),
            prevent_initial_call=True
        )
        def display_summary_phonon_plot(clickData, graph_id):
            model_name = graph_id['index']
            if clickData is None:
                return html.Div("Click on a point to view its phonon dispersion plot.")
            point_index = clickData['points'][0]['pointIndex']
            mp_id, _ = scatter_to_dispersion_map[model_name]['point_map'][point_index]
            img_path = scatter_to_dispersion_map[model_name]['img_paths'][mp_id]
            encoded_img = base64.b64encode(open(img_path, 'rb').read()).decode()
            return html.Img(
                src=f'data:image/png;base64,{encoded_img}',
                style={
                    "width": "80%",
                    "height": "80%",
                    "objectFit": "contain",
                    "border": "2px solid black"
                }
            )

        # Callback for clicking on stability pair plot (distinct ID namespace)
        @app.callback(
            Output({'type': 'stability-plot-display', 'index': MATCH}, 'children'),
            Input({'type': 'stability-pair-plot', 'index': MATCH}, 'clickData'),
            State({'type': 'stability-pair-plot', 'index': MATCH}, 'id'),
            prevent_initial_call=True
        )
        def display_stability_phonon_plot(clickData, graph_id):
            model_name = graph_id['index']
            if clickData is None:
                return html.Div("Click on a point to view its phonon dispersion plot.")
            point_index = clickData['points'][0]['pointIndex']
            mp_id, _ = scatter_to_dispersion_map[model_name]['point_map'][point_index]
            img_path = scatter_to_dispersion_map[model_name]['img_paths'][mp_id]
            encoded_img = base64.b64encode(open(img_path, 'rb').read()).decode()
            return html.Img(
                src=f'data:image/png;base64,{encoded_img}',
                style={
                    "width": "80%",
                    "height": "80%",
                    "objectFit": "contain",
                    "border": "2px solid black"
                }
            )



    def generate_phonon_report(
        mae_summary_df,
        models_list,
    ):
        """Generates a markdown and pdf report contraining the MAE summary table, scatter plots and phonon dispersions
        """
        markdown_path = Path("benchmark_stats/bulk_crystal_benchmark/phonons/phonon_benchmark_report.md")
        pdf_path = markdown_path.with_suffix(".pdf")

        md = []

        md.append("# Phonon Dispersion Report\n")

        # MAE Summary table
        md.append("## Phonon MAE Summary Table (300K)\n")
        md.append(mae_summary_df.to_markdown(index=False))
        md.append("\n")
        
        # function for adding images to the markdown 
        def add_image_rows(md_lines, image_paths, n_cols = 4):
            """Append n images per row"""
            for i in range(0, len(image_paths), n_cols):
                image_set = image_paths[i:i+n_cols]
                width = 100 // n_cols
                line = " ".join(f"![]({img.resolve()}){{ width={width}% }}" for img in image_set)
                md_lines.append(line + "\n")


        # Scatter Plots
        md.append("## Scatter Plots\n")
        for model in models_list:
            md.append(f"### {model}\n")
            scatter_plot_dir = Path(f"benchmark_stats/bulk_crystal_benchmark/phonons/{model}/scatter_plots")
            images = sorted(scatter_plot_dir.glob("*.png"))
            add_image_rows(md, images)            
                    
        # Dispersion Plots
        md.append("## Dispersion Plots\n")
        for model in models_list:
            md.append(f"### {model}\n")
            dispersion_plot_dir = Path(f"benchmark_stats/bulk_crystal_benchmark/phonons/{model}/phonon_plots")
            images = sorted(dispersion_plot_dir.glob("*.png"))
            add_image_rows(md, images)
    
    
        # Save Markdown file
        markdown_path.write_text("\n".join(md))

        # Replace unicode terms with LaTeX
        text = markdown_path.read_text()
        text = text.replace("ω_max", "$\\omega_{max}$")
        
        markdown_path.write_text(text)

        print(f"Markdown report saved to: {markdown_path}")

        # Generate PDF with Pandoc
        try:
            subprocess.run(
                ["pandoc", str(markdown_path), "-o", str(pdf_path), "--pdf-engine=xelatex", "--variable=geometry:top=1.5cm,bottom=2cm,left=1cm,right=1cm"],
                check=True
            )
            print(f"PDF report saved to {pdf_path}")

        except subprocess.CalledProcessError as e:
            print(f"PDF generation failed: {e}")


        return markdown_path



    def process_reference_data(ref_node_dict, mp_id, ref_band_data_dict, ref_benchmarks_dict, pred_benchmarks_dict):
        """
        Process reference data for a given structure.
        
        Returns:
            bool: True if processing succeeded, False if files were missing
        """
        if mp_id not in ref_node_dict:
            return False
            
        node_ref = ref_node_dict[mp_id]
        
        try:
            band_structure_ref = node_ref.band_structure
        except FileNotFoundError:
            print(f"Skipping {node_ref.name} — band_structure file not found at {node_ref.band_structure_path}")
            return False
            
        try:
            dos_freqs_ref, dos_values_ref = node_ref.dos
            # remove values with zero density to avoid artifacts in the plot + min/max freq calulations
            dos_freqs_ref, dos_values_ref = dos_freqs_ref[dos_values_ref > 0], dos_values_ref[dos_values_ref > 0]
        except FileNotFoundError:
            print(f"Skipping {node_ref.name} — dos file not found at {node_ref.dos_path}")
            return False
        
        # Store reference data
        ref_benchmarks_dict[mp_id] = {}
        ref_band_data_dict[mp_id] = {}
        pred_benchmarks_dict[mp_id] = {}
        
        ref_band_data_dict[mp_id] = band_structure_ref
        
        # Calculate reference benchmarks
        T_300K_index = node_ref.get_thermal_properties['temperatures'].index(300)
        ref_benchmarks_dict[mp_id]['max_freq'] = np.max(dos_freqs_ref)
        ref_benchmarks_dict[mp_id]['min_freq'] = np.min(dos_freqs_ref)
        ref_benchmarks_dict[mp_id]['S'] = node_ref.get_thermal_properties['entropy'][T_300K_index]
        ref_benchmarks_dict[mp_id]['F'] = node_ref.get_thermal_properties['free_energy'][T_300K_index]
        ref_benchmarks_dict[mp_id]['C_V'] = node_ref.get_thermal_properties['heat_capacity'][T_300K_index]
        
        return True


    def process_prediction_data(pred_nodes, node_ref, mp_id, ref_benchmarks_dict, pred_benchmarks_dict,
                            model_benchmarks_dict, plot_stats_dict, scatter_to_dispersion_map, 
                            phonon_plot_paths, point_index_to_id, benchmarks):
        """
        Process prediction data for all models for a given structure.
        """
        for model_name in pred_nodes.keys():
            try:
                dos_freqs_pred, dos_values_pred = pred_nodes[model_name].dos
                # remove values with zero density to avoid artifacts in the plot + min/max freq calulations
                dos_freqs_pred, dos_values_pred = dos_freqs_pred[dos_values_pred > 0], dos_values_pred[dos_values_pred > 0]
            except FileNotFoundError:
                print(f"Skipping {model_name} — dos file not found at {pred_nodes[model_name].dos_path}")
                continue
            
            # Calculate prediction benchmarks
            pred_benchmarks_dict[mp_id][model_name] = {}
            T_300K_index = pred_nodes[model_name].get_thermal_properties['temperatures'].index(300)
            
            pred_benchmarks_dict[mp_id][model_name]['max_freq'] = np.max(dos_freqs_pred)
            pred_benchmarks_dict[mp_id][model_name]['min_freq'] = np.min(dos_freqs_pred)
            pred_benchmarks_dict[mp_id][model_name]['S'] = pred_nodes[model_name].get_thermal_properties['entropy'][T_300K_index]
            pred_benchmarks_dict[mp_id][model_name]['F'] = pred_nodes[model_name].get_thermal_properties['free_energy'][T_300K_index]
            pred_benchmarks_dict[mp_id][model_name]['C_V'] = pred_nodes[model_name].get_thermal_properties['heat_capacity'][T_300K_index]
            
            ref_freqs = node_ref.band_structure["frequencies"]
            pred_freqs = pred_nodes[model_name].band_structure["frequencies"]

            
            # Generate phonon dispersion plot
            phonon_plot_path = PhononDispersion.compare_reference(
                node_pred=pred_nodes[model_name],
                node_ref=node_ref,
                correlation_plot_mode=True,
                model_name=model_name
            )
            
            phonon_plot_paths[mp_id][model_name] = phonon_plot_path
            point_index_to_id.append((mp_id, model_name))
            
            # Initialize model data if needed
            PhononDispersion.initialize_model_data(model_name, model_benchmarks_dict, plot_stats_dict, scatter_to_dispersion_map, benchmarks)
            
            # Initialize if needed
            if model_name not in scatter_to_dispersion_map:
                scatter_to_dispersion_map[model_name] = {'hover': [], 'point_map': [], 'img_paths': {}}
            if "band_errors" not in scatter_to_dispersion_map[model_name]:
                scatter_to_dispersion_map[model_name]["band_errors"] = {}

            # Compute and store band errors
            band_errors = np.mean(np.abs(np.concatenate([
                np.array(p) - np.array(r)
                for p, r in zip(pred_freqs, ref_freqs)
            ])))
            scatter_to_dispersion_map[model_name]["band_errors"][mp_id] = band_errors.flatten()

            
            
            # Update benchmark data for the model
            PhononDispersion.update_model_benchmarks(mp_id, model_name, ref_benchmarks_dict, pred_benchmarks_dict, 
                                model_benchmarks_dict, benchmarks)
            
            # Calculate and store statistics
            PhononDispersion.calculate_benchmark_statistics(model_name, model_benchmarks_dict, plot_stats_dict, benchmarks)
            
            # Update plotting data
            PhononDispersion.update_plotting_data(mp_id, model_name, phonon_plot_path, scatter_to_dispersion_map)


    def initialize_model_data(model_name, model_benchmarks_dict, plot_stats_dict, scatter_to_dispersion_map, benchmarks):
        """Initialize data structures for a model if they don't exist yet."""
        # Initialize model benchmarks dictionary if needed
        if model_name not in model_benchmarks_dict:
            model_benchmarks_dict[model_name] = {b: {'ref': [], 'pred': []} for b in benchmarks}
        
        # Initialize plot stats dictionary if needed
        if model_name not in plot_stats_dict:
            plot_stats_dict[model_name] = {b: {'RMSE': [], 'MAE': []} for b in benchmarks}
        
        # Initialize model plotting data if needed
        if model_name not in scatter_to_dispersion_map:
            scatter_to_dispersion_map[model_name] = {'hover': [], 'point_map': [], 'img_paths': {}}


    def update_model_benchmarks(mp_id, model_name, ref_benchmarks_dict, pred_benchmarks_dict, 
                            model_benchmarks_dict, benchmarks):
        """Update model benchmarks with the current structure's data."""
        for benchmark in benchmarks:
            ref_value = ref_benchmarks_dict[mp_id][benchmark]
            pred_value = pred_benchmarks_dict[mp_id][model_name][benchmark]
            
            model_benchmarks_dict[model_name][benchmark]['ref'].append(ref_value)
            model_benchmarks_dict[model_name][benchmark]['pred'].append(pred_value)


    def calculate_benchmark_statistics(model_name, model_benchmarks_dict, plot_stats_dict, benchmarks):
        """Calculate RMSE and MAE statistics for benchmarks."""
        for benchmark in benchmarks:
            ref_values = np.array(model_benchmarks_dict[model_name][benchmark]['ref'])
            pred_values = np.array(model_benchmarks_dict[model_name][benchmark]['pred'])
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((ref_values - pred_values)**2))
            
            # Calculate MAE
            mae = np.mean(np.abs(ref_values - pred_values))
            
            # Store statistics
            plot_stats_dict[model_name][benchmark]['RMSE'].append(rmse)
            plot_stats_dict[model_name][benchmark]['MAE'].append(mae)


    def update_plotting_data(mp_id, model_name, phonon_plot_path, scatter_to_dispersion_map):
        """Update visualization data for the current structure and model."""
        scatter_to_dispersion_map[model_name]['hover'].append(f"{mp_id} ({model_name})")
        scatter_to_dispersion_map[model_name]['point_map'].append((mp_id, model_name))
        scatter_to_dispersion_map[model_name]['img_paths'][mp_id] = phonon_plot_path


    def calculate_summary_statistics(plot_stats_dict, pretty_benchmark_labels, benchmarks):
        """Calculate and format summary statistics for all models."""
        mae_summary_dict = {}
        
        for model_name in plot_stats_dict:
            mae_summary_dict[model_name] = {}
            for benchmark in benchmarks:
                pretty_label = pretty_benchmark_labels[benchmark]
                mae_value = plot_stats_dict[model_name][benchmark]['MAE'][-1]
                mae_summary_dict[model_name][pretty_label] = round(mae_value, 3)
        
        # Create DataFrame from dictionary
        mae_summary_df = pd.DataFrame.from_dict(mae_summary_dict, orient='index')
        mae_summary_df.index.name = 'Model'
        mae_summary_df.reset_index(inplace=True)
        
        mae_cols = [col for col in mae_summary_df.columns if col not in ['Model']]
        mae_summary_df['Phonon Score \u2193'] = mae_summary_df[mae_cols].mean(axis=1).round(3)
        mae_summary_df['Rank'] = mae_summary_df['Phonon Score \u2193'].rank(method='min', ascending=True).astype(int)

        
        return mae_summary_df


    def generate_and_save_plots(scatter_to_dispersion_map, model_benchmarks_dict, plot_stats_dict, 
                            pretty_benchmark_labels, benchmarks):
        """Generate and save visualization plots for all models."""
        model_figures_dict = {}
        results_dir = Path("benchmark_stats/bulk_crystal_benchmark/phonons/")
        results_dir.mkdir(exist_ok=True)
        
        for model_name in scatter_to_dispersion_map:
            model_figures_dict[model_name] = {}
            
            scatter_dir = results_dir / model_name / "scatter_plots"
            scatter_dir.mkdir(exist_ok=True)
            
            for benchmark in benchmarks:
                ref_vals = model_benchmarks_dict[model_name][benchmark]["ref"]
                pred_vals = model_benchmarks_dict[model_name][benchmark]["pred"]
                
                # Save plot data to CSV
                pd.DataFrame({"Reference": ref_vals, "Predicted": pred_vals}).to_csv(
                    scatter_dir / f"{benchmark}.csv", index=False
                )
                
                # Get statistics
                rmse = plot_stats_dict[model_name][benchmark]['RMSE'][-1]
                mae = plot_stats_dict[model_name][benchmark]['MAE'][-1]
                
                # Create scatter plot
                fig = PhononDispersion.create_scatter_plot(
                    ref_vals, 
                    pred_vals, 
                    model_name, 
                    benchmark, 
                    mae, 
                    pretty_benchmark_labels
                )
                

                
                model_figures_dict[model_name][benchmark] = fig
                fig.write_image(scatter_dir / f"{benchmark}.png", width=800, height=600)
        
        # Save summary table to CSV
        mae_summary_df = PhononDispersion.calculate_summary_statistics(plot_stats_dict, pretty_benchmark_labels, benchmarks)
        mae_summary_df.to_csv(results_dir / "mae_phonons.csv", index=False)


    def create_scatter_plot(ref_vals, pred_vals, model_name, benchmark, mae, pretty_benchmark_labels):
        """Create a scatter plot comparing reference and predicted values."""
        combined_min = min(min(ref_vals), min(pred_vals), 0)
        combined_max = max(max(ref_vals), max(pred_vals))
        
        pretty_label = pretty_benchmark_labels[benchmark]
        
        fig = px.scatter(
            x=ref_vals,
            y=pred_vals,
            labels={
                "x": f"Reference {pretty_label}",
                "y": f"Predicted {pretty_label}",
            },
            title=f"{model_name} - {pretty_label}",
        )
        
        # y=x
        fig.add_shape(
            type="line", 
            x0=combined_min, 
            y0=combined_min, 
            x1=combined_max, 
            y1=combined_max,
            xref='x', 
            yref='y', 
            line=dict(color="black", dash="dash")
        )
        
        fig.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white", 
            font_color="black",
            xaxis=dict(showgrid=True, gridcolor="lightgray", scaleanchor="y", scaleratio=1), 
            yaxis=dict(showgrid=True, gridcolor="lightgray")
        )
        
        # legend
        fig.add_annotation(
            xref="paper", 
            yref="paper", 
            x=0.02, 
            y=0.98, 
            text=f"MAE: {mae:.3f}", 
            showarrow=False, 
            align="left", 
            font=dict(size=12, color="black"), 
            bordercolor="black", 
            borderwidth=1, 
            borderpad=4, 
            bgcolor="white", 
            opacity=0.8
        )
        
        return fig





    @staticmethod
    def create_stability_scatter_plot(ref_vals, pred_vals, labels):
        """Create a scatter plot of predicted vs reference, colored by classification (TP, TN, FP, FN)"""
        import plotly.express as px
        color_map = {
            "TP": "green",
            "TN": "blue",
            "FP": "orange",
            "FN": "red"
        }
        fig = px.scatter(
            x=ref_vals,
            y=pred_vals,
            color=labels,
            color_discrete_map=color_map,
            labels={
                "x": "Reference ω_min [THz]",
                "y": "Predicted ω_min [THz]",
                "color": "Class"
            },
            title="Stability Classification: Predicted vs Reference",
        )
        # y=x
        combined_min = min(min(ref_vals), min(pred_vals), 0)
        combined_max = max(max(ref_vals), max(pred_vals))
        fig.add_shape(
            type="line",
            x0=combined_min,
            y0=combined_min,
            x1=combined_max,
            y1=combined_max,
            xref='x',
            yref='y',
            line=dict(color="black", dash="dash")
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="black",
            xaxis=dict(showgrid=True, gridcolor="lightgray", scaleanchor="y", scaleratio=1),
            yaxis=dict(showgrid=True, gridcolor="lightgray")
        )
        return fig


    @staticmethod
    def create_confusion_matrix_figure(conf_matrix, labels, model_name):
        """Create a Plotly heatmap for a confusion matrix with labels and percent per quadrant."""
        import plotly.graph_objects as go
        import numpy as np
        total = np.sum(conf_matrix)
        percent = conf_matrix / total * 100 if total > 0 else np.zeros_like(conf_matrix, dtype=float)
        quadrant_labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]

        fig = go.Figure(
            data=go.Heatmap(
                z=conf_matrix,
                x=["Stable", "Not Stable"],  # Predicted
                y=["Stable", "Not Stable"],  # True
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Count"),
                hovertemplate='%{y} (DFT) vs %{x} (MLIP): %{z}<extra></extra>'
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title=model_name,
            yaxis_title="DFT PBE",
            xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Stable", "Not Stable"], side="top"),
            yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Stable", "Not Stable"], autorange="reversed"),
            font=dict(color="black"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            
            annotations=[
                dict(
                    x=j,
                    y=i,
                    text=f"{quadrant_labels[i][j]}<br>{percent[i, j]:.1f}%",
                    showarrow=False,
                    font=dict(size=16, color="black")
                )
                for i in range(2) for j in range(2)
            ]
        )
        return fig

    @staticmethod
    def add_stability_classification_column(mae_summary_df, model_benchmarks_dict):
        from sklearn.metrics import f1_score

        threshold = -0.05
        f1_scores = []
        for model in mae_summary_df["Model"]:
            ref_vals = np.array(model_benchmarks_dict[model]["min_freq"]["ref"])
            pred_vals = np.array(model_benchmarks_dict[model]["min_freq"]["pred"])
            y_true = ref_vals > threshold
            y_pred = pred_vals > threshold
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(round(f1, 3))

        # Insert the new column after "C_V"
        cv_col = next(col for col in mae_summary_df.columns if col.startswith("C_V"))
        insert_idx = list(mae_summary_df.columns).index(cv_col) + 1
        mae_summary_df.insert(insert_idx, "Stability Classification (F1)", f1_scores)
        return mae_summary_df
    
    @staticmethod
    def add_band_mae_column(mae_summary_df, scatter_to_dispersion_map):
        band_maes = []
        for model in mae_summary_df["Model"]:
            band_error_dict = scatter_to_dispersion_map.get(model, {}).get("band_errors", {})
            all_errors = np.concatenate(list(band_error_dict.values())) if band_error_dict else np.array([])
            avg_band_mae = np.mean(np.abs(all_errors)) if all_errors.size > 0 else np.nan
            band_maes.append(round(avg_band_mae, 3))
        # Insert the new column after "C_V" or at the end
        if "ω_min [THz]" in mae_summary_df.columns:
            insert_idx = mae_summary_df.columns.get_loc("ω_min [THz]") + 1
            mae_summary_df.insert(insert_idx, "Avg BZ MAE", band_maes)
        else:
            mae_summary_df["Avg BZ MAE"] = band_maes
        return mae_summary_df