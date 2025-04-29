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




class PhononDispersion(zntrack.Node):
    """Compute the phonon dispersion from a phonopy object
    """
    # inputs
    phonopy_yaml_path: pathlib.Path = zntrack.deps()
    thermal_properties_path: pathlib.Path = zntrack.deps()
    
    node_idx: int = zntrack.params(None)
    total_no_nodes: int = zntrack.params(None)
    

    qpoints_input_path: t.Optional[pathlib.Path] = zntrack.deps(None)
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

        
        if self.qpoints_input_path:
            # calculate phonon structure along the reference path to ensure a valid comparison
            with open(self.qpoints_input_path, "rb") as f:
                qpoints = pickle.load(f)
            with open(self.labels_input_path, "r") as f:
                labels = json.load(f)
            with open(self.connections_input_path, "r") as f:
                connections = json.load(f)
            

            
            #phonons.run_band_structure(paths=qpoints, labels=labels, path_connections=connections)
            #print(len(phonons.get_band_structure_dict()["distances"][0]))

            phonons.auto_band_structure()
            

            # save, zntrack requires each output declared to be an output
            with open(self.qpoints_path, "wb") as f:
                pickle.dump(qpoints, f)
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
            
            with open(self.qpoints_path, "wb") as f:
                pickle.dump(qpoints, f)
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

    @property
    def qpoints(self):
        with open(self.qpoints_path, "rb") as f:
            return pickle.load(f)

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
            
            if not os.path.exists(f"benchmark_stats/phonons/{model_name}/phonon_plots"):
                os.makedirs(f"benchmark_stats/phonons/{model_name}/phonon_plots")
            
            phonon_plot_path = f"benchmark_stats/phonons/{model_name}/phonon_plots/dispersion_{model_name}_{mp_id}.png"
            fig.savefig(phonon_plot_path, bbox_inches='tight')
            plt.close(fig)
            return phonon_plot_path
        
        else:
            plt.show()
            
        return
    

        




















    @staticmethod
    def benchmark_interactive(pred_node_dict, ref_node_dict, ui = None, dont_run = False):
        """
        Main benchmarking function that coordinates the benchmarking process.
        
        Args:
            pred_node_dict: Dictionary of prediction nodes
            ref_node_dict: Dictionary of reference nodes
            ui: Optional UI object for interactive display
            
        Returns:
            Various benchmarking data and plots
        """

        ref_band_data_dict = {}
        ref_benchmarks_dict = {}
        pred_benchmarks_dict = {}
        model_plotting_data = {}     # model_name -> {property -> {ref, pred}, hover, point_map, img_paths}
        phonon_plot_paths = {}
        point_index_to_id = []
        
        benchmarks = [
            'max_freq', 
            'S',
            'F',
            'C_V',
        ]
        benchmark_units = {
            'max_freq': 'THz', 
            'S': '[J/K/mol]',
            'F': '[kJ/mol]',
            'C_V': '[J/K/mol]',
        }
        
        pretty_benchmark_labels = {
            'max_freq': 'ω_max [THz]',
            'S': 'S [J/mol·K]',
            'F': 'F [kJ/mol]',
            'C_V': 'C_V [J/mol·K]'
        }
        label_to_key = {v: k for k, v in pretty_benchmark_labels.items()}

        model_benchmarks_dict = {} # max_freq, min_freq ...
        plot_stats_dict = {}
        
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
                model_plotting_data,
                phonon_plot_paths,
                point_index_to_id,
                benchmarks
            )
        
        # Calculate summary statistics and generate plots
        mae_summary_df = PhononDispersion.calculate_summary_statistics(
            plot_stats_dict, 
            pretty_benchmark_labels, 
            benchmarks
        )
        
        PhononDispersion.generate_and_save_plots(
            model_plotting_data,
            model_benchmarks_dict,
            plot_stats_dict,
            pretty_benchmark_labels,
            benchmarks
        )
        
        model_list = list(pred_node_dict[list(pred_node_dict.keys())[0]].keys())
        
        PhononDispersion.generate_phonon_markdown_report(
            mae_summary_df=mae_summary_df,
            models_list=model_list
        )
        
        if ui is None:
            if dont_run:
                pass
            else:
                return




        
        # --------------------------------- Dash app ---------------------------------

        
        # Dash app
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H2("Phonon dispersion MAE Summary Table (300 K)", style={"color": "black"}),

            dash_table.DataTable(
                id='mae-summary-table',
                columns=[{"name": col, "id": col} for col in mae_summary_df.columns],
                data=mae_summary_df.to_dict('records'),
                style_cell={'textAlign': 'center', 'fontSize': '14px'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                cell_selectable=True
            ),

            html.Br(),

            #html.H2("Pair Correlation Plot: Predicted vs Reference", style={"color": "black"}),

            html.Div(id="model-graphs-container")
        ], style={"backgroundColor": "white", "padding": "20px"})

        # Callback to generate plot for a single model/property cell
        @app.callback(
            Output("model-graphs-container", "children"),
            Input("mae-summary-table", "active_cell"),
            State("mae-summary-table", "data")
        )
        def update_single_model_plot(active_cell, table_data):
            if active_cell is None:
                raise PreventUpdate

            col = active_cell['column_id']
            row = active_cell['row']
            if col == 'Model':
                raise PreventUpdate

            model_name = table_data[row]['Model']
            selected_property = label_to_key.get(col)
            if selected_property is None:
                return html.Div("Invalid property selected.")

            # Gather data
            plotting_data = model_plotting_data[model_name]
            data = model_benchmarks_dict[model_name]

            ref_vals = data[selected_property]['ref']
            pred_vals = data[selected_property]['pred']

            x_min, x_max = min(ref_vals), max(ref_vals)
            y_min, y_max = min(pred_vals), max(pred_vals)
            x_range = [x_min - 0.05 * abs(x_min), x_max + 0.05 * abs(x_max)]
            y_range = [y_min - 0.05 * abs(y_min), y_max + 0.05 * abs(y_max)]

            combined_min = min(min(ref_vals), min(pred_vals), 0)
            combined_max = max(max(ref_vals), max(pred_vals))

            stats = plot_stats_dict[model_name][selected_property]
            metrics_text = (
                f"RMSE: {stats['RMSE'][-1]:.2f}<br>"
                f"MAE: {stats['MAE'][-1]:.2f}<br>"
            )

            units = benchmark_units.get(selected_property, "")

            scatter_fig = px.scatter(
                x=ref_vals,
                y=pred_vals,
                hover_name=plotting_data['hover'],
                labels={
                    'x': f'Reference {selected_property.replace("_", " ").title()} {units}',
                    'y': f'Predicted {selected_property.replace("_", " ").title()} {units}'
                }
            )
            scatter_fig.add_shape(
                type="line",
                x0=combined_min, y0=combined_min,
                x1=combined_max, y1=combined_max,
                xref='x', yref='y',
                line=dict(color="black", dash="dash"),
                name="y = x"
            )

            scatter_fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='black',
                margin=dict(t=40, r=30, b=40, l=50),
                xaxis=dict(range=x_range, showgrid=True, gridcolor='lightgray', scaleanchor='y', scaleratio=1),
                yaxis=dict(range=y_range, showgrid=True, gridcolor='lightgray')
            )

            scatter_fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=metrics_text,
                showarrow=False,
                align="left",
                font=dict(size=12, color="black"),
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )

            return html.Div([
                html.H4(f"{model_name} - {selected_property.replace('_', ' ').title()}", style={"color": "black"}),
                html.Div([
                    dcc.Graph(
                        id={'type': 'pair-plot', 'index': model_name},
                        figure=scatter_fig,
                        style={"width": "45vw", "height": "60vh"}
                    ),
                    html.Div(
                        id={'type': 'plot-display', 'index': model_name},
                        style={
                            "width": "50vw",
                            "height": "60vh",
                            "marginLeft": "2vw",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "border": "1px solid #ccc",
                            "padding": "10px",
                            #"backgroundColor": "#f9f9f9"
                        }
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "40px"
                })
            ])

        # Callback to show phonon dispersion image on scatter click
        @app.callback(
            Output({'type': 'plot-display', 'index': MATCH}, 'children'),
            Input({'type': 'pair-plot', 'index': MATCH}, 'clickData'),
            State({'type': 'pair-plot', 'index': MATCH}, 'id')
        )
        def display_phonon_plot(clickData, graph_id):
            model_name = graph_id['index']
            if clickData is None:
                return html.Div("Click on a point to view its phonon dispersion plot.")

            point_index = clickData['points'][0]['pointIndex']
            mp_id, _ = model_plotting_data[model_name]['point_map'][point_index]
            img_path = model_plotting_data[model_name]['img_paths'][mp_id]

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
            
            
        def reserve_free_port():
            s = socket.socket()
            s.bind(('', 0))
            port = s.getsockname()[1]
            return s, port  # you must close `s` later


        def run_app(app, ui):
            sock, port = reserve_free_port()
            url = f"http://localhost:{port}"
            sock.close()

            def _run_server():
                app.run(debug=True, use_reloader=False, port=port)
                
                
            # if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
            #     import threading
            #     print(f"\n Detected SSH session — skipping browser launch.")
            #     #threading.Thread(target=_run_server, daemon=True).start()
            #     return

                
            if ui == "browser":
                import webbrowser
                import threading
                #threading.Thread(target=_run_server, daemon=True).start()
                _run_server()
                time.sleep(1.5)
                #webbrowser.open(url)
            elif ui == "notebook":
                _run_server()
            
            else:
                print(f"Unknown UI option: {ui}. Please use 'browser', or 'notebook'.")
                return


        
        if dont_run:
            return app, plot_stats_dict


        return run_app(app, ui=ui)








    # --------------------------------- Helper Functions ---------------------------------




    def generate_phonon_markdown_report(
        mae_summary_df,
        models_list,
    ):
        markdown_path = Path("benchmark_stats/phonons/phonon_benchmark_report.md")
        pdf_path = markdown_path.with_suffix(".pdf")

        md = []

        md.append("# Phonon Dispersion Report\n")

        # MAE Summary table
        md.append("## Phonon Dispersion MAE Summary (300K)\n")
        md.append(mae_summary_df.to_markdown(index=False))
        md.append("\n")
        
        def add_image_rows(md_lines, image_paths):
            """Append 2-per-row image rows using Pandoc's attribute syntax."""
            for i in range(0, len(image_paths), 2):
                pair = image_paths[i:i+2]
                line = " ".join(f"![]({img.resolve()}){{ width=48% }}" for img in pair)
                md_lines.append(line + "\n")


        # Scatter Plots
        md.append("## Scatter Plots\n")
        for model in models_list:
            md.append(f"### {model}\n")
            scatter_plot_dir = Path(f"benchmark_stats/phonons/{model}/scatter_plots")
            # for img_path in sorted(scatter_plot_dir.glob("*.png")):
            #     title = img_path.stem.replace("_", " ").title()
            #     md.append(f"### {title}\n")
            #     md.append(f"![]({img_path.resolve()})\n")  # <-- full path
            images = sorted(scatter_plot_dir.glob("*.png"))
            add_image_rows(md, images)            
                    
        # Dispersion Plots
        md.append("## Dispersion Plots\n")
        for model in models_list:
            md.append(f"### {model}\n")
            dispersion_plot_dir = Path(f"benchmark_stats/phonons/{model}/phonon_plots")
            # for img_path in sorted(dispersion_plot_dir.glob("*.png")):
            #     title = model + img_path.stem.split("_")[1]
            #     md.append(f"### {title}\n")
            #     md.append(f"![]({img_path.resolve()})\n")  # <-- full path
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
                ["pandoc", str(markdown_path), "-o", str(pdf_path), "--pdf-engine=xelatex"],
                check=True
            )
            print(f"PDF report saved to {pdf_path}")

        except subprocess.CalledProcessError as e:
            print(f"PDF generation failed: {e}")



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
        ref_benchmarks_dict[mp_id]['S'] = node_ref.get_thermal_properties['entropy'][T_300K_index]
        ref_benchmarks_dict[mp_id]['F'] = node_ref.get_thermal_properties['free_energy'][T_300K_index]
        ref_benchmarks_dict[mp_id]['C_V'] = node_ref.get_thermal_properties['heat_capacity'][T_300K_index]
        
        return True


    def process_prediction_data(pred_nodes, node_ref, mp_id, ref_benchmarks_dict, pred_benchmarks_dict,
                            model_benchmarks_dict, plot_stats_dict, model_plotting_data, 
                            phonon_plot_paths, point_index_to_id, benchmarks):
        """
        Process prediction data for all models for a given structure.
        """
        for model_name in pred_nodes.keys():
            try:
                dos_freqs_pred, dos_values_pred = pred_nodes[model_name].dos
            except FileNotFoundError:
                print(f"Skipping {model_name} — dos file not found at {pred_nodes[model_name].dos_path}")
                continue
            
            # Calculate prediction benchmarks
            pred_benchmarks_dict[mp_id][model_name] = {}
            T_300K_index = pred_nodes[model_name].get_thermal_properties['temperatures'].index(300)
            
            pred_benchmarks_dict[mp_id][model_name]['max_freq'] = np.max(dos_freqs_pred)
            pred_benchmarks_dict[mp_id][model_name]['S'] = pred_nodes[model_name].get_thermal_properties['entropy'][T_300K_index]
            pred_benchmarks_dict[mp_id][model_name]['F'] = pred_nodes[model_name].get_thermal_properties['free_energy'][T_300K_index]
            pred_benchmarks_dict[mp_id][model_name]['C_V'] = pred_nodes[model_name].get_thermal_properties['heat_capacity'][T_300K_index]
            
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
            PhononDispersion.initialize_model_data(model_name, model_benchmarks_dict, plot_stats_dict, model_plotting_data, benchmarks)
            
            # Update benchmark data for the model
            PhononDispersion.update_model_benchmarks(mp_id, model_name, ref_benchmarks_dict, pred_benchmarks_dict, 
                                model_benchmarks_dict, benchmarks)
            
            # Calculate and store statistics
            PhononDispersion.calculate_benchmark_statistics(model_name, model_benchmarks_dict, plot_stats_dict, benchmarks)
            
            # Update plotting data
            PhononDispersion.update_plotting_data(mp_id, model_name, phonon_plot_path, model_plotting_data)


    def initialize_model_data(model_name, model_benchmarks_dict, plot_stats_dict, model_plotting_data, benchmarks):
        """Initialize data structures for a model if they don't exist yet."""
        # Initialize model benchmarks dictionary if needed
        if model_name not in model_benchmarks_dict:
            model_benchmarks_dict[model_name] = {b: {'ref': [], 'pred': []} for b in benchmarks}
        
        # Initialize plot stats dictionary if needed
        if model_name not in plot_stats_dict:
            plot_stats_dict[model_name] = {b: {'RMSE': [], 'MAE': []} for b in benchmarks}
        
        # Initialize model plotting data if needed
        if model_name not in model_plotting_data:
            model_plotting_data[model_name] = {'hover': [], 'point_map': [], 'img_paths': {}}


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


    def update_plotting_data(mp_id, model_name, phonon_plot_path, model_plotting_data):
        """Update visualization data for the current structure and model."""
        model_plotting_data[model_name]['hover'].append(f"{mp_id} ({model_name})")
        model_plotting_data[model_name]['point_map'].append((mp_id, model_name))
        model_plotting_data[model_name]['img_paths'][mp_id] = phonon_plot_path


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
        
        return mae_summary_df


    def generate_and_save_plots(model_plotting_data, model_benchmarks_dict, plot_stats_dict, 
                            pretty_benchmark_labels, benchmarks):
        """Generate and save visualization plots for all models."""
        model_figures_dict = {}
        results_dir = Path("benchmark_stats/phonons/")
        results_dir.mkdir(exist_ok=True)
        
        for model_name in model_plotting_data:
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
                    rmse, 
                    mae, 
                    pretty_benchmark_labels
                )
                
                model_figures_dict[model_name][benchmark] = fig
                fig.write_image(scatter_dir / f"{benchmark}.png")
        
        # Save summary table to CSV
        mae_summary_df = PhononDispersion.calculate_summary_statistics(plot_stats_dict, pretty_benchmark_labels, benchmarks)
        mae_summary_df.to_csv(results_dir / "mae_phonons.csv", index=False)


    def create_scatter_plot(ref_vals, pred_vals, model_name, benchmark, rmse, mae, pretty_benchmark_labels):
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
            text=f"RMSE: {rmse:.3f}<br>MAE: {mae:.3f}", 
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





 



    



    # @staticmethod
    # def bulk_crystal_benchmark(phonon_pred_node_dict, phonon_ref_node_dict, elasticity_dict , ui = None):
        
    #     """Combines all the benchmarks related to bulk crystals, including:
    #     - Phonon Dispersion
    #     - Elasticity (bulk and shear moduli)
    #     """
        

    #     app, phonon_plot_stats_dict = PhononDispersion.benchmark_interactive(
    #         phonon_pred_node_dict, 
    #         phonon_ref_node_dict,
    #         ui = "browser",
    #         dont_run=True,
    #     )
        
    #     print(phonon_plot_stats_dict)


    #     benchmarks = [
    #         'K_vrh', 
    #         'G_vrh',
    #     ]
    #     benchmark_units = {
    #         'K_vrh': '[GPa]', 
    #         'G_vrh': '[GPa]',
    #     }
    #     benchmark_labels = {
    #         'K_vrh': 'K_bulk',
    #         'G_vrh': 'K_shear',
    #     }
        
    #     label_to_key = {v: k for k, v in benchmark_labels.items()}
        
    #     mae_df_elas = pd.DataFrame()

    #     for model in elasticity_dict.keys():
    #         results_df = elasticity_dict[model].results
    #         mae_K = np.abs(results_df[f'K_vrh_{model}'].values - results_df['K_vrh_DFT'].values).mean()
    #         mae_G = np.abs(results_df[f'G_vrh_{model}'].values - results_df['G_vrh_DFT'].values).mean()
    #         mae_df_elas.loc[model, 'K_bulk [GPa]'] = mae_K
    #         mae_df_elas.loc[model, 'K_shear [GPa]'] = mae_G

    #     mae_df_elas = mae_df_elas.reset_index().rename(columns={'index': 'Model'})
    #     mae_df_elas = mae_df_elas.round(3)


        
    #     def bulk_crystal_benchmark_score(phonon_plot_stats_dict, mae_df_elas):
    #         """ Currently avg mae
    #             (problem with other explored metrics: if we normalise by the max mae, then the models in this test are comparable to each other but not models run in a different test, as they will be normalised differently)
    #         """
    #         # Initialize scores
    #         maes = {}
    #         scores = {}

            
    #         # Calculate scores for each model
    #         for model in phonon_plot_stats_dict.keys():
    #             scores[model] = 0
                                                
    #             for benchmark in phonon_plot_stats_dict[model].keys():
    #                 mae = phonon_plot_stats_dict[model][benchmark]['MAE'][0]
    #                 scores[model] += mae
                
    #             for benchmark in mae_df_elas.columns[1:]: # first col model
    #                 mae = mae_df_elas.loc[mae_df_elas['Model'] == model, benchmark].values[0]
    #                 scores[model] += mae
                
    #             scores[model] = scores[model] / (len(phonon_plot_stats_dict[model]) + len(mae_df_elas.columns[1:]))
                
    #         return pd.DataFrame.from_dict(scores, orient='index', columns=['Avg MAE \u2193']).reset_index().rename(columns={'index': 'Model'})

    #     bulk_crystal_benchmark_score_df = bulk_crystal_benchmark_score(phonon_plot_stats_dict, mae_df_elas).round(3)
        
    #     score_min = bulk_crystal_benchmark_score_df['Avg MAE \u2193'].min()
    #     score_max = bulk_crystal_benchmark_score_df['Avg MAE \u2193'].max()

    #     def color_from_score(val):
    #         if score_max == score_min:
    #             return 'rgb(0, 255, 0)'  # If all scores are same
    #         ratio = (val - score_min) / (score_max - score_min)
    #         red = int(255 * ratio)
    #         green = int(255 * (1 - ratio))
    #         return f'rgb({red}, {green}, 0)'

    #     # Create color conditions
    #     style_data_conditional = [
    #         {
    #             'if': {'filter_query': f'{{Avg MAE \u2193}} = {score}', 'column_id': 'Avg MAE \u2193'},
    #             'backgroundColor': color_from_score(score),
    #             'color': 'white' if score > (score_min + score_max) / 2 else 'black'
    #         }
    #         for score in bulk_crystal_benchmark_score_df['Avg MAE \u2193']
    #     ]






    #     elasticity_layout = html.Div([
    #         html.H2("Bulk and Shear Moduli MAEs", style={'color': 'Black', 'padding': '1rem'}),
    #         dash_table.DataTable(
    #             id='mae-table',
    #             columns=[{"name": col, "id": col} for col in mae_df_elas.columns],
    #             data=mae_df_elas.to_dict('records'),
    #             style_cell={'textAlign': 'center', 'fontSize': '14px'},
    #             style_header={'fontWeight': 'bold'},
    #         ),
    #         dcc.Graph(id='scatter-plot'),
    #     ],
    #     style={
    #         'backgroundColor': 'white',
    #     })
        
        
    #     # benchmark score table
    #     benchmark_score_table = html.Div([
    #         html.H2("Benchmark Score Table", style={'color': 'Black', 'padding': '1rem'}),
    #         dash_table.DataTable(
    #             id='benchmark-score-table',
    #             columns=[{"name": col, "id": col} for col in bulk_crystal_benchmark_score_df.columns],
    #             data=bulk_crystal_benchmark_score_df.to_dict('records'),
    #             style_cell={'textAlign': 'center', 'fontSize': '14px'},
    #             style_header={'fontWeight': 'bold'},
    #             style_data_conditional=style_data_conditional,
    #         ),
    #     ])
        
        
    #     original_layout = app.layout
        
    #     app.layout = html.Div([
    #         html.H1("Bulk Crystal Benchmark", style={"color": "black"}),
    #         html.Div(benchmark_score_table, style={
    #             "backgroundColor": "white",
    #             "padding": "20px",
    #             "border": "2px solid black",
    #             "marginBottom": "30px"
    #         }),
    #         html.H2("Phonon Dispersion MAE Summary Table (300 K)", style={"color": "black"}),
    #         html.Div(original_layout.children, style={
    #             "backgroundColor": "white",
    #             "padding": "20px",
    #             "border": "2px solid black",
    #             "marginBottom": "30px"
    #         }),
    #         html.Div(elasticity_layout, style={
    #             "backgroundColor": "white",
    #             "padding": "20px",
    #             "border": "2px solid gray"
    #         }),
    #     ], style={"backgroundColor": "#f8f8f8"})

            

    #     @app.callback(
    #         Output('scatter-plot', 'figure'),
    #         Input('mae-table', 'active_cell')
    #     )
    #     def update_scatter_plot(active_cell):
    #         if active_cell is None: 
    #             raise PreventUpdate
    #             #return px.scatter(title="Click on a cell to view scatter plot")

    #         row = active_cell['row']
    #         col = active_cell['column_id']
    #         model = mae_df_elas.loc[row, 'Model']

    #         if col not in mae_df_elas.columns or col == 'Model':
    #             return px.scatter(title="Invalid column clicked")

    #         label = col.split()[0]  # "K_bulk" or "K_shear"
    #         prop = label_to_key.get(label, None)
    #         if prop is None:
    #             return px.scatter(title="Unknown property")

    #         df = elasticity_dict[model].results
    #         fig = px.scatter(
    #             data_frame=df,
    #             x=f'{prop}_DFT',
    #             y=f'{prop}_{model}',
    #             labels={
    #                 f'{prop}_DFT': f'{label} DFT [GPa]',
    #                 f'{prop}_{model}': f'{label} Predicted [GPa]'
    #             },
    #             title=f'{label} Scatter Plot - {model}',
    #             hover_data=['mp_id', 'formula', f'{prop}_DFT', f'{prop}_{model}'],
    #         )
    #         fig.add_shape(type='line', x0=df[f'{prop}_DFT'].min(), y0=df[f'{prop}_DFT'].min(),
    #                     x1=df[f'{prop}_DFT'].max(), y1=df[f'{prop}_DFT'].max(),
    #                     line=dict(dash='dash'))

    #         fig.update_layout(
    #             plot_bgcolor='white',
    #             paper_bgcolor='white',
    #             font_color='black',
    #             xaxis_showgrid=True,
    #             yaxis_showgrid=True,
    #             xaxis=dict(gridcolor='lightgray'),
    #             yaxis=dict(gridcolor='lightgray')
    #         )

            
            
    #         fig.add_annotation(
    #             xref="paper", yref="paper",
    #             x=0.02, y=0.98,
    #             text=f"{label} MAE: {mae_df_elas.loc[row, col]} GPa",
    #             showarrow=False,
    #             align="left",
    #             font=dict(size=12, color="black"),
    #             bordercolor="black",
    #             borderwidth=1,
    #             borderpad=4,
    #             bgcolor="white",
    #             opacity=0.8
    #         )


    #         return fig

        

    #     def reserve_free_port():
    #         s = socket.socket()
    #         s.bind(('', 0))
    #         port = s.getsockname()[1]
    #         return s, port  # you must close `s` later


    #     def run_app(app, ui):
    #         sock, port = reserve_free_port()
    #         url = f"http://localhost:{port}"
    #         sock.close()

    #         def _run_server():
    #             app.run(debug=True, use_reloader=False, port=port)
                
                
                
    #         if ui == "browser":
    #             import webbrowser
    #             import threading
    #             #threading.Thread(target=_run_server, daemon=True).start()
    #             _run_server()
    #             time.sleep(1.5)
    #             #webbrowser.open(url)
    #         elif ui == "notebook":
    #             _run_server()
            
    #         else:
    #             print(f"Unknown UI option: {ui}. Please use 'browser', or 'notebook'.")
    #             return


    #         print(f"Dash app running at {url}")
        
    #     return run_app(app, ui=ui)