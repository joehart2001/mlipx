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
from mlipx import PhononDispersion, Elasticity



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class BulkCrystalBenchmark(zntrack.Node):
    """ Node to combine all bulk crystal benchmarks
    """
    # inputs
    phonon_ref_list: List[PhononDispersion] = zntrack.deps()
    phonon_pred_list: List[PhononDispersion] = zntrack.deps()
    elasticity_list: List[Elasticity] = zntrack.deps()
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files
    #elasticity_dict: pathlib.Path = zntrack.outs_path(zntrack.nwd / "elasticity_dict.json")
    #phonon_ref_dict: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_ref_dict.json")
    #phonon_pred_dict: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_pred_dict.json")

    
    def run(self):
        pass
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    @staticmethod
    def benchmark_interactive(elasticity_data: List[Elasticity] | Dict[str, Elasticity],
                            phonon_ref_data: List[PhononDispersion] | Dict[str, PhononDispersion],
                            phonon_pred_data: List[PhononDispersion] | Dict[str, Dict[str, PhononDispersion]],
                            ui: str = "browser"
    ):
        
        
        """ Interactive dashboard + saving plots and data for all bulk crystal benchmarks
        """
        
        def process_data(data, key_extractor, value_extractor):
            if isinstance(data, list):
                result = {}
                for node in data:
                    key = key_extractor(node)
                    value = value_extractor(node)

                    if isinstance(value, dict):
                        # Merge nested dictionaries
                        if key not in result:
                            result[key] = {}
                        result[key].update(value)
                    else:
                        result[key] = value
                return result

            elif isinstance(data, dict):
                return data

            else:
                raise ValueError(f"{data} should be a list or dict")


        elasticity_dict = process_data(
            elasticity_data,
            key_extractor=lambda node: node.name.split("_Elasticity")[0],
            value_extractor=lambda node: node
        )

        phonon_dict_ref = process_data(
            phonon_ref_data,
            key_extractor=lambda node: node.name.split("PhononDispersion_")[1],
            value_extractor=lambda node: node
        )

        phonon_dict_pred = process_data(
            phonon_pred_data,
            key_extractor=lambda node: node.name.split("PhononDispersion_")[1],
            value_extractor=lambda node: {node.name.split("_phonons-dispersion")[0]: node}
        )                
        
        
        app_phonon, phonon_plot_stats_dict, phonon_md_path = PhononDispersion.benchmark_interactive(
            pred_node_dict=phonon_dict_pred,
            ref_node_dict=phonon_dict_ref,
            run_interactive=False,
        )
        
        app_elasticity, mae_df_elas, elas_md_path = mlipx.Elasticity.mae_plot_interactive(
            node_dict=elasticity_dict,
            run_interactive=False,
        )
    
    
    
    
        def bulk_crystal_benchmark_score(phonon_plot_stats_dict, mae_df_elas):
            """ Currently avg mae
                (problem with other explored metrics: if we normalise by the max mae, then the models in this test are comparable to each other but not models run in a different test, as they will be normalised differently)
            """
            # Initialize scores
            maes = {}
            scores = {}

            
            # Calculate scores for each model
            for model in phonon_plot_stats_dict.keys():
                scores[model] = 0
                                                
                for benchmark in phonon_plot_stats_dict[model].keys():
                    mae = phonon_plot_stats_dict[model][benchmark]['MAE'][0]
                    scores[model] += mae
                
                for benchmark in mae_df_elas.columns[1:]: # first col model
                    mae = mae_df_elas.loc[mae_df_elas['Model'] == model, benchmark].values[0]
                    scores[model] += mae
                
                scores[model] = scores[model] / (len(phonon_plot_stats_dict[model]) + len(mae_df_elas.columns[1:]))
                
            return pd.DataFrame.from_dict(scores, orient='index', columns=['Avg MAE \u2193']).reset_index().rename(columns={'index': 'Model'})

        bulk_crystal_benchmark_score_df = bulk_crystal_benchmark_score(phonon_plot_stats_dict, mae_df_elas).round(3)
        
        if not os.path.exists("benchmark_stats/"):
            os.makedirs("benchmark_stats/")
        bulk_crystal_benchmark_score_df.to_csv("benchmark_stats/bulk_crystal_benchmark_score.csv", index=False)
        
        score_min = bulk_crystal_benchmark_score_df['Avg MAE \u2193'].min()
        score_max = bulk_crystal_benchmark_score_df['Avg MAE \u2193'].max()

        def color_from_score(val):
            if score_max == score_min:
                return 'rgb(0, 255, 0)'  # If all scores are same
            ratio = (val - score_min) / (score_max - score_min)
            red = int(255 * ratio)
            green = int(255 * (1 - ratio))
            return f'rgb({red}, {green}, 0)'

        # Create color conditions
        style_data_conditional = [
            {
                'if': {'filter_query': f'{{Avg MAE \u2193}} = {score}', 'column_id': 'Avg MAE \u2193'},
                'backgroundColor': color_from_score(score),
                'color': 'white' if score > (score_min + score_max) / 2 else 'black'
            }
            for score in bulk_crystal_benchmark_score_df['Avg MAE \u2193']
        ]
        
        
        md_path_list = [elas_md_path, phonon_md_path]
    
        BulkCrystalBenchmark.generate_report(
            bulk_crystal_benchmark_score_df=bulk_crystal_benchmark_score_df,
            md_report_paths=md_path_list,
            markdown_path="benchmark_stats/bulk_crystal_benchmark_report.md",
        )


    
        
        # benchmark score table
        benchmark_score_table = html.Div([
            html.H2("Benchmark Score Table", style={'color': 'Black', 'padding': '1rem'}),
            dash_table.DataTable(
                id='benchmark-score-table',
                columns=[{"name": col, "id": col} for col in bulk_crystal_benchmark_score_df.columns],
                data=bulk_crystal_benchmark_score_df.to_dict('records'),
                style_cell={'textAlign': 'center', 'fontSize': '14px'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=style_data_conditional,
            ),
        ])
        
        if ui is None:
            return
        
        
        phohon_layout = app_phonon.layout
        elasticity_layout = app_elasticity.layout
        
        app_phonon.layout = html.Div([
            html.H1("Bulk Crystal Benchmark", style={"color": "black"}),
            html.Div(benchmark_score_table, style={
                "backgroundColor": "white",
                "padding": "20px",
                "border": "2px solid black",
                "marginBottom": "30px"
            }),
            html.Div(phohon_layout.children, style={
                "backgroundColor": "white",
                "padding": "20px",
                "border": "2px solid black",
                "marginBottom": "30px"
            }),
            html.Div(elasticity_layout.children, style={
                "backgroundColor": "white",
                "padding": "20px",
                "border": "2px solid gray"
            }),
        ], style={"backgroundColor": "#f8f8f8"})

        Elasticity.register_elasticity_callbacks(
            app_phonon, mae_df_elas, elasticity_dict,
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


            print(f"Dash app running at {url}")
        
        return run_app(app_phonon, ui=ui)
    
    
    
    
    
    
    














    # --------------------------------- Helper Functions ---------------------------------


    def generate_report(
        bulk_crystal_benchmark_score_df: pd.DataFrame,
        md_report_paths: List[str],
        markdown_path: str,
    ):
        markdown_path = Path(markdown_path)
        pdf_path = markdown_path.with_suffix(".pdf")
        combined_md = []
        
        # Benchmark score Summary table
        combined_md.append("# Bulk Crystal Benchmark Report\n")
        combined_md.append("## Benchmark Score Table \n")
        combined_md.append(bulk_crystal_benchmark_score_df.to_markdown(index=False))
        combined_md.append("\n")

        for path in md_report_paths:
            path = Path(path)
            if not path.exists():
                print(f"Skipping {path} — file not found")
                continue
            with open(path, 'r') as f:
                md = f.read()
                #combined_md.append(f"# {path.stem}\n\n")
                combined_md.append(md)
                combined_md.append('\n\\newpage\n\n')
        
        Path(markdown_path).write_text("\n".join(combined_md))
        
        
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
        results_dir = Path("phonon-dispersion-stats-plots")
        results_dir.mkdir(exist_ok=True)
        
        for model_name in model_plotting_data:
            model_figures_dict[model_name] = {}
            
            model_dir = results_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            for benchmark in benchmarks:
                ref_vals = model_benchmarks_dict[model_name][benchmark]["ref"]
                pred_vals = model_benchmarks_dict[model_name][benchmark]["pred"]
                
                # Save plot data to CSV
                pd.DataFrame({"Reference": ref_vals, "Predicted": pred_vals}).to_csv(
                    model_dir / f"{benchmark}.csv", index=False
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
                fig.write_image(model_dir / f"{benchmark}.png")
        
        # Save summary table to CSV
        mae_summary_df = PhononDispersion.calculate_summary_statistics(plot_stats_dict, pretty_benchmark_labels, benchmarks)
        mae_summary_df.to_csv(results_dir / "mae_summary.csv", index=False)


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