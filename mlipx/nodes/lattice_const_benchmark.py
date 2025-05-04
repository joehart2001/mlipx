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


import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class LatticeConstant(zntrack.Node):
    """ Node to combine all bulk crystal benchmarks
    """
    # inputs
    structure: List[mlipx.StructureOptimization] = zntrack.deps()
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files
    lattice_const_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_const.json")

    
    
    def run(self):

        structure = self.structure[-1]
        
        lattice_const = structure.cell.lengths()[0]
        lattice_type = structure.info['lattice_type']
        # primitive cell lattice const -> conventional cell lattice const

        primitive_length = structure.cell.lengths()[0]

        if lattice_type in ['fcc', 'diamond' , 'rocksalt', 'zincblende']:
            lattice_const = primitive_length * np.sqrt(2)
        elif lattice_type == 'bcc':
            lattice_const = primitive_length * 2 / np.sqrt(3)
        elif lattice_type == 'sc':
            lattice_const = primitive_length

        with open(self.lattice_const_output, 'w') as f:
            json.dump(lattice_const, f)
            print(f"Saved lattice constant to {self.lattice_const_output}")
                
    
    @property
    def get_lattice_const(self):
        with open(self.lattice_const_output, 'r') as f:
            lattice_const = json.load(f)
        return lattice_const
    
    
    
    
    
    @staticmethod
    def mae_plot_interactive(
        node_dict,
        ref_node,
        ui: str | None = None,
        run_interactive: bool = True,
    ):

        ref_dict = ref_node.get_ref
        
        full_data = {}
        for formula, model_data in node_dict.items():
            formula_ref = {'SiC_a': 'SiC(a)', 'SiC_c': 'SiC(c)'}.get(formula, formula)
            ref_val = ref_dict[formula_ref]
            full_data[formula] = {"ref": ref_val}
            for model_name, node in model_data.items():
                full_data[formula][model_name] = node.get_lattice_const

        lat_const_df = pd.DataFrame.from_dict(full_data, orient="index")
        lat_const_df = lat_const_df[["ref"] + [col for col in lat_const_df.columns if col != "ref"]]
        lat_const_df = lat_const_df.round(3)

        # mae
        mae_data = []
        for model_name in lat_const_df.columns:
            if model_name == "ref":
                continue
            diff = lat_const_df[model_name] - lat_const_df["ref"]
            mae = np.mean(np.abs(diff))
            mae_data.append({"Model": model_name, "MAE (Å)": round(mae, 3)})

        mae_df = pd.DataFrame(mae_data)

        # save
        LatticeConstant.save_lattice_const_plots_tables(lat_const_df)
        
        # report
        models_list = [col for col in lat_const_df.columns if col != "ref"]
        md_path = LatticeConstant.generate_lattice_const_report(
            mae_df=mae_df,
            models_list=models_list,
            markdown_path="benchmark_stats/lattice_constants/lattice_const_report.md",
            lat_const_df=lat_const_df,
        )

        

        if ui is None and run_interactive:
            return lat_const_df, mae_df

        # === 5. Dash App ===
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H2("Lattice Constants MAE Summary Table", style={"color": "black"}),

            dash_table.DataTable(
                id='lat-mae-score-table',
                columns=[{"name": col, "id": col} for col in mae_df.columns],
                data=mae_df.to_dict('records'),
                style_cell={'textAlign': 'center', 'fontSize': '14px'},
                style_header={'fontWeight': 'bold'},
                cell_selectable=True,
            ),

            html.Br(),
            
            dcc.Store(id='lattice-table-last-clicked'),
            html.Div(id="lattice-const-table"),
            
        ], style={"backgroundColor": "white", "padding": "20px"})

        LatticeConstant.register_callbacks(app, mae_df, lat_const_df)
        


        def reserve_free_port():
            s = socket.socket()
            s.bind(('', 0))
            port = s.getsockname()[1]
            return s, port

        def run_app(app, ui):
            sock, port = reserve_free_port()
            url = f"http://localhost:{port}"
            sock.close()

            def _run_server():
                app.run(debug=True, use_reloader=False, port=port)

            if ui == "browser":
                _run_server()
            elif ui == "notebook":
                _run_server()
            else:
                print(f"Unknown UI option: {ui}. Please use 'browser' or 'notebook'.")
                return

            print(f"Dash app running at {url}")

        if not run_interactive:
            return app, mae_df, lat_const_df, md_path

        return run_app(app, ui=ui)
    
    
    
    
    # -------- helper functions --------
    
    def generate_lattice_const_report(
        mae_df,
        models_list,
        markdown_path,
        lat_const_df,
    ):
        """Generate a markdown + PDF report with MAE table, difference tables, and scatter plots."""
        markdown_path = Path(markdown_path)
        
        pdf_path = markdown_path.with_suffix(".pdf")

        md = []

        md.append("# Lattice Constants Report\n")

        # MAE Summary Table
        md.append("## Lattice Constants MAEs\n")
        md.append(mae_df.to_markdown(index=False))
        md.append("\n")

        def add_image_rows(md_lines, image_paths, n_cols=1):
            for i in range(0, len(image_paths), n_cols):
                image_set = image_paths[i:i+n_cols]
                width = 100 // n_cols
                line = " ".join(f"![]({img.resolve()}){{ width={width}% }}" for img in image_set)
                md_lines.append(line + "\n")

        # Per-model Δ Tables and Scatter Plots
        md.append("## Per-Model Tables and Scatter Plots\n")

        for model in models_list:
            if model == "ref":
                continue

            md.append(f"### {model}\n")

            # diff Table
            ref_vals = lat_const_df["ref"]
            pred_vals = lat_const_df[model]
            abs_diff = pred_vals - ref_vals
            pct_diff = 100 * abs_diff / ref_vals
            formulas = lat_const_df.index.tolist()

            table_df = pd.DataFrame({
                "Element": formulas,
                "DFT": ref_vals,
                model: pred_vals,
                "Δ": abs_diff.round(3),
                "Δ/%": pct_diff.round(2)
            }).round(3)

            mae_val = mae_df[mae_df["Model"] == model]["MAE (Å)"].values[0]
            table_df.loc["MAE"] = ["", "", "", round(mae_val, 3), ""]

            md.append(table_df.to_markdown(index=False))
            md.append("\n")

            # Scatter plot image(s)
            scatter_plot_dir = markdown_path.parent / f"{model}/scatter_plots"
            images = sorted(scatter_plot_dir.glob("*.png"))
            add_image_rows(md, images)

        # Save Markdown file
        markdown_path.write_text("\n".join(md))
        print(f"Markdown report saved to: {markdown_path}")

        # Optional PDF export
        try:
            import subprocess
            subprocess.run(
                [
                    "pandoc", str(markdown_path),
                    "-o", str(pdf_path),
                    "--pdf-engine=xelatex",
                    "--variable=geometry:top=1.5cm,bottom=2cm,left=1cm,right=1cm"
                ],
                check=True
            )
            print(f"PDF report saved to {pdf_path}")
        except Exception as e:
            print(f"PDF generation failed: {e}")

        return markdown_path
    
    
    @staticmethod
    def save_lattice_const_plots_tables(lat_const_df, save_dir="benchmark_stats/lattice_constants"):
        """Save lattice constant scatter plots, Δ tables, and summary table to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        mae_summary = []

        for model_name in lat_const_df.columns:
            if model_name == "ref":
                continue

            model_dir = save_dir / model_name / "scatter_plots"
            model_dir.mkdir(parents=True, exist_ok=True)

            ref_vals = lat_const_df["ref"]
            pred_vals = lat_const_df[model_name]
            formulas = lat_const_df.index.tolist()

            # === 1. Save reference/prediction CSV
            pd.DataFrame({
                "Formula": formulas,
                "Reference": ref_vals,
                "Predicted": pred_vals
            }).to_csv(model_dir / "lattice_constants.csv", index=False)

            # === 2. Δ and %Δ table
            abs_diff = pred_vals - ref_vals
            pct_diff = 100 * abs_diff / ref_vals

            table_df = pd.DataFrame({
                "Element": formulas,
                "DFT": ref_vals,
                model_name: pred_vals,
                "Δ": abs_diff.round(3),
                "Δ/%": pct_diff.round(2)
            }).round(3)

            mae = np.mean(np.abs(abs_diff))
            table_df.loc["MAE"] = ["", "", "", round(mae, 3), ""]

            # Save diff table
            table_df.to_csv(model_dir / "full_table.csv", index=False)

            fig = LatticeConstant.create_scatter_plot(ref_vals, pred_vals, model_name, mae, formulas)
            fig.write_image(model_dir / "lattice_constants.png", width=800, height=600)

            mae_summary.append({"Model": model_name, "MAE (Å)": round(mae, 3)})

        pd.DataFrame(mae_summary).to_csv(save_dir / "mae_summary.csv", index=False)
    

    @staticmethod
    def create_scatter_plot(ref_vals, pred_vals, model_name, mae, hover=None):
        """Create a lattice constant scatter plot comparing ref vs predicted."""
        combined_min = min(min(ref_vals), min(pred_vals))
        combined_max = max(max(ref_vals), max(pred_vals))

        fig = px.scatter(
            x=ref_vals,
            y=pred_vals,
            hover_name=hover,
            labels={
                "x": "Reference Lattice Constant (Å)",
                "y": f"{model_name} Prediction (Å)",
            },
            title=f"{model_name} - Lattice Constant Prediction"
        )

        fig.add_shape(
            type="line",
            x0=combined_min, y0=combined_min,
            x1=combined_max, y1=combined_max,
            xref='x', yref='y',
            line=dict(color="black", dash="dash")
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="black",
            xaxis=dict(showgrid=True, gridcolor="lightgray", scaleanchor="y", scaleratio=1),
            yaxis=dict(showgrid=True, gridcolor="lightgray"),
        )

        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text=f"MAE (Å): {mae:.3f}",
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
    def register_callbacks(app, mae_df, lat_const_df):
        
        @app.callback(
            Output("lattice-const-table", "children"),
            Output("lattice-table-last-clicked", "data"),
            Input("lat-mae-score-table", "active_cell"),
            State("lat-mae-score-table", "data"),
            State("lattice-table-last-clicked", "data")
        )
        def update_lattice_const_plot(active_cell, table_data, last_clicked):
            if active_cell is None:
                raise PreventUpdate

            row = active_cell["row"]
            model_name = table_data[row]["Model"]

            # Toggle behavior: if the same model is clicked again, collapse
            if last_clicked == model_name:
                return html.Div(), None  # Collapse

            # Else, render the plot + table
            mae_val = (lat_const_df[model_name] - lat_const_df["ref"]).abs().mean()

            ref_vals = lat_const_df["ref"]
            pred_vals = lat_const_df[model_name]
            formulas = lat_const_df.index.tolist()

            fig = LatticeConstant.create_scatter_plot(ref_vals, pred_vals, model_name, mae_val, formulas)

            abs_diff = pred_vals - ref_vals
            pct_diff = 100 * abs_diff / ref_vals

            table_df = pd.DataFrame({
                "Element": formulas,
                "DFT": ref_vals,
                model_name: pred_vals,
                "Δ": abs_diff.round(3),
                "Δ/%": pct_diff.round(2)
            }).round(3)

            table_df.loc["MAE"] = ["", "", "", round(mae_val, 3), ""]

            summary_table = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in table_df.columns],
                data=table_df.reset_index(drop=True).to_dict('records'),
                style_cell={'textAlign': 'center', 'fontSize': '14px'},
                style_header={'fontWeight': 'bold'},
                style_table={'overflowX': 'auto'},
            )

            return html.Div([
                dcc.Tabs([
                    dcc.Tab(label="Scatter Plot", children=[dcc.Graph(figure=fig)]),
                    dcc.Tab(label="Δ Table", children=[html.Div(summary_table, style={"padding": "20px"})])
                ])
            ]), model_name