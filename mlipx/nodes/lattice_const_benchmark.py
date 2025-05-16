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
        
        if structure.info['lattice_type'] == '2H-SiC' or structure.info['lattice_type'] == 'hcp':
            lattice_const_a = structure.cell.lengths()[0]
            lattice_const_c = structure.cell.lengths()[2]
            lattice_const = {"a": lattice_const_a, "c": lattice_const_c}
        
        else:
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
        normalise_to_model: t.Optional[str] = None,
    ):

        ref_dict = ref_node.get_ref
        
        
        full_data = {}
        for formula, model_data in node_dict.items():
            if formula == "SiC":
                formula_a = "SiC(a)"
                formula_c = "SiC(c)"
                full_data[formula_a] = {"ref": ref_dict[formula_a]}
                full_data[formula_c] = {"ref": ref_dict[formula_c]}
            else:
                full_data[formula] = {"ref": ref_dict[formula]}

            for model_name, node in model_data.items():
                if isinstance(node.get_lattice_const, dict):
                    # for SiC
                    full_data["SiC(a)"][model_name] = node.get_lattice_const["a"]
                    full_data["SiC(c)"][model_name] = node.get_lattice_const["c"]
                else:
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
            mae_data.append({"Model": model_name, "Lat Const MAE [Å]": round(mae, 3)})

        mae_df = pd.DataFrame(mae_data)
        
        if normalise_to_model is not None:
            for model in mae_df["Model"]:
                mae_df.loc[mae_df["Model"] == model, "Lat Const Score \u2193"] = mae_df.loc[mae_df["Model"] == model, "Lat Const MAE [Å]"] / mae_df.loc[mae_df["Model"] == normalise_to_model, "Lat Const MAE [Å]"].values[0]
        
        else:
            mae_df["Lat Const Score \u2193"] = mae_df["Lat Const MAE [Å]"]

        mae_df = mae_df.round(3)
        mae_df['Rank'] = mae_df['Lat Const Score \u2193'].rank(method='min', ascending=True).astype(int)


        # save
        LatticeConstant.save_lattice_const_plots_tables(lat_const_df)
        
        # report
        models_list = [col for col in lat_const_df.columns if col != "ref"]
        md_path = LatticeConstant.generate_lattice_const_report(
            mae_df=mae_df,
            models_list=models_list,
            markdown_path="benchmark_stats/bulk_crystal_benchmark/lattice_constants/lattice_const_report.md",
            lat_const_df=lat_const_df,
        )

        

        if ui is None and run_interactive:
            return lat_const_df, mae_df


        # === 5. Dash App ===
        app = dash.Dash(__name__)
        

        from mlipx.dash_utils import dash_table_interactive
        app.layout = dash_table_interactive(
            df=mae_df,
            id="lat-mae-score-table",
            title="Lattice Constants MAE Summary Table",
            extra_components=[
                html.Div(id="lattice-const-table"),
                dcc.Store(id="lattice-table-last-clicked", data=None),
            ],
        )
        

        LatticeConstant.register_callbacks(app, mae_df, lat_const_df)
        


        from mlipx.dash_utils import run_app

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
        md.append("## Lattice Constants MAE Table\n")
        md.append(mae_df.to_markdown(index=False))
        md.append("\n")

        def add_image_rows(md_lines, image_paths, n_cols=1):
            for i in range(0, len(image_paths), n_cols):
                image_set = image_paths[i:i+n_cols]
                width = 100 // n_cols
                line = " ".join(f"![]({img.resolve()}){{ width={width}% }}" for img in image_set)
                md_lines.append(line + "\n")

        # Per-model diff Tables and Scatter Plots
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
                "DFT (Å)": ref_vals,
                f"{model} (Å)": pred_vals,
                "Δ": abs_diff.round(3),
                "Δ/%": pct_diff.round(2)
            }).round(3)


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
    def save_lattice_const_plots_tables(lat_const_df, save_dir="benchmark_stats/bulk_crystal_benchmark/lattice_constants"):
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

            # Save diff table
            table_df.to_csv(model_dir / "full_table.csv", index=False)

            from mlipx.dash_utils import create_scatter_plot
            fig = create_scatter_plot(
                ref_vals = ref_vals, 
                pred_vals = pred_vals, 
                model_name = model_name, 
                mae = mae, 
                metric_label = ("Lattice Constant", "Å"),
                hover_data = (formulas, "Formula"),
            )

            fig.write_image(model_dir / "lattice_constants.png", width=800, height=600)

            mae_summary.append({"Model": model_name, "MAE (Å)": round(mae, 3)})

        pd.DataFrame(mae_summary).to_csv(save_dir / "mae_summary.csv", index=False)
    


    



    @staticmethod
    def register_callbacks(app, mae_df, lat_const_df):
        
        # decorator tells dash the function below is a callback
        @app.callback(
            Output("lattice-const-table", "children"),
            Output("lattice-table-last-clicked", "data"),
            Input("lat-mae-score-table", "active_cell"),
            State("lattice-table-last-clicked", "data")
        )
        def update_lattice_const_plot(active_cell, last_clicked):
            if active_cell is None:
                raise PreventUpdate

            row = active_cell["row"]
            col = active_cell["column_id"]
            model_name = mae_df.loc[row, "Model"]
            # vital for closing plot
            if col not in mae_df.columns or col == "Model":
                return None, active_cell

            # Toggle behavior: if the same model is clicked again, collapse
            if last_clicked is not None and (
                active_cell["row"] == last_clicked.get("row") and
                active_cell["column_id"] == last_clicked.get("column_id")
            ):
                return None, None

            # Else, render the plot + table
            mae_val = (lat_const_df[model_name] - lat_const_df["ref"]).abs().mean()

            ref_vals = lat_const_df["ref"]
            pred_vals = lat_const_df[model_name]
            formulas = lat_const_df.index.tolist()

            from mlipx.dash_utils import create_scatter_plot
            fig = create_scatter_plot(
                ref_vals = ref_vals, 
                pred_vals = pred_vals, 
                model_name = model_name, 
                mae = mae_val, 
                metric_label = ("Lattice Constant", "Å"),
                hover_data = (formulas, "Formula"),
            )
                
                
            abs_diff = pred_vals - ref_vals
            pct_diff = 100 * abs_diff / ref_vals

            table_df = pd.DataFrame({
                "Element": formulas,
                "DFT (Å)": ref_vals,
                f"{model_name} (Å)": pred_vals,
                "Δ": abs_diff.round(3),
                "Δ/%": pct_diff.round(2)
            }).round(3)


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
            ]), active_cell