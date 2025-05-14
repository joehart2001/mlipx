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
from mlipx import PhononDispersion, Elasticity, LatticeConstant



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
    lattice_const_list: List[LatticeConstant] = zntrack.deps()
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files
    #elasticity_dict: pathlib.Path = zntrack.outs_path(zntrack.nwd / "elasticity_dict.json")
    #phonon_ref_dict: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_ref_dict.json")
    #phonon_pred_dict: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_pred_dict.json")

    
    def run(self):
        pass
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    @staticmethod
    def benchmark_interactive(
        elasticity_data: List[Elasticity] | Dict[str, Elasticity],
        lattice_const_data: List[LatticeConstant] | Dict[str, Dict[str, LatticeConstant]],
        lattice_const_ref_node: LatticeConstant,
        phonon_ref_data: List[PhononDispersion] | Dict[str, PhononDispersion],
        phonon_pred_data: List[PhononDispersion] | Dict[str, Dict[str, PhononDispersion]],
        ui: str = "browser",
        full_benchmark: bool = False,
        report: bool = True,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all bulk crystal benchmarks
        """
        
        from mlipx.dash_utils import process_data

        # Lattice constant
        lattice_const_dict = process_data(
            lattice_const_data,
            key_extractor=lambda node: node.name.split("LatticeConst-")[1],
            value_extractor=lambda node: {node.name.split("_lattice-constant-pred")[0]: node}
        )            

        # elasticity
        elasticity_dict = process_data(
            elasticity_data,
            key_extractor=lambda node: node.name.split("_Elasticity")[0],
            value_extractor=lambda node: node
        )

        # phonons
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
        
        
        # get apps
        app_phonon, phonon_mae_df, phonon_scatter_to_dispersion_map, phonon_benchmarks_scatter_dict, phonon_md_path = PhononDispersion.benchmark_interactive(
            pred_node_dict=phonon_dict_pred,
            ref_node_dict=phonon_dict_ref,
            run_interactive=False,
        )
        
        app_elasticity, mae_df_elas, elas_md_path = mlipx.Elasticity.mae_plot_interactive(
            node_dict=elasticity_dict,
            run_interactive=False,
        )
        
        
        app_lattice_const, mae_df_lattice_const, lattice_const_dict_with_ref, lattice_const_md_path = LatticeConstant.mae_plot_interactive(
            node_dict=lattice_const_dict,
            ref_node = lattice_const_ref_node,
            run_interactive=False,
        )

        # Combine all MAE tables
        combined_mae_table = BulkCrystalBenchmark.combine_mae_tables(
            mae_df_elas,
            mae_df_lattice_const,
            phonon_mae_df,
        ) # TODO: use this in the benchmark score func for ease
    

        bulk_crystal_benchmark_score_df = BulkCrystalBenchmark.bulk_crystal_benchmark_score(phonon_mae_df, mae_df_elas, mae_df_lattice_const).round(3)
        bulk_crystal_benchmark_score_df = bulk_crystal_benchmark_score_df.sort_values(by='Avg MAE \u2193', ascending=True)
        bulk_crystal_benchmark_score_df = bulk_crystal_benchmark_score_df.reset_index(drop=True)
        bulk_crystal_benchmark_score_df['Rank'] = bulk_crystal_benchmark_score_df.index + 1
        
        if not os.path.exists("benchmark_stats/bulk_crystal_benchmark/"):
            os.makedirs("benchmark_stats/bulk_crystal_benchmark")
        bulk_crystal_benchmark_score_df.to_csv("benchmark_stats/bulk_crystal_benchmark/bulk_crystal_benchmark_score.csv", index=False)
        
        from mlipx.dash_utils import colour_table
        # Viridis-style colormap for Dash DataTable
        style_data_conditional = colour_table(bulk_crystal_benchmark_score_df, col_name="Avg MAE \u2193")
        
        
        md_path_list = [elas_md_path, lattice_const_md_path, phonon_md_path]
    
        if report:
            BulkCrystalBenchmark.generate_report(
                bulk_crystal_benchmark_score_df=bulk_crystal_benchmark_score_df,
                md_report_paths=md_path_list,
                markdown_path="benchmark_stats/bulk_crystal_benchmark/bulk_crystal_benchmark_report.md",
                combined_mae_table=combined_mae_table,
            )


        if ui is None and not full_benchmark:
            return
    
        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        from mlipx.dash_utils import combine_apps
        apps_list = [app_phonon, app_lattice_const, app_elasticity]
        layout = combine_apps(
            benchmark_score_df=bulk_crystal_benchmark_score_df,
            benchmark_title="Bulk Crystal Benchmark",
            apps_list=apps_list,
            style_data_conditional=style_data_conditional,
        )
        #apps_list[0].layout = layout
        app.layout = layout
        
        # Register callbacks for each app
        PhononDispersion.register_callbacks(
            app, phonon_mae_df, phonon_scatter_to_dispersion_map, phonon_benchmarks_scatter_dict,
        )
        Elasticity.register_callbacks(
            app, mae_df_elas, elasticity_dict,
        )
        LatticeConstant.register_callbacks(
            app, mae_df_lattice_const, lattice_const_dict_with_ref,
        )
        
        
        from mlipx.dash_utils import run_app

        if full_benchmark:
            return app, bulk_crystal_benchmark_score_df, lambda app: (
                PhononDispersion.register_callbacks(app, phonon_mae_df, phonon_scatter_to_dispersion_map, phonon_benchmarks_scatter_dict),
                Elasticity.register_callbacks(app, mae_df_elas, elasticity_dict),
                LatticeConstant.register_callbacks(app, mae_df_lattice_const, lattice_const_dict_with_ref),
            )

        return run_app(app, ui=ui)
    
    
    
    
    
    
    







    # --------------------------------- Helper Functions ---------------------------------




    def bulk_crystal_benchmark_score(
        phonon_mae_df, 
        mae_df_elas, 
        mae_df_lattice_const,
    ):
        """ Currently avg mae
            (problem with other explored metrics: if we normalise by the max mae, then the models in this test are comparable to each other but not models run in a different test, as they will be normalised differently)
        """
        # Initialize scores
        maes = {}
        scores = {}
        
        # Calculate scores for each model
        model_list = phonon_mae_df['Model'].values.tolist()
        
        for model in model_list:
            scores[model] = 0
            
            for benchmark in mae_df_lattice_const.columns[1:]: # first col model
                mae = mae_df_lattice_const.loc[mae_df_lattice_const['Model'] == model, benchmark].values[0]
                scores[model] += mae     
                
            for benchmark in phonon_mae_df.columns[1:]: # first col model
                mae = phonon_mae_df.loc[phonon_mae_df['Model'] == model, benchmark].values[0]
                scores[model] += mae                    
                
            
            for benchmark in mae_df_elas.columns[1:]: # first col model
                mae = mae_df_elas.loc[mae_df_elas['Model'] == model, benchmark].values[0]
                scores[model] += mae
            
            scores[model] = scores[model] / (len(phonon_mae_df.columns[1:]) + len(mae_df_elas.columns[1:]) + len(mae_df_lattice_const.columns[1:]))
            
        return pd.DataFrame.from_dict(scores, orient='index', columns=['Avg MAE \u2193']).reset_index().rename(columns={'index': 'Model'})




    def combine_mae_tables(*mae_dfs):
        """ combine mae tables from different nodes for a summary table
        """
        combined_parts = []

        for df in mae_dfs:
            df = df.copy()
            df_cols = df.columns.tolist()
            if "Model" not in df_cols:
                raise ValueError("Each input dataframe must contain a 'Model' column.")
            other_cols = [col for col in df.columns if col != "Model"]
            df = df.set_index("Model")
            df.columns = other_cols
            combined_parts.append(df)

        combined = pd.concat(combined_parts, axis=1)

        combined.reset_index(inplace=True)
        return combined

    def generate_report(
        bulk_crystal_benchmark_score_df: pd.DataFrame,
        md_report_paths: List[str],
        markdown_path: str,
        combined_mae_table: pd.DataFrame,
    ):
        # TODO: colour tables
        # TODO: if number of phonon plots >  28 (whole page), then only show erronous ones 
        markdown_path = Path(markdown_path)
        pdf_path = markdown_path.with_suffix(".pdf")
        combined_md = []
        
        def latexify_column(col):
            import re
            if isinstance(col, str) and '_' in col:
                return re.sub(r'(\w+)_(\w+)', r'$\1_{\2}$', col)
            return col

        # colored_df = bulk_crystal_benchmark_score_df.copy()
        
        # from matplotlib import cm
        # viridis = cm.get_cmap("viridis_r")
        # vmin = colored_df["Avg MAE \u2193"].min()
        # vmax = colored_df["Avg MAE \u2193"].max()

        # def cellcolor(val):
        #     norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0
        #     r, g, b, _ = viridis(norm)
        #     r, g, b = int(r * 255), int(g * 255), int(b * 255)
        #     return f'<span style="background-color: rgb({r},{g},{b}); padding:2px;">{val:.3f}</span>'


        # # Apply LaTeX cellcolor to MAE column
        # colored_df["Avg MAE \u2193"] = colored_df["Avg MAE \u2193"].apply(cellcolor)

        # Add the required LaTeX package (included in the preamble by Pandoc)
        combined_md.append("## Benchmark Score Table \n")
        #combined_md.append("\\rowcolors{2}{gray!10}{white}\n")
        combined_md.append(bulk_crystal_benchmark_score_df.to_markdown(index=False))
        combined_md.append("\n")

        
        # Benchmark score Summary table
        # combined_md.append("# Bulk Crystal Benchmark Report\n")
        # combined_md.append("## Benchmark Score Table \n")
        # combined_md.append(bulk_crystal_benchmark_score_df.to_markdown(index=False))
        # combined_md.append("\n")
        
        combined_md.append("## Combined MAE Table \n")
        combined_mae_table.columns = [latexify_column(col) for col in combined_mae_table.columns]
        combined_md.append(combined_mae_table.to_markdown(index=False))
        combined_md.append('\n\\newpage\n\n')
        
        # summary page - extract MAE tables from all reports
        

       
        
        
        # rest of reports
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
                [
                    "pandoc",
                    str(markdown_path),
                    "-o", 
                    str(pdf_path),
                    "--pdf-engine=xelatex",
                    "--variable=geometry:top=1.5cm,bottom=2cm,left=1cm,right=1cm",
                    "--from", "markdown+raw_tex",  # Important for handling raw LaTeX
                ],
                check=True
            )
            print(f"PDF report saved to {pdf_path}")

        except subprocess.CalledProcessError as e:
            print(f"PDF generation failed: {e}")
        
        
        return markdown_path



    def generate_report1(
        bulk_crystal_benchmark_score_df: pd.DataFrame,
        md_report_paths: List[str],
        markdown_path: str,
        combined_mae_table: pd.DataFrame,
    ):
        markdown_path = Path(markdown_path)
        pdf_path = markdown_path.with_suffix(".pdf")
        combined_md = []

        def latexify_column(col):
            import re
            if isinstance(col, str) and '_' in col:
                return re.sub(r'(\w+)_(\w+)', r'$\1_{\2}$', col)
            return col

        # Add LaTeX preamble with required packages
        latex_preamble = """
            ---
            header-includes:
            - \\usepackage{colortbl}
            - \\usepackage{xcolor}
            - \\usepackage{booktabs}
            - \\usepackage{array}
            ---

            """        # Add LaTeX preamble to the beginning of the markdown
        combined_md.append(latex_preamble)
        
        # Create a copy of the DataFrame
        colored_df = bulk_crystal_benchmark_score_df.copy()
        mae_col = "Avg MAE \u2193"
        
        # Normalize MAE values for color mapping
        vmin = colored_df[mae_col].min()
        vmax = colored_df[mae_col].max()
        
        # Create a custom formatter function for the MAE column
        # This avoids direct LaTeX commands in table cells, which is problematic
        def format_mae(val):
            # Format to 3 decimal places
            return f"{val:.3f}"
        
        # Apply the formatting to the MAE column
        colored_df[mae_col] = colored_df[mae_col].apply(format_mae)
        
        # Add benchmark score table with markdown
        combined_md.append("## Benchmark Score Table\n")
        
        # Add custom LaTeX for the colored table
        combined_md.append("```{=latex}")
        combined_md.append("\\begin{table}[h]")
        combined_md.append("\\centering")
        combined_md.append("\\rowcolors{2}{gray!10}{white}")
        
        # Create column definitions
        cols = colored_df.columns
        col_defs = "l" * len(cols)  # Left-align all columns
        
        combined_md.append("\\begin{tabular}{" + col_defs + "}")
        combined_md.append("\\toprule")
        
        # Add header row
        combined_md.append(" & ".join([str(col) for col in cols]) + " \\\\")
        combined_md.append("\\midrule")
        
        # Add data rows with color for MAE column
        mae_idx = list(cols).index(mae_col)
        for _, row in colored_df.iterrows():
            row_vals = []
            for i, val in enumerate(row):
                if i == mae_idx:
                    # Calculate color based on the original value before formatting
                    orig_val = bulk_crystal_benchmark_score_df.loc[_, mae_col]
                    norm = (orig_val - vmin) / (vmax - vmin) if vmax != vmin else 0
                    intensity = int(norm * 100)
                    row_vals.append(f"\\cellcolor{{blue!{intensity}}} {val}")
                else:
                    row_vals.append(str(val))
            combined_md.append(" & ".join(row_vals) + " \\\\")
        
        combined_md.append("\\bottomrule")
        combined_md.append("\\end{tabular}")
        combined_md.append("\\end{table}")
        combined_md.append("```\n")
        
        # Combined MAE Table
        combined_md.append("## Combined MAE Table\n")
        combined_mae_table.columns = [latexify_column(col) for col in combined_mae_table.columns]
        combined_md.append(combined_mae_table.to_markdown(index=False))
        combined_md.append('\n\\newpage\n\n')
        
        # Add the rest of the reports
        for path in md_report_paths:
            path = Path(path)
            if not path.exists():
                print(f"Skipping {path} — file not found")
                continue
                
            with open(path, 'r') as f:
                md = f.read()
            combined_md.append(md)
            combined_md.append('\n\\newpage\n\n')
        
        Path(markdown_path).write_text("\n".join(combined_md))
        print(f"Markdown report saved to: {markdown_path}")
        
        # Generate PDF with Pandoc
        try:
            subprocess.run(
                [
                    "pandoc",
                    str(markdown_path),
                    "-o", 
                    str(pdf_path),
                    "--pdf-engine=pdflatex",
                    "--variable=geometry:top=1.5cm,bottom=2cm,left=1cm,right=1cm",
                    "--from", "markdown+raw_tex",  # Important for handling raw LaTeX
                ],
                check=True
            )
            print(f"PDF report saved to {pdf_path}")
        except subprocess.CalledProcessError as e:
            print(f"PDF generation failed: {e}")
            
        return markdown_path