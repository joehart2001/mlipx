import io
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
from typing import List, Dict, Any, Optional
import cctk
from ase.io.trajectory import Trajectory
from plotly.io import write_image
from ase.io import read
import mlipx
from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator
import time

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
from matcalc.benchmark import ElasticityBenchmark
from mlipx.dash_utils import dash_table_interactive
from mlipx.dash_utils import run_app


class Elasticity(zntrack.Node):
    """Bulk and shear moduli benchmark model against all available MP data.
    """
    # inputs
    #dataset_path: pathlib.Path = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    n_jobs: int = zntrack.params(-1)  # -1 for all available cores
    
    norm_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.1, -0.05, 0.05, 0.1))
    shear_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.02, -0.01, 0.01, 0.02))
    # norm_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.01, -0.005, 0.005, 0.01)) # mp0 and uma values
    # shear_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.06, -0.03, 0.03, 0.06))
    relax_structure: bool = zntrack.params(True)
    relax_deformed_structures: bool = zntrack.params(False)
    use_checkpoint: bool = zntrack.params(True)
    n_materials: int = zntrack.params(10)
    fmax: float = zntrack.params(0.05)

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    results_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "moduli_results.csv")
    mae_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae.csv")
    



    def run(self):
        
        calc = self.model.get_calculator()
        
        
        # from matcalc
        benchmark = ElasticityBenchmark(n_samples=self.n_materials, seed=2025, 
                                        fmax=self.fmax, 
                                        relax_structure=self.relax_structure,
                                        relax_deformed_structures=self.relax_deformed_structures,
                                        norm_strains = self.norm_strains,
                                        shear_strains = self.shear_strains,
                                        benchmark_name = "mp-pbe-elasticity-2025.3.json.gz"
        )
        
        
        print(self.model_name)
        
        if self.use_checkpoint:
            checkpoint_path = "elasticity_benchmark_checkpoint.json"
        
        else:
            checkpoint_path = None
        
        start_time = time.perf_counter()
        results = benchmark.run(calc, self.model_name, n_jobs=self.n_jobs,
                                checkpoint_file=checkpoint_path,
                                checkpoint_freq=100,
                                delete_checkpoint_on_finish=False)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.2f} seconds")
        
        results.to_csv(self.results_path, index=False)
        
        mae_df = pd.DataFrame()
        mae_K = np.abs(results[f'K_vrh_{self.model_name}'].values - results['K_vrh_DFT'].values).mean()
        mae_G = np.abs(results[f'G_vrh_{self.model_name}'].values - results['G_vrh_DFT'].values).mean()
        mae_df.loc[self.model_name, 'K_bulk [GPa]'] = mae_K
        mae_df.loc[self.model_name, 'K_shear [GPa]'] = mae_G
        mae_df = mae_df.reset_index().rename(columns={'index': 'Model'})
        mae_df.to_csv(self.mae_path, index=False)        
        
        
        
        
    @property
    def results(self) -> pd.DataFrame:
        """Load the results from the benchmark
        """
        results = pd.read_csv(self.results_path)
        return results
    
    
    
    
    
    @staticmethod
    def mae_plot_interactive(
        node_dict,
        ui=None,
        run_interactive=True,
        report=False,
        normalise_to_model: t.Optional[str] = None,
    ):
        """Interactive MAE table -> scatter plot for bulk and shear moduli for each model
        """
        benchmarks = [
            'K_vrh',
            'G_vrh',
        ]
        benchmark_units = {
            'K_vrh': '[GPa]',
            'G_vrh': '[GPa]',
        }
        benchmark_labels = {
            'K_vrh': 'K_bulk',
            'G_vrh': 'K_shear',
        }
        label_to_key = {v: k for k, v in benchmark_labels.items()}

        mae_df = pd.DataFrame()
        for model in node_dict.keys():
            results_df = node_dict[model].results
            full_count = len(results_df)
            mask_K = (results_df[f'K_vrh_{model}'] > -50) & (results_df[f'K_vrh_{model}'] < 600)
            mask_G = (results_df[f'G_vrh_{model}'] > -50) & (results_df[f'G_vrh_{model}'] < 600)
            valid_mask = mask_K & mask_G
            results_df = results_df[valid_mask].copy()
            #print(f"Model: {model}, Valid entries: {len(results_df)}")
            mae_K = np.abs(results_df[f'K_vrh_{model}'].values - results_df['K_vrh_DFT'].values).mean()
            mae_G = np.abs(results_df[f'G_vrh_{model}'].values - results_df['G_vrh_DFT'].values).mean()
            mae_df.loc[model, 'K_bulk [GPa]'] = mae_K
            mae_df.loc[model, 'K_shear [GPa]'] = mae_G
            mae_df.loc[model, "excluded count"] = full_count - len(results_df)
        mae_df = mae_df.reset_index().rename(columns={'index': 'Model'})
        mae_df = mae_df
        mae_cols = [col for col in mae_df.columns if col not in ['Model']]
        if normalise_to_model is not None:
            for model in node_dict.keys():
                score = 0
                for col in mae_cols:
                    if col == 'excluded count':
                        continue
                    else:
                        score += mae_df.loc[mae_df['Model'] == model, col].values[0] / mae_df.loc[mae_df['Model'] == normalise_to_model, col].values[0]
                mae_df.loc[mae_df['Model'] == model, 'Elasticity Score \u2193'] = score / (len(mae_cols) - 1)  # -1 to exclude 'excluded count'
        else:
            mae_df['Elasticity Score \u2193'] = mae_df[mae_cols].mean(axis=1)
        mae_df = mae_df.round(3)
        mae_df['Rank'] = mae_df['Elasticity Score \u2193'].rank(method='min', ascending=True).astype(int)

        Elasticity.save_scatter_plots_stats(
            node_dict=node_dict,
            mae_df=mae_df,
            save_path=f"benchmark_stats/bulk_crystal_benchmark/elasticity/"
        )
        models_list = list(node_dict.keys())
        if report:
            md_path = Elasticity.generate_elasticity_report(
                mae_df=mae_df,
                models_list=models_list,
            )
        else:
            md_path = None
        if ui is None and run_interactive:
            return

        # Dash app
        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Use the class method to build the layout
        app.layout = Elasticity.build_layout(mae_df)

        results_dict = {model: node.results for model, node in node_dict.items()}
        if run_interactive:
            Elasticity.register_callbacks(app, mae_df, results_dict)

        from mlipx.dash_utils import run_app
        if not run_interactive:
            return app, mae_df, md_path, results_dict
        return run_app(app, ui=ui)


    @staticmethod
    def build_layout(mae_df):
        """Return the Dash layout for the elasticity MAE dashboard."""
        return html.Div([
            dash_table_interactive(
                df=mae_df,
                id='elas-mae-table',
                title="Bulk and Shear Moduli MAEs",
                extra_components=[
                    dcc.Store(id="stored-cell-points"),
                    dcc.Store(id="stored-results-df"),
                    dcc.Store(id='elas-mae-table-last-clicked'),
                    html.Div(id='scatter-plot-container'),
                    dash_table.DataTable(id="material-table"),
                ],
                tooltip_header={
                    "Model": "Name of the MLIP model",
                    "K_bulk [GPa]": "Mean Absolute Error of Voigt-Reuss-Hill (VRH) average bulk modulus",
                    "K_shear [GPa]": "Mean Absolute Error of Voigt-Reuss-Hill (VRH) average shear modulus",
                    "excluded count": "Number of materials excluded in the MAE calculations due to K > 600 GPa or K < -50 GPa",
                    "Elasticity Score ↓": "Average MAE (lower is better)",
                    "Rank": "Ranking based on Elasticity Score (1 = best)",
                }
            )
        ],
        style={
            'backgroundColor': 'white',
        })
                






 # -------------- helper functions ----------------
    


    def generate_elasticity_report(
        mae_df,
        models_list,
    ):
        """Generates a markdown and pdf report contraining the MAE summary table, scatter plots and phonon dispersions
        """
        markdown_path = Path("benchmark_stats/bulk_crystal_benchmark/elasticity/elasticity_benchmark_report.md")
        pdf_path = markdown_path.with_suffix(".pdf")

        md = []

        md.append("# Elasticity Report\n")

        # MAE Summary table
        md.append("## Bulk and Shear Moduli MAE Table\n")
        md.append(mae_df.to_markdown(index=False))
        md.append("\n")
        
        # function for adding images to the markdown 
        def add_image_rows(md_lines, image_paths, n_cols = 2):
            """Append n images per row"""
            for i in range(0, len(image_paths), n_cols):
                image_set = image_paths[i:i+n_cols]
                width = 100 // n_cols
                line = " ".join(f"![]({img.resolve()}){{ width={width}% }}" for img in image_set)
                md_lines.append(line + "\n")


        md.append("## Scatter and density Plots\n")
        for model in models_list:
            md.append(f"### {model}\n")
            scatter_plot_dir = Path(f"benchmark_stats/bulk_crystal_benchmark/elasticity/{model}/scatter_plots")
            images = sorted(scatter_plot_dir.glob("*.png"))
            add_image_rows(md, images)
            
        markdown_path.write_text("\n".join(md))

        print(f"Markdown report saved to: {markdown_path}")

        try:
            subprocess.run(
                ["pandoc", str(markdown_path), "-o", str(pdf_path), "--pdf-engine=xelatex", "--variable=geometry:top=1.5cm,bottom=2cm,left=1cm,right=1cm"],
                check=True
            )
            print(f"PDF report saved to {pdf_path}")

        except subprocess.CalledProcessError as e:
            print(f"PDF generation failed: {e}")
        
        
        return markdown_path

            
            

    @staticmethod
    def save_scatter_plots_stats(
        node_dict: Dict[str, zntrack.Node],
        mae_df: pd.DataFrame,
        save_path: str = "benchmark_stats/bulk_crystal_benchmark/elasticity/"
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        mae_df.to_csv(f"{save_path}/mae_elasticity.csv", index=False)
        mae_df = mae_df.set_index('Model')

        for model, node in node_dict.items():
            path = Path(f"{save_path}/{model}/")
            (path / "scatter_plots").mkdir(parents=True, exist_ok=True)
            
            results_df = node.results
            results_df.to_csv(f"{path}/results.csv", index=False)

            for col in mae_df.columns:
                if "K_bulk" in col:
                    prop, label = "K_vrh", "K_bulk"
                elif "K_shear" in col:
                    prop, label = "G_vrh", "K_shear"
                else:
                    continue

                mae = mae_df.loc[model, col]
                x = results_df[f'{prop}_DFT']
                y = results_df[f'{prop}_{model}']

                # Standard scatter
                fig_std = px.scatter(
                    results_df, x=f'{prop}_DFT', y=f'{prop}_{model}',
                    labels={f'{prop}_DFT': f'{label} DFT [GPa]', f'{prop}_{model}': f'{label} Predicted [GPa]'},
                    title=f'{label} Scatter Plot - {model}'
                )
                combined_min, combined_max = min(x.min(), y.min()), max(x.max(), y.max())
                fig_std.add_shape(type='line', x0=combined_min, y0=combined_min, x1=combined_max, y1=combined_max,
                                  line=dict(dash='dash'))
                fig_std.add_annotation(
                    xref="paper", yref="paper", x=0.02, y=0.98,
                    text=f"{label} MAE: {mae} GPa",
                    showarrow=False, font=dict(size=14, color="black"),
                    bordercolor="black", borderwidth=1,
                    borderpad=4, bgcolor="white", opacity=0.8
                )
                fig_std.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(size=16, color="black"),
                    xaxis=dict(gridcolor='lightgray'), yaxis=dict(gridcolor='lightgray')
                )
                fig_std.write_image(f"{path}/scatter_plots/{label}_scatter.png", width=800, height=600)

                # Heatmap-style scatter
                from mlipx.plotting_utils import generate_density_scatter_figure
                fig_heatmap, *_ = generate_density_scatter_figure(x, y, label, model, mae)
                fig_heatmap.write_image(f"{path}/scatter_plots/{label}_density.png", width=800, height=600)

    @staticmethod
    def register_callbacks(app, mae_df, results_dict):

        @app.callback(
            Output('scatter-plot-container', 'children'),
            Output('stored-cell-points', 'data'),
            Output('stored-results-df', 'data'),
            Output('elas-mae-table-last-clicked', 'data'),
            Output('material-table', 'data', allow_duplicate=True),
            Output('material-table', 'columns', allow_duplicate=True),
            Input('elas-mae-table', 'active_cell'),
            State('elas-mae-table-last-clicked', 'data'),
            prevent_initial_call=True
        )
        def update_scatter_plot(active_cell, last_clicked):
            if active_cell is None:
                raise PreventUpdate
            if last_clicked is not None and (
                active_cell["row"] == last_clicked.get("row") and
                active_cell["column_id"] == last_clicked.get("column_id")
            ):
                return None, None, None, None, None, None

            row = active_cell['row']
            col = active_cell['column_id']
            model = mae_df.loc[row, 'Model']

            if col not in mae_df.columns or col == 'Model':
                return None, None, None, active_cell, None, None

            if "K_bulk" in col:
                prop, label = "K_vrh", "K_bulk"
            elif "K_shear" in col:
                prop, label = "G_vrh", "K_shear"
            else:
                return None, None, None, active_cell, None, None

            df = results_dict[model]
            
            # rm outliers
            mask_K = (df[f'K_vrh_{model}'] > -50) & (df[f'K_vrh_{model}'] < 600)
            mask_G = (df[f'G_vrh_{model}'] > -50) & (df[f'G_vrh_{model}'] < 600)
            valid_mask = mask_K & mask_G
            df = df[valid_mask].copy()
            
            x = df[f'{prop}_DFT']
            y = df[f'{prop}_{model}']
            mae = mae_df.loc[row, col]

            from mlipx.plotting_utils import generate_density_scatter_figure
            fig, cell_points, *_ = generate_density_scatter_figure(x, y, label, model, mae)

            # Serialize for Dash
            serialized = {f"{cx}_{cy}": indices for (cx, cy), indices in cell_points.items()}

            graph = html.Div([
                html.P(
                    "Info: Click on a point to view the underlying materials.",
                    style={"fontSize": "14px", "color": "#555"}
                ),
                dcc.Graph(
                    id={'type': 'scatter-plot', 'index': 'main'},
                    figure=fig,
                    style={"height": "60vh"}
                )
            ])

            # when returning a new plot, clear the material table until clickData occurs
            return graph, json.dumps(serialized), df.to_json(orient='split'), active_cell, None, None

        # update the material table based on clickData from the scatter plot
        @app.callback(
            Output('material-table', 'data'),
            Output('material-table', 'columns'),
            Input({'type': 'scatter-plot', 'index': 'main'}, 'clickData'),
            State('stored-cell-points', 'data'),
            State('stored-results-df', 'data')
        )
        def update_material_table(clickData, cell_points_json, results_json):
            if clickData is None:
                raise PreventUpdate

            cx, cy = clickData['points'][0]['customdata']
            cell_key = f"{cx}_{cy}"
            cell_points = json.loads(cell_points_json)
            indices = cell_points.get(cell_key, [])

            df = pd.read_json(io.StringIO(results_json), orient='split').round(3)
            subset = df.iloc[indices]

            return subset.to_dict('records'), [{"name": col, "id": col} for col in subset.columns]


    

    @staticmethod
    def benchmark_precompute(
        node_dict,
        cache_dir: str = "app_cache/bulk_crystal_benchmark/elasticity_cache",
        ui=None,
        run_interactive: bool = False,
        report: bool = False,
        normalise_to_model: t.Optional[str] = None,
    ):
        """
        Precompute all data for elasticity benchmarking and save to cache_dir.
        """
        os.makedirs(cache_dir, exist_ok=True)
        app, mae_df, md_path, results_dict = Elasticity.mae_plot_interactive(
            node_dict=node_dict,
            ui=ui,
            run_interactive=False,
            report=report,
            normalise_to_model=normalise_to_model,
        )
        mae_df.to_pickle(f"{cache_dir}/mae_summary.pkl")
        with open(f"{cache_dir}/results_dict.pkl", "wb") as f: # cant use to_pickle on plain dict
            pickle.dump(results_dict, f)
    
        # for model, df in results_dict.items():
        #     df.to_csv(f"{cache_dir}/{model}_results.csv", index=False)
            
        return

    # @staticmethod
    # def build_layout(mae_df):
    #     return html.Div([
    #         dash_table_interactive(
    #             df=mae_df,
    #             id='elas-mae-table',
    #             title="Bulk and Shear Moduli MAEs",
    #             extra_components=[
    #                 dcc.Store(id="stored-cell-points"),
    #                 dcc.Store(id="stored-results-df"),
    #                 dcc.Store(id='elas-mae-table-last-clicked'),
    #                 html.Div(id='scatter-plot-container'),
    #                 dash_table.DataTable(id="material-table"),
    #             ]
    #         )
    #     ],
    #     style={'backgroundColor': 'white'})


    @staticmethod
    def launch_dashboard(cache_dir: str = "app_cache/bulk_crystal_benchmark/elasticity_cache", ui=None):
        mae_df = pd.read_pickle(f"{cache_dir}/mae_summary.pkl")
        with open(f"{cache_dir}/results_dict.pkl", "rb") as f:
            results_dict = pickle.load(f)
        #results_dict = pd.read_pickle(f"{cache_dir}/results_dict.pkl")
        # results_dict = {
        #     row["Model"]: pd.read_csv(f"{cache_dir}/{row['Model']}_results.csv")
        #     for _, row in mae_df.iterrows()
        # }
        app = dash.Dash(__name__)
        app.layout = Elasticity.build_layout(mae_df)
        Elasticity.register_callbacks(app, mae_df, results_dict)
        return run_app(app, ui=ui)