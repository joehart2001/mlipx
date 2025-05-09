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



class Elasticity(zntrack.Node):
    """Bulk and shear moduli benchmark model against all available MP data.
    """
    # inputs
    #dataset_path: pathlib.Path = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    
    norm_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.1, -0.05, 0.05, 0.1))
    shear_strains: t.Tuple[float, float, float, float] = zntrack.params((-0.02, -0.01, 0.01, 0.02))
    relax_structure: bool = zntrack.params(True)
    n_materials: int = zntrack.params(10)
    fmax: float = zntrack.params(0.05)

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    results_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "moduli_results.csv")
    mae_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae.csv")
    



    def run(self):
        
        calc = self.model.get_calculator()
        
        # with open(self.dataset_path, "r") as f:
        #     ref_data = json.load(f)
        
        
        # from matcalc
        benchmark = ElasticityBenchmark(n_samples=self.n_materials, seed=2025, 
                                        fmax=self.fmax, 
                                        relax_structure=self.relax_structure,
                                        norm_strains = self.norm_strains,
                                        shear_strains = self.shear_strains,
        )
        
        
        print(self.model_name)
        results = benchmark.run(calc, self.model_name)
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
    def mae_plot_interactive(node_dict, ui = None, run_interactive = True):
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
            mae_K = np.abs(results_df[f'K_vrh_{model}'].values - results_df['K_vrh_DFT'].values).mean()
            mae_G = np.abs(results_df[f'G_vrh_{model}'].values - results_df['G_vrh_DFT'].values).mean()
            mae_df.loc[model, 'K_bulk [GPa]'] = mae_K
            mae_df.loc[model, 'K_shear [GPa]'] = mae_G

        mae_df = mae_df.reset_index().rename(columns={'index': 'Model'})
        mae_df = mae_df.round(3)
        

        Elasticity.save_scatter_plots_stats(
            node_dict=node_dict,
            mae_df=mae_df,
            save_path=f"benchmark_stats/elasticity/"
        )
        
        models_list = list(node_dict.keys())
        md_path = Elasticity.generate_elasticity_report(
            mae_df=mae_df,
            models_list=models_list,
            
        )
        
        if ui is None and run_interactive:
            return
        

        # Dash app
        app = dash.Dash(__name__, suppress_callback_exceptions=True)


        app.layout = html.Div([
            html.H2("Bulk and Shear Moduli MAEs", style={'color': 'Black', 'padding': '1rem'}),
            dash_table.DataTable(
                id='elas-mae-table',
                columns=[{"name": col, "id": col} for col in mae_df.columns],
                data=mae_df.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'},
            ),
            dcc.Store(id="stored-cell-points"),
            dcc.Store(id="stored-results-df"),
            dcc.Store(id='elas-mae-table-last-clicked'),
            
            html.Div(id='scatter-plot-container'),
            
            dash_table.DataTable(id="material-table"),
        ],
            style={
                'backgroundColor': 'white',
            }
        )
        
        
        # made into a function so can be called from outside (in the bulk crystal benchmark)
        Elasticity.register_callbacks(app, mae_df, node_dict)



        from mlipx.dash_utils import run_app

        if not run_interactive:
            return app, mae_df, md_path

        return run_app(app, ui=ui)
                






 # -------------- helper functions ----------------
    


    def generate_elasticity_report(
        mae_df,
        models_list,
    ):
        """Generates a markdown and pdf report contraining the MAE summary table, scatter plots and phonon dispersions
        """
        markdown_path = Path("benchmark_stats/elasticity/elasticity_benchmark_report.md")
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


        # Scatter Plots
        md.append("## Scatter and density Plots\n")
        for model in models_list:
            md.append(f"### {model}\n")
            scatter_plot_dir = Path(f"benchmark_stats/elasticity/{model}/scatter_plots")
            images = sorted(scatter_plot_dir.glob("*.png"))
            add_image_rows(md, images)
            
        # Save Markdown file
        markdown_path.write_text("\n".join(md))

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

            
            

    @staticmethod
    def generate_density_scatter_figure(x, y, label, model, mae=None, max_points=500):
        x, y = np.array(x), np.array(y)
        x_min, x_max = x.min() - 0.05, x.max() + 0.05 # add a small margin otherwise the egde points are cut off
        y_min, y_max = y.min() - 0.05, y.max() + 0.05
        num_cells = 50
        cell_w = (x_max - x_min) / num_cells
        cell_h = (y_max - y_min) / num_cells

        cell_counts = np.zeros((num_cells, num_cells))
        cell_points = { (i, j): [] for i in range(num_cells) for j in range(num_cells) }

        for idx, (xi, yi) in enumerate(zip(x, y)):
            cx = int((xi - x_min) // cell_w)
            cy = int((yi - y_min) // cell_h)
            if 0 <= cx < num_cells and 0 <= cy < num_cells:
                cell_counts[cx, cy] += 1
                cell_points[(cx, cy)].append(idx)  # store index, not coordinates

        plot_x, plot_y, plot_colors = [], [], []
        point_cells = []

        # plotting points logic
        # cell has more than 5 points: randomly select one
        # cell has less than 5 points: plot all points
        # helps for large datasets but keeps outliers visiable
        for (cx, cy), indices in cell_points.items():
            if len(indices) > 5:
                indices = [np.random.choice(indices)]
                if len(plot_x) < max_points:
                    plot_x.append(x[idx])
                    plot_y.append(y[idx])
                    plot_colors.append(cell_counts[cx, cy])
                    point_cells.append((cx, cy))
            else:
                indices = indices
            for idx in indices:
                plot_x.append(x[idx])
                plot_y.append(y[idx])
                plot_colors.append(cell_counts[cx, cy])
                point_cells.append((cx, cy))

        fig = go.Figure(go.Scatter(
            x=plot_x, y=plot_y, mode='markers',
            marker=dict(color=plot_colors, size=5, colorscale='Viridis', showscale=True,
                        colorbar=dict(title="Density")),
            text=[f"Density: {int(c)}" for c in plot_colors],
            customdata=point_cells
        ))

        fig.update_layout(
            title=f'{label} Scatter Plot - {model}',
            xaxis_title=f'{label} DFT [GPa]',
            yaxis_title=f'{label} Predicted [GPa]',
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(size=16, color='black'),
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True),
            hovermode='closest'
        )

        combined_min, combined_max = min(x_min, y_min), max(x_max, y_max)
        fig.add_shape(type='line', x0=combined_min, y0=combined_min, x1=combined_max, y1=combined_max,
                      line=dict(dash='dash'))

        if mae is not None:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.98,
                text=f"{label} MAE: {mae} GPa",
                showarrow=False,
                font=dict(size=14, color="black"),
                bordercolor="black", borderwidth=1,
                borderpad=4, bgcolor="white", opacity=0.8
            )

        return fig, cell_points, cell_w, cell_h, x_min, y_min

    @staticmethod
    def save_scatter_plots_stats(
        node_dict: Dict[str, zntrack.Node],
        mae_df: pd.DataFrame,
        save_path: str = "benchmark_stats/elasticity/"
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
                fig_heatmap, *_ = Elasticity.generate_density_scatter_figure(x, y, label, model, mae)
                fig_heatmap.write_image(f"{path}/scatter_plots/{label}_density.png", width=800, height=600)

    @staticmethod
    def register_callbacks(app, mae_df, node_dict):

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

            df = node_dict[model].results
            x = df[f'{prop}_DFT']
            y = df[f'{prop}_{model}']
            mae = mae_df.loc[row, col]

            fig, cell_points, *_ = Elasticity.generate_density_scatter_figure(x, y, label, model, mae)

            # Serialize for Dash
            serialized = {f"{cx}_{cy}": indices for (cx, cy), indices in cell_points.items()}

            graph = dcc.Graph(
                id={'type': 'scatter-plot', 'index': 'main'},
                figure=fig,
                style={"height": "60vh"}
            )

            # When returning a new plot, clear the material table until clickData occurs
            return graph, json.dumps(serialized), df.to_json(orient='split'), active_cell, None, None

        # New callback: update the material table based on clickData from the scatter plot
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


    
