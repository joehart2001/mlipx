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
    n_samples: int = zntrack.params(10)
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
        benchmark = ElasticityBenchmark(n_samples=self.n_samples, seed=2025, 
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
    def mae_plot_interactive(node_dict, ui = None, dont_run = False):
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
        
        # save stats
        # if not os.path.exists(f"benchmark_stats/elasticity/"):
        #     os.makedirs(f"benchmark_stats/elasticity/")
            
        #mae_df.to_csv('benchmark_stats/elasticity/mae_elasticity.csv', index=False)  # save stats
        
        Elasticity.save_scatter_plots_stats(
            node_dict=node_dict,
            mae_df=mae_df,
            save_path=f"benchmark_stats/elasticity/scatter_plots/"
        )
        
        if ui is None:
            if dont_run:
                pass
            else:
                return
        

        # Dash app
        app = dash.Dash(__name__)


        app.layout = html.Div([
            html.H2("Bulk and Shear Moduli MAEs", style={'color': 'Black', 'padding': '1rem'}),
            dash_table.DataTable(
                id='mae-table',
                columns=[{"name": col, "id": col} for col in mae_df.columns],
                data=mae_df.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'},
            ),
            dcc.Graph(id='scatter-plot'),
        ],
            style={
                'backgroundColor': 'white',
            }
        )
        
        
        # made into a function so can be called from outside (in the bulk crystal benchmark)
        Elasticity.register_elasticity_callbacks(app, mae_df, node_dict)

        

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
        
        if dont_run:
            return app, mae_df
        print(app)
        
        return run_app(app, ui=ui)




    @staticmethod
    def save_scatter_plots_stats(
        node_dict: Dict[str, zntrack.Node], 
        mae_df: pd.DataFrame, 
        save_path: str = "benchmark_stats/elasticity/scatter_plots/"
    ):
        """
        Generate and save scatter plots for each model and property.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        mae_df.to_csv(f"{save_path}/../mae_elasticity.csv", index=False)  # save stats

        mae_df = mae_df.set_index('Model')
        for model in node_dict.keys():
            results_df = node_dict[model].results
            results_df.to_csv(f"{save_path}/{model}_results.csv", index=False)
            for col in mae_df.columns:
                if "K_bulk" in col:
                    prop = "K_vrh"
                    label = "K_bulk"
                elif "K_shear" in col:
                    prop = "G_vrh"
                    label = "K_shear"
                
                # plot
                fig = px.scatter(
                    data_frame=results_df,
                    x=f'{prop}_DFT',
                    y=f'{prop}_{model}',
                    labels={
                        f'{prop}_DFT': f'{label} DFT [GPa]',
                        f'{prop}_{model}': f'{label} Predicted [GPa]'
                    },
                    title=f'{label} Scatter Plot - {model}')
                fig.add_shape(type='line', x0=results_df[f'{prop}_DFT'].min(), y0=results_df[f'{prop}_DFT'].min(),
                            x1=results_df[f'{prop}_DFT'].max(), y1=results_df[f'{prop}_DFT'].max(),
                            line=dict(dash='dash'))
                fig.update_layout(plot_bgcolor='white',paper_bgcolor='white',font_color='black',xaxis_showgrid=True,yaxis_showgrid=True,xaxis=dict(gridcolor='lightgray'),yaxis=dict(gridcolor='lightgray'), font=dict(size=16, color="black")
                )
                fig.add_annotation(xref="paper", yref="paper",x=0.02, y=0.98,text=f"{prop} MAE: {mae_df.loc[model, col]} GPa",showarrow=False,align="left",font=dict(size=16, color="black"),bordercolor="black",borderwidth=1,borderpad=4,bgcolor="white",opacity=0.8
                )
                fig.write_image(f"{save_path}/{model}_{label}.png", width=800, height=600)
            
                    
        
        
                
                
                
    
    @staticmethod
    def register_elasticity_callbacks(app, mae_df, node_dict):
        
        @app.callback(
            Output('scatter-plot', 'figure'),
            Input('mae-table', 'active_cell')
        )
        
        
        def update_scatter_plot(active_cell):
            if active_cell is None: 
                raise PreventUpdate
                #return px.scatter(title="Click on a cell to view scatter plot")

            row = active_cell['row']
            col = active_cell['column_id']
            model = mae_df.loc[row, 'Model']

            if col not in mae_df.columns or col == 'Model':
                return px.scatter(title="Invalid column clicked")

            # label = col.split()[0]  # "K_bulk" or "K_shear"
            # prop = label_to_key.get(label, None)
            
            if "K_bulk" in col:
                prop = "K_vrh"
                label = "K_bulk"
            elif "K_shear" in col:
                prop = "G_vrh"
                label = "K_shear"
            else:
                return px.scatter(title="Unknown property")
    
    
            # if prop is None:
            #     return px.scatter(title="Unknown property")

            df = node_dict[model].results
            fig = px.scatter(
                data_frame=df,
                x=f'{prop}_DFT',
                y=f'{prop}_{model}',
                labels={
                    f'{prop}_DFT': f'{label} DFT [GPa]',
                    f'{prop}_{model}': f'{label} Predicted [GPa]'
                },
                title=f'{label} Scatter Plot - {model}',
                hover_data=['mp_id', 'formula', f'{prop}_DFT', f'{prop}_{model}'],
            )
            fig.add_shape(type='line', x0=df[f'{prop}_DFT'].min(), y0=df[f'{prop}_DFT'].min(),
                        x1=df[f'{prop}_DFT'].max(), y1=df[f'{prop}_DFT'].max(),
                        line=dict(dash='dash'))

            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='black',
                xaxis_showgrid=True,
                yaxis_showgrid=True,
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )

            
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=f"{label} MAE: {mae_df.loc[row, col]} GPa",
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