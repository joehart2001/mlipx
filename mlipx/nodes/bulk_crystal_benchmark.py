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
from mlipx import PhononAllRef, PhononAllBatch



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from typing import Union
from mlipx.dash_utils import colour_table



class BulkCrystalBenchmark(zntrack.Node):
    """ Node to combine all bulk crystal benchmarks
    """
    # inputs
    phonon_ref: Union[List[PhononDispersion], PhononAllRef] = zntrack.deps()
    phonon_pred_list: Union[List[PhononDispersion], List[PhononAllBatch]] = zntrack.deps()
    elasticity_list: List[Elasticity] = zntrack.deps()
    lattice_const_list: List[LatticeConstant] = zntrack.deps()
    
    
    def run(self):
        pass
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    @staticmethod
    def register_callbacks(
        app,
        phonon_mae_df,
        mae_df_elas,
        mae_df_lattice_const,
        normalise_to_model=None,
    ):
        """
        Register callbacks for the interactive benchmark dashboard, including the weight update callback.
        """
        # --- Callback to update benchmark table based on weights (now uses input fields) ---
        @app.callback(
            Output("phonon-benchmark-score-table", "data"),
            Output("phonon-benchmark-score-table", "style_data_conditional"),
            Output("bulk-crystal-weights", "data"),
            Input("phonon-weight-input", "value"),
            Input("elasticity-weight-input", "value"),
            Input("lattice-const-weight-input", "value"),
            State("bulk-crystal-weights", "data"),
        )
        def update_benchmark_table_and_store(phonon_w, elas_w, lat_w, stored_weights):
            if None in (phonon_w, elas_w, lat_w):
                # Try to fall back to stored weights
                if stored_weights is None:
                    raise PreventUpdate
                phonon_w = stored_weights.get("phonon", 1.0)
                elas_w = stored_weights.get("elasticity", 1.0)
                lat_w = stored_weights.get("lattice_const", 0.2)
        
            weights = {"phonon": phonon_w, "elasticity": elas_w, "lattice_const": lat_w}
            
            updated_df = BulkCrystalBenchmark.bulk_crystal_benchmark_score(
                phonon_mae_df,
                mae_df_elas,
                mae_df_lattice_const,
                normalise_to_model=normalise_to_model,
                weights=weights
            ).round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)
            
            updated_df["Rank"] = updated_df['Avg MAE \u2193'].rank(ascending=True)
            style_data_conditional = colour_table(updated_df, all_cols=True)
            
            return (
                updated_df.to_dict("records"), 
                style_data_conditional,
                weights,
            )


        # --- Callbacks to sync sliders and input fields ---
        @app.callback(
            Output("phonon-weight", "value"),
            Output("phonon-weight-input", "value"),
            Input("phonon-weight", "value"),
            Input("phonon-weight-input", "value"),
            prevent_initial_call=True,
        )
        def sync_phonon_weight(slider_val, input_val):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return (input_val, input_val) if "input" in triggered_id else (slider_val, slider_val)



        @app.callback(
            Output("elasticity-weight", "value"),
            Output("elasticity-weight-input", "value"),
            Input("elasticity-weight", "value"),
            Input("elasticity-weight-input", "value"),
            prevent_initial_call=True,
        )
        def sync_elasticity_weight(slider_val, input_val):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return (input_val, input_val) if "input" in triggered_id else (slider_val, slider_val)

        @app.callback(
            Output("lattice-const-weight", "value"),
            Output("lattice-const-weight-input", "value"),
            Input("lattice-const-weight", "value"),
            Input("lattice-const-weight-input", "value"),
            prevent_initial_call=True,
        )
        def sync_lattice_const_weight(slider_val, input_val):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return (input_val, input_val) if "input" in triggered_id else (slider_val, slider_val)

        # --- Callback to update overall benchmark score table when weights are changed ---
        @app.callback(
            Output("overall-score-table", "data"),
            Input("bulk-crystal-weights", "data"),
            prevent_initial_call=True
        )
        def update_overall_table_from_store(weights):
            if weights is None:
                raise dash.exceptions.PreventUpdate

            updated_df = BulkCrystalBenchmark.bulk_crystal_benchmark_score(
                phonon_mae_df,
                mae_df_elas,
                mae_df_lattice_const,
                normalise_to_model=normalise_to_model,
                weights=weights
            ).round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)

            updated_df["Rank"] = updated_df['Avg MAE \u2193'].rank(ascending=True)
            return updated_df.to_dict("records")
        

    @staticmethod
    def benchmark_precompute(
        elasticity_data: List[Elasticity] | Dict[str, Elasticity],
        lattice_const_data: List[LatticeConstant] | Dict[str, Dict[str, LatticeConstant]],
        lattice_const_ref_node_dict: LatticeConstant,
        phonon_ref_data: List[PhononDispersion] | Dict[str, PhononDispersion] | PhononAllRef,
        phonon_pred_data: List[PhononDispersion] | Dict[str, Dict[str, PhononDispersion]] | List[PhononAllBatch] | Dict[str, PhononAllBatch],
        cache_dir: str = "app_cache/bulk_crystal_benchmark",
        ui = None,
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        
        """
        Precompute and cache all sub-benchmarks for bulk crystal benchmark.
        """

        from mlipx.dash_utils import process_data
        from mlipx.phonons_utils import convert_batch_to_node_dict

        # ---- convert all data to dicts if not already----
        lattice_const_dict = process_data(
            lattice_const_data,
            key_extractor=lambda node: node.name.split("LatticeConst-")[1],
            value_extractor=lambda node: {node.name.split("_lattice-constant-pred")[0]: node}
        )            
        elasticity_dict = process_data(
            elasticity_data,
            key_extractor=lambda node: node.name.split("_Elasticity")[0],
            value_extractor=lambda node: node
        )
        phonon_dict_pred = process_data(
            phonon_pred_data,
            key_extractor=lambda node: node.name.split("_phonons-dispersion")[0],
            value_extractor=lambda node: node
        ) 
            
        
        os.makedirs(cache_dir, exist_ok=True)

        # Run precomputes for all sub-benchmarks
        PhononAllBatch.benchmark_precompute(
            pred_node_dict=phonon_dict_pred,
            ref_phonon_node=phonon_ref_data,
            ui=ui,
            run_interactive=False,
            report=report,
            normalise_to_model=normalise_to_model,
        )
        Elasticity.benchmark_precompute(
            node_dict=elasticity_dict,
            ui=ui,
            run_interactive=False,
            report=report,
            normalise_to_model=normalise_to_model,
        )
        LatticeConstant.benchmark_precompute(
            node_dict=lattice_const_dict,
            ref_node_dict=lattice_const_ref_node_dict,
            ui=ui,
            run_interactive=False,
            report=report,
            normalise_to_model=normalise_to_model,
        )

        # Load precomputed data
        mae_df_elas = pd.read_pickle(f"{cache_dir}/elasticity_cache/mae_summary.pkl")
        mae_df_lattice_const = pd.read_pickle(f"{cache_dir}/lattice_cache/mae_summary.pkl")
        phonon_mae_df = pd.read_pickle(f"{cache_dir}/phonons_cache/mae_summary.pkl")

        benchmark_score_df = BulkCrystalBenchmark.bulk_crystal_benchmark_score(
            phonon_mae_df,
            mae_df_elas,
            mae_df_lattice_const,
            normalise_to_model=normalise_to_model,
        ).round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)
        benchmark_score_df['Rank'] = benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)

        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
                
        callback_fn = BulkCrystalBenchmark.callback_fn_from_cache(
            cache_dir=cache_dir,
            phonon_mae_df=phonon_mae_df,
            mae_df_elas=mae_df_elas,
            mae_df_lattice_const=mae_df_lattice_const,
            normalise_to_model=normalise_to_model,
        )

        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(BulkCrystalBenchmark.callback_fn_from_cache, f)
            
            
            
    @staticmethod
    def callback_fn_from_cache(cache_dir, phonon_mae_df, mae_df_elas, mae_df_lattice_const, normalise_to_model=None):
        from mlipx import PhononDispersion, Elasticity, LatticeConstant

        def callback_fn(app):
            with open(f"{cache_dir}/phonons_cache/scatter_to_dispersion_map.pkl", "rb") as f:
                scatter_to_dispersion_map = pickle.load(f)
            with open(f"{cache_dir}/phonons_cache/model_benchmarks_dict.pkl", "rb") as f:
                model_benchmarks_dict = pickle.load(f)
            lat_const_df = pd.read_pickle(f"{cache_dir}/lattice_cache/lat_const_df.pkl")
            with open(f"{cache_dir}/elasticity_cache/results_dict.pkl", "rb") as f:
                results_dict = pickle.load(f)
                
            PhononDispersion.register_callbacks(app, phonon_mae_df, scatter_to_dispersion_map, model_benchmarks_dict)
            Elasticity.register_callbacks(app, mae_df_elas, results_dict)
            LatticeConstant.register_callbacks(app, mae_df_lattice_const, lat_const_df)
            BulkCrystalBenchmark.register_callbacks(app, phonon_mae_df, mae_df_elas, mae_df_lattice_const, normalise_to_model=normalise_to_model)

        return callback_fn

        
        
    
    
    
    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/bulk_crystal_benchmark",
        ui=None,
        full_benchmark: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        import pandas as pd
        import pickle
        from mlipx.dash_utils import run_app
        import dash

        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")
        phonon_mae_df = pd.read_pickle(f"{cache_dir}/phonons_cache/mae_summary.pkl")
        mae_df_elas = pd.read_pickle(f"{cache_dir}/elasticity_cache/mae_summary.pkl")
        mae_df_lattice_const = pd.read_pickle(f"{cache_dir}/lattice_cache/mae_summary.pkl")
        callback_fn = BulkCrystalBenchmark.callback_fn_from_cache(
            cache_dir, phonon_mae_df, mae_df_elas, mae_df_lattice_const
        )

        app = dash.Dash(__name__)

        layout = BulkCrystalBenchmark.build_layout(
            benchmark_score_df=benchmark_score_df,
            phonon_mae_df=phonon_mae_df,
            mae_df_elas=mae_df_elas,
            mae_df_lattice_const=mae_df_lattice_const,
            normalise_to_model=normalise_to_model,
        )
        
        if full_benchmark:
            return layout, callback_fn
        
        app.layout = layout
        callback_fn(app)

        return run_app(app, ui=ui)
    
    
    
    







    # --------------------------------- Helper Functions ---------------------------------




    # def bulk_crystal_benchmark_score(
    #     phonon_mae_df, 
    #     mae_df_elas, 
    #     mae_df_lattice_const,
    #     normalise_to_model: Optional[str] = None,
    #     weights: Dict[str, float] = None
    # ):
    #     if weights is None:
    #         weights = {"phonon": 1.0, "elasticity": 1.0, "lattice_const": 0.2}

    #     scores = {}
    #     model_list = phonon_mae_df['Model'].values.tolist()

    #     for model in model_list:
    #         score = 0
    #         score += weights["lattice_const"] * mae_df_lattice_const.loc[mae_df_lattice_const['Model'] == model, "Lat Const Score \u2193 (PBE)"].values[0]
    #         score += weights["phonon"] * phonon_mae_df.loc[phonon_mae_df['Model'] == model, "Phonon Score \u2193"].values[0]
    #         score += weights["elasticity"] * mae_df_elas.loc[mae_df_elas['Model'] == model, "Elasticity Score \u2193"].values[0]
    #         scores[model] = score / sum(weights.values())

    #     if normalise_to_model:
    #         scores = {k: v / scores[normalise_to_model] for k, v in scores.items()}

    #     return pd.DataFrame.from_dict(scores, orient='index', columns=['Avg MAE \u2193']).reset_index().rename(columns={'index': 'Model'})

    @staticmethod
    def bulk_crystal_benchmark_score(
        phonon_mae_df, 
        mae_df_elas, 
        mae_df_lattice_const,
        normalise_to_model: Optional[str] = None,
        weights: Dict[str, float] = None
    ):
        if weights is None:
            weights = {"phonon": 1.0, "elasticity": 1.0, "lattice_const": 0.2}

        scores = {}
        model_list = phonon_mae_df['Model'].values.tolist()

        for model in model_list:
            phonon_score = phonon_mae_df.loc[phonon_mae_df['Model'] == model, "Phonon Score \u2193"].values[0]
            #print(mae_df_elas)
            #print(model)
            elas_score = mae_df_elas.loc[mae_df_elas['Model'] == model, "Elasticity Score \u2193"].values[0]
            lat_score = mae_df_lattice_const.loc[mae_df_lattice_const['Model'] == model, "Lat Const Score \u2193 (PBE)"].values[0]

            weighted_avg = (
                weights["phonon"] * phonon_score +
                weights["elasticity"] * elas_score +
                weights["lattice_const"] * lat_score
            ) / sum(weights.values())

            scores[model] = {
                "Phonon Score \u2193": phonon_score,
                "Elasticity Score \u2193": elas_score,
                "Lat Const Score \u2193 (PBE)": lat_score,
                "Avg MAE \u2193": weighted_avg,
            }

        df = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(columns={"index": "Model"})

        if normalise_to_model:
            norm_val = df.loc[df["Model"] == normalise_to_model, "Avg MAE \u2193"].values[0]
            df["Avg MAE \u2193"] = df["Avg MAE \u2193"] / norm_val

        return df

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
        normalise_to_model: Optional[str] = None,
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


        info_str = f"(Normalised to {normalise_to_model})" if normalise_to_model else ""
        combined_md.append(f"## Benchmark Score Table {info_str} \n")
        #combined_md.append("\\rowcolors{2}{gray!10}{white}\n")
        combined_md.append(bulk_crystal_benchmark_score_df.to_markdown(index=False))
        combined_md.append("\n")

        
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

        try:
            subprocess.run(
                [
                    "pandoc",
                    str(markdown_path),
                    "-o", 
                    str(pdf_path),
                    "--pdf-engine=xelatex",
                    "--variable=geometry:top=1.5cm,bottom=2cm,left=1cm,right=1cm",
                    "--from", "markdown+raw_tex",
                ],
                check=True
            )
            print(f"PDF report saved to {pdf_path}")

        except subprocess.CalledProcessError as e:
            print(f"PDF generation failed: {e}")
        
        
        return markdown_path

    @staticmethod
    def build_layout(benchmark_score_df, phonon_mae_df, mae_df_elas, mae_df_lattice_const, normalise_to_model=None):
        from dash import html, dcc
        from mlipx.dash_utils import combine_apps
        from mlipx import PhononDispersion, LatticeConstant, Elasticity

        def weight_control(label, slider_id, input_id, default_value):
            return html.Div([
                html.Label(label),
                html.Div([
                    html.Div(
                        dcc.Slider(
                            id=slider_id,
                            min=0,
                            max=5,
                            step=0.1,
                            value=default_value,
                            tooltip={"always_visible": False},
                            marks=None,
                        ),
                        style={"flex": "1 1 80%"},
                    ),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        value=default_value,
                        step=0.1,
                        style={"width": "80px"},
                    ),
                ], style={"display": "flex", "gap": "10px", "alignItems": "center"})
            ])

        weight_controls = html.Div([
            weight_control("Phonon Weight", "phonon-weight", "phonon-weight-input", 1.0),
            weight_control("Elasticity Weight", "elasticity-weight", "elasticity-weight-input", 1.0),
            weight_control("Lattice Const Weight", "lattice-const-weight", "lattice-const-weight-input", 0.2),
        ], style={"margin": "20px"})

        # Remove dcc.Store(id="bulk-crystal-weights", ...) from here.
        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Bulk Crystal Benchmark",
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
            apps_or_layouts_list=[
                PhononDispersion.build_layout(phonon_mae_df),
                LatticeConstant.build_layout(mae_df_lattice_const),
                Elasticity.build_layout(mae_df_elas),
            ],
            id="phonon-benchmark-score-table",
            weights_components=[weight_controls],
            shared_stores=[
                #dcc.Store(id="bulk-crystal-weights", storage_type="session")
            ]
        )

        return layout