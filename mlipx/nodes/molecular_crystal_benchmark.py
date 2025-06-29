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
from mlipx import PhononDispersion, Elasticity, LatticeConstant, X23Benchmark, DMCICE13Benchmark



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class MolecularCrystalBenchmark(zntrack.Node):
    """ Node to combine all molecular crystal benchmarks
    """
    # inputs
    X23_list: List[X23Benchmark] = zntrack.deps()
    DMC_ICE_list: List[DMCICE13Benchmark] = zntrack.deps()
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files

    
    def run(self):
        pass
        


    
    

    @staticmethod
    def benchmark_interactive(
        X23_data: List[X23Benchmark] | Dict[str, X23Benchmark],
        DMC_ICE_data: List[DMCICE13Benchmark] | Dict[str, DMCICE13Benchmark],
        ui: str = "browser",
        full_benchmark: bool = False,
        normalise_to_model: t.Optional[str] = None,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all molecular crystal benchmarks
        """

        from mlipx.dash_utils import process_data
        
        # X23
        X23_dict = process_data(
            X23_data,
            key_extractor=lambda node: node.name.split("_X23Benchmark")[0],
            value_extractor=lambda node: node
        )
        
        
        # DMC-ICE-13
        DMC_ICE_dict = process_data(
            DMC_ICE_data,
            key_extractor=lambda node: node.name.split("_DMCICE13Benchmark")[0],
            value_extractor=lambda node: node
        )

    
        
        app_X23, mae_df_X23 = mlipx.X23Benchmark.mae_plot_interactive(
            node_dict=X23_dict,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )
        app_DMC_ICE, mae_df_DMC_ICE, rel_poly_dfs_DMC_ICE = mlipx.DMCICE13Benchmark.mae_plot_interactive(
            node_dict=DMC_ICE_dict,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )
    
        from mlipx.dash_utils import combine_mae_tables
        combined_mae_table = combine_mae_tables(
            mae_df_X23,
            mae_df_DMC_ICE,
        ) # TODO: use this in the benchmark score func for ease
    
    

        mol_crystal_benchmark_score_df = MolecularCrystalBenchmark.mol_crystal_benchmark_score(mae_df_X23, mae_df_DMC_ICE).round(3)
        mol_crystal_benchmark_score_df = mol_crystal_benchmark_score_df.sort_values(by='Avg MAE \u2193', ascending=True)
        mol_crystal_benchmark_score_df = mol_crystal_benchmark_score_df.reset_index(drop=True)
        mol_crystal_benchmark_score_df['Rank'] = mol_crystal_benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)
        
        if not os.path.exists("benchmark_stats/molecular_crystal_benchmark/"):
            os.makedirs("benchmark_stats/molecular_crystal_benchmark/")
        mol_crystal_benchmark_score_df.to_csv("benchmark_stats/molecular_crystal_benchmark/mol_crystal_benchmark_score.csv", index=False)


        from mlipx.dash_utils import colour_table
        # Viridis-style colormap for Dash DataTable
        style_data_conditional = colour_table(mol_crystal_benchmark_score_df, col_name="Avg MAE \u2193")
        
        
        # md_path_list = [elas_md_path, lattice_const_md_path, phonon_md_path]
    
        # BulkCrystalBenchmark.generate_report(
        #     bulk_crystal_benchmark_score_df=bulk_crystal_benchmark_score_df,
        #     md_report_paths=md_path_list,
        #     markdown_path="benchmark_stats/bulk_crystal_benchmark_report.md",
        #     combined_mae_table=combined_mae_table,
        # )
        
    
        
        if ui is None and not full_benchmark:
            return
        
        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        from mlipx.dash_utils import combine_apps
        apps_list = [app_X23, app_DMC_ICE]
        benchmark_table_info = f"Scores normalised to: {normalise_to_model}" if normalise_to_model else ""
        layout = combine_apps(
            benchmark_score_df=mol_crystal_benchmark_score_df,
            benchmark_title="Molecular Crystal Benchmark",
            apps_list=apps_list,
            benchmark_table_info=benchmark_table_info,
            #style_data_conditional=style_data_conditional,
        )
        app.layout = layout
        
        # app_list[0] is the main app now
        DMCICE13Benchmark.register_callbacks(app, rel_poly_dfs_DMC_ICE)
        
        
        from mlipx.dash_utils import run_app

        if full_benchmark:
            return app, mol_crystal_benchmark_score_df, lambda app: (
                DMCICE13Benchmark.register_callbacks(app, rel_poly_dfs_DMC_ICE),
            )
        
        return run_app(app, ui=ui)        
        
        
        
        
    # ------------- helper functions -------------

    def mol_crystal_benchmark_score(
        mae_df_X23,
        mae_df_DMC_ICE,
        normalise_to_model: t.Optional[str] = None,
        weights: Dict[str, float] = None,
    ):
        """Weighted molecular crystal benchmark scoring."""
        if weights is None:
            weights = {"X23": 1.0, "DMC_ICE": 1.0}

        scores = {}
        model_list = mae_df_X23['Model'].values.tolist()

        for model in model_list:
            x23_score = mae_df_X23.loc[mae_df_X23['Model'] == model, 'Score'].values[0]
            dmc_ice_score = mae_df_DMC_ICE.loc[mae_df_DMC_ICE['Model'] == model, 'Score'].values[0]

            weighted_avg = (
                weights["X23"] * x23_score +
                weights["DMC_ICE"] * dmc_ice_score
            ) / sum(weights.values())

            scores[model] = {
                "X23 Score \u2193": x23_score,
                "DMC-ICE13 Score \u2193": dmc_ice_score,
                "Avg MAE \u2193": weighted_avg,
            }

        df = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(columns={'index': 'Model'})

        if normalise_to_model is not None:
            norm_val = df.loc[df["Model"] == normalise_to_model, "Avg MAE \u2193"].values[0]
            df["Avg MAE \u2193"] = df["Avg MAE \u2193"] / norm_val

        return df





    @staticmethod
    def benchmark_precompute(
        X23_data: List[X23Benchmark] | Dict[str, X23Benchmark],
        DMC_ICE_data: List[DMCICE13Benchmark] | Dict[str, DMCICE13Benchmark],
        cache_dir: str = "app_cache/molecular_crystal_benchmark",
        ui=None,
        run_interactive=False,
        report: bool = False,
        normalise_to_model: str | None = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)

        from mlipx.dash_utils import process_data
        # list -> dict or dict -> dict
        # GMTKN55
        X23_dict = process_data(
            X23_data,
            key_extractor=lambda node: node.name.split("_X23Benchmark")[0],
            value_extractor=lambda node: node
        )
        
        # Homonuclear Diatomics
        DMC_ICE_dict = process_data(
            DMC_ICE_data,
            key_extractor=lambda node: node.name.split("_DMCICE13Benchmark")[0],
            value_extractor=lambda node: node
        )
        
        
        
        os.makedirs(cache_dir, exist_ok=True)
        X23Benchmark.benchmark_precompute(
            node_dict=X23_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )
        DMCICE13Benchmark.benchmark_precompute(
            node_dict=DMC_ICE_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )
        

        
        mae_df_X23 = pd.read_pickle(f"{cache_dir}/X23_cache/mae_df.pkl")
        abs_error_df_X23 = pd.read_pickle(f"{cache_dir}/X23_cache/abs_error_df.pkl")
        lattice_e_df_X23 = pd.read_pickle(f"{cache_dir}/X23_cache/lattice_e_df.pkl")
        
        mae_df_ICE = pd.read_pickle(f"{cache_dir}/DMC_ICE_13_cache/mae_df.pkl")
        lattice_e_all_df_ICE = pd.read_pickle(f"{cache_dir}/DMC_ICE_13_cache/lattice_e_all_df.pkl")
        with open(f"{cache_dir}/DMC_ICE_13_cache/rel_poly_dfs.pkl", "rb") as f:
            rel_poly_dfs_ICE = pickle.load(f)
        

        
        benchmark_score_df = MolecularCrystalBenchmark.mol_crystal_benchmark_score(
            mae_df_X23=mae_df_X23,
            mae_df_DMC_ICE=mae_df_ICE,
            normalise_to_model=normalise_to_model,
        ).round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)
        benchmark_score_df['Rank'] = benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)
        
        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
        
        callback_fn = MolecularCrystalBenchmark.callback_fn_from_cache(
            mae_df_X23=mae_df_X23,
            rel_poly_dfs_DMC_ICE=rel_poly_dfs_ICE,
        )
        
        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(MolecularCrystalBenchmark.callback_fn_from_cache, f)

        return



    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/molecular_crystal_benchmark",
        ui=None,
        full_benchmark: bool = False,
        normalise_to_model: str | None = None,
    ):
        import pandas as pd
        from mlipx.dash_utils import run_app, combine_apps

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        mae_df_X23 = pd.read_pickle(f"{cache_dir}/X23_cache/mae_df.pkl")
        abs_error_df_X23 = pd.read_pickle(f"{cache_dir}/X23_cache/abs_error_df.pkl")
        lattice_e_df_X23 = pd.read_pickle(f"{cache_dir}/X23_cache/lattice_e_df.pkl")
        
        mae_df_ICE = pd.read_pickle(f"{cache_dir}/DMC_ICE_13_cache/mae_df.pkl")
        lattice_e_all_df_ICE = pd.read_pickle(f"{cache_dir}/DMC_ICE_13_cache/lattice_e_all_df.pkl")
        with open(f"{cache_dir}/DMC_ICE_13_cache/rel_poly_dfs.pkl", "rb") as f:
            rel_poly_dfs_ICE = pickle.load(f)
            
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")


        callback_fn = MolecularCrystalBenchmark.callback_fn_from_cache(
            #cache_dir=cache_dir,
            mae_df_X23=mae_df_X23,
            rel_poly_dfs_DMC_ICE=rel_poly_dfs_ICE,
        )


        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        

        layout = MolecularCrystalBenchmark.build_layout(
            benchmark_score_df=benchmark_score_df,
            apps_or_layouts_list=[
                X23Benchmark.build_layout(mae_df_X23, abs_error_df_X23, lattice_e_df_X23),
                DMCICE13Benchmark.build_layout(mae_df_ICE, rel_poly_dfs_ICE, lattice_e_all_df_ICE),
            ],
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
        )

        if full_benchmark:
            return layout, callback_fn

        app.layout = layout
        callback_fn(app)
        
        return run_app(app, ui=ui)
    
    @staticmethod
    def build_layout(
        benchmark_score_df: pd.DataFrame,
        apps_or_layouts_list: t.List[t.Union[dash.Dash, t.Callable, html.Div]],
        benchmark_table_info: str = "",
        id: str = "molecular-benchmark-score-table",
    ):
        """ Build the layout for the molecular crystal benchmark dashboard
        """
        from mlipx.dash_utils import combine_apps
        
        return combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Molecular Crystal Benchmark",
            apps_or_layouts_list=apps_or_layouts_list,
            benchmark_table_info=benchmark_table_info,
            id=id,
            static_coloured_table=True,
        )
    
    @staticmethod
    def callback_fn_from_cache(
        mae_df_X23,
        rel_poly_dfs_DMC_ICE,
    ):
        from mlipx import X23Benchmark, DMCICE13Benchmark

        def callback_fn(app):
            #X23Benchmark.register_callbacks(app, mae_df_X23)
            DMCICE13Benchmark.register_callbacks(app, rel_poly_dfs_DMC_ICE)

        return callback_fn