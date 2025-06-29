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
from mlipx import GMTKN55Benchmark, HomonuclearDiatomics



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class MolecularBenchmark(zntrack.Node):
    """ Node to combine all molecular benchmarks
    """
    # inputs
    GMTKN55_list: List[GMTKN55Benchmark] = zntrack.deps()
    diatomic_list: List[HomonuclearDiatomics] = zntrack.deps()


    def run(self):
        pass
        


    
    

    @staticmethod
    def benchmark_precompute(
        GMTKN55_data: List[GMTKN55Benchmark] | Dict[str, GMTKN55Benchmark],
        HD_data: List[HomonuclearDiatomics] | Dict[str, HomonuclearDiatomics],
        cache_dir: str = "app_cache/molecular_benchmark/",
        ui: str = "browser",
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all molecular benchmarks
        """
        
        from mlipx.dash_utils import process_data
        # list -> dict or dict -> dict
        # GMTKN55
        GMTKN55_dict = process_data(
            GMTKN55_data,
            key_extractor=lambda node: node.name.split("_GMTKN55Benchmark")[0],
            value_extractor=lambda node: node
        )
        
        # Homonuclear Diatomics
        HD_dict = process_data(
            HD_data,
            key_extractor=lambda node: node.name.split("_homonuclear-diatomics")[0],
            value_extractor=lambda node: node
        )
        
        
        os.makedirs(cache_dir, exist_ok=True)
        GMTKN55Benchmark.benchmark_precompute(
            node_dict=GMTKN55_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )
        HomonuclearDiatomics.benchmark_precompute(
            node_dict=HD_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )
        
        
        # Load precomputed data
        wtmad_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/wtmad_table.pkl")
        mae_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/mae_df.pkl")
        
        stats_df_HD = pd.read_pickle(f"{cache_dir}/diatomics_cache/stats_df.pkl")
        with open(f"{cache_dir}/diatomics_cache/results_dict.pkl", "rb") as f:
            results_dict = pickle.load(f)

        
        benchmark_score_df = MolecularBenchmark.mol_benchmark_score(
            wtmad_df_GMTKN55=wtmad_df_GMTKN55,
            stats_df_HD=stats_df_HD,
            normalise_to_model=normalise_to_model,
        ).round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)
        benchmark_score_df['Rank'] = benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)
        
        benchmark_score_df.to_pickle(f"{cache_dir}/benchmark_score.pkl")
        
        callback_fn = MolecularBenchmark.callback_fn_from_cache(
            mae_df_GMTKN55=mae_df_GMTKN55,
            all_model_dicts_GMTKN55=GMTKN55_data,
            results_dict_HD=results_dict,
            normalise_to_model=normalise_to_model,
        )
        
        with open(f"{cache_dir}/callback_data.pkl", "wb") as f:
            pickle.dump(MolecularBenchmark.callback_fn_from_cache, f)
        
        return 
        
        # from mlipx.dash_utils import process_data
        
        # # GMTKN55
        # GMTKN55_dict = process_data(
        #     GMTKN55_data,
        #     key_extractor=lambda node: node.name.split("_GMTKN55Benchmark")[0],
        #     value_extractor=lambda node: node
        # )
        
        # # Homonuclear Diatomics
        # HD_dict = process_data(
        #     HD_data,
        #     key_extractor=lambda node: node.name.split("_homonuclear-diatomics")[0],
        #     value_extractor=lambda node: node
        # )
        
        
    
        # app_GMTKN55, wtmad_df_GMTKN55, mae_df_GMTKN55 = mlipx.GMTKN55Benchmark.mae_plot_interactive(
        #     node_dict=GMTKN55_dict,
        #     run_interactive=False,
        #     normalise_to_model=normalise_to_model,
        # )
        
        # app_HD, results_df_HD, stats_df_HD = mlipx.HomonuclearDiatomics.mae_plot_interactive(
        #     node_dict=HD_dict,
        #     run_interactive=False,
        #     normalise_to_model=normalise_to_model,
        # )
        

        # # from mlipx.dash_utils import combine_mae_tables
        # # combined_mae_table = combine_mae_tables(
        # #     mae_df_X23,
        # #     mae_df_DMC_ICE,
        # # ) # TODO: use this in the benchmark score func for ease
    


        # mol_benchmark_score_df = MolecularBenchmark.mol_benchmark_score(wtmad_df_GMTKN55, stats_df_HD).round(3)
        # mol_benchmark_score_df = mol_benchmark_score_df.sort_values(by='Avg MAE \u2193', ascending=True)
        # mol_benchmark_score_df = mol_benchmark_score_df.reset_index(drop=True)
        # mol_benchmark_score_df['Rank'] = mol_benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)
        
        # if not os.path.exists("benchmark_stats/molecular_benchmark/"):
        #     os.makedirs("benchmark_stats/molecular_benchmark/")
        # mol_benchmark_score_df.to_csv("benchmark_stats/molecular_benchmark/mol_benchmark_score.csv", index=False)


        # from mlipx.dash_utils import colour_table
        # # Viridis-style colormap for Dash DataTable
        # style_data_conditional = colour_table(mol_benchmark_score_df, col_name="Avg MAE \u2193")
        
        
        # md_path_list = [elas_md_path, lattice_const_md_path, phonon_md_path]
    
        # BulkCrystalBenchmark.generate_report(
        #     bulk_crystal_benchmark_score_df=bulk_crystal_benchmark_score_df,
        #     md_report_paths=md_path_list,
        #     markdown_path="benchmark_stats/bulk_crystal_benchmark_report.md",
        #     combined_mae_table=combined_mae_table,
        # )
        
    
        
        # if ui is None and not full_benchmark:
        #     return
        
        # app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # from mlipx.dash_utils import combine_apps
        # apps_list = [app_GMTKN55, app_HD]
        # benchmark_table_info = f"Scores normalised to: {normalise_to_model}" if normalise_to_model else ""
        # layout = combine_apps(
        #     benchmark_score_df=mol_benchmark_score_df,
        #     benchmark_title="Molecular Benchmark",
        #     apps_list=apps_list,
        #     benchmark_table_info=benchmark_table_info,
        #     #style_data_conditional=style_data_conditional,
        # )
        # app.layout = layout
        
        # # app_list[0] is the main app now
        # GMTKN55Benchmark.register_callbacks(app, GMTKN55_dict, mae_df_GMTKN55)
        # HomonuclearDiatomics.register_callbacks(app, HD_dict, results_df_HD)
        
        # from mlipx.dash_utils import run_app

        # if full_benchmark:
        #     return app, mol_benchmark_score_df, lambda app: (
        #         GMTKN55Benchmark.register_callbacks(app, GMTKN55_dict, mae_df_GMTKN55),
        #         HomonuclearDiatomics.register_callbacks(app, HD_dict, results_df_HD),
        #     )
        
        # return run_app(app, ui=ui)        
        
        
    @staticmethod
    def callback_fn_from_cache(
        mae_df_GMTKN55,
        all_model_dicts_GMTKN55,
        results_dict_HD,
        normalise_to_model=None,
    ):
        from mlipx import GMTKN55Benchmark, HomonuclearDiatomics

        def callback_fn(app):
            GMTKN55Benchmark.register_callbacks(app, mae_df_GMTKN55, all_model_dicts_GMTKN55)
            HomonuclearDiatomics.register_callbacks(app, results_dict_HD)

        return callback_fn
    # ------------- helper functions -------------

    @staticmethod
    def mol_benchmark_score(
        wtmad_df_GMTKN55, 
        stats_df_HD,
        normalise_to_model: Optional[str] = None,
        weights: Dict[str, float] = None,
    ):
        """Weighted scoring for molecular benchmarks"""
        if weights is None:
            weights = {"gmtkn55": 1.0, "diatomics": 1.0}

        scores = {}
        model_list = wtmad_df_GMTKN55['Model'].values.tolist()

        for model in model_list:
            gmtkn55_score = wtmad_df_GMTKN55.loc[wtmad_df_GMTKN55['Model'] == model, 'Score'].values[0]
            diatomic_score = stats_df_HD.loc[stats_df_HD['Model'] == model, 'Score'].values[0]

            weighted_avg = (
                weights["gmtkn55"] * gmtkn55_score +
                weights["diatomics"] * diatomic_score
            ) / sum(weights.values())

            scores[model] = {
                "GMTKN55 Score \u2193": gmtkn55_score,
                "Diatomics Score \u2193": diatomic_score,
                "Avg MAE \u2193": weighted_avg,
            }

        df = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(columns={'index': 'Model'})

        if normalise_to_model:
            norm_val = df.loc[df["Model"] == normalise_to_model, "Avg MAE \u2193"].values[0]
            df["Avg MAE \u2193"] = df["Avg MAE \u2193"] / norm_val

        return df


    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/molecular_benchmark/",
        ui=None,
        full_benchmark: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        import pandas as pd
        import pickle
        import dash
        from mlipx.dash_utils import run_app, combine_apps
        from mlipx import GMTKN55Benchmark, HomonuclearDiatomics

        # Load precomputed results
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")
        wtmad_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/wtmad_table.pkl")
        benchmark_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/benchmark_df.pkl")
        mae_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/mae_df.pkl")
        stats_df_HD = pd.read_pickle(f"{cache_dir}/diatomics_cache/stats_df.pkl")

        with open(f"{cache_dir}/GMTKN55_cache/all_model_dicts.pkl", "rb") as f:
            all_model_dicts_GMTKN55 = pickle.load(f)
        with open(f"{cache_dir}/diatomics_cache/results_dict.pkl", "rb") as f:
            results_dict_HD = pickle.load(f)

        callback_fn = MolecularBenchmark.callback_fn_from_cache(
            #cache_dir=cache_dir,
            mae_df_GMTKN55=mae_df_GMTKN55,
            all_model_dicts_GMTKN55=all_model_dicts_GMTKN55,
            results_dict_HD=results_dict_HD,
            normalise_to_model=normalise_to_model,
        )

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Molecular Benchmark",
            apps_or_layouts_list=[
                GMTKN55Benchmark.build_layout(wtmad_df_GMTKN55, benchmark_df_GMTKN55),
                HomonuclearDiatomics.build_layout(stats_df_HD, results_dict_HD),
            ],
            benchmark_table_info=f"Scores normalised to: {normalise_to_model}" if normalise_to_model else "",
            id="molecular-benchmark-score-table",
            static_coloured_table=True,
        )
        
        if full_benchmark:
            return layout, callback_fn

        app.layout = layout
        callback_fn(app)
        
        
        
        return run_app(app, ui=ui)