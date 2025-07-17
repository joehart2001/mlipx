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
from mlipx import GMTKN55Benchmark, HomonuclearDiatomics, Wiggle150



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
    wiggle150_list: List[Wiggle150] = zntrack.deps()


    def run(self):
        pass
        


    
    

    @staticmethod
    def benchmark_precompute(
        GMTKN55_data: List[GMTKN55Benchmark] | Dict[str, GMTKN55Benchmark],
        HD_data: List[HomonuclearDiatomics] | Dict[str, HomonuclearDiatomics],
        Wiggle150_data: List[Wiggle150] | Dict[str, Wiggle150] = None,
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
        
        wiggle150_dict = process_data(
            Wiggle150_data,
            key_extractor=lambda node: node.name.split("_Wiggle150")[0],
            value_extractor=lambda node: node
        ) if Wiggle150_data else None
        
        
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
        Wiggle150.benchmark_precompute(
            node_dict=wiggle150_dict,
            ui=ui,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        ) if Wiggle150_data else None
        
        
        # ------- Load precomputed data -------
        # GMTKN55
        wtmad_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/wtmad_table.pkl")
        mae_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/mae_df.pkl")
        # HD
        stats_df_HD = pd.read_pickle(f"{cache_dir}/diatomics_cache/stats_df.pkl")
        with open(f"{cache_dir}/diatomics_cache/results_dict.pkl", "rb") as f:
            results_dict = pickle.load(f)
        # Wiggle150
        if Wiggle150_data:
            wig150_rel_energy_df = pd.read_pickle(os.path.join(cache_dir, "wiggle150_cache/rel_energy_df.pkl"))
            wig150_mae_df = pd.read_pickle(os.path.join(cache_dir, "wiggle150_cache/mae_df.pkl"))
            
        
        from mlipx.data_utils import category_weighted_benchmark_score

        benchmark_score_df = category_weighted_benchmark_score(
            gmtkn55=wtmad_df_GMTKN55,
            diatomics=stats_df_HD,
            wiggle150=wig150_mae_df,
            normalise_to_model="my-best-model",
            weights={"gmtkn55": 1.0, "diatomics": 0.5, "wiggle150": 0.5},
        )
        # benchmark_score_df = benchmark_score_df.round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)
        # benchmark_score_df['Rank'] = benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)
        
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
        
        

    # ------------- helper functions -------------

    # @staticmethod
    # def category_weighted_benchmark_score(
    #     normalise_to_model: Optional[str] = None,
    #     weights: Optional[Dict[str, float]] = None,
    #     **score_dfs: pd.DataFrame,
    # ):
    #     """Weighted scoring for molecular benchmarks from multiple input score DataFrames."""
    #     if weights is None:
    #         weights = {name: 1.0 for name in score_dfs.keys()}

    #     scores = {}
    #     # Only consider DataFrames that are not None
    #     valid_dfs = {name: df for name, df in score_dfs.items() if df is not None}
    #     if not valid_dfs:
    #         return pd.DataFrame()
    #     # Find intersection of all model/method names across all input DataFrames
    #     all_models = set.intersection(
    #         *(set(df["Model"] if "Model" in df.columns else df["Method"]) for df in valid_dfs.values())
    #     )

    #     for model in all_models:
    #         entry = {}
    #         total = 0.0
    #         denom = 0.0
    #         for name, df in score_dfs.items():
    #             if df is None:
    #                 continue
    #             col = "Model" if "Model" in df.columns else "Method"
    #             # Try to find a score column with "Score" in name, prefer "\u2193" if present
    #             score_col = None
    #             for c in df.columns:
    #                 if "Score" in c:
    #                     score_col = c
    #                     if "\u2193" in c:
    #                         break
    #             if score_col is None:
    #                 continue
    #             # Only add if model present in this df
    #             if model not in set(df[col]):
    #                 continue
    #             score = df.loc[df[col] == model, score_col].values[0]
    #             entry[f"{name.capitalize()} {score_col}"] = score
    #             total += weights.get(name, 1.0) * score
    #             denom += weights.get(name, 1.0)
    #         entry["Avg MAE \u2193"] = total / denom if denom > 0 else None
    #         scores[model] = entry

    #     df = pd.DataFrame.from_dict(scores, orient="index").reset_index().rename(columns={"index": "Model"})

    #     if normalise_to_model:
    #         norm_val = df.loc[df["Model"] == normalise_to_model, "Avg MAE \u2193"].values[0]
    #         df["Avg MAE \u2193"] = df["Avg MAE \u2193"] / norm_val

    #     return df


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

        # ------- Load precomputed results -------
        # GMTKN55
        wtmad_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/wtmad_table.pkl")
        benchmark_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/benchmark_df.pkl")
        mae_df_GMTKN55 = pd.read_pickle(f"{cache_dir}/GMTKN55_cache/mae_df.pkl")
        # HD
        stats_df_HD = pd.read_pickle(f"{cache_dir}/diatomics_cache/stats_df.pkl")
        # wiggle150
        wig150_rel_energy_df = pd.read_pickle(os.path.join(cache_dir, "wiggle150_cache/rel_energy_df.pkl"))
        wig150_mae_df = pd.read_pickle(os.path.join(cache_dir, "wiggle150_cache/mae_df.pkl"))
        # mol becnchmark
        benchmark_score_df = pd.read_pickle(f"{cache_dir}/benchmark_score.pkl")

        with open(f"{cache_dir}/GMTKN55_cache/all_model_dicts.pkl", "rb") as f:
            all_model_dicts_GMTKN55 = pickle.load(f)
        with open(f"{cache_dir}/diatomics_cache/results_dict.pkl", "rb") as f:
            results_dict_HD = pickle.load(f)

        callback_fn = MolecularBenchmark.callback_fn_from_cache(
            mae_df_GMTKN55=mae_df_GMTKN55,
            all_model_dicts_GMTKN55=all_model_dicts_GMTKN55,
            results_dict_HD=results_dict_HD,
            wig150_mae_df=wig150_mae_df,
            wig150_rel_energy_df=wig150_rel_energy_df,
            normalise_to_model=normalise_to_model,
        )

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Molecular Benchmark",
            apps_or_layouts_list=[
                GMTKN55Benchmark.build_layout(wtmad_df_GMTKN55, benchmark_df_GMTKN55),
                HomonuclearDiatomics.build_layout(stats_df_HD, results_dict_HD),
                Wiggle150.build_layout(wig150_mae_df),
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
    
    
    
    @staticmethod
    def callback_fn_from_cache(
        mae_df_GMTKN55,
        all_model_dicts_GMTKN55,
        results_dict_HD,
        wig150_mae_df,
        wig150_rel_energy_df,
        normalise_to_model=None,
    ):
        from mlipx import GMTKN55Benchmark, HomonuclearDiatomics

        def callback_fn(app):
            GMTKN55Benchmark.register_callbacks(app, mae_df_GMTKN55, all_model_dicts_GMTKN55)
            HomonuclearDiatomics.register_callbacks(app, results_dict_HD)
            Wiggle150.register_callbacks(app, wig150_mae_df, wig150_rel_energy_df)
        return callback_fn