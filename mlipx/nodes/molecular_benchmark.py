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
from mlipx import PhononDispersion, Elasticity, LatticeConstant, X23Benchmark, DMCICE13Benchmark, GMTKN55Benchmark



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
    
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files

    
    def run(self):
        pass
        


    
    

    @staticmethod
    def benchmark_interactive(
        GMTKN55_data: List[X23Benchmark] | Dict[str, X23Benchmark],
        ui: str = "browser",
        full_benchmark: bool = False,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all molecular benchmarks
        """

        from mlipx.dash_utils import process_data
        
        # GMTKN55
        GMTKN55_dict = process_data(
            GMTKN55_data,
            key_extractor=lambda node: node.name.split("_GMTKN55Benchmark")[0],
            value_extractor=lambda node: node
        )
        
    
        app_GMTKN55, wtmad_df_GMTKN55, mae_df_GMTKN55 = mlipx.GMTKN55Benchmark.mae_plot_interactive(
            node_dict=GMTKN55_dict,
            run_interactive=False,
        )

        # from mlipx.dash_utils import combine_mae_tables
        # combined_mae_table = combine_mae_tables(
        #     mae_df_X23,
        #     mae_df_DMC_ICE,
        # ) # TODO: use this in the benchmark score func for ease
    


        mol_benchmark_score_df = MolecularBenchmark.mol_benchmark_score(wtmad_df_GMTKN55).round(3)
        mol_benchmark_score_df = mol_benchmark_score_df.sort_values(by='Avg MAE \u2193', ascending=True)
        mol_benchmark_score_df = mol_benchmark_score_df.reset_index(drop=True)
        mol_benchmark_score_df['Rank'] = mol_benchmark_score_df.index + 1
        
        if not os.path.exists("benchmark_stats/molecular_benchmark/"):
            os.makedirs("benchmark_stats/molecular_benchmark/")
        mol_benchmark_score_df.to_csv("benchmark_stats/molecular_benchmark/mol_crystal_benchmark_score.csv", index=False)


        from mlipx.dash_utils import colour_table
        # Viridis-style colormap for Dash DataTable
        style_data_conditional = colour_table(mol_benchmark_score_df, col_name="Avg MAE \u2193")
        
        
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
        apps_list = [app_GMTKN55]
        layout = combine_apps(
            benchmark_score_df=mol_benchmark_score_df,
            benchmark_title="Molecular Benchmark",
            apps_list=apps_list,
            style_data_conditional=style_data_conditional,
        )
        app.layout = layout
        
        # app_list[0] is the main app now
        GMTKN55Benchmark.register_callbacks(app, GMTKN55_dict, mae_df_GMTKN55)
        
        
        from mlipx.dash_utils import run_app

        if full_benchmark:
            return app, mol_benchmark_score_df, lambda app: (
                GMTKN55Benchmark.register_callbacks(app, GMTKN55_dict, mae_df_GMTKN55),
            )
        
        return run_app(app, ui=ui)        
        
        
        
        
        
        
    # ------------- helper functions -------------

    def mol_benchmark_score(
        wtmad_df_GMTKN55, 
        #mae_df_DMC_ICE, 
    ):
        """ Currently avg mae
        """
        # Initialize scores
        maes = {}
        scores = {}
        
        # Calculate scores for each model
        model_list = wtmad_df_GMTKN55['Model'].values.tolist()
        
        for model in model_list:
            scores[model] = 0
            
            for benchmark in wtmad_df_GMTKN55.columns[1:]: # first col model
                mae = wtmad_df_GMTKN55.loc[wtmad_df_GMTKN55['Model'] == model, benchmark].values[0]
                scores[model] += mae     
                
               
            
            
            scores[model] = scores[model] / (len(wtmad_df_GMTKN55.columns[1:]))
            
        return pd.DataFrame.from_dict(scores, orient='index', columns=['Avg MAE \u2193']).reset_index().rename(columns={'index': 'Model'})
