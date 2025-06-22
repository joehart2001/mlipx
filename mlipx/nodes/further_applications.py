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
from mlipx import MolecularDynamics



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class FutherApplications(zntrack.Node):
    """ Node to combine all molecular benchmarks
    """
    # inputs
    MD_list: List[MolecularDynamics] = zntrack.deps()


    def run(self):
        pass
        


    
    

    @staticmethod
    def benchmark_interactive(
        MD_data: List[MolecularDynamics] | Dict[str, MolecularDynamics],
        ui: str = "browser",
        full_benchmark: bool = False,
        normalise_to_model: t.Optional[str] = None,
    ):
        
        
        """ Interactive dashboard + saving plots and data for all molecular benchmarks
        """

        from mlipx.dash_utils import process_data
        
        # MD water
        MD_dict = process_data(
            MD_data,
            key_extractor=lambda node: node.name.split("_config-0_MolecularDynamics")[0],
            value_extractor=lambda node: node
        )
    
        app_MD, (mae_df_oo_MD, mae_df_oh_MD, mae_df_hh_MD), properties_dict_MD = mlipx.MolecularDynamics.mae_plot_interactive(
            node_dict=MD_dict,
            run_interactive=False,
            normalise_to_model=normalise_to_model,
        )

        benchmark_score_df = FutherApplications.benchmark_score((mae_df_oo_MD, mae_df_oh_MD, mae_df_hh_MD)).round(3)
        benchmark_score_df = benchmark_score_df.sort_values(by='Avg MAE \u2193', ascending=True)
        benchmark_score_df = benchmark_score_df.reset_index(drop=True)
        benchmark_score_df['Rank'] = benchmark_score_df['Avg MAE \u2193'].rank(ascending=True)
        
        os.makedirs("benchmark_stats/further_applications", exist_ok=True)
        benchmark_score_df.to_csv("benchmark_stats/further_applications/benchmark_score.csv", index=False)


        from mlipx.dash_utils import colour_table
        # Viridis-style colormap for Dash DataTable
        style_data_conditional = colour_table(benchmark_score_df, col_name="Avg MAE \u2193")
        
        
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
        apps_list = [app_MD]
        benchmark_table_info = f"Scores normalised to: {normalise_to_model}" if normalise_to_model else ""
        layout = combine_apps(
            benchmark_score_df=benchmark_score_df,
            benchmark_title="Further Applications",
            apps_list=apps_list,
            benchmark_table_info=benchmark_table_info,
            style_data_conditional=style_data_conditional,
        )
        app.layout = layout
        
        # app_list[0] is the main app now
        mlipx.MolecularDynamics.register_callbacks(
            app, [
                ("rdf-mae-score-table-oo", "rdf-table-details-oo", "rdf-table-last-clicked-oo", "g_r_oo"),
                ("rdf-mae-score-table-oh", "rdf-table-details-oh", "rdf-table-last-clicked-oh", "g_r_oh"),
                ("rdf-mae-score-table-hh", "rdf-table-details-hh", "rdf-table-last-clicked-hh", "g_r_hh"),
            ], mae_df_list=[mae_df_oo_MD, mae_df_oh_MD, mae_df_hh_MD], properties_dict=properties_dict_MD
        )
        
        from mlipx.dash_utils import run_app

        if full_benchmark:
            return app, benchmark_score_df, lambda app: (
                mlipx.MolecularDynamics.register_callbacks(
                    app, [
                        ("rdf-mae-score-table-oo", "rdf-table-details-oo", "rdf-table-last-clicked-oo", "g_r_oo"),
                        ("rdf-mae-score-table-oh", "rdf-table-details-oh", "rdf-table-last-clicked-oh", "g_r_oh"),
                        ("rdf-mae-score-table-hh", "rdf-table-details-hh", "rdf-table-last-clicked-hh", "g_r_hh"),
                    ], mae_df_list=[mae_df_oo_MD, mae_df_oh_MD, mae_df_hh_MD], properties_dict=properties_dict_MD
                ),
            )
        
        return run_app(app, ui=ui)      
    
    
    
    
    
    
    

    def benchmark_score(
        mae_df_MD,
    ):
        """Average MAE over all RDF properties."""

        # Unpack the tuple of MAE DataFrames
        mae_df_oo, mae_df_oh, mae_df_hh = mae_df_MD

        # Ensure all DataFrames have the same set of models
        models = mae_df_oo['Model'].tolist()

        # Initialize dictionary to collect MAEs per model
        scores = {model: [] for model in models}

        # Loop through each model and collect PBE scores from each MAE DataFrame
        for model in models:
            for df in [mae_df_oo, mae_df_oh, mae_df_hh]:
                value = df.loc[df['Model'] == model, 'Score ↓ (PBE)'].values
                if len(value) > 0:
                    scores[model].append(value[0])

        # Compute average
        import numpy as np
        avg_scores = {model: np.mean(vals) for model, vals in scores.items()}

        import pandas as pd
        return pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Avg MAE \u2193']).reset_index().rename(columns={'index': 'Model'})
