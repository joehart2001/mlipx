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
from mlipx import MolecularCrystalBenchmark, BulkCrystalBenchmark, PhononDispersion, Elasticity, LatticeConstant, X23Benchmark, DMCICE13Benchmark, GMTKN55Benchmark, MolecularBenchmark, HomonuclearDiatomics
from mlipx import PhononAllRef, PhononAllBatch, MolecularDynamics, FutherApplications, NEBFutherApplications, NEB2



import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


BENCHMARK_STRUCTURE = {
    "Overall Benchmark": [],
    "Bulk Crystal Score": ["Phonon Dispersion", "Lattice Constants", "Elasticity"],
    "Molecular Crystal Score": ["X23 Benchmark", "DMC-ICE13 Benchmark"],
    "Molecular Score": ["GMTKN55 Benchmark", "Homonuclear Diatomics"],
    "Water MD": ["Water RDFs"],
    "NEBs": ["LiFePO4 b and c paths"]#, "Si 64 and 216"],
}

# # Mapping from benchmark label to their dash table id
# BENCHMARK_TABLE_IDS = {
#     "Elasticity": "elas-mae-table",
#     "Lattice Constants": "lat-mae-score-table",
#     "Phonon Dispersion": "phonon-mae-summary-table",
#     "X23 Benchmark": "x23-mae-table",
#     "DMC-ICE13 Benchmark": "dmc-ice-mae-table",
#     "GMTKN55 Benchmark": "GMTKN55-wtmad-table",
#     "Homonuclear Diatomics": "diatomics-stats-table",
#     "Molecular Dynamics": "rdf-mae-score-table-oo",
# }

class FullBenchmark(zntrack.Node):
    """ Node to combine all bulk crystal benchmarks
    """
    # inputs
    bulk_crystal_benchmark: List[BulkCrystalBenchmark] = zntrack.deps()
    mol_crystal_benchmark: List[MolecularCrystalBenchmark] = zntrack.deps()
    mol_benchmark: List[MolecularBenchmark] = zntrack.deps()
    further_apps_benchmark: List[FutherApplications] = zntrack.deps()
    neb_further_apps_benchmark: List[NEBFutherApplications] = zntrack.deps()
    
    # outputs
    # nwd: ZnTrack's node working directory for saving files

    
    def run(self):
        pass
        


    
    





    

    @staticmethod
    def benchmark_precompute(
        elasticity_data: List[Elasticity] | Dict[str, Elasticity],
        lattice_const_data: List[LatticeConstant] | Dict[str, Dict[str, LatticeConstant]],
        lattice_const_ref_node_dict: LatticeConstant,
        phonon_ref_data: List[PhononDispersion] | Dict[str, PhononDispersion] | PhononAllRef,
        phonon_pred_data: List[PhononDispersion] | Dict[str, Dict[str, PhononDispersion]] | List[PhononAllBatch] | Dict[str, PhononAllBatch],
        X23_data: List[X23Benchmark] | Dict[str, X23Benchmark],
        DMC_ICE_data: List[DMCICE13Benchmark] | Dict[str, DMCICE13Benchmark],
        GMTKN55_data: List[GMTKN55Benchmark] | Dict[str, GMTKN55Benchmark],
        HD_data: List[HomonuclearDiatomics] | Dict[str, HomonuclearDiatomics],
        MD_data: List[MolecularDynamics] | Dict[str, MolecularDynamics] = None,
        NEB_data: List[NEB2] | Dict[str, NEB2] = None,
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        # Create directory
        cache_dir = Path("app_cache/")
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("Precomputing Bulk Crystal Benchmark (1/4)...")
        BulkCrystalBenchmark.benchmark_precompute(
            elasticity_data=elasticity_data,
            lattice_const_data=lattice_const_data,
            lattice_const_ref_node_dict=lattice_const_ref_node_dict,
            phonon_ref_data=phonon_ref_data,
            phonon_pred_data=phonon_pred_data,
            cache_dir=str(cache_dir / "bulk_crystal_benchmark"),
            report=report,
            normalise_to_model=normalise_to_model,
        )

        print("Precomputing Molecular Crystal Benchmark (2/4)...")
        MolecularCrystalBenchmark.benchmark_precompute(
            X23_data=X23_data,
            DMC_ICE_data=DMC_ICE_data,
            cache_dir=str(cache_dir / "molecular_crystal_benchmark"),
            report=report,
            normalise_to_model=normalise_to_model,
        )

        print("Precomputing Molecular Benchmark (3/4)...")
        MolecularBenchmark.benchmark_precompute(
            GMTKN55_data=GMTKN55_data,
            HD_data=HD_data,
            cache_dir=str(cache_dir / "molecular_benchmark"),
            report=report,
            normalise_to_model=normalise_to_model,
        )
        
        print("Precomputing Further Applications Benchmark (4/4)...")
        FutherApplications.benchmark_precompute(
            MD_data=MD_data,
            cache_dir=str(cache_dir / "further_applications_benchmark"),
            report=report,
            normalise_to_model=normalise_to_model,
        )
        
        NEBFutherApplications.benchmark_precompute(
            neb_data=NEB_data,
            cache_dir=str(cache_dir / "nebs_further_apps"),
            report=report,
            normalise_to_model=normalise_to_model,
        )



    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/",
        ui="browser",
        normalise_to_model: Optional[str] = None,
        return_app: bool = False
    ):
        # Load score DataFrames
        bulk_df = pd.read_pickle(Path(cache_dir) / "bulk_crystal_benchmark" / "benchmark_score.pkl")
        mol_crystal_df = pd.read_pickle(Path(cache_dir) / "molecular_crystal_benchmark" / "benchmark_score.pkl")
        molecular_df = pd.read_pickle(Path(cache_dir) / "molecular_benchmark" / "benchmark_score.pkl")

        scores_all_df = FullBenchmark.get_overall_score_df(
            (bulk_df, "Bulk Crystal"),
            (mol_crystal_df, "Molecular Crystal"),
            (molecular_df, "Molecular"),
        )
        scores_all_df.to_csv(Path(cache_dir) / "overall_benchmark.csv", index=False)

        with open(f"{cache_dir}/nebs_further_apps/nebs_cache/all_group_data.pkl", "rb") as f:
            all_group_data = pickle.load(f)
        #assets_dir = next(iter(all_group_data.values()))[2]
        assets_dir = "assets"
        from mlipx.dash_utils import run_app
        print("Serving assets from:", assets_dir)
        app_summary = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=assets_dir)

        bulk_crystal_layout, bulk_crystal_callback_fn = BulkCrystalBenchmark.launch_dashboard(full_benchmark=True)
        mol_crystal_layout, mol_crystal_callback_fn = MolecularCrystalBenchmark.launch_dashboard(full_benchmark=True)
        mol_layout, mol_callback_fn = MolecularBenchmark.launch_dashboard(full_benchmark=True)
        # Add further applications layout and callback
        further_layout, further_callback_fn = FutherApplications.launch_dashboard(full_benchmark=True)
        neb_further_layout, neb_further_callback_fn = NEBFutherApplications.launch_dashboard(full_benchmark=True)



        # Patch layouts removed

        component_layouts = {
            "Bulk Crystal Score": bulk_crystal_layout,
            "Molecular Crystal Score": mol_crystal_layout,
            "Molecular Score": mol_layout,
            "Water MD": further_layout,
            "NEBs": neb_further_layout,
        }

        layout, tab_layouts = FullBenchmark.build_layout(scores_all_df, component_layouts, normalise_to_model)
        app_summary.layout = layout

        bulk_crystal_callback_fn(app_summary)
        mol_crystal_callback_fn(app_summary)
        mol_callback_fn(app_summary)
        further_callback_fn(app_summary)
        neb_further_callback_fn(app_summary)

        @app_summary.callback(
            dash.Output("tab-content", "children"),
            dash.Input("tabs", "value"),
        )
        
        def render_tab(tab_name):
            return tab_layouts[tab_name]

        # # --- Scroll to anchor on request (clientside) ---
        # app_summary.clientside_callback(
        #     """
        #     function(targetId) {
        #         if (!targetId) return;
        #         const el = document.getElementById(targetId);
        #         if (el) {
        #             el.scrollIntoView({ behavior: "smooth" });
        #         }
        #         return '';
        #     }
        #     """,
        #     dash.Output("scroll-anchor", "children"),
        #     dash.Input("scroll-target", "data")
        # )

        # --- Keep overall summary in sync with Bulk‑Crystal weighting (cached version) ---
        @app_summary.callback(
            dash.Output("summary-table", "data"),
            dash.Output("summary-table", "style_data_conditional"),
            dash.Input("bulk-benchmark-score-table", "data"),
        )
        def update_summary_table(bulk_data):
            """Re-calculate overall scores whenever the BulkCrystal scores change."""
            from mlipx.dash_utils import colour_table
            import pandas as pd
            if bulk_data is None:
                raise dash.exceptions.PreventUpdate

            bulk_df_current = pd.DataFrame(bulk_data)

            combined_df = FullBenchmark.get_overall_score_df(
                (bulk_df_current, "Bulk Crystal"),
                (mol_crystal_df, "Molecular Crystal"),
                (molecular_df, "Molecular"),
            )

            style_conditional = colour_table(combined_df, all_cols=True)
            return combined_df.to_dict("records"), style_conditional

        # Add callback for Table of Contents navigation (generalized)
        inputs = [dash.Input(f"toc-{name.lower().replace(' ', '-')}", "n_clicks") for name in BENCHMARK_STRUCTURE.keys()]

        @app_summary.callback(
            dash.Output("tabs", "value"),
            inputs,
            prevent_initial_call=True
        )
        def navigate_from_toc(*clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
            for name in BENCHMARK_STRUCTURE:
                if btn_id == f"toc-{name.lower().replace(' ', '-')}":
                    return name
            raise dash.exceptions.PreventUpdate

        # # --- Add callbacks for subtest jump buttons to scroll to section ---
        # jump_inputs = []
        # jump_id_map = {}
        # for cat, subtests in BENCHMARK_STRUCTURE.items():
        #     for sub in subtests:
        #         btn_id = f"jump-{cat.lower().replace(' ', '-')}-{sub.lower().replace(' ', '-')}"
        #         target_id = BENCHMARK_TABLE_IDS[sub]
        #         jump_inputs.append(dash.Input(btn_id, "n_clicks"))
        #         jump_id_map[btn_id] = target_id

        # @app_summary.callback(
        #     dash.Output("scroll-target", "data"),
        #     jump_inputs,
        #     prevent_initial_call=True
        # )
        # def scroll_to_section(*args):
        #     ctx = dash.callback_context
        #     if not ctx.triggered:
        #         raise dash.exceptions.PreventUpdate
        #     btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        #     return jump_id_map.get(btn_id, None)

        if return_app == True:
            return app_summary
        else:
            return run_app(app_summary, ui=ui)


    @staticmethod
    def build_layout(
        scores_all_df,
        component_layouts,
        normalise_to_model=None
    ):
        from mlipx.dash_utils import colour_table
        from dash import html, dcc
        from dash import dash_table

        style_data_conditional = colour_table(scores_all_df, all_cols=True)

        def make_toc_buttons():
            #BENCHMARK_STRUCTURE_categories = {tab_name: subtests for tab_name, subtests in BENCHMARK_STRUCTURE.items() if tab_name != "Overall Benchmark"}
            return html.Div([
                html.H1("Table of Contents"),
                html.Ul([
                    html.Li([
                        html.Button(tab_name, id=f"toc-{tab_name.lower().replace(' ', '-')}", n_clicks=0),
                        html.Ul([
                            html.Li(
                                html.Span(sub)
                            )
                            for sub in subtests
                        ])
                    ]) for tab_name, subtests in BENCHMARK_STRUCTURE.items()
                ])
            ])

        toc_buttons = make_toc_buttons()

        summary_layout = html.Div([
            html.H1("Overall Benchmark Scores (avg MAE)"),
            html.P("Scores are avg MAEs normalised to: " + normalise_to_model if normalise_to_model else "Scores are avg MAEs"),
            html.Div([
                dash_table.DataTable(
                    id="summary-table",
                    columns=[{"name": i, "id": i} for i in scores_all_df.columns],
                    data=scores_all_df.to_dict("records"),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'fontSize': '14px'},
                    style_header={'fontWeight': 'bold'},
                    style_data_conditional=style_data_conditional,
                )
            ]),
            toc_buttons,
        ])

        tab_layouts = {
            "Overall Benchmark": summary_layout,
            **{name: layout for name, layout in component_layouts.items()}
        }

        full_layout = html.Div([
            dcc.Tabs(
                id="tabs",
                value="Overall Benchmark",
                #persistence=True,
                #persistence_type="session",
                children=[
                    dcc.Tab(label=tab, value=tab) for tab in tab_layouts
                ]
            ),
            html.Div(id="tab-content"),
            #dcc.Store(id="scroll-target"),
            #html.Div(id="scroll-anchor")
        ], style={
            "backgroundColor": "white",
            "padding": "20px",
            "border": "2px solid black",
        })

        return full_layout, tab_layouts
    
    
    
    
    # ------------- helper functions -------------

    @staticmethod
    def get_overall_score_df(
        *dfs_with_names: t.Tuple[pd.DataFrame, str]
    ) -> pd.DataFrame:
        """Combine multiple benchmark DataFrames into an overall score DataFrame.
        """
        merged_df = None
        
        for df, name in dfs_with_names:
            df_renamed = df[["Model", "Avg MAE \u2193"]].rename(columns={"Avg MAE \u2193": name})
            if merged_df is None:
                merged_df = df_renamed
            else:
                merged_df = pd.merge(merged_df, df_renamed, on="Model", how="outer")

        # Compute average MAE across all benchmarks
        benchmark_cols = [name for _, name in dfs_with_names]
        merged_df["Overall Score \u2193"] = merged_df[benchmark_cols].mean(axis=1)

        # Sort and rank
        merged_df = merged_df.sort_values(by="Overall Score \u2193", ascending=True)
        merged_df = merged_df.reset_index(drop=True)
        merged_df['Rank'] = merged_df['Overall Score \u2193'].rank(ascending=True)

        return merged_df.round(3)