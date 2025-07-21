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
from mlipx import MolecularCrystalBenchmark, BulkCrystalBenchmark, MolecularBenchmark, FurtherApplications
from mlipx import NEBFurtherApplications, SurfaceBenchmark, SupramolecularComplexBenchmark, PhysicalityBenchmark, MOFBenchmark
from mlipx import OC157Benchmark, S24Benchmark, S30LBenchmark, LNCI16Benchmark, ProteinLigandBenchmark
from mlipx import PhononDispersion, Elasticity, LatticeConstant, X23Benchmark, DMCICE13Benchmark, GMTKN55Benchmark, HomonuclearDiatomics
from mlipx import PhononAllRef, PhononAllBatch, MolecularDynamics, NEB2, Wiggle150, SlabExtensivityBenchmark, QMOFBenchmark


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
    "Molecular Score": ["GMTKN55 Benchmark", "Homonuclear Diatomics", "Wiggle150"],
    "Water MD": ["Water RDFs"],
    "NEBs": ["LiFePO4 b and c paths"],
    "Surfaces": ["OC157 Benchmark", "S24 Benchmark"],
    "Supramolecular Complexes": ["S30L Benchmark", "LNCI16 Benchmark", "Protein-Ligand Benchmark"],
    "Physicality": ["Ghost Atom Benchmark", "Slab Extensivity Benchmark"],
    "MOF": ["QMOF Benchmark"],
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
    further_apps_benchmark: List[FurtherApplications] = zntrack.deps()
    neb_further_apps_benchmark: List[NEBFurtherApplications] = zntrack.deps()
    surface_benchmark: List[SurfaceBenchmark] = zntrack.deps()
    supramolecular_complex_benchmark: List[SupramolecularComplexBenchmark] = zntrack.deps()
    physicality_benchmark: List[PhysicalityBenchmark] = zntrack.deps()
    mof_benchmark: List[MOFBenchmark] = zntrack.deps()
    
    
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
        Wiggle150_data: List[Wiggle150] | Dict[str, Wiggle150],
        MD_data: List[MolecularDynamics] | Dict[str, MolecularDynamics],
        NEB_data: List[NEB2] | Dict[str, NEB2],
        OC157_data: List[OC157Benchmark] | Dict[str, OC157Benchmark],
        S24_data: List[S24Benchmark] | Dict[str, S24Benchmark],
        S30L_data: List[S30LBenchmark] | Dict[str, S30LBenchmark],
        LNCI16_data: List[LNCI16Benchmark] | Dict[str, LNCI16Benchmark],
        protein_ligand_data: List[mlipx.ProteinLigandBenchmark] | Dict[str, mlipx.ProteinLigandBenchmark],
        ghost_atom_data: List[mlipx.GhostAtomBenchmark] | Dict[str, mlipx.GhostAtomBenchmark],
        slab_extensivity_data: List[SlabExtensivityBenchmark] | Dict[str, SlabExtensivityBenchmark],
        QMOF_data: List[QMOFBenchmark] | Dict[str, QMOFBenchmark],
        report: bool = False,
        normalise_to_model: Optional[str] = None,
    ):
        # Create directory
        cache_dir = Path("app_cache/")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        n_categories = 9

        # print(f"Precomputing Bulk Crystal Benchmark (1/{n_categories})...")
        # BulkCrystalBenchmark.benchmark_precompute(
        #     elasticity_data=elasticity_data,
        #     lattice_const_data=lattice_const_data,
        #     lattice_const_ref_node_dict=lattice_const_ref_node_dict,
        #     phonon_ref_data=phonon_ref_data,
        #     phonon_pred_data=phonon_pred_data,
        #     cache_dir=str(cache_dir / "bulk_crystal_benchmark"),
        #     report=report,
        #     normalise_to_model=normalise_to_model,
        # )

        # print(f"Precomputing Molecular Crystal Benchmark (2/{n_categories})...")
        # MolecularCrystalBenchmark.benchmark_precompute(
        #     X23_data=X23_data,
        #     DMC_ICE_data=DMC_ICE_data,
        #     cache_dir=str(cache_dir / "molecular_crystal_benchmark"),
        #     report=report,
        #     normalise_to_model=normalise_to_model,
        # )

        # print(f"Precomputing Molecular Benchmark (3/{n_categories})...")
        # MolecularBenchmark.benchmark_precompute(
        #     GMTKN55_data=GMTKN55_data,
        #     HD_data=HD_data,
        #     Wiggle150_data=Wiggle150_data,
        #     cache_dir=str(cache_dir / "molecular_benchmark"),
        #     report=report,
        #     normalise_to_model=normalise_to_model,
        # )
        
        # print(f"Precomputing Water MD Benchmark (4/{n_categories})...")
        # FurtherApplications.benchmark_precompute(
        #     MD_data=MD_data,
        #     cache_dir=str(cache_dir / "further_applications_benchmark"),
        #     report=report,
        #     normalise_to_model=normalise_to_model,
        # )
        
        # print(f"Precomputing NEB Benchmark (5/{n_categories})...")
        # NEBFurtherApplications.benchmark_precompute(
        #     neb_data=NEB_data,
        #     cache_dir=str(cache_dir / "nebs_further_apps"),
        #     report=report,
        #     normalise_to_model=normalise_to_model,
        # )
        
        print(f"Precomputing Surface Benchmark (6/{n_categories})...")
        SurfaceBenchmark.benchmark_precompute(
            OC157_data=OC157_data,
            S24_data=S24_data,
            cache_dir=str(cache_dir / "surface_benchmark"),
            normalise_to_model=normalise_to_model,
        )
        
        print(f"Precomputing Supramolecular Complex Benchmark (7/{n_categories})...")
        SupramolecularComplexBenchmark.benchmark_precompute(
            S30L_data=S30L_data,
            LNCI16_data=LNCI16_data,
            protein_ligand_data=protein_ligand_data,
            cache_dir=str(cache_dir / "supramolecular_complexes"),
            normalise_to_model=normalise_to_model,
        )
        
        print(f"Precomputing Physicality Benchmark (8/{n_categories})...")
        PhysicalityBenchmark.benchmark_precompute(
            ghost_atom_data=ghost_atom_data,
            slab_extensivity_data=slab_extensivity_data,
            cache_dir=str(cache_dir / "physicality_benchmark"),
            normalise_to_model=normalise_to_model,
        )
        
        print(f"Precomputing MOF Benchmark (9/{n_categories})...")
        MOFBenchmark.benchmark_precompute(
            QMOF_data=QMOF_data,
            cache_dir=str(cache_dir / "MOF"),
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
        water_md_df = pd.read_pickle(Path(cache_dir) / "further_applications_benchmark" / "benchmark_score.pkl")
        nebs_df = pd.read_pickle(Path(cache_dir) / "nebs_further_apps" / "benchmark_score.pkl")
        surface_df = pd.read_pickle(Path(cache_dir) / "surface_benchmark" / "benchmark_score.pkl")
        supramolecular_df = pd.read_pickle(Path(cache_dir) / "supramolecular_complexes" / "benchmark_score.pkl")
        physicality_df = pd.read_pickle(Path(cache_dir) / "physicality_benchmark" / "benchmark_score.pkl")
        mof_df = pd.read_pickle(Path(cache_dir) / "MOF" / "benchmark_score.pkl")

        scores_all_df = FullBenchmark.get_overall_score_df(
            (bulk_df, "Bulk Crystal"),
            (mol_crystal_df, "Molecular Crystal"),
            (molecular_df, "Molecular"),
            (water_md_df, "Water MD"),
            (nebs_df, "NEBs"),
            (surface_df, "Surfaces"),
            (supramolecular_df, "Supramolecular Complexes"),
            (physicality_df, "Physicality"),
            (mof_df, "MOF"),
        )
        scores_all_df.to_csv(Path(cache_dir) / "overall_benchmark.csv", index=False)

        # with open(f"{cache_dir}/nebs_further_apps/nebs_cache/all_group_data.pkl", "rb") as f:
        #     all_group_data = pickle.load(f)

        
        assets_dir = os.path.abspath("assets")
        from mlipx.dash_utils import run_app
        print("Serving assets from:", assets_dir)
        app_summary = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=assets_dir)

        bulk_crystal_layout, bulk_crystal_callback_fn = BulkCrystalBenchmark.launch_dashboard(full_benchmark=True)
        mol_crystal_layout, mol_crystal_callback_fn = MolecularCrystalBenchmark.launch_dashboard(full_benchmark=True)
        mol_layout, mol_callback_fn = MolecularBenchmark.launch_dashboard(full_benchmark=True)
        further_layout, further_callback_fn = FurtherApplications.launch_dashboard(full_benchmark=True)
        neb_further_layout, neb_further_callback_fn = NEBFurtherApplications.launch_dashboard(full_benchmark=True)
        surface_layout, surface_callback_fn = SurfaceBenchmark.launch_dashboard(full_benchmark=True)
        supramolecular_layout, supramolecular_callback_fn = SupramolecularComplexBenchmark.launch_dashboard(full_benchmark=True)
        physicality_layout, physicality_callback_fn = PhysicalityBenchmark.launch_dashboard(full_benchmark=True)
        mof_layout, mof_callback_fn = MOFBenchmark.launch_dashboard(full_benchmark=True)


        # Patch layouts removed

        component_layouts = {
            "Bulk Crystal Score": bulk_crystal_layout,
            "Molecular Crystal Score": mol_crystal_layout,
            "Molecular Score": mol_layout,
            "Water MD": further_layout,
            "NEBs": neb_further_layout,
            "Surfaces": surface_layout,
            "Supramolecular Complexes": supramolecular_layout,
            "Physicality": physicality_layout,
            "MOF": mof_layout,
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


    

        # --- Keep overall summary in sync with Bulk‑Crystal weighting (cached version) ---
        @app_summary.callback(
            dash.Output("summary-table", "data"),
            dash.Output("summary-table", "style_data_conditional"),
            dash.Input("bulk-crystal-weights", "data"),
        )
        def update_summary_table(_):
            from mlipx.dash_utils import colour_table
            import pandas as pd
            weights = _
            if weights is None:
                weights = {"phonon": 1.0, "elasticity": 1.0, "lattice_const": 0.2}
            cache_dir = "app_cache/bulk_crystal_benchmark"
            phonon_mae_df = pd.read_pickle(f"{cache_dir}/phonons_cache/mae_summary.pkl")
            mae_df_elas = pd.read_pickle(f"{cache_dir}/elasticity_cache/mae_summary.pkl")
            mae_df_lattice_const = pd.read_pickle(f"{cache_dir}/lattice_cache/mae_summary.pkl")
            bulk_df = BulkCrystalBenchmark.bulk_crystal_benchmark_score(
                phonon_mae_df,
                mae_df_elas,
                mae_df_lattice_const,
                weights=weights,
            ).round(3).sort_values(by='Avg MAE \u2193').reset_index(drop=True)
            bulk_df["Rank"] = bulk_df['Avg MAE \u2193'].rank(ascending=True)

            combined_df = FullBenchmark.get_overall_score_df(
                (bulk_df, "Bulk Crystal"),
                (mol_crystal_df, "Molecular Crystal"),
                (molecular_df, "Molecular"),
                (water_md_df, "Water MD"),
                (nebs_df, "NEBs"),
                (surface_df, "Surfaces"),
                (supramolecular_df, "Supramolecular Complexes"),
                (physicality_df, "Physicality"),
                (mof_df, "MOF"),
                
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
            dcc.Store(
                id="bulk-crystal-weights", 
                storage_type="session",
                data={"phonon": 1.0, "elasticity": 1.0, "lattice_const": 0.2}
            ),
            dcc.Tabs(
                id="tabs",
                value="Overall Benchmark",
                children=[
                    dcc.Tab(label=tab, value=tab) for tab in tab_layouts
                ]
            ),
            html.Div(id="tab-content"),
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