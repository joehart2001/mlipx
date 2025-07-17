import pathlib
import typing as t
import json

import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import zntrack
from ase import Atoms, units
from ase.build import bulk, surface
from ase.io import write, read
import subprocess

import warnings
from pathlib import Path
from typing import Any, Callable
from ase.calculators.calculator import Calculator
from tqdm import tqdm
from phonopy.api_phonopy import Phonopy
from mlipx.abc import ComparisonResults, NodeWithCalculator
from mlipx.dash_utils import dash_table_interactive
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
from mlipx import OC157Benchmark, S24Benchmark
import os
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
from copy import deepcopy



class SlabExtensivityBenchmark(zntrack.Node):
    """ Benchmark comparing: E_slab1
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    slab_energy_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "slab_energy_extensivity.csv")
    
    
    def run(self):

        calc = self.model.get_calculator()
        
        
        # ------ setup -------
        
        sym1, sym2   = "Al", "Ni"         # element of slab-1 and slab-2
        layers       = 8
        size_xy      = (4, 4)
        vacuum_z     = 100.0               # Å vacuum on isolated slab
        gap          = 100.0               # Å gap between slabs in combined cell
        
        
        
        def make_slab(symbol):
            slab = surface(bulk(symbol, "fcc"), (1, 1, 1), layers, vacuum=vacuum_z)
            slab = slab.repeat((size_xy[0], size_xy[1], 1))
            slab.center(axis=2, vacuum=vacuum_z)
            slab.pbc = True
            return slab

        def energy(calc, atoms):
            """Return potential energy on a deep copy; retry without PBC if needed."""
            a = atoms
            a.calc = calc
            
            try:
                return a.get_potential_energy()
            except Exception as err:
                warnings.warn(f"{type(calc).__name__}: {err} → retry with pbc=False")
                a.pbc = False
                a.calc = calc
                return a.get_potential_energy()
            


        # -------- build slabs --------
        slab1 = make_slab(sym1)
        slab2 = make_slab(sym2);  slab2.translate([0, 0, gap])

        combined = slab1 + slab2
        tall_cell = slab1.cell.copy();  tall_cell[2, 2] += gap
        combined.set_cell(tall_cell);  combined.pbc = True

        # --- put isolated slabs in the same tall cell ---
        slab1_big = slab1.copy();  slab1_big.set_cell(tall_cell, scale_atoms=False)
        slab2_big = slab2.copy();  slab2_big.set_cell(tall_cell, scale_atoms=False)

        # write xyz files (skip silently on read-only FS)
        # for fn, at in [("slab1.xyz", slab1_big), ("slab2.xyz", slab2_big), ("combined.xyz", combined)]:
        #     try: write(fn, at, format="extxyz")
        #     except OSError: pass

       
        e1  = energy(calc, slab1_big)
        e2  = energy(calc, slab2_big)
        e12 = energy(calc, combined)
        delta_meV = (e12 - (e1 + e2)) * 1_000

        # Prepare dataframe and write to CSV
        df = pd.DataFrame([{
            "Model": self.model_name,
            "E1 (eV)": e1,
            "E2 (eV)": e2,
            "E12 (eV)": e12,
            "Delta (meV)": delta_meV
        }])
        df.to_csv(self.slab_energy_output, index=False)



    @property
    def get_results(self):
        """Load results from CSV file."""
        return pd.read_csv(self.slab_energy_output)

    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "SlabExtensivityBenchmark"],
        cache_dir: str = "app_cache/physicality_benchmark/slab_extensivity_cache/",
        normalise_to_model: Optional[str] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        df_list = []
        for model_name, node in node_dict.items():
            df = node.get_results.copy()
            df["Model"] = model_name  # ensure model name column
            df_list.append(df)

        full_df = pd.concat(df_list, ignore_index=True)

        full_df["Score"] = full_df["Delta (meV)"].abs()

        if normalise_to_model:
            norm_value = full_df.loc[full_df["Model"] == normalise_to_model, "Score"].values[0]
            full_df["Score"] /= norm_value

        full_df["Rank"] = full_df["Score"].rank(ascending=True, method="min")

        full_df.to_pickle(os.path.join(cache_dir, "results_df.pkl"))



    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/physicality_benchmark/slab_extensivity_cache/",
        app: dash.Dash | None = None,
        ui=None,
    ):
        from mlipx.dash_utils import run_app

        results_df = pd.read_pickle(os.path.join(cache_dir, "results_df.pkl"))

        layout = SlabExtensivityBenchmark.build_layout(results_df)

        if app is None:
            app = dash.Dash(__name__)
            app.layout = layout
            return run_app(app, ui=ui)
        else:
            return layout, lambda _: None

    @staticmethod
    def build_layout(results_df):
        return html.Div([
            dash_table_interactive(
                df=results_df.round(3),
                id="slab-extensivity-table",
                benchmark_info="Evaluates extensivity of slab energies by comparing E1 + E2 vs E12.",
                title="Slab Energy Extensivity Benchmark",
                tooltip_header={
                    "Model": "Name of the MLIP model",
                    "E1 (eV)": "Energy of isolated slab 1",
                    "E2 (eV)": "Energy of isolated slab 2",
                    "E12 (eV)": "Energy of both slabs in a single box",
                    "Delta (meV)": "Deviation from extensivity: E12 - (E1 + E2)",
                    "Score": "Absolute value of Delta (meV); normalized if specified",
                    "Rank": "Ranking of model by Score (lower is better)"
                }
            )
        ])
        
        
    @staticmethod
    def register_callbacks(app: dash.Dash, results_df: pd.DataFrame):
        pass