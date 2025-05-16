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
# import mlipx
import tempfile


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

from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mlipx.benchmark_download_utils import get_benchmark_data



class X23Benchmark(zntrack.Node):
    """Benchmark model against X23
    """
    
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    abs_error_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "abs_error.csv")
    lattice_e_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_e.csv")
    mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae.json")
    

    def run(self):
        calc = self.model.get_calculator()
        ev_to_kjmol = 96.485
        
        # download X23 dataset
        lattice_energy_dir = get_benchmark_data("lattice_energy.zip") / "lattice_energy"
        
        with open(lattice_energy_dir / "list", "r") as f:
            systems = f.read().splitlines()
        

        def get_lattice_energy(model, sol, mol, nmol):
            sol.calc = model
            mol.calc = model
            return sol.get_potential_energy() / nmol - mol.get_potential_energy()


        error_rows = []
        lattice_e_dict = {}
        
        def safe_read_poscar(path: Path):
            """Read a POSCAR file safely, stripping any extra lines to prevent velocity parsing errors."""
            try:
                return read(path, index=0, format='vasp')
            except Exception:
                # Strip after atomic coordinates
                with open(path) as f:
                    lines = f.readlines()

                # Detect the number of atoms
                atom_counts_line = lines[6].split()
                total_atoms = sum(int(x) for x in atom_counts_line)
                coord_start_line = 8
                coord_end_line = coord_start_line + total_atoms

                trimmed_lines = lines[:coord_end_line]

                # Write to temp file and load
                with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
                    tmp.writelines(trimmed_lines)
                    tmp_path = tmp.name

                return read(tmp_path, index=0, format='vasp')


        for system in systems:
            mol_path = lattice_energy_dir / system / "POSCAR_molecule"
            sol_path = lattice_energy_dir / system / "POSCAR_solid"
            ref_path = lattice_energy_dir / system / "lattice_energy_DMC"
            nmol_path = lattice_energy_dir / system / "nmol"
            
            mol = safe_read_poscar(mol_path)
            sol = safe_read_poscar(sol_path)
            # pol = safe_read_poscar(pol_path)
            # water = safe_read_poscar(water_path)

            
            # mol = read(mol_path, '0', format='vasp', struct_fmt="poscar")
            # sol = read(sol_path, '0', format='vasp', struct_fmt="poscar")
            ref = np.loadtxt(ref_path)[0]
            nmol = np.loadtxt(nmol_path)

            lat_energy = get_lattice_energy(calc, sol, mol, nmol) * ev_to_kjmol
            error = abs(lat_energy - ref)

            lattice_e_dict[system] = {
                'ref': ref,
                self.model_name: lat_energy
            }
            error_rows.append({'System': system, self.model_name: error})

        error_data = pd.DataFrame(error_rows)
        mae = error_data[self.model_name].mean()
        lattice_e_df = pd.DataFrame.from_dict(lattice_e_dict, orient='index')
        lattice_e_df.index.name = "System"

            
        with open(self.abs_error_output, "w") as f:
            error_data.to_csv(f, index=False)
        with open(self.lattice_e_output, "w") as f:
            lattice_e_df.to_csv(f, index=True)
        with open(self.mae_output, "w") as f:
            json.dump(mae, f)
            
            
    
    @property
    def get_abs_error(self):
        """Absolute error"""
        return pd.read_csv(self.abs_error_output)
    @property
    def get_lattice_e(self):
        """Lattice energy"""
        return pd.read_csv(self.lattice_e_output)
    @property
    def get_mae(self):
        """Mean absolute error"""
        with open(self.mae_output, "r") as f:
            return json.load(f)



    @staticmethod
    def mae_plot_interactive(
        node_dict: dict[str, "X23Benchmark"],
        ui: str | None = None,
        run_interactive: bool = True,
        normalise_to_model: t.Optional[str] = None,

    ):
        from mlipx.dash_utils import run_app, dash_table_interactive
        import plotly.express as px
        from dash import Dash, dcc, html

        # Collect data
        mae_dict = {}
        abs_error_df_all = None
        lattice_e_df_all = None

        for model_name, node in node_dict.items():
            mae_dict[model_name] = node.get_mae
            abs_df = node.get_abs_error
            lat_df = node.get_lattice_e

            if abs_error_df_all is None:
                abs_error_df_all = abs_df
            else:
                abs_error_df_all = abs_error_df_all.merge(abs_df, on="System")

            if lattice_e_df_all is None:
                lattice_e_df_all = lat_df
            else:
                lattice_e_df_all = lattice_e_df_all.merge(lat_df.drop(columns=["ref"]), on="System")

        mae_df = pd.DataFrame(mae_dict.items(), columns=["Model", "MAE (kJ/mol)"])
        
        if normalise_to_model is not None:
            for model_name in node_dict.keys():
                mae_df['Score'] = mae_df['MAE (kJ/mol)'] / mae_df.loc[mae_df['Model'] == normalise_to_model, 'MAE (kJ/mol)'].values[0]
                
        else:
            mae_df['Score'] = mae_df['MAE (kJ/mol)']
        
        mae_df['Rank'] = mae_df['Score'].rank(ascending=True)
        
        save_path = Path("benchmark_stats/molecular_crystal_benchmark/X23")
        save_path.mkdir(parents=True, exist_ok=True)
        mae_df.to_csv("benchmark_stats/molecular_crystal_benchmark/X23/mae.csv", index=False)
        abs_error_df_all.to_csv("benchmark_stats/molecular_crystal_benchmark/X23/abs_error.csv", index=False)
        lattice_e_df_all.to_csv("benchmark_stats/molecular_crystal_benchmark/X23/lattice_e.csv", index=False)


        if ui is None and run_interactive:
            return mae_df, abs_error_df_all, lattice_e_df_all

        # Plotly interactive app
        df_abs_long = abs_error_df_all.melt(id_vars="System", var_name="Model", value_name="Absolute Error")
        df_lat_long = lattice_e_df_all.melt(id_vars="System", var_name="Model", value_name="Energy")

        fig_abs = px.line(df_abs_long, x="System", y="Absolute Error", color="Model", markers=True)
        fig_lat = px.line(df_lat_long, x="System", y="Energy", color="Model", markers=True)

        app = Dash(__name__)
        
        tabs = X23Benchmark.create_tabs_from_figures(
            tab1_label = "Predicted Lattice Energies", 
            tab1_fig = fig_lat, 
            tab2_label = "Absolute Errors", 
            tab2_fig = fig_abs
        )

        app.layout = html.Div([
            html.H1("X23 Dataset"),

            html.H2("MAE (kJ/mol)"),
            dash_table_interactive(
                df=mae_df.round(3),
                id="x23-mae-table",
                title="MAE (kJ/mol)",
            ),

            tabs,

        ], style={"backgroundColor": "white", "padding": "20px"})

        if not run_interactive:
            return app, mae_df #, md_path

        return run_app(app, ui=ui)



    def create_tabs_from_figures(
        tab1_label: str,
        tab1_fig,
        tab2_label: str,
        tab2_fig,
    ) -> dcc.Tabs:
        """Return a Dash Tabs component with two figures."""
        return dcc.Tabs([
            dcc.Tab(label=tab1_label, children=[
                html.H3(tab1_label),
                dcc.Graph(figure=tab1_fig)
            ]),
            dcc.Tab(label=tab2_label, children=[
                html.H3(tab2_label),
                dcc.Graph(figure=tab2_fig)
            ])
        ])



