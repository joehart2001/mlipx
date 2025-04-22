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
from ase.build import bulk
from ase.phonons import Phonons
from ase.dft.kpoints import bandpath
from ase.optimize import LBFGS
from dataclasses import field

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


from mlipx.phonons_utils import get_fc2_and_freqs, init_phonopy, load_phonopy, get_chemical_formula
from phonopy.structure.atoms import PhonopyAtoms
from seekpath import get_path
import zntrack.node
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class CohesiveEnergies(zntrack.Node):
    """Benchmark model against X23 and DMC-ICE13
    """
    # inputs
    lattice_energy_dir: pathlib.Path = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    abs_error_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "abs_error.csv")
    lattice_e_ouptut: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_e.csv")
    mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae.json")
    



    def run(self):
        
        calc = self.model.get_calculator()
        
        with open(self.lattice_energy_dir + "/list", "r") as f:
            systems = f.read().splitlines()
        
        
        error_data = pd.DataFrame(columns=['System'] + [self.model_name])
        ev_to_kjmol = 96.485
        
        
        lattice_e_dict = {}

        for system in systems:
            # Read molecule and solid structures
            mol = read(f"{self.lattice_energy_dir}/{system}/POSCAR_molecule", '0')
            sol = read(f"{self.lattice_energy_dir}/{system}/POSCAR_solid", '0')
            # Reference lattice energy
            ref = np.loadtxt(f"{self.lattice_energy_dir}/{system}/lattice_energy_DMC")
            nmol = np.loadtxt(f"{self.lattice_energy_dir}/{system}/nmol")
            
            lattice_e_dict[system] = {}
            lattice_e_dict[system]['ref'] = ref[0]
            
            
            def get_lattice_energy(model, sol, mol, nmol):
                # Assign calculator to structures
                sol.calc = model
                mol.calc = model
                # Compute energies
                energy_solid = sol.get_potential_energy()
                energy_molecule = mol.get_potential_energy()
                return energy_solid / nmol - energy_molecule
            
            
            system_errors = {'System': system}

            lat_energy = get_lattice_energy(calc, sol, mol, nmol)
            lat_energy_kjmol = lat_energy * ev_to_kjmol  # Convert to kJ/mol
            # absolute error
            error = abs(lat_energy_kjmol - ref[0])
            system_errors[self.model_name] = error
            
            # lattice energy
            lattice_e_dict[system][self.model_name] = lat_energy_kjmol
            
            # absolute error for each system
            error_data = pd.concat([error_data, pd.DataFrame([system_errors])], ignore_index=True)
            
            # mae for the model model
            mae = error_data[self.model_name].mean()
        
        lattice_e_df = pd.DataFrame.from_dict(lattice_e_dict, orient='index')
        lattice_e_df.index.name = "System"
            
        with open(self.abs_error_output, "w") as f:
            error_data.to_csv(f, index=False)
        with open(self.lattice_e_ouptut, "w") as f:
            lattice_e_df.to_csv(f, index=True)
        with open(self.mae_output, "w") as f:
            json.dump(mae, f)
            
        
    
    @property
    def abs_error(self):
        """Absolute error"""
        return pd.read_csv(self.abs_error_output)
    @property
    def lattice_e(self):
        """Lattice energy"""
        return pd.read_csv(self.lattice_e_ouptut)
    @property
    def mae(self):
        """Mean absolute error"""
        with open(self.mae_output, "r") as f:
            return json.load(f)
        
        
        
    @staticmethod
    def mae_plot_interactive(node_dict, ui = None):
        
        mae_dict = {}
        abs_error_df_all = None
        lattice_e_df_all = None
        
        for model_name, node in node_dict.items():
            # mae
            mae_dict[model_name] = node.mae
        
            # absolute error
            abs_error_df = node.abs_error
            if abs_error_df_all is None:
                abs_error_df_all = abs_error_df
            else:
                abs_error_df_all = abs_error_df_all.merge(abs_error_df, on="System")
                
            # lattice energy
            lattice_e_df = node.lattice_e
            if lattice_e_df_all is None:
                lattice_e_df_all = lattice_e_df
            else:
                lattice_e_df_all = lattice_e_df_all.merge(lattice_e_df.drop(columns=["ref"]), on="System")

        mae_df = pd.DataFrame(mae_dict.items(), columns=["Model", "MAE"])
        
        # save plots/csvs
        path = Path("cohesive-energy-benchmark-stats/X23")
        path.mkdir(parents=True, exist_ok=True)
        mae_df.to_csv(path/"X23-mae.csv", index=False)
        abs_error_df_all.to_csv(path/"X23-abs_error.csv", index=False)
        lattice_e_df_all.to_csv(path/"X23-lattice_e.csv", index=False)

        
        
        if ui is None:
            return
        
        
        # --- Dash app ---
        app = dash.Dash(__name__)
        app.title = "Cohesive Energy Dashboard"
        
        # Melt it to long format (needed for hover labels)
        df_long_abs = abs_error_df_all.melt(id_vars="System", var_name="Model", value_name="Absolute Error")
        df_long_lat_e = lattice_e_df_all.melt(id_vars="System", var_name="Model", value_name="Energy")
        all_models = sorted(set(df_long_abs["Model"]).union(df_long_lat_e["Model"]))
        color_sequence = px.colors.qualitative.Plotly
        model_colors = {model: color_sequence[i % len(color_sequence)] for i, model in enumerate(all_models)}

        # Absolute error figure
        

        # Create the figure
        abs_error_fig = px.line(
            df_long_abs,
            x="System",
            y="Absolute Error",
            color="Model",
            markers=True,
            custom_data=["Model", "System", "Absolute Error"],
            color_discrete_map=model_colors,
        )

        # Update layout
        abs_error_fig.update_layout(
            legend_title_text="Models",
            xaxis_title="System",
            yaxis_title="Absolute Error (KJ/mol)"
        )

        abs_error_fig.update_traces(
            hovertemplate="Model: %{customdata[0]}<br>System: %{customdata[1]}<br>Abs Error = %{customdata[2]:.3f} kJ/mol<extra></extra>"
        )
        
        for trace in abs_error_fig.data:
            trace.line.dash = "dash"
            
        # Predicted energy figure

        lattice_e_fig = px.line(
            df_long_lat_e,
            x="System",
            y="Energy",
            color="Model",
            markers=True,
            custom_data=["Model", "System", "Energy"],
            color_discrete_map=model_colors,
        )
        
        for trace in lattice_e_fig.data:
            model = trace.name
            if model == "ref":
                trace.line.dash = "solid"
            else:
                trace.line.dash = "dash"

        lattice_e_fig.update_layout(
            legend_title_text="Models",
            xaxis_title="System",
            yaxis_title="Lattice Energy (KJ/mol)"
        )

        lattice_e_fig.update_traces(
            hovertemplate="Model = %{customdata[0]}<br>System = %{customdata[1]}<br>Lattice E = %{customdata[2]:.3f} kJ/mol<extra></extra>"
        )
            

        app.layout = html.Div(
            style={"backgroundColor": "white", "padding": "20px"},  # <- set white background here
            children=[
                html.H1("Cohesive Energy Benchmarking Dashboard", style={"color": "black"}),

                html.H2("X23 Dataset Lattice Energy MAE (KJ/mol)", style={"color": "black", "marginTop": "20px"}),
                dash_table.DataTable(
                    data=mae_df.round(3).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in mae_df.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "minWidth": "100px", "border": "1px solid black"},
                    style_header={"backgroundColor": "lightgray", "fontWeight": "bold"},
                ),
                
                html.H2("Predicted Lattice Energies", style={"marginTop": "40px"}),
                dcc.Graph(figure=lattice_e_fig),

                html.H2("Absolute Error Comparison (vs reference)", style={"marginTop": "40px"}),
                dcc.Graph(figure=abs_error_fig),

            ]
)






        def get_free_port():
            """Find an unused local port."""
            s = socket.socket()
            s.bind(('', 0))  # let OS pick a free port
            port = s.getsockname()[1]
            s.close()
            return port


        def run_app(app, ui):
            port = get_free_port()
            url = f"http://localhost:{port}"

            def _run_server():
                app.run(debug=True, use_reloader=False, port=port)
                
            if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
                import threading
                print(f"\n Detected SSH session — skipping browser launch.")
                #threading.Thread(target=_run_server, daemon=True).start()
                return
            elif ui == "browser":
                import webbrowser
                import threading
                #threading.Thread(target=_run_server, daemon=True).start()
                time.sleep(1.5)
                _run_server()
                #webbrowser.open(url)
            elif ui == "notebook":
                _run_server()
            
            else:
                print(f"Unknown UI option: {ui}. Please use, 'browser', or 'notebook'.")
                return

            print(f"Dash app running at {url}")
            
        return run_app(app, ui=ui)