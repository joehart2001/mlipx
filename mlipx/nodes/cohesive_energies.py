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

from mlipx.benchmark_download_utils import get_benchmark_data



class CohesiveEnergies(zntrack.Node):
    """Benchmark model against X23 and DMC-ICE13
    """
    # inputs
    #lattice_energy_dir: pathlib.Path = zntrack.params()
    #dmc_ice_dir: pathlib.Path = zntrack.params()
    #ice_ref: t.Dict[str, float] = zntrack.params()
    
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    abs_error_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "abs_error.csv")
    lattice_e_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_e.csv")
    mae_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae.json")
    
    lattice_e_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_e_ice.csv")
    mae_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae_ice.json")
    ref_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "ref_ice.csv")
    

    def run(self):
        calc = self.model.get_calculator()
        ev_to_kjmol = 96.485
        
        # download X23 and DMC-ICE13 datasets
        lattice_energy_dir = get_benchmark_data("lattice_energy.zip") / "lattice_energy"
        dmc_ice_dir = get_benchmark_data("dmc-ice13-main.zip") / "dmc-ice13-main/INPUT/VASP"
        
        
        with open(dmc_ice_dir / "../../ice_polymorph_ref_PBE_D3.json", "r") as f:
            ice_ref = json.load(f)
        
        with open(lattice_energy_dir / "list", "r") as f:
            systems = f.read().splitlines()
        

        # polymorphs = [name for name in os.listdir(dmc_ice_dir)
        #             if os.path.isdir(os.path.join(dmc_ice_dir, name)) and name != "water"]
        polymorphs = [p.name for p in dmc_ice_dir.iterdir() if p.is_dir() and p.name != "water"]
        
        water = read(dmc_ice_dir / "water/POSCAR", '0')

        def get_lattice_energy(model, sol, mol, nmol):
            sol.calc = model
            mol.calc = model
            return sol.get_potential_energy() / nmol - mol.get_potential_energy()


        error_rows = []
        lattice_e_dict = {}

        for system in systems:
            mol_path = lattice_energy_dir / system / "POSCAR_molecule"
            sol_path = lattice_energy_dir / system / "POSCAR_solid"
            ref_path = lattice_energy_dir / system / "lattice_energy_DMC"
            nmol_path = lattice_energy_dir / system / "nmol"
            
            mol = read(mol_path, '0')
            sol = read(sol_path, '0')
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

        # ICE part
        ice_error_rows = []
        ice_lattice_e_dict = {}

        for polymorph in polymorphs:
            pol_path = dmc_ice_dir / polymorph / "POSCAR"
            pol = read(pol_path, '0')
            nmol = len(pol.arrays['numbers']) / 3
            ref = ice_ref[polymorph]

            lat_energy = get_lattice_energy(calc, pol, water, nmol) * 1000  # to meV
            error = abs(lat_energy - ref)

            ice_lattice_e_dict[polymorph] = {
                self.model_name: lat_energy
            }
            ice_error_rows.append({'Polymorph': polymorph, self.model_name: error})

        ice_error_data = pd.DataFrame(ice_error_rows)
        mae_ice = ice_error_data[self.model_name].mean()
        ice_lattice_e_df = pd.DataFrame.from_dict(ice_lattice_e_dict, orient='index')
        ice_lattice_e_df.index.name = "Polymorph"
        
        order = ['Ih', 'II', 'III', 'IV', 'VI', 'VII', 'VIII', 'IX', 'XI', 'XIII', 'XIV', 'XV', 'XVII']
        ice_lattice_e_df = ice_lattice_e_df.reindex(order)
        ref_df = pd.DataFrame(ice_ref, index=[0]).T.rename(columns={0: "Ref"})
        ref_df.index.name = "Polymorph"
        


            
        with open(self.abs_error_output, "w") as f:
            error_data.to_csv(f, index=False)
        with open(self.lattice_e_output, "w") as f:
            lattice_e_df.to_csv(f, index=True)
        with open(self.mae_output, "w") as f:
            json.dump(mae, f)
            
        with open(self.lattice_e_ice_output, "w") as f:
            ice_lattice_e_df.to_csv(f, index=True)
        with open(self.mae_ice_output, "w") as f:
            json.dump(mae_ice, f)
        with open(self.ref_ice_output, "w") as f:
            ref_df.to_csv(f, index=True)
        
            
        
    
    @property
    def abs_error(self):
        """Absolute error"""
        return pd.read_csv(self.abs_error_output)
    @property
    def lattice_e(self):
        """Lattice energy"""
        return pd.read_csv(self.lattice_e_output)
    @property
    def mae(self):
        """Mean absolute error"""
        with open(self.mae_output, "r") as f:
            return json.load(f)
    @property
    def lattice_e_ice(self):
        """Lattice energy ICE"""
        return pd.read_csv(self.lattice_e_ice_output).set_index("Polymorph")
    @property
    def mae_ice(self):
        """Mean absolute error ICE"""
        with open(self.mae_ice_output, "r") as f:
            return json.load(f)
    @property
    def ref_ice(self):
        """Reference ICE lattice energy"""
        return pd.read_csv(self.ref_ice_output).set_index("Polymorph")
        
        
    @staticmethod
    def mae_plot_interactive(
        node_dict, 
        ui: str | None = None,
        run_interactive: bool = True,
        ):
        
        mae_dict = {}
        abs_error_df_all = None
        lattice_e_df_all = None
        
        mae_ice_dict = {}
        mae_rel_ih_dict = {}
        mae_rel_all_dict = {}
        lattice_e_ice_df_all_models = None
        lattice_e_ice_all_polymorphs_dict_of_dfs = {}
        rel_lattice_e_ice_df_ih = pd.DataFrame()
        
        
        
        
        
        for model_name, node in node_dict.items():
            # --- X23 ---
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
            
            
            
            
            # --- DMC-ICE13 ---
            # mae
            mae_ice_dict[model_name] = node.mae_ice
            
            # lattice e
            
            if lattice_e_ice_df_all_models is None:
                lattice_e_ice_df_all_models = node.ref_ice
                lattice_e_ice_df_all_models = lattice_e_ice_df_all_models.merge(node.lattice_e_ice[model_name], on="Polymorph")
            else:
                lattice_e_ice_df_all_models = lattice_e_ice_df_all_models.merge(node.lattice_e_ice[model_name], on="Polymorph")
                
            
            # set the reference for the rel Ih lattice energy
            rel_ref_ice_df = node.ref_ice - node.ref_ice.loc['Ih']
            rel_ref_ice_df = rel_ref_ice_df[rel_ref_ice_df.index != 'Ih']
            rel_lattice_e_ice_df_ih['Ref'] = rel_ref_ice_df
            
            # Ih relative lattice energy + mae
            rel_lattice_e_ih_df = node.lattice_e_ice - node.lattice_e_ice.loc['Ih']
            rel_lattice_e_ih_df = rel_lattice_e_ih_df[rel_lattice_e_ih_df.index != 'Ih']
            rel_lattice_e_ice_df_ih[model_name] = rel_lattice_e_ih_df[model_name]
            
            mae_rel_ih_dict[model_name] = abs(rel_lattice_e_ih_df[model_name] - rel_lattice_e_ice_df_ih['Ref']).mean()

            # all relative lattice energies + avg mae
            mae_all_rel = 0
            for polymorph in node.lattice_e_ice.index:
                
                rel_lattice_e = node.lattice_e_ice - node.lattice_e_ice.loc[polymorph]
                rel_lattice_e = rel_lattice_e[rel_lattice_e.index != polymorph]

                rel_ref_df = node.ref_ice - node.ref_ice.loc[polymorph]
                rel_ref_df = rel_ref_df[rel_ref_df.index != polymorph]
                
                mae_all_rel += abs(rel_lattice_e[model_name] - rel_ref_df['Ref']).mean()
                

                if polymorph not in lattice_e_ice_all_polymorphs_dict_of_dfs:
                    lattice_e_ice_all_polymorphs_dict_of_dfs[polymorph] = rel_ref_df

                lattice_e_ice_all_polymorphs_dict_of_dfs[polymorph] = lattice_e_ice_all_polymorphs_dict_of_dfs[polymorph].merge(rel_lattice_e, on="Polymorph")


                
                
            mae_rel_all_dict[model_name] = mae_all_rel / len(node.lattice_e_ice.index)

        
        mae_df = pd.DataFrame(mae_dict.items(), columns=["Model", "MAE"])
        
        mae_ice_df = pd.DataFrame(mae_ice_dict.items(), columns=["Model", "MAE"])
        mae_rel_ih_df = pd.DataFrame(mae_rel_ih_dict.items(), columns=["Model", "MAE (relative to Ih)"])
        mae_rel_all_df = pd.DataFrame(mae_rel_all_dict.items(), columns=["Model", "Avg MAE (relative to all polymorphs)"])
        # merge
        mae_ice_df = mae_ice_df.merge(mae_rel_ih_df, on="Model")
        mae_ice_df = mae_ice_df.merge(mae_rel_all_df, on="Model")
        
        
        # save plots/csvs
        path = Path("cohesive-energy-benchmark-stats/X23")
        path.mkdir(parents=True, exist_ok=True)
        mae_df.to_csv(path/"X23-mae.csv", index=False)
        abs_error_df_all.to_csv(path/"X23-abs_error.csv", index=False)
        lattice_e_df_all.to_csv(path/"X23-lattice_e.csv", index=False)

        
        
        if ui is None and run_interactive:
            return mae_df
        
        
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
        
        
        # ICE figure
        fig_ice = px.line(
            lattice_e_ice_df_all_models,
            markers=True,
        )
        fig_ice.update_layout(
            xaxis_title="Polymorph",
            yaxis_title="Absolute Lattice Energy (meV)"
        )
            

        app.layout = html.Div(
            style={"backgroundColor": "white", "padding": "20px"},  # <- set white background here
            children=[
                html.H1("Cohesive Energy Benchmarking Dashboard", style={"color": "black"}),

                # X23
                html.H2("X23 Dataset Lattice Energy MAE (KJ/mol)", style={"color": "black", "marginTop": "20px"}),
                dash_table.DataTable(
                    data=mae_df.round(3).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in mae_df.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "minWidth": "100px", "border": "1px solid black"},
                    style_header={"backgroundColor": "lightgray", "fontWeight": "bold"},
                ),
                
                html.H2("X23 - Predicted Lattice Energies", style={"marginTop": "40px"}),
                dcc.Graph(figure=lattice_e_fig),

                html.H2("X23 - Absolute Error Comparison (vs reference)", style={"marginTop": "40px"}),
                dcc.Graph(figure=abs_error_fig),

                html.Hr(style={"marginTop": "40px", "marginBottom": "30px", "borderTop": "2px solid #bbb"}),
                
                # DMC-ICE13
                html.H2("DMC-ICE13 Dataset Lattice Energy MAE (meV)", style={"color": "black", "marginTop": "20px"}),
                dash_table.DataTable(
                    data=mae_ice_df.round(3).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in mae_ice_df.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "minWidth": "100px", "border": "1px solid black"},
                    style_header={"backgroundColor": "lightgray", "fontWeight": "bold"},
                ),
                
                html.H2("DMC-ICE13 - Absolute lattice energies", style={"marginTop": "40px"}),
                dcc.Graph(figure=fig_ice),
            
                html.H2("DMC-ICE13 - Relative Lattice Energies"),
                dcc.Dropdown(
                    id="polymorph-dropdown",
                    options=[{"label": poly, "value": poly} for poly in lattice_e_ice_all_polymorphs_dict_of_dfs],
                    value=list(lattice_e_ice_all_polymorphs_dict_of_dfs.keys())[0]
                ),
                dcc.Graph(id="relative-energy-plot")
                                
            ])
        


        @app.callback(
            Output("relative-energy-plot", "figure"),
            Input("polymorph-dropdown", "value")
        )
        
        def update_plot(selected_poly):
            df = lattice_e_ice_all_polymorphs_dict_of_dfs[selected_poly].reset_index()
            df_melt = df.melt(id_vars="Polymorph", var_name="Model", value_name="Relative Energy")

            fig = px.line(df_melt, x="Polymorph", y="Relative Energy", color="Model", markers=True,
                        title=f"Relative to {selected_poly}")
            fig.update_layout(xaxis_title="Polymorph", yaxis_title="Relative Lattice Energy (meV)")
            return fig




        from mlipx.dash_utils import run_app

        if not run_interactive:
            return app, mae_df

        return run_app(app, ui=ui)