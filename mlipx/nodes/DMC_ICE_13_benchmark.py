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
from mlipx.dash_utils import dash_table_interactive
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



class DMCICE13Benchmark(zntrack.Node):
    """Benchmark model against DMC-ICE13
    """
    
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    
    lattice_e_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lattice_e_ice.csv")
    mae_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "mae_ice.json")
    ref_ice_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "ref_ice.csv")
    

    def run(self):
        calc = self.model.get_calculator()
        ev_to_kjmol = 96.485
        
        # download DMC-ICE13 dataset
        dmc_ice_dir = get_benchmark_data("dmc-ice13-main.zip") / "dmc-ice13-main/INPUT/VASP"
        
        
        with open(dmc_ice_dir / "../../ice_polymorph_ref_PBE_D3.json", "r") as f:
            ice_ref = json.load(f)
    

        polymorphs = [p.name for p in dmc_ice_dir.iterdir() if p.is_dir() and p.name != "water"]
        
        water = read(dmc_ice_dir / "water/POSCAR", '0')

        def get_lattice_energy(model, sol, mol, nmol):
            sol.calc = model
            mol.calc = model
            return sol.get_potential_energy() / nmol - mol.get_potential_energy()


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
    
            
        with open(self.lattice_e_ice_output, "w") as f:
            ice_lattice_e_df.to_csv(f, index=True)
        with open(self.mae_ice_output, "w") as f:
            json.dump(mae_ice, f)
        with open(self.ref_ice_output, "w") as f:
            ref_df.to_csv(f, index=True)
        
            
        
    @property
    def get_lattice_e(self):
        """Lattice energy ICE"""
        return pd.read_csv(self.lattice_e_ice_output).set_index("Polymorph")
    @property
    def get_mae(self):
        """Mean absolute error ICE"""
        with open(self.mae_ice_output, "r") as f:
            return json.load(f)
    @property
    def get_ref(self):
        """Reference ICE lattice energy"""
        return pd.read_csv(self.ref_ice_output).set_index("Polymorph")
    
    
    


    @staticmethod
    def mae_plot_interactive(
        node_dict: dict[str, "DMCICE13Benchmark"],
        ui: str | None = None,
        run_interactive: bool = True,
        normalise_to_model: t.Optional[str] = None,

    ):
        from mlipx.dash_utils import run_app

        mae_dict = {}
        rel_ih_dict = {}
        rel_poly_mae_dict = {}
        lattice_e_all_df = None
        ref_df = None
        rel_poly_dfs = {}

        for model_name, node in node_dict.items():
            lat_df = node.get_lattice_e[[model_name]]
            ref = node.get_ref

            if lattice_e_all_df is None:
                lattice_e_all_df = node.get_ref
                lattice_e_all_df = lattice_e_all_df.merge(node.get_lattice_e[model_name], on="Polymorph")
            
            else:
                lattice_e_all_df = lattice_e_all_df.merge(node.get_lattice_e[model_name], on="Polymorph")


            mae_dict[model_name] = node.get_mae

            # Relative to Ih
            rel_model = lat_df - lat_df.loc["Ih"]
            rel_ref = ref - ref.loc["Ih"]
            rel_model = rel_model[rel_model.index != "Ih"]
            rel_ref = rel_ref[rel_ref.index != "Ih"]

            rel_ih_dict[model_name] = abs(rel_model[model_name] - rel_ref["Ref"]).mean()

            # All relative plots
            per_poly_maes = []
            for poly in lat_df.index:
                rel_model = lat_df - lat_df.loc[poly]
                rel_ref = ref - ref.loc[poly]
                rel_model = rel_model[rel_model.index != poly]
                rel_ref = rel_ref[rel_ref.index != poly]

                if poly not in rel_poly_dfs:
                    rel_poly_dfs[poly] = rel_ref
                rel_poly_dfs[poly] = rel_poly_dfs[poly].merge(rel_model, left_index=True, right_index=True)

                per_poly_maes.append(abs(rel_model[model_name] - rel_ref["Ref"]).mean())
            
            rel_poly_mae_dict[model_name] = np.mean(per_poly_maes)

            if ref_df is None:
                ref_df = ref

        mae_df = (
            pd.DataFrame(mae_dict.items(), columns=["Model", "MAE (meV)"])
            .merge(pd.DataFrame(rel_poly_mae_dict.items(), columns=["Model", "Avg MAE relative to all polymorphs (meV)"]), on="Model")
        )            
        
        mae_df["Score"] = mae_df[["MAE (meV)", "Avg MAE relative to all polymorphs (meV)"]].mean(axis=1)
        
        if normalise_to_model is not None:
            mae_df["Score"] = mae_df["Score"] / mae_df[mae_df["Model"] == normalise_to_model]["Score"].values[0]

        mae_df['Rank'] = mae_df['Score'].rank(ascending=True)


        save_path = Path("benchmark_stats/molecular_crystal_benchmark/DMC-ICE13")
        save_path.mkdir(parents=True, exist_ok=True)
        mae_df.to_csv(save_path / "mae.csv", index=False)
        lattice_e_all_df.to_csv(save_path / "lattice_e.csv", index=True)

        if ui is None and run_interactive:
            return mae_df, lattice_e_all_df

        # Plotly figures
        # need to melt for custom axes labels
        
        df_abs_melt = lattice_e_all_df.reset_index().melt(id_vars="Polymorph", var_name="Model", value_name="Absolute Lattice Energy (meV)")
        fig_abs_lat = px.line(
            df_abs_melt.reset_index(), 
            x="Polymorph", 
            y="Absolute Lattice Energy (meV)",
            color="Model",
            markers=True,
            #labels={"Polymorph": "Polymorph", "Absolute Lattice Energy (meV)": "Absolute Lattice Energy"}
        )

        # Default relative figure (vs Ih)
        default_poly = "Ih" if "Ih" in rel_poly_dfs else list(rel_poly_dfs.keys())[0]
        rel_df = rel_poly_dfs[default_poly].reset_index()
        df_rel_melt = rel_df.melt(id_vars="Polymorph", var_name="Model", value_name="Relative Energy")
        fig_rel_lat = px.line(
            df_rel_melt, 
            x="Polymorph", 
            y="Relative Energy", 
            color="Model", 
            markers=True
        )

        app = dash.Dash(__name__)
        tabs = DMCICE13Benchmark.create_tabs_from_figures(
            tab1_label="Absolute Lattice Energies",
            tab1_fig=fig_abs_lat,
            tab2_label="Relative Lattice Energies",
            tab2_fig=fig_rel_lat,
            tab2_extra_content=[
                html.Label("Select reference polymorph:"),
                dcc.Dropdown(
                    id="poly-dropdown",
                    options=[{"label": poly, "value": poly} for poly in rel_poly_dfs],
                    value=default_poly,
                    style={"width": "300px", "marginBottom": "20px"}
                ),
            ]
        )


        app.layout = html.Div([
            dash_table_interactive(
                df=mae_df.round(3),
                id="dmc-ice-mae-table",
                info= "This table is not interactive.",
                title="DMC-ICE-13 Dataset: MAE Table (meV)",
            ),

            tabs,
            
        ], style={"backgroundColor": "white", "padding": "20px"})

        # Register callbacks
        DMCICE13Benchmark.register_callbacks(app, rel_poly_dfs)



        if not run_interactive:
            return app, mae_df, rel_poly_dfs

        return run_app(app, ui=ui)





    # -------------- helper functions -----------------

    @staticmethod
    def create_tabs_from_figures(
        tab1_label: str,
        tab1_fig,
        tab2_label: str,
        tab2_fig,
        tab1_extra_content: list | None = None,
        tab2_extra_content: list | None = None, # for the ice polymorph dropdown (or any other features)
    ) -> dcc.Tabs:
        """Return a Dash Tabs component with two figures and optional extra content in tab 2."""
        return dcc.Tabs([
            dcc.Tab(label=tab1_label, children=[
                html.H3(tab1_label),
                *(tab1_extra_content or []),
                dcc.Graph(figure=tab1_fig)
            ]),
            dcc.Tab(label=tab2_label, children=[
                html.H3(tab2_label),
                *(tab2_extra_content or []),
                dcc.Graph(id="rel-lattice-graph", figure=tab2_fig)
            ])
        ])


    @staticmethod
    def register_callbacks(app, rel_poly_dfs):
    
        @app.callback(
            Output("rel-lattice-graph", "figure"),
            Input("poly-dropdown", "value")
        )
        def update_relative_plot(selected_poly):
            rel_df = rel_poly_dfs[selected_poly].reset_index()
            melt_df = rel_df.melt(id_vars="Polymorph", var_name="Model", value_name="Relative Energy")
            return px.line(melt_df, x="Polymorph", y="Relative Energy", color="Model", markers=True,
                        title=f"Relative to {selected_poly}",
                        labels={"Polymorph": "Polymorph", "Relative Energy": "Relative Energy (meV)"})
