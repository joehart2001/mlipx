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
from ase.units import Bohr
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
from mlipx.benchmark_download_utils import get_benchmark_data
import logging



class LNCI16Benchmark(zntrack.Node):
    """ """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    lnci16_results_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lnci16_pred.csv")
    lnci16_mae_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lnci16_mae.json")
    lnci16_complex_atoms_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "lnci16_complex_atoms.xyz")

    def run(self):

        calc = self.model.get_calculator()

        base_dir = get_benchmark_data("LNCI16_data.zip") / "LNCI16_data/benchmark-LNCI16-main"


        LNCI16_REFERENCE_ENERGIES = {
            "BpocBenz": -6.81,
            "BpocMeOH": -6.19,
            "BNTube": -14.32,
            "GramA": -36.30,
            "DHComplex": -57.57,
            "DNA": -363.30,
            "SH3": -25.65,
            "TYK2": -72.31,
            "FXa": -70.73,
            "2xHB238": -74.92,
            "FullGraph": -74.13,
            "DithBrCap": -45.63,
            "BrCap": -21.12,
            "MolMus": -62.58,
            "Rotax": -55.89,
            "Nylon": -566.23,
        }

        # System charges from Table 1
        LNCI16_CHARGES = {
            "BpocBenz": 0,
            "BpocMeOH": 0,
            "BNTube": 0,
            "GramA": 0,
            "DHComplex": 0,
            "DNA": 0,
            "SH3": 0,
            "TYK2": +1,
            "FXa": -2,
            "2xHB238": 0,
            "FullGraph": 0,
            "DithBrCap": 0,
            "BrCap": 0,
            "MolMus": 0,
            "Rotax": 0,
            "Nylon": 0,
        }
        
        KCAL_TO_EV = 0.04336414
        EV_TO_KCAL = 1.0 / KCAL_TO_EV
        
        
        def interaction_energy(frags: Dict[str, Atoms], calc: Calculator) -> float:
            frags["complex"].calc = calc
            e_complex = frags["complex"].get_potential_energy()
            frags["host"].calc = calc
            e_host = frags["host"].get_potential_energy()
            frags["guest"].calc = calc
            e_guest = frags["guest"].get_potential_energy()
            return e_complex - e_host - e_guest

        # ------------------------------------------------------------
        # File I/O functions for LNCI16 format
        # ------------------------------------------------------------
        def read_turbomole_xyz(filepath: Path) -> Atoms:
            """Read Turbomole format xyz file (coordinates in atomic units)"""
            atoms = read(filepath, format='xyz')
            return atoms

        def read_charge_file(filepath: Path) -> float:
            """Read charge from .CHRG file"""
            if not filepath.exists():
                return 0.0
            
            try:
                with open(filepath, 'r') as f:
                    charge_str = f.read().strip()
                    charge = float(charge_str)
                    
                logging.debug(f"Read charge {charge} from {filepath}")
                return charge
            except Exception as e:
                logging.warning(f"Failed to read charge from {filepath}: {e}")
                return 0.0

        def load_lnci16_system(system_name: str) -> Dict[str, Atoms]:
            """Load complex, host, and guest structures for a LNCI16 system"""
            system_dir = base_dir / system_name
            if not system_dir.exists():
                raise FileNotFoundError(f"System directory not found: {system_dir}")
            # Load structures
            complex_atoms = read_turbomole_xyz(system_dir / "complex" / "struc.xyz")
            host_atoms = read_turbomole_xyz(system_dir / "host" / "struc.xyz")
            guest_atoms = read_turbomole_xyz(system_dir / "guest" / "struc.xyz")
            # Read charges
            complex_charge = read_charge_file(system_dir / "complex" / ".CHRG")
            host_charge = read_charge_file(system_dir / "host" / ".CHRG")
            guest_charge = read_charge_file(system_dir / "guest" / ".CHRG")
            # Set charges in atoms info
            complex_atoms.info['charge'] = complex_charge
            host_atoms.info['charge'] = host_charge
            guest_atoms.info['charge'] = guest_charge
            # Add system identifier
            complex_atoms.info['system'] = system_name
            host_atoms.info['system'] = system_name
            guest_atoms.info['system'] = system_name
            return {
                "complex": complex_atoms,
                "host": host_atoms,
                "guest": guest_atoms,
            }

        # ------------------------------------------------------------
        # Benchmarking functions
        # ------------------------------------------------------------
        def benchmark_lnci16(calc: Calculator, model_name: str) -> pd.DataFrame:
            """Benchmark LNCI16 dataset"""
            logging.info(f"Benchmarking LNCI16 with {model_name}...")
            # Check if calculator supports charges
            supports_charges = any(hasattr(calc, attr) for attr in ['set_charge', 'charge', 'total_charge_key'])
            if supports_charges:
                logging.info(f"  Calculator {model_name} supports charge handling")
            else:
                logging.warning(f"  Calculator {model_name} may not support charge handling")
            results = []
            complex_atoms_list = []
            for system_name in tqdm(LNCI16_REFERENCE_ENERGIES.keys(), desc="LNCI16"):
                try:
                    # Load system structures
                    frags = load_lnci16_system(system_name)
                    complex_atoms = frags["complex"]
                    host_atoms = frags["host"]
                    guest_atoms = frags["guest"]
                    # Log charge information for charged systems
                    if complex_atoms.info['charge'] != 0:
                        logging.info(f"  Processing charged system {system_name} "
                                f"(charge = {complex_atoms.info['charge']:+.0f})")
                    # Compute interaction energy
                    E_int_model = interaction_energy(frags, calc)
                    # Reference energy in kcal/mol, convert to eV
                    E_int_ref_kcal = LNCI16_REFERENCE_ENERGIES[system_name]
                    E_int_ref_eV = E_int_ref_kcal * KCAL_TO_EV
                    # Calculate errors
                    error_eV = E_int_model - E_int_ref_eV
                    error_kcal = error_eV * EV_TO_KCAL
                    results.append({
                        'system': system_name,
                        'E_int_ref_kcal': E_int_ref_kcal,
                        #'E_int_ref_eV': E_int_ref_eV,
                        #'E_int_model_eV': E_int_model,
                        'E_int_model_kcal': E_int_model * EV_TO_KCAL,
                        #'error_eV': error_eV,
                        'error_kcal': error_kcal,
                        #'relative_error_pct': (error_eV / E_int_ref_eV) * 100,
                        'complex_atoms': len(complex_atoms),
                        'host_atoms': len(host_atoms),
                        'guest_atoms': len(guest_atoms),
                        'complex_charge': complex_atoms.info['charge'],
                        'host_charge': host_atoms.info['charge'],
                        'guest_charge': guest_atoms.info['charge'],
                        'is_charged': complex_atoms.info['charge'] != 0,
                    })
                    complex_atoms.info['model'] = model_name
                    complex_atoms.info['E_int_model_kcal'] = E_int_model * EV_TO_KCAL
                    complex_atoms.info['E_int_ref_kcal'] = E_int_ref_kcal
                    complex_atoms.info['error_kcal'] = error_kcal
                    complex_atoms.info['system'] = system_name
                    complex_atoms.info['charged'] = complex_atoms.info['charge'] != 0
                    complex_atoms.info['complex_charge'] = complex_atoms.info['charge']
                    complex_atoms.info['host_charge'] = host_atoms.info['charge']
                    complex_atoms.info['guest_charge'] = guest_atoms.info['charge']

                    complex_atoms_list.append(complex_atoms)
                    logging.info(f"  {system_name}: E_int = {E_int_model:.6f} eV "
                                f"(ref: {E_int_ref_eV:.6f} eV, error: {error_kcal:.2f} kcal/mol)")
                except Exception as e:
                    logging.error(f"Error processing {system_name}: {e}")
                    continue
            return pd.DataFrame(results), complex_atoms_list

        def validate_lnci16_dataset():
            """Validate that all LNCI16 systems can be loaded"""
            
            logging.info("Validating LNCI16 dataset...")
            
            missing_systems = []
            loaded_systems = []
            
            for system_name in LNCI16_REFERENCE_ENERGIES.keys():
                try:
                    complex_atoms, host_atoms, guest_atoms = load_lnci16_system(system_name)
                    
                    # Basic validation
                    assert len(complex_atoms) > 0, f"Empty complex for {system_name}"
                    assert len(host_atoms) > 0, f"Empty host for {system_name}"
                    assert len(guest_atoms) > 0, f"Empty guest for {system_name}"
                    
                    # Check charge consistency (should sum to complex charge)
                    expected_charge = host_atoms.info['charge'] + guest_atoms.info['charge']
                    actual_charge = complex_atoms.info['charge']
                    reference_charge = LNCI16_CHARGES.get(system_name, 0)
                    
                    if abs(expected_charge - actual_charge) > 1e-6:
                        logging.warning(f"{system_name}: Charge inconsistency - "
                                    f"expected {expected_charge}, got {actual_charge}")
                    
                    if abs(actual_charge - reference_charge) > 1e-6:
                        logging.warning(f"{system_name}: Reference charge mismatch - "
                                    f"loaded {actual_charge}, reference {reference_charge}")
                    
                    loaded_systems.append(system_name)
                    
                    # Log charge information
                    charge_info = ""
                    if actual_charge != 0:
                        charge_info = f", charge = {actual_charge:+.0f}"
                    
                    logging.info(f"  ✓ {system_name}: {len(complex_atoms)} atoms{charge_info}")
                    if host_atoms.info['charge'] != 0 or guest_atoms.info['charge'] != 0:
                        logging.info(f"    → Host: {len(host_atoms)} atoms, charge = {host_atoms.info['charge']:+.0f}")
                        logging.info(f"    → Guest: {len(guest_atoms)} atoms, charge = {guest_atoms.info['charge']:+.0f}")
                    
                except Exception as e:
                    logging.error(f"  ✗ {system_name}: {e}")
                    missing_systems.append(system_name)
            
            if missing_systems:
                logging.error(f"Missing systems: {missing_systems}")
                return False
            
            # Summary of charged systems
            charged_systems = [name for name in LNCI16_CHARGES.keys() if LNCI16_CHARGES[name] != 0]
            if charged_systems:
                logging.info(f"\nCharged systems in LNCI16: {charged_systems}")
                for sys in charged_systems:
                    logging.info(f"  {sys}: {LNCI16_CHARGES[sys]:+d}")
            
            logging.info(f"\nSuccessfully validated {len(loaded_systems)} systems")
            return True
        
        
        
        results, complex_atoms_list = benchmark_lnci16(calc, self.model_name)

        # Save the entire results DataFrame to lnci16_results_path as CSV
        results.to_csv(self.lnci16_results_path, index=False)
        
        # Save the complex atoms as an xyz file
        write(self.lnci16_complex_atoms_path, complex_atoms_list)


        # Compute MAE from the "error_kcal" column and save as JSON
        mae = results["error_kcal"].abs().mean()
        with open(self.lnci16_mae_path, "w") as f:
            json.dump(mae, f)


    @property
    def get_results(self) -> pd.DataFrame:
        """Load predicted data from CSV file."""
        return pd.read_csv(self.lnci16_results_path)

    @property
    def get_mae(self):
        """Load MAE from JSON file."""
        with open(self.lnci16_mae_path, "r") as f:
            data = json.load(f)
        return data
    
    @property
    def get_complex_atoms(self) -> List[Atoms]:
        """Load complex atoms from xyz file."""
        return read(self.lnci16_complex_atoms_path, index=":")




    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "LNCI16Benchmark"],
        cache_dir: str = "app_cache/supramolecular_complexes/LNCI16_cache/",
        normalise_to_model: Optional[str] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        mae_dict = {}
        pred_dfs = []
        
        # save images for WEAS viewer
        complex_atoms = list(node_dict.values())[0].get_complex_atoms
        save_dir = os.path.abspath(f"assets/LNCI16/")
        os.makedirs(save_dir, exist_ok=True)
        write(os.path.join(save_dir, "complex_atoms.xyz"), complex_atoms)

        for model_name, node in node_dict.items():
            mae_dict[model_name] = node.get_mae
            pred_df = node.get_results.rename(columns={
                "E_int_model_kcal": "E_model (kcal)",
                "E_int_ref_kcal": "E_ref (kcal)",
            })
            pred_df["Model"] = model_name
            pred_df["Error (kcal/mol)"] = pred_df["E_model (kcal)"] - pred_df["E_ref (kcal)"]
            pred_dfs.append(pred_df)

        mae_df = pd.DataFrame.from_dict(mae_dict, orient="index", columns=["MAE (kcal/mol)"]).reset_index()
        mae_df = mae_df.rename(columns={"index": "Model"})

        pred_full_df = pd.concat(pred_dfs, ignore_index=True)

        mae_df["Score"] = mae_df["MAE (kcal/mol)"]

        if normalise_to_model:
            norm_value = mae_df.loc[mae_df["Model"] == normalise_to_model, "Score"].values[0]
            mae_df["Score"] /= norm_value

        mae_df["Rank"] = mae_df["Score"].rank(ascending=True, method="min")

        mae_df.to_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        pred_full_df.to_pickle(os.path.join(cache_dir, "results_df.pkl"))
        


    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/supramolecular_complexes/LNCI16_cache/",
        app: dash.Dash | None = None,
        ui=None,
    ):
        from mlipx.dash_utils import run_app

        mae_df = pd.read_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        results_df = pd.read_pickle(os.path.join(cache_dir, "results_df.pkl"))

        layout = LNCI16Benchmark.build_layout(mae_df)

        def callback_fn(app_instance):
            LNCI16Benchmark.register_callbacks(app_instance, results_df)

        if app is None:
            assets_dir = os.path.abspath("assets")
            print("Serving assets from:", assets_dir)
            app = dash.Dash(__name__, assets_folder=assets_dir)

            app.layout = layout
            callback_fn(app)
            return run_app(app, ui=ui)
        else:
            return layout, callback_fn


    @staticmethod
    def build_layout(mae_df):
        return html.Div([
            dash_table_interactive(
                df=mae_df.round(3),
                id="LNCI16-table",
                benchmark_info="Benchmark info:",
                title="LNCI16 Benchmark",
                tooltip_header={
                    "Model": "Name of the MLIP model",
                    "Score": "Absolute value of Delta (meV); normalized if specified",
                    "Rank": "Ranking of model by Score (lower is better)"
                },
                extra_components=[
                    html.Div(
                        children=[
                            html.Div(
                                "Click on the points to see the structure!",
                                style={
                                    "color": "red",
                                    "fontWeight": "bold",
                                    "marginBottom": "10px"
                                }
                            ),
                            dcc.Graph(id="LNCI16-plot")
                        ],
                        id="LNCI16-plot-container",
                        style={"display": "none"},
                    ),
                    html.Div(id="weas-viewer-LNCI16", style={'marginTop': '20px'}),
                ]
            )
        ])
        
        
    @staticmethod
    def register_callbacks(app, results_df):
        # Import shared WEAS viewer callback utility
        from mlipx.dash_utils import weas_viewer_callback
        
        @app.callback(
            Output("LNCI16-plot", "figure"),
            Output("LNCI16-plot-container", "style"),
            Input("LNCI16-table", "active_cell"),
            State("LNCI16-table", "data"),
        )
        def update_LNCI16_plot(active_cell, table_data):
            if not active_cell:
                raise PreventUpdate

            row = active_cell["row"]
            clicked_model = table_data[row]["Model"]
            col = active_cell["column_id"]

            if col == "Model":
                return None

            df = results_df.copy()
            df = df[results_df["Model"] == clicked_model]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["E_ref (kcal)"],
                    y=df["E_model (kcal)"],
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    text=[
                        f"System: {sys}<br>Ref: {eref:.2f} kcal/mol<br>Model: {emod:.2f} kcal/mol<br>Error: {err:.2f} kcal/mol<br>Charged: {charged}<br>Atoms: C={ca}, H={ha}, G={ga}"
                        for sys, eref, emod, err, charged, ca, ha, ga in zip(
                            df["system"],
                            df["E_ref (kcal)"],
                            df["E_model (kcal)"],
                            df["error_kcal"],
                            df["is_charged"],
                            df["complex_atoms"],
                            df["host_atoms"],
                            df["guest_atoms"],
                        )
                    ],
                    hoverinfo="text",
                    name=clicked_model
                )
            )
            fig.add_trace(go.Scatter(
                x=[df["E_ref (kcal)"].min(), df["E_ref (kcal)"].max()],
                y=[df["E_ref (kcal)"].min(), df["E_ref (kcal)"].max()],
                mode="lines",
                line=dict(dash="dash", color="black", width=1),
                showlegend=False
            ))

            mae = df["error_kcal"].abs().mean()

            fig.update_layout(
                title=f"{clicked_model} vs Reference Interaction Energies",
                xaxis_title="Reference Energy [kcal/mol]",
                yaxis_title=f"{clicked_model} Energy [kcal/mol]",
                annotations=[
                    dict(
                        text=f"N = {len(df)}<br>MAE = {mae:.2f} kcal/mol",
                        xref="paper", yref="paper",
                        x=0.01, y=0.99, showarrow=False,
                        align="left", bgcolor="white", font=dict(size=10)
                    )
                ]
            )

            return fig, {"display": "block"}

        @app.callback(
            Output("weas-viewer-LNCI16", "children"),
            Output("weas-viewer-LNCI16", "style"),
            Input("LNCI16-plot", "clickData"),
        )
        def update_weas_viewer(clickData):
            return weas_viewer_callback(
                clickData, 
                "assets/LNCI16/complex_atoms.xyz",
                mode="info",
                info_key="system",
                )


