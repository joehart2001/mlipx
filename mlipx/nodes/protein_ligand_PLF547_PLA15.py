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
from typing import List, Dict, Any, Optional, Tuple
import mlipx
from mlipx import OC157Benchmark, S24Benchmark
import os
import dash
from dash import dcc, html, Input, Output, State, MATCH
from dash import ctx
import base64
import csv
import warnings
from copy import deepcopy
from scipy.stats import kendalltau, pearsonr, spearmanr
import logging
from mlipx.benchmark_download_utils import get_benchmark_data



class ProteinLigandBenchmark(zntrack.Node):
    """ 
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    
    plf547_ref_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plf547_ref.csv")
    plf547_pred_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plf547_pred.csv")
    plf547_mae_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plf547_mae.json")
    plf547_complex_atoms_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plf547_complex_atoms.xyz")

    pla15_ref_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "pla15_ref.csv")
    pla15_pred_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "pla15_pred.csv")
    pla15_mae_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "pla15_mae.json")
    pla15_complex_atoms_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "pla15_complex_atoms.xyz")
    
    def run(self):
        base_dir = get_benchmark_data("protein-ligand-data_PLA15_PLF547.zip") / "protein-ligand-data_PLA15_PLF547"
        PLF547_DIR = base_dir / "PLF547_pdbs"
        PLA15_DIR = base_dir / "PLA15_pdbs"
        PLF547_REF_FILE = PLF547_DIR / "reference_energies.txt"
        PLA15_REF_FILE = PLA15_DIR / "reference_energies.txt"

        KCAL_TO_EV = 0.04336414
        EV_TO_KCAL = 1.0 / KCAL_TO_EV

        # ------------------------------------------------------------
        # PDB processing functions
        # ------------------------------------------------------------
        def extract_charge_and_selections(pdb_path: Path) -> Tuple[float, float, float, str, str]:
            """Extract charge and selection information from PDB REMARK lines"""
            total_charge = qa = qb = 0.0
            selection_a = selection_b = ""
            
            with open(pdb_path, 'r') as f:
                for line in f:
                    if not line.startswith('REMARK'):
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            break
                        continue
                    
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                        
                    tag = parts[1].lower()
                    
                    if tag == 'charge':
                        total_charge = float(parts[2])
                    elif tag == 'charge_a':
                        qa = float(parts[2])
                    elif tag == 'charge_b': 
                        qb = float(parts[2])
                    elif tag == 'selection_a':
                        selection_a = ' '.join(parts[2:])
                    elif tag == 'selection_b':
                        selection_b = ' '.join(parts[2:])
            
            return total_charge, qa, qb, selection_a, selection_b

        def separate_protein_ligand_simple(pdb_path: Path):
            """Simple separation based on residue names"""
            import MDAnalysis as mda
            
            # Load with MDAnalysis
            u = mda.Universe(str(pdb_path))
            
            # Simple separation: ligand = UNK residues, protein = everything else
            protein_atoms = []
            ligand_atoms = []
            
            for atom in u.atoms:
                if atom.resname.strip().upper() in ['UNK', 'LIG', 'MOL']:
                    ligand_atoms.append(atom)
                else:
                    protein_atoms.append(atom)
            
            return u.atoms, protein_atoms, ligand_atoms

        def mda_atoms_to_ase(atom_list, charge: float, identifier: str) -> "ase.Atoms":
            """Convert MDAnalysis atoms to ASE Atoms object"""
            from ase import Atoms
            
            if not atom_list:
                atoms = Atoms()
                print("charge", charge, "identifier", identifier)
                atoms.info.update({'charge': charge, 'identifier': identifier})
                
                return atoms
            
            symbols = []
            positions = []
            
            for atom in atom_list:
                # Get element symbol
                try:
                    elem = (atom.element or "").strip().title()
                except:
                    elem = ""
                
                if not elem:
                    # Fallback: first letter of atom name
                    elem = "".join([c for c in atom.name if c.isalpha()])[:1].title() or "C"
                
                symbols.append(elem)
                positions.append(atom.position)
            
            atoms = Atoms(symbols=symbols, positions=np.array(positions))
            atoms.info.update({'charge': int(round(charge)), 'identifier': identifier})
            return atoms

        def benchmark_plf547(calc: Calculator, model_name: str):
            """Benchmark PLF547 dataset - protein fragment-ligand interactions"""
            plf547_refs = parse_plf547_references(PLF547_REF_FILE)
            pdb_files = list(PLF547_DIR.glob("*.pdb"))
            results = []
            complex_atoms_list = []
            for pdb_file in tqdm(pdb_files, desc="PLF547"):
                identifier = pdb_file.stem
                if identifier not in plf547_refs:
                    continue
                fragments = process_pdb_file(pdb_file)
                if not fragments:
                    continue
                try:
                    # Direct energy computation
                    fragments['complex'].calc = calc
                    E_complex = fragments['complex'].get_potential_energy()
                    fragments['protein'].calc = calc
                    E_protein = fragments['protein'].get_potential_energy()
                    fragments['ligand'].calc = calc
                    E_ligand = fragments['ligand'].get_potential_energy()
                    E_int_model = E_complex - E_protein - E_ligand
                    E_int_ref = plf547_refs[identifier]
                    results.append({
                        'identifier': identifier,
                        'dataset': 'PLF547',
                        'E_int_ref': E_int_ref,
                        f'E_int_{model_name}': E_int_model,
                        'error_eV': E_int_model - E_int_ref,
                        'error_kcal': (E_int_model - E_int_ref) * EV_TO_KCAL,
                        'complex_atoms': len(fragments['complex']),
                        'protein_atoms': len(fragments['protein']),
                        'ligand_atoms': len(fragments['ligand']),
                        'complex': fragments['complex']
                    })
                    complex_atoms_list.append(fragments['complex'])
                except Exception as e:
                    continue
            df = pd.DataFrame(results)
            return df, complex_atoms_list

        def process_pdb_file(pdb_path: Path) -> Dict[str, "ase.Atoms"]:
            """Process one PDB file and return complex + separated fragments"""
            
            total_charge, charge_a, charge_b, _, _ = extract_charge_and_selections(pdb_path)
            
            try:
                all_atoms, protein_atoms, ligand_atoms = separate_protein_ligand_simple(pdb_path)
                
                if len(ligand_atoms) == 0:
                    logging.warning(f"No ligand atoms found in {pdb_path.name}")
                    return {}
                
                if len(protein_atoms) == 0:
                    logging.warning(f"No protein atoms found in {pdb_path.name}")
                    return {}
                
                base_id = pdb_path.stem
                
                # Create ASE objects
                complex_atoms = mda_atoms_to_ase(list(all_atoms), total_charge, base_id)
                protein_frag = mda_atoms_to_ase(protein_atoms, charge_a, base_id) 
                ligand = mda_atoms_to_ase(ligand_atoms, charge_b, base_id)
                
                return {
                    'complex': complex_atoms,
                    'protein': protein_frag,
                    'ligand': ligand
                }
                
            except Exception as e:
                logging.warning(f"Error processing {pdb_path}: {e}")
                return {}

        # ------------------------------------------------------------
        # Reference energy parsing
        # ------------------------------------------------------------
        def parse_plf547_references(path: Path) -> Dict[str, float]:
            """Parse PLF547 reference interaction energies (kcal/mol -> eV)"""
            ref: Dict[str, float] = {}
            
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.lower().startswith("no.") or line.startswith("-"):
                    continue
                    
                parts = line.split()
                if len(parts) < 3:
                    continue
                    
                try:
                    energy_kcal = float(parts[-1])
                except ValueError:
                    continue
                    
                identifier = parts[1].replace(".pdb", "")
                energy_eV = energy_kcal * KCAL_TO_EV  # Convert to eV
                ref[identifier] = energy_eV
                
            return ref

        def parse_plf547_references(path: Path) -> Dict[str, float]:
            """Parse PLF547 reference interaction energies (kcal/mol -> eV)"""
            ref: Dict[str, float] = {}
            
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.lower().startswith("no.") or line.startswith("-"):
                    continue
                    
                parts = line.split()
                if len(parts) < 3:
                    continue
                    
                try:
                    energy_kcal = float(parts[-1])
                except ValueError:
                    continue
                    
                # Extract full identifier with residue type
                full_identifier = parts[1].replace(".pdb", "")
                
                # Extract base identifier by removing residue type suffix
                # Format: "2P4Y_28_met" -> "2P4Y_28"
                identifier_parts = full_identifier.split('_')
                if len(identifier_parts) >= 3:
                    # Assume last part is residue type (met, arg, bbn, etc.)
                    base_identifier = '_'.join(identifier_parts[:-1])
                else:
                    # Fallback: use full identifier if format is unexpected
                    base_identifier = full_identifier
                    
                energy_eV = energy_kcal * KCAL_TO_EV  # Convert to eV
                ref[base_identifier] = energy_eV
                
            return ref

        def parse_pla15_references(path: Path) -> Dict[str, float]:
            """Parse PLA15 reference total energies (kcal/mol -> eV)"""
            ref: Dict[str, float] = {}
            
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.lower().startswith("no.") or line.startswith("-"):
                    continue
                    
                parts = line.split()
                if len(parts) < 3:
                    continue
                    
                try:
                    energy_kcal = float(parts[-1])
                except ValueError:
                    continue
                    
                # Extract full identifier with residue type
                full_identifier = parts[1].replace(".pdb", "")
                
                # Extract base identifier by removing residue type suffix
                # Format: "1ABC_15_lys" -> "1ABC_15"
                identifier_parts = full_identifier.split('_')
                if len(identifier_parts) >= 3:
                    # Assume last part is residue type (lys, arg, asp, etc.)
                    base_identifier = '_'.join(identifier_parts[:-1])
                else:
                    # Fallback: use full identifier if format is unexpected
                    base_identifier = full_identifier
                    
                energy_eV = energy_kcal * KCAL_TO_EV  # Convert to eV
                ref[base_identifier] = energy_eV
                
            return ref

        # ------------------------------------------------------------
        # PLA15 benchmark (complete active site interactions)
        # ------------------------------------------------------------
        def benchmark_pla15(calc: Calculator, model_name: str):
            """Benchmark PLA15 dataset - complete active site-ligand interactions"""
            pla15_refs = parse_pla15_references(PLA15_REF_FILE)
            pdb_files = list(PLA15_DIR.glob("*.pdb"))
            results = []
            complex_atoms_list = []
            for pdb_file in tqdm(pdb_files, desc="PLA15"):
                identifier = pdb_file.stem
                if identifier not in pla15_refs:
                    continue
                fragments = process_pdb_file(pdb_file)
                if not fragments:
                    continue
                try:
                    fragments['complex'].calc = calc
                    E_complex = fragments['complex'].get_potential_energy()
                    fragments['protein'].calc = calc
                    E_protein = fragments['protein'].get_potential_energy()
                    fragments['ligand'].calc = calc
                    E_ligand = fragments['ligand'].get_potential_energy()
                    E_int_model = E_complex - E_protein - E_ligand
                    E_int_ref = pla15_refs[identifier]
                    results.append({
                        'identifier': identifier,
                        'dataset': 'PLA15',
                        'E_int_ref': E_int_ref,
                        f'E_int_{model_name}': E_int_model,
                        'error_eV': E_int_model - E_int_ref,
                        'error_kcal': (E_int_model - E_int_ref) * EV_TO_KCAL,
                        'complex_atoms': len(fragments['complex']),
                        'protein_atoms': len(fragments['protein']),
                        'ligand_atoms': len(fragments['ligand']),
                        'complex': fragments['complex']
                    })
                    
                    fragments['complex'].info['identifier'] = identifier
                    fragments['complex'].info['dataset'] = 'PLA15'
                    fragments['complex'].info['model'] = model_name
                    fragments['complex'].info[f'E_int_{model_name}'] = E_int_model
                    fragments['complex'].info['E_int_ref'] = E_int_ref
                    fragments['complex'].info['error_eV'] = E_int_model - E_int_ref
                    fragments['complex'].info['error_kcal'] = (E_int_model - E_int_ref) * EV_TO_KCAL

                    complex_atoms_list.append(fragments['complex'])
                except Exception as e:
                    continue
            df = pd.DataFrame(results)
            return df, complex_atoms_list



        calc = self.model.get_calculator()
        model_name = self.model_name

        # Run PLF547 benchmark
        plf547_df, plf547_complex_atoms_list = benchmark_plf547(calc, model_name)
        # Save PLF547 results
        plf547_df.drop(columns=[f'E_int_{model_name}']).to_csv(self.plf547_ref_path, index=False)
        plf547_df.drop(columns=["E_int_ref"]).to_csv(self.plf547_pred_path, index=False)
        if not plf547_df.empty:
            plf547_mae = float(np.abs(plf547_df[f'E_int_{model_name}'] - plf547_df['E_int_ref']).mean() * EV_TO_KCAL)
        else:
            plf547_mae = None
        with open(self.plf547_mae_path, "w") as f:
            json.dump({"mae_kcal": plf547_mae}, f)
        if plf547_complex_atoms_list:
            write(self.plf547_complex_atoms_path, plf547_complex_atoms_list)

        # Run PLA15 benchmark
        pla15_df, pla15_complex_atoms_list = benchmark_pla15(calc, model_name)
        # Save PLA15 results
        pla15_df.to_csv(self.pla15_ref_path, index=False)
        pla15_df.to_csv(self.pla15_pred_path, index=False)
        pla15_df.drop(columns=[f'E_int_{model_name}']).to_csv(self.pla15_ref_path, index=False)
        pla15_df.drop(columns=["E_int_ref"]).to_csv(self.pla15_pred_path, index=False)
        if not pla15_df.empty:
            pla15_mae = float(np.abs(pla15_df[f'E_int_{model_name}'] - pla15_df['E_int_ref']).mean() * EV_TO_KCAL)
        else:
            pla15_mae = None
        with open(self.pla15_mae_path, "w") as f:
            json.dump({"mae_kcal": pla15_mae}, f)
        if pla15_complex_atoms_list:
            write(self.pla15_complex_atoms_path, pla15_complex_atoms_list)



    @property
    def get_ref_PLF547(self) -> pd.DataFrame:
        """Get reference energies for PLF547 dataset."""
        if not self.plf547_ref_path.exists():
            raise FileNotFoundError(f"Reference file {self.plf547_ref_path} does not exist.")
        return pd.read_csv(self.plf547_ref_path)
    @property
    def get_pred_PLF547(self) -> pd.DataFrame:
        """Get predicted energies for PLF547 dataset."""
        if not self.plf547_pred_path.exists():
            raise FileNotFoundError(f"Prediction file {self.plf547_pred_path} does not exist.")
        return pd.read_csv(self.plf547_pred_path)
    @property
    def get_mae_PLF547(self) -> Dict[str, float]:
        """Get MAE for PLF547 dataset."""
        if not self.plf547_mae_path.exists():
            raise FileNotFoundError(f"MAE file {self.plf547_mae_path} does not exist.")
        with open(self.plf547_mae_path, "r") as f:
            return json.load(f)
    @property
    def get_complex_atoms_PLF547(self) -> Atoms:
        """Get complex atoms for PLF547 dataset."""
        if not self.plf547_complex_atoms_path.exists():
            raise FileNotFoundError(f"Complex atoms file {self.plf547_complex_atoms_path} does not exist.")
        return read(self.plf547_complex_atoms_path, index=":")

    @property
    def get_ref_PLA15(self) -> pd.DataFrame:
        """Get reference energies for PLA15 dataset."""
        if not self.pla15_ref_path.exists():
            raise FileNotFoundError(f"Reference file {self.pla15_ref_path} does not exist.")
        return pd.read_csv(self.pla15_ref_path)
    @property
    def get_pred_PLA15(self) -> pd.DataFrame:
        """Get predicted energies for PLA15 dataset."""
        if not self.pla15_pred_path.exists():
            raise FileNotFoundError(f"Prediction file {self.pla15_pred_path} does not exist.")
        return pd.read_csv(self.pla15_pred_path)
    @property
    def get_mae_PLA15(self) -> Dict[str, float]:
        """Get MAE for PLA15 dataset."""
        if not self.pla15_mae_path.exists():
            raise FileNotFoundError(f"MAE file {self.pla15_mae_path} does not exist.")
        with open(self.pla15_mae_path, "r") as f:
            return json.load(f)
    @property
    def get_complex_atoms_PLA15(self) -> Atoms:
        """Get complex atoms for PLA15 dataset."""
        if not self.pla15_complex_atoms_path.exists():
            raise FileNotFoundError(f"Complex atoms file {self.pla15_complex_atoms_path} does not exist.")
        return read(self.pla15_complex_atoms_path, index=":") 








    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "ProteinLigandBenchmark"],
        cache_dir: str = "app_cache/supramolecular_complexes/PLA15_PLA547_cache/",
        normalise_to_model: Optional[str] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        mae_dict = {}
        PLA_15_pred_dfs = []
        PLF_547_pred_dfs = []

        # save images for WEAS viewer
        PLA_15_complex_atoms = list(node_dict.values())[0].get_complex_atoms_PLA15
        PLF_547_complex_atoms = list(node_dict.values())[0].get_complex_atoms_PLF547
        PLA_15_save_dir = os.path.abspath(f"assets/PLA15/")
        PLF_547_save_dir = os.path.abspath(f"assets/PLF547/")
        os.makedirs(PLA_15_save_dir, exist_ok=True)
        os.makedirs(PLF_547_save_dir, exist_ok=True)
        write(os.path.join(PLA_15_save_dir, "complex_atoms.xyz"), PLA_15_complex_atoms)
        write(os.path.join(PLF_547_save_dir, "complex_atoms.xyz"), PLF_547_complex_atoms)

        PLA_15_ref_df = list(node_dict.values())[0].get_ref_PLA15.rename(columns={"E_int_ref": "E_ref (eV)"})
        PLF_547_ref_df = list(node_dict.values())[0].get_ref_PLF547.rename(columns={"E_int_ref": "E_ref (eV)"})

        # New for-loop for PLA15 and PLF547 predictions
        for model_name, node in node_dict.items():
            mae_dict[model_name] = {
                "PLA15 MAE (kcal/mol)": node.get_mae_PLA15["mae_kcal"],
                "PLA547 MAE (kcal/mol)": node.get_mae_PLF547["mae_kcal"]
            }

            pla15_pred_df = node.get_pred_PLA15.rename(columns={f"E_int_{model_name}": "E_model (eV)"})
            pla15_merged = pd.merge(pla15_pred_df, PLA_15_ref_df, on="identifier")
            pla15_merged["Model"] = model_name
            pla15_merged["Error (eV)"] = pla15_merged["E_model (eV)"] - pla15_merged["E_ref (eV)"]
            pla15_merged["Error (kcal/mol)"] = pla15_merged["Error (eV)"] / 0.04336414
            PLA_15_pred_dfs.append(pla15_merged)

            plf547_pred_df = node.get_pred_PLF547.rename(columns={f"E_int_{model_name}": "E_model (eV)"})
            plf547_merged = pd.merge(plf547_pred_df, PLF_547_ref_df, on="identifier")
            plf547_merged["Model"] = model_name
            plf547_merged["Error (eV)"] = plf547_merged["E_model (eV)"] - plf547_merged["E_ref (eV)"]
            plf547_merged["Error (kcal/mol)"] = plf547_merged["Error (eV)"] / 0.04336414
            PLF_547_pred_dfs.append(plf547_merged)

        mae_df = pd.DataFrame.from_dict(mae_dict, orient="index").reset_index()
        mae_df = mae_df.rename(columns={"index": "Model"})

        # Concatenate and save new prediction dataframes
        plf547_merged_df = pd.concat(PLF_547_pred_dfs, ignore_index=True)
        pla15_merged_df = pd.concat(PLA_15_pred_dfs, ignore_index=True)



        mae_df["Score"] = (mae_df["PLA15 MAE (kcal/mol)"] + mae_df["PLA547 MAE (kcal/mol)"]) / 2

        if normalise_to_model:
            norm_value = mae_df.loc[mae_df["Model"] == normalise_to_model, "Score"].values[0]
            mae_df["Score"] /= norm_value
            

        mae_df["Rank"] = mae_df["Score"].rank(ascending=True, method="min")

        mae_df.to_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        plf547_merged_df.to_pickle(os.path.join(cache_dir, "plf547_predictions_df.pkl"))
        pla15_merged_df.to_pickle(os.path.join(cache_dir, "pla15_predictions_df.pkl"))





    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/supramolecular_complexes/PLA15_PLA547_cache/",
        app: dash.Dash | None = None,
        ui=None,
    ):
        from mlipx.dash_utils import run_app

        mae_df = pd.read_pickle(os.path.join(cache_dir, "mae_df.pkl"))
        plf547_df = pd.read_pickle(os.path.join(cache_dir, "plf547_predictions_df.pkl"))
        pla15_df = pd.read_pickle(os.path.join(cache_dir, "pla15_predictions_df.pkl"))

        layout = ProteinLigandBenchmark.build_layout(mae_df)

        def callback_fn(app_instance):
            ProteinLigandBenchmark.register_callbacks(app_instance, pla15_df, plf547_df)

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
                id="ProteinLigand-table",
                benchmark_info="Benchmark info: Protein-Ligand Benchmark. Interaction energies for protein-ligand complexes (PLA15/PLF547 datasets).",
                title="Protein-Ligand Benchmark",
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
                            dcc.Graph(id="ProteinLigand-plot")
                        ],
                        id="ProteinLigand-plot-container",
                        style={"display": "none"},
                    ),
                    dcc.Store(id="ProteinLigand-weas-structure-path", data=""),
                    html.Div(id="weas-viewer-ProteinLigand", style={'marginTop': '20px'}),
                ]
            )
        ])
        
        
    @staticmethod
    def register_callbacks(
        app,
        pla15_df,
        plf547_df
    ):
        from mlipx.dash_utils import weas_viewer_callback

        @app.callback(
            Output("ProteinLigand-plot", "figure"),
            Output("ProteinLigand-plot-container", "style"),
            Output("ProteinLigand-weas-structure-path", "data"),
            Input("ProteinLigand-table", "active_cell"),
            State("ProteinLigand-table", "data"),
        )
        def update_proteinligand_plot(active_cell, table_data):
            if not active_cell:
                raise PreventUpdate

            row = active_cell["row"]
            clicked_model = table_data[row]["Model"]
            col = active_cell["column_id"]

            if col == "Model":
                return dash.no_update, {"display": "none"}, ""

            if col == "PLA15 MAE (kcal/mol)":
                df = pla15_df
                structure_path = "assets/PLA15/complex_atoms.xyz"
                print("PLA15 structure path:", structure_path)
            elif col == "PLA547 MAE (kcal/mol)":
                df = plf547_df
                structure_path = "assets/PLF547/complex_atoms.xyz"
                print("PLF547 structure path:", structure_path)
            else:
                return dash.no_update, {"display": "none"}, ""

            df = df[df["Model"] == clicked_model]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["E_ref (eV)"] / 0.04336414,
                    y=df["E_model (eV)"] / 0.04336414,
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    customdata=df["identifier"],
                    text=[
                        f"identifier: {identifier}<br>DFT: {e_ref:.3f} eV<br>{clicked_model}: {e_model:.3f} eV"
                        for identifier, e_ref, e_model in zip(
                            df["identifier"], 
                            df["E_ref (eV)"], 
                            df["E_model (eV)"]
                        )
                    ],
                    hoverinfo="text",
                    name=clicked_model
                )
            )
            # Draw diagonal line for perfect agreement
            min_val = min(
                (df["E_ref (eV)"] / 0.04336414).min(),
                (df["E_model (eV)"] / 0.04336414).min()
            )
            max_val = max(
                (df["E_ref (eV)"] / 0.04336414).max(),
                (df["E_model (eV)"] / 0.04336414).max()
            )
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(dash="dash", color="black", width=1),
                showlegend=False
            ))

            mae = df["Error (kcal/mol)"].abs().mean()

            fig.update_layout(
                title=f"{clicked_model} vs DFT Interaction Energies",
                xaxis_title="DFT Energy [kcal/mol]",
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

            return fig, {"display": "block"}, structure_path

        @app.callback(
            Output("weas-viewer-ProteinLigand", "children"),
            Output("weas-viewer-ProteinLigand", "style"),
            Input("ProteinLigand-plot", "clickData"),
            State("ProteinLigand-weas-structure-path", "data"),
        )
        def update_weas_viewer(clickData, structure_path):
            if not clickData:
                raise PreventUpdate
            return weas_viewer_callback(
                clickData,
                structure_path,
                mode="info",
                info_key="identifier"
            )