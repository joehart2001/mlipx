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
from ase.build import molecule
from ase.geometry import geometry
from ase.data import atomic_numbers
from ase.calculators.calculator import Calculator
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
import random


class GhostAtomBenchmark(zntrack.Node):
    """ Benchmark comparing: E_slab1
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    ghost_atom_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "ghost_atom_benchmark.csv")



    def smiles_to_ase(self, smiles: str) -> Atoms:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        conf = mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        return Atoms(symbols=symbols, positions=positions)

    def get_forces(self, atoms: Atoms, calc: Calculator) -> np.ndarray:
        atoms.calc = calc
        try:
            forces = calc.get_forces(atoms)
        except Exception:
            # fallback: calculate forces by finite difference or other method if needed
            forces = calc.get_forces(atoms)
        return forces

    def create_ghost_atoms_system(self, solute: Atoms, ghost_Ne: int, ghost_dist: float, calc: Calculator) -> Atoms:
        # Place ghost Ne atoms at least ghost_dist from solute center of mass
        com = solute.get_center_of_mass()
        ghost_positions = []
        rng = np.random.default_rng(42)
        while len(ghost_positions) < ghost_Ne:
            pos = rng.uniform(low=0, high=60, size=3)
            if np.linalg.norm(pos - com) >= ghost_dist:
                ghost_positions.append(pos)
        ghost_atoms = Atoms(symbols=["Ne"] * ghost_Ne, positions=ghost_positions)
        combined = solute + ghost_atoms
        combined.calc = calc
        return combined

    def add_random_H(self, atoms: Atoms, min_dist: float, max_dist: float, rng: np.random.Generator, calc: Calculator) -> Atoms:
        com = atoms.get_center_of_mass()
        for _ in range(100):
            # random point on sphere shell between min_dist and max_dist
            r = rng.uniform(min_dist, max_dist)
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            x = com[0] + r * np.sin(phi) * np.cos(theta)
            y = com[1] + r * np.sin(phi) * np.sin(theta)
            z = com[2] + r * np.cos(phi)
            pos = np.array([x, y, z])
            # check min distance to existing atoms
            dists = np.linalg.norm(atoms.positions - pos, axis=1)
            if np.all(dists > 1.0):
                new_atoms = atoms.copy()
                new_atoms += Atoms("H", positions=[pos])
                new_atoms.calc = calc
                return new_atoms
        # fallback: add H at min_dist along x axis
        pos = com + np.array([min_dist, 0, 0])
        new_atoms = atoms.copy()
        new_atoms += Atoms("H", positions=[pos])
        new_atoms.calc = calc
        return new_atoms



    def run(self):

        calc = self.model.get_calculator()

        # ------ setup -------

        solute_smiles = "CC(=O)C"    # SMILES (RDKit required)
        box_L         = 60.0         # Å edge length of cubic cell
        ghost_Ne      = 20           # number of ghost atoms for Test 3
        ghost_dist    = 40.0         # place all Ne ≥ this many Å from solute COM
        rand_trials   = 30           # Test 6 repetitions
        rand_min_dist = 20.0         # inner shell radius for random H
        rand_max_dist = 50.0         # outer shell radius
        SEED = 42

        # Generate solute structure
        solute = self.smiles_to_ase(solute_smiles)
        solute.set_cell([box_L, box_L, box_L])
        solute.center()
        solute.pbc = True
        solute.calc = calc

        # Test 1: Add ghost atoms and compare forces
        system_ghost = self.create_ghost_atoms_system(solute, ghost_Ne, ghost_dist, calc)
        F_solute = self.get_forces(solute, calc)
        F_ghost = self.get_forces(system_ghost, calc)
        # difference in forces on solute atoms only
        dF_test3 = np.linalg.norm(F_solute - F_ghost[:len(solute)], axis=1)
        max_dF_test3 = np.max(dF_test3)

        # Test 2: Add random H atoms and compare forces
        rng = np.random.default_rng(SEED)
        dF_test6_list = []
        for _ in range(rand_trials):
            system_randH = self.add_random_H(solute, rand_min_dist, rand_max_dist, rng, calc)
            F_randH = self.get_forces(system_randH, calc)
            # difference in forces on solute atoms only
            dF = np.linalg.norm(F_solute - F_randH[:len(solute)], axis=1)
            dF_test6_list.append(np.mean(dF))
        mean_dF_test6 = np.mean(dF_test6_list)
        std_dF_test6 = np.std(dF_test6_list)

        # Collect results into DataFrame
        df = pd.DataFrame({
            "Model": [self.model_name],
            "test1 max ΔF": [max_dF_test3],
            "test2 mean ΔF": [mean_dF_test6],
            "test2 std ΔF": [std_dF_test6],
        })

        df.to_csv(self.ghost_atom_output, index=False)
        
        
    @property
    def get_results(self):
        # load df
        return pd.read_csv(self.ghost_atom_output)
    
    
    @staticmethod
    def benchmark_precompute(
        node_dict: dict[str, "GhostAtomBenchmark"],
        cache_dir: str = "app_cache/physicality_benchmark/ghost_atom_cache/",
        normalise_to_model: Optional[str] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        df_list = []
        for model_name, node in node_dict.items():
            df = node.get_results.copy()
            #df["Model"] = model_name  # ensure model name column
            df_list.append(df)
            
        full_df = pd.concat(df_list, ignore_index=True)
        
        full_df["Score"] = full_df["test2 mean ΔF"]
        
        if normalise_to_model:
            norm_value = full_df.loc[full_df["Model"] == normalise_to_model, "Score"].values[0]
            full_df["Score"] /= norm_value
        
        full_df["Rank"] = full_df["Score"].rank(ascending=True, method="min")
        
        full_df.to_pickle(os.path.join(cache_dir, "results_df.pkl"))




    @staticmethod
    def launch_dashboard(
        cache_dir: str = "app_cache/physicality_benchmark/ghost_atom_cache/",
        app: dash.Dash | None = None,
        ui=None,
    ):
        from mlipx.dash_utils import run_app

        results_df = pd.read_pickle(os.path.join(cache_dir, "results_df.pkl"))

        layout = GhostAtomBenchmark.build_layout(results_df)
        
        def callback_fn(app_instance):
            GhostAtomBenchmark.register_callbacks(
                app_instance,
                results_df=results_df,
            )

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
                df=results_df.round(5),
                id="ghost-mae-table",
                benchmark_info="Evaluates force sensitivity to ghost atoms (test 1) and randomly placed H atoms (test 2).",
                title="Ghost Atom Benchmark",
                tooltip_header={
                    "Model": "Name of the MLIP model",
                    "test1 max ΔF": "Max ΔF (eV/Å) on solute atoms due to ghost atoms placed far away",
                    "test2 mean ΔF": "Mean ΔF (eV/Å) on solute atoms from random H-atom placements",
                    "test2 std ΔF": "Standard deviation of ΔF for random H-atom placements",
                    "Score": "Same as test2 mean ΔF (lower is better); normalized if specified",
                    "Rank": "Ranking of model by Score (lower is better)"
                }
            )
        ])
        
        
        
    @staticmethod
    def register_callbacks(app, results_df):
        pass