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
from dash import dcc, html, Input, Output
import base64

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




class PhononDispersion(zntrack.Node):
    """Compute the phonon dispersion from a phonopy object
    """
    # inputs
    phonopy_yaml_path: pathlib.Path = zntrack.deps()
    
    # qpoints: list[np.ndarray] = zntrack.deps(None)
    # labels: list[str] = zntrack.deps(None)
    # connections: list[bool] = zntrack.deps(None)
    qpoints_input_path: t.Optional[pathlib.Path] = zntrack.deps(None)
    labels_input_path: t.Optional[pathlib.Path] = zntrack.deps(None)
    connections_input_path: t.Optional[pathlib.Path] = zntrack.deps(None)

    # outputs
    # nwd: ZnTrack's node working directory for saving files
    band_structure_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "band_structure.npz")
    dos_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "dos.npz")
    phonon_obj_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "phonon_obj.yaml")
    qpoints_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "qpoints.pkl")
    labels_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "labels.json")
    connections_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "connections.json")
    



    def run(self):        
        
        phonons = load_phonopy(str(self.phonopy_yaml_path))
        
        if self.qpoints_input_path:
            # calculate phonon structure along the reference path to ensure a valid comparison
            with open(self.qpoints_input_path, "rb") as f:
                qpoints = pickle.load(f)
            with open(self.labels_input_path, "r") as f:
                labels = json.load(f)
            with open(self.connections_input_path, "r") as f:
                connections = json.load(f)
            

            
            #phonons.run_band_structure(paths=qpoints, labels=labels, path_connections=connections)
            #print(len(phonons.get_band_structure_dict()["distances"][0]))

            phonons.auto_band_structure()

            # save, zntrack requires each output declared to be an output
            with open(self.qpoints_path, "wb") as f:
                pickle.dump(qpoints, f)
            with open(self.labels_path, "w") as f:
                json.dump(labels, f)
            with open(self.connections_path, "w") as f:
                json.dump(connections, f)
                
                            
        else:
            #phonons.auto_band_structure() # uses seekpath
            
            qpoints, labels, connections = get_band_qpoints_by_seekpath(
                phonons.primitive, npoints=101, is_const_interval=True
            )
            
            phonons.run_band_structure(
                paths=qpoints,
                labels=labels,
                path_connections=connections,
            )

            # qpoints = phonons.band_structure.qpoints
            # labels = phonons.band_structure.labels
            # connections = phonons.band_structure.path_connections
            
            with open(self.qpoints_path, "wb") as f:
                pickle.dump(qpoints, f)
            with open(self.labels_path, "w") as f:
                json.dump(labels, f)
            with open(self.connections_path, "w") as f:
                json.dump(connections, f)
                
            

                
        band_structure = phonons.get_band_structure_dict()
        phonons.auto_total_dos()
        dos = phonons.get_total_dos_dict()
        
        
        with self.band_structure_path.open("wb") as f:
            pickle.dump(band_structure, f)        
        print(f"Band structure saved to: {self.band_structure_path}")
        with self.dos_path.open("wb") as f:
            pickle.dump(dos, f)
        
        phonons.save(filename=self.phonon_obj_path, settings={"force_constants": True})

        

    @property
    def qpoints(self):
        with open(self.qpoints_path, "rb") as f:
            return pickle.load(f)

    @property
    def labels(self):
        with open(self.labels_path, "r") as f:
            return json.load(f)

    @property
    def connections(self):
        with open(self.connections_path, "r") as f:
            return json.load(f)
        
    @property
    def band_structure(self):
        with self.band_structure_path.open("rb") as f:
            return pickle.load(f)
        
    @property
    def dos(self):
        with self.dos_path.open("rb") as f:
            dos = pickle.load(f)
        return dos["frequency_points"], dos["total_dos"]
    
    @property
    def band_width(self):
        with self.band_structure_path.open("rb") as f:
            band_structure = pickle.load(f)
        
        freqs = np.concatenate(band_structure["frequencies"])
        return np.max(freqs) - np.min(freqs)
    
    @property
    def max_freq(self):
        with self.band_structure_path.open("rb") as f:
            band_structure = pickle.load(f)
        
        freqs = np.concatenate(band_structure["frequencies"])
        return np.max(freqs)
    

        
    @property
    def plot_auto_band_structure(self):
        phonons = load_phonopy(self.phonon_obj_path)
        phonons.auto_band_structure(plot=True)
        return
    
    def _load_band_and_dos(self, phonon_path):
        phonons = load_phonopy(phonon_path)
        phonons.auto_band_structure()
        phonons.auto_total_dos()
        return phonons.get_band_structure_dict(), phonons.get_total_dos_dict(), phonons


    def _build_xticks(self, distances, labels, connections):
        # begins with _ as this is a private method only for internal use
        xticks, xticklabels = [], []
        cumulative_dist, i = 0.0, 0
        connections = [True] + connections
        for seg_dist, connected in zip(distances, connections):
            start, end = labels[i], labels[i+1]
            pos_start = cumulative_dist
            pos_end = cumulative_dist + (seg_dist[-1] - seg_dist[0])
            xticks.append(pos_start)
            xticklabels.append(f"{start}|{end}" if not connected else start)
            i += 2 if not connected else 1
            cumulative_dist = pos_end
        xticks.append(cumulative_dist)
        xticklabels.append(labels[-1])
        return xticks, xticklabels
    
    
    @property
    def plot_band_structure(self):
        band_structure, dos, phonons = self._load_band_and_dos(self.phonon_obj_path)


        dos_freqs = dos["frequency_points"]
        dos_values = dos["total_dos"]


        # plotting
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])
        
        for dist_segment, freq_segment in zip(band_structure['distances'], band_structure['frequencies']):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, color='red', lw=1)

        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])  # DOS plot on the right
        ax2.plot(dos_values, dos_freqs, color="red", lw=1)
        

        # sorting out xticks
        labels = phonons.band_structure.labels
        connections = phonons.band_structure.path_connections
        distances = band_structure["distances"]
        print(labels)

        
        xticks, xticklabels = self._build_xticks(distances, labels, connections)

        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)
        
        ax1.set_xticks(xticks, xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)")
        ax1.set_xlabel("Wave Vector")
        
        max_freq, min_freq = np.max(np.concatenate(band_structure["frequencies"])), np.min(np.concatenate(band_structure["frequencies"]))
        ax1.set_ylim(min_freq - 0.4, max_freq + 0.4)
        
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("DOS")
        
        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        return phonons





    @staticmethod
    def compare_models(node_dict, selected_models=None):

        if selected_models is not None:
            node_dict = {k: v for k, v in node_dict.items() if k in selected_models}

        # ---- Setup subplot grid ----
        fig = plt.figure(figsize=(9, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])  # band structure
        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])  # DOS

        # Load tick info from the first node
        first_node = next(iter(node_dict.values()))
        band_structure = first_node.band_structure
        distances = band_structure["distances"]

        phonons = load_phonopy(first_node.phonon_obj_path)
        phonons.auto_band_structure()
        phonons.auto_total_dos()

        labels = phonons.band_structure.labels
        connections = phonons.band_structure.path_connections
        connections = [True] + connections

        # ---- X-tick Construction ----
        xticks = []
        xticklabels = []
        cumulative_dist = 0.0
        i = 0

        for segment_dist, connected in zip(distances, connections):
            start_label = labels[i]
            end_label = labels[i + 1]

            start_pos = cumulative_dist
            end_pos = cumulative_dist + (segment_dist[-1] - segment_dist[0])

            if not connected:
                merged_label = f"{start_label}|{end_label}"
                xticks.append(start_pos)
                xticklabels.append(merged_label)
                i += 2
            else:
                xticks.append(start_pos)
                xticklabels.append(start_label)
                i += 1

            cumulative_dist = end_pos

        xticks.append(cumulative_dist)
        xticklabels.append(labels[-1])

        # ---- Plot each model ----
        for idx, (model_name, node) in enumerate(node_dict.items()):
            color = f"C{idx}"
            band_structure = node.band_structure
            distances = band_structure["distances"]
            frequencies = band_structure["frequencies"]

            # Plot band structure
            for dist_segment, freq_segment in zip(distances, frequencies):
                for band in freq_segment.T:
                    ax1.plot(dist_segment, band, lw=1, label=model_name, color=color)

            # Plot DOS using each node’s phonon object
            phonons = load_phonopy(node.phonon_obj_path)
            phonons.auto_total_dos()
            dos = phonons.get_total_dos_dict()
            dos_freqs = dos["frequency_points"]
            dos_values = dos["total_dos"]

            ax2.plot(dos_values, dos_freqs, lw=1.2, color=color)

        # ---- Decorations ----
        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)

        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)")
        ax1.set_xlabel("Wave Vector")

        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("DOS")

        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)

        # Clean legend (no duplicates)
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()



    @staticmethod
    def compare_reference_old(node, reference_dir, mp_id):
        
        # ----------setup ticks using node------------
        fig = plt.figure(figsize=(9, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])  # band structure
        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])  # DOS

        
        band_structure = node.band_structure
        distances = band_structure["distances"]
        frequencies = band_structure["frequencies"]
        
        phonons = load_phonopy(node.phonon_obj_path)
        phonons.auto_band_structure()
        phonons.auto_total_dos()
        dos = phonons.get_total_dos_dict()
        dos_freqs = dos["frequency_points"]
        dos_values = dos["total_dos"]
        
        labels = phonons.band_structure.labels
        connections = phonons.band_structure.path_connections
        connections = [True] + connections

        xticks = []
        xticklabels = []
        cumulative_dist = 0.0
        i = 0

        for segment_dist, connected in zip(distances, connections):
            start_label = labels[i]
            end_label = labels[i + 1]

            start_pos = cumulative_dist
            end_pos = cumulative_dist + (segment_dist[-1] - segment_dist[0])

            if not connected:
                merged_label = f"{start_label}|{end_label}"
                xticks.append(start_pos)
                xticklabels.append(merged_label)
                i += 2
            else:
                xticks.append(start_pos)
                xticklabels.append(start_label)
                i += 1

            cumulative_dist = end_pos

        xticks.append(cumulative_dist)
        xticklabels.append(labels[-1])
        
        
        
        # -------------reference data----------------
        # reference file layed out as e.g. mp-1234-formuala-bands.json and mp-1234-formuala-dos.json
        reference_dir = pathlib.Path(reference_dir)
        band_file = None
        dos_file = None

        for file in reference_dir.glob(f"mp-{mp_id}-*-bands.json"):
            band_file = file
        for file in reference_dir.glob(f"mp-{mp_id}-*-dos.json"):
            dos_file = file

        if not band_file or not dos_file:
            raise FileNotFoundError(f"Reference files for {mp_id} not found in {reference_dir}")

        with open(band_file, "r") as f:
            ref_bands = json.load(f)

        with open(dos_file, "r") as f:
            ref_dos = json.load(f)
            
            
        ref_band_data = (ref_bands['bands'])
        ref_dos_densities, ref_dos_freq = ref_dos['densities'], ref_dos['frequencies']
        
        #-----------------------normalise reference data-----------------------
        # normalise the x-axis to match the node's distance for each segment
        segment_lengths = [segment[-1] - segment[0] for segment in distances]
        n_segments = len(segment_lengths)
        ref_bands = np.array(ref_bands['bands'])  # shape (n_bands, total_kpoints)
        total_kpoints = ref_bands.shape[1]

        if total_kpoints % n_segments != 0:
            raise ValueError("Cannot evenly split reference bands into same number of segments as computed data.")

        ref_points_per_segment = total_kpoints // n_segments

        ref_band_data_x = []
        cumulative = 0.0

        for seg_len in segment_lengths:
            segment_x = np.linspace(cumulative, cumulative + seg_len, ref_points_per_segment, endpoint=False)
            ref_band_data_x.append(segment_x)
            cumulative += seg_len

        ref_band_data_x = np.concatenate(ref_band_data_x)
        

        #-----------------------plotting-----------------------
        # node band structure
        model = node.name.split("/")[0]
        for dist_segment, freq_segment in zip(distances, frequencies):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='--', label=model, color='red')

        ax2.plot(dos_values, dos_freqs, lw=1.2, color="red")
        
        # reference band structure
        for band in ref_band_data:
            ax1.plot(ref_band_data_x, band, lw=1, label="Reference", color="blue")
            
        ax2.plot(ref_dos_densities, ref_dos_freq, lw=1.2, color="blue")
            
        

        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)
        
        ax1.set_xticks(xticks, xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)")
        ax1.set_xlabel("Wave Vector")
        
        comp_freqs = np.concatenate(band_structure["frequencies"]).flatten()
        ref_freqs = np.array(ref_band_data).flatten()
        all_freqs = np.concatenate([comp_freqs, ref_freqs])

        ax1.set_ylim(all_freqs.min() - 0.4, all_freqs.max() + 0.4)
        ax2.set_ylim(ax1.get_ylim())
                
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("DOS")
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        return phonons
    
    



    #@staticmethod
    def compare_reference(self, node_pred, node_ref, correlation_plot_mode = False, model_name = None):
        
        # ----------setup ticks using node------------
        fig = plt.figure(figsize=(9, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])  # band structure
        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])  # DOS

        phonons_pred = load_phonopy(node_pred.phonon_obj_path)
        phonons_ref = load_phonopy(node_ref.phonon_obj_path)

        band_structure_pred = node_pred.band_structure
        distances_pred = band_structure_pred["distances"]
        frequencies_pred = band_structure_pred["frequencies"]
        dos_freqs_pred, dos_values_pred = node_pred.dos
        

        
        band_structure_ref = node_ref.band_structure
        distances_ref = band_structure_ref["distances"]
        frequencies_ref = band_structure_ref["frequencies"]
        dos_freqs_ref, dos_values_ref = node_ref.dos

        # print("distances pred", len(distances_pred[0]))
        # print("distances ref", len(distances_ref[0]))
        
        
        labels = node_ref.labels
        connections = node_ref.connections
        connections = [True] + connections
        
        xticks, xticklabels = self._build_xticks(distances_ref, labels, connections)
        

        #-----------------------plotting-----------------------
        # pred band structure
        model = node_pred.name.split("/")[0]
        for dist_segment, freq_segment in zip(distances_pred, frequencies_pred):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='--', label=model, color='red')

        ax2.plot(dos_values_pred, dos_freqs_pred, lw=1.2, color="red", linestyle='--')
        
        # reference band structure
        for dist_segment, freq_segment in zip(distances_ref, frequencies_ref):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='-', label="PBE", color='blue')
            
        ax2.plot(dos_values_ref, dos_freqs_ref, lw=1.2, color="blue")
            
        

        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)
        
        ax1.set_xticks(xticks, xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)")
        ax1.set_xlabel("Wave Vector")
        
        pred_freqs_flat = np.concatenate(frequencies_pred).flatten()
        ref_freqs_flat = np.concatenate(frequencies_ref).flatten()
        all_freqs = np.concatenate([pred_freqs_flat, ref_freqs_flat])

        
        ax1.set_ylim(all_freqs.min() - 0.4, all_freqs.max() + 0.4)
        ax2.set_ylim(ax1.get_ylim())
                
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("DOS")
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc='upper center',
                    bbox_to_anchor=(0.8, 1.02),  # Right of center title
                    frameon=False,
                    ncol=2,
                    fontsize=14,    
                    )
        
        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        
        chemical_formula = get_chemical_formula(phonons_ref)
        mp_id = node_ref.name.split("_")[1]
        plt.suptitle(f"{chemical_formula} ({mp_id})", x=0.4, fontsize = 14)
            
            
        
        if correlation_plot_mode:
            
            if not os.path.exists(f"nodes/{model_name}/phonons-dispersion/phonon_plots"):
                os.makedirs(f"nodes/{model_name}/phonons-dispersion/phonon_plots")
            
            phonon_plot_path = f"nodes/{model_name}/phonons-dispersion/phonon_plots/dispersion_{model_name}_{mp_id}.png"
            fig.savefig(phonon_plot_path, bbox_inches='tight')
            plt.close(fig)
            return phonon_plot_path
        
        else:
            plt.show()
            
        return
    
    

        
        
        

        
        
    
    
    def _plot_phonons(self, band_structure, dos, labels, connections, reference_phonons, ref_dos, mp_id, chemical_formula, model_name):

        phonons_freq = band_structure["frequencies"]
        phonons_dist = band_structure["distances"]
        dos_freqs = dos["frequency_points"]
        dos_values = dos["total_dos"]
        
        ref_dos_densities, ref_dos_freq = ref_dos['densities'], ref_dos['frequencies']
        
        #-----------------------normalise reference data-----------------------
        # normalise the x-axis to match the node's distance for each segment
        segment_lengths = [segment[-1] - segment[0] for segment in phonons_dist]
        n_segments = len(segment_lengths)
        ref_bands = np.array(reference_phonons)  # shape (n_bands, total_kpoints)
        total_kpoints = ref_bands.shape[1]

        if total_kpoints % n_segments != 0:
            print(f"{mp_id} Remainder:", total_kpoints % n_segments)
            #raise ValueError("Cannot evenly split reference bands into same number of segments as computed data.")
            return

        ref_points_per_segment = total_kpoints // n_segments

        ref_band_data_x = []
        cumulative = 0.0

        for seg_len in segment_lengths:
            segment_x = np.linspace(cumulative, cumulative + seg_len, ref_points_per_segment, endpoint=False)
            ref_band_data_x.append(segment_x)
            cumulative += seg_len

        ref_band_data_x = np.concatenate(ref_band_data_x)
        

        #-----------------------plotting-----------------------
        # node band structure
        #model = node.name.split("/")[0]
        
        fig = plt.figure(figsize=(9, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])
        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])  # DOS
        
        for dist_segment, freq_segment in zip(phonons_dist, phonons_freq):
            for band in freq_segment.T:
                ax1.plot(dist_segment, band, lw=1, linestyle='--', color='red', label = model_name)

        ax2.plot(dos_values, dos_freqs, lw=1.2, color="red", linestyle='--')
        
        # reference band structure
        for band in reference_phonons:
            ax1.plot(ref_band_data_x, band, lw=1, label="Reference", color="blue")
            
        ax2.plot(ref_dos_densities, ref_dos_freq, lw=1.2, color="blue")
            
        #print("phonons_dist", phonons_dist)
        #print("labels", labels)
        #print("connections", connections)
        xticks, xticklabels = self._build_xticks(phonons_dist, labels, connections)

        for x in xticks:
            ax1.axvline(x=x, color='k', linewidth=1)
        
        ax1.axhline(0, color='k', linewidth=1)
        ax2.axhline(0, color='k', linewidth=1)
        
        ax1.set_xticks(xticks, xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylabel("Frequency (THz)", fontsize = 14)
        ax1.set_xlabel("Wave Vector", fontsize = 14)
        
        comp_freqs = np.concatenate(phonons_freq).flatten()
        ref_freqs = np.array(reference_phonons).flatten()
        all_freqs = np.concatenate([comp_freqs, ref_freqs])

        ax1.set_ylim(all_freqs.min() - 0.4, all_freqs.max() + 0.4)
        ax2.set_ylim(ax1.get_ylim())
                
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_xlabel("DOS", fontsize = 14)
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc='upper center',
                    bbox_to_anchor=(0.8, 1.02),  # Right of center title
                    frameon=False,
                    ncol=2,
                    fontsize=14,    
                    )
        
        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        
        plt.suptitle(f"{chemical_formula} ({mp_id})", x=0.4, fontsize = 14)
        
        if not os.path.exists(f"nodes/{model_name}/phonons-dispersion/phonon_plots"):
            os.makedirs(f"nodes/{model_name}/phonons-dispersion/phonon_plots")
        
        phonon_plot_path = f"nodes/{model_name}/phonons-dispersion/phonon_plots/dispersion_{model_name}_{mp_id}.png"
        fig.savefig(phonon_plot_path, bbox_inches='tight')
        plt.close(fig)
        
        return phonon_plot_path
    
    
    
    

    #@property
    def max_freq_benchmark_interactive_old(self, phonon_dict, reference_dir, model):
        
        
        band_strucuture_dict = {}
        for mp_id in phonon_dict.keys():
            for model_name, phonon in phonon_dict[mp_id].items():
                band_strucuture_dict.setdefault(mp_id, {})[model_name] = phonon.band_structure
                


        
        # -------------reference data----------------
        # reference file layed out as e.g. mp-1234-formuala-bands.json and mp-1234-formuala-dos.json
        reference_dir = pathlib.Path(reference_dir)
        ref_band_data_dict = {}
        ref_benchmarks_dict = {}
        phonon_plot_paths = {}
        benchmark_dict_calc = {}
        point_index_to_id = [] # for plotting later

        for mp_id in phonon_dict.keys():
            ref_benchmarks_dict[mp_id] = {}
            ref_band_data_dict[mp_id] = {}
        
            band_file = None
            dos_file = None
            
            
            for file in reference_dir.glob(f"{mp_id}-*-bands.json"):
                band_file = file
            for file in reference_dir.glob(f"{mp_id}-*-dos.json"):
                dos_file = file

            if not band_file or not dos_file:
                raise FileNotFoundError(f"Reference files for {mp_id} not found in {reference_dir}")

            with open(band_file, "r") as f:
                ref_bands = json.load(f)

            with open(dos_file, "r") as f:
                ref_dos = json.load(f)
                
                
            ref_band_data = (ref_bands['bands'])
            ref_band_data_dict[mp_id] = ref_band_data
            
            
            # calculate dos dependent benchmarks
            ref_dos_densities, ref_dos_freq = ref_dos['densities'], ref_dos['frequencies']
            ref_benchmarks_dict[mp_id]['max_freq'] = np.max(ref_dos_freq)
            
            phonon_plot_paths[mp_id] = {}
            for model_name in phonon_dict[mp_id].keys():
                
                # phonon plots
                band_structure, dos, phonon = self._load_band_and_dos(phonon_dict[mp_id][model].phonon_obj_path)
                chemical_formula = get_chemical_formula(phonon)

                # max freq from models
                benchmark_dict_calc[mp_id] = {}
                for model_name in band_strucuture_dict[mp_id].keys():
                    benchmark_dict_calc[mp_id][model_name] = {}
                    benchmark_dict_calc[mp_id][model_name]['max_freq'] = np.max(dos['frequency_points'])
                
                labels = phonon.band_structure.labels
                connections = phonon.band_structure.path_connections
                connections = [True] + connections
                
                phonon_plot_path = self._plot_phonons(band_structure, dos,
                                labels, connections,
                                ref_band_data_dict[mp_id], ref_dos,
                                mp_id,
                                chemical_formula,
                                model_name)

                phonon_plot_paths[mp_id][model_name] = phonon_plot_path
                point_index_to_id.append((mp_id, model_name))
            


        
        #------------------------pair plot---------------------
        
        # # create array for plotting
        max_freqs_calc = []
        max_freqs_ref = []
        for (mp_id, model_name) in point_index_to_id:
            max_freqs_calc.append(benchmark_dict_calc[mp_id][model_name]['max_freq'])
            max_freqs_ref.append(ref_benchmarks_dict[mp_id]['max_freq'])
        

        hover_labels = [f"{mp_id} ({model_name})" for (mp_id, model_name) in point_index_to_id]
        max_val = max(max(max_freqs_calc), max(max_freqs_ref)) * 1.05

        scatter_fig = px.scatter(
            x=max_freqs_ref,
            y=max_freqs_calc,
            hover_name=hover_labels,
            labels={
                'x': 'Reference Max Frequency (THz)',
                'y': 'Calculated Max Frequency (THz)'
            }
        )

        scatter_fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='y = x',
            line=dict(color='black', dash='dash')
        ))

        scatter_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black',
            xaxis=dict(range=[0, max_val], showgrid=True, gridcolor='lightgray'),
            yaxis=dict(range=[0, max_val], showgrid=True, gridcolor='lightgray')
        )

        # --- Dash app ---
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H2("Pair Correlation Plot: Max Frequency vs Reference", style={"color": "black"}),

            html.Div([
                dcc.Graph(
                    id='pair-correlation-plot',
                    figure=scatter_fig,
                    style={"width": "45vw", "height": "80vh", "paddingBottom": "40px"}
                ),
                html.Div(
                    id='phonon-plot-display',
                    style={
                        "width": "50vw",
                        "height": "80vh",
                        "marginLeft": "2vw",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "backgroundColor": "#f9f9f9"
                    }
                )
            ], style={
                "display": "flex",
                "flexDirection": "row",
                "alignItems": "center",
                "justifyContent": "space-between"
            })
        ], style={"backgroundColor": "white", "padding": "20px"})

        @app.callback(
            Output('phonon-plot-display', 'children'),
            Input('pair-correlation-plot', 'clickData')
        )
        def display_phonon_plot(clickData):
            if clickData is None:
                return html.Div("Click on a point to view its phonon dispersion plot.")

            point_index = clickData['points'][0]['pointIndex']
            mp_id, model_name = point_index_to_id[point_index]
            img_path = phonon_plot_paths[mp_id][model_name]

            encoded_img = base64.b64encode(open(img_path, 'rb').read()).decode()

            return html.Img(
                src=f'data:image/png;base64,{encoded_img}',
                style={
                    "width": "80%",
                    "height": "80%",
                    "objectFit": "contain",
                    "border": "2px solid black"
                }
            )

        def run_app(app):
            app.run(debug=True, port=8051)

        return run_app(app)

            






        
        
        
        
    def max_freq_benchmark_interactive(self, pred_node_dict, ref_node_dict, model):
        
        
        band_strucuture_dict_pred = {}
        for mp_id in pred_node_dict.keys():
            for model_name, phonon in pred_node_dict[mp_id].items():
                band_strucuture_dict_pred.setdefault(mp_id, {})[model_name] = phonon.band_structure
        
        
        # band_structure_pred = node_pred.band_structure
        # distances_pred = band_structure_pred["distances"]
        # frequencies_pred = band_structure_pred["frequencies"]
        # dos_freqs_pred, dos_values_pred = node_pred.dos
        
        # band_structure_ref = node_ref.band_structure
        # distances_ref = band_structure_ref["distances"]
        # frequencies_ref = band_structure_ref["frequencies"]
        # dos_freqs_ref, dos_values_ref = node_ref.dos


        
        # -------------reference data----------------
        ref_band_data_dict = {}
        ref_benchmarks_dict = {}
        pred_benchmarks_dict = {}
        phonon_plot_paths = {}
        point_index_to_id = [] # for plotting later

        for mp_id in pred_node_dict.keys():
            ref_benchmarks_dict[mp_id] = {}
            ref_band_data_dict[mp_id] = {}
            pred_benchmarks_dict[mp_id] = {}
            
            node_ref = ref_node_dict[mp_id]
            band_structure_ref = node_ref.band_structure
            ref_band_data_dict[mp_id] = band_structure_ref
            distances_ref, frequencies_ref = band_structure_ref["distances"], band_structure_ref["frequencies"]
            dos_freqs_ref, dos_values_ref = node_ref.dos
                
                
            
            # calculate dos dependent benchmarks
            ref_dos_freq, ref_dos_densities = node_ref.dos
            ref_benchmarks_dict[mp_id]['max_freq'] = np.max(ref_dos_freq)
            
            
            # thermal properties
            
            
            
            phonon_plot_paths[mp_id] = {}
            for model_name in pred_node_dict[mp_id].keys():
                
                dos_freqs_pred, dos_values_pred = pred_node_dict[mp_id][model_name].dos
                
                # max freq from models
                
                for model_name in band_strucuture_dict_pred[mp_id].keys():
                    pred_benchmarks_dict[mp_id][model_name] = {}
                    pred_benchmarks_dict[mp_id][model_name]['max_freq'] = np.max(dos_freqs_pred)
                
                # labels = phonon.band_structure.labels
                # connections = phonon.band_structure.path_connections
                # connections = [True] + connections
                
                phonon_plot_path = self.compare_reference(
                                    node_pred=pred_node_dict[mp_id][model_name],
                                    node_ref=ref_node_dict[mp_id],
                                    correlation_plot_mode=True,
                                    model_name=model_name
                                )
                

                phonon_plot_paths[mp_id][model_name] = phonon_plot_path
                point_index_to_id.append((mp_id, model_name))
            


        
        #------------------------pair plot---------------------
        
        # # create array for plotting
        max_freqs_calc = []
        max_freqs_ref = []
        for (mp_id, model_name) in point_index_to_id:
            max_freqs_calc.append(pred_benchmarks_dict[mp_id][model_name]['max_freq'])
            max_freqs_ref.append(ref_benchmarks_dict[mp_id]['max_freq'])
        

        hover_labels = [f"{mp_id} ({model_name})" for (mp_id, model_name) in point_index_to_id]
        max_val = max(max(max_freqs_calc), max(max_freqs_ref)) * 1.05

        scatter_fig = px.scatter(
            x=max_freqs_ref,
            y=max_freqs_calc,
            hover_name=hover_labels,
            labels={
                'x': 'Reference Max Frequency (THz)',
                'y': 'Calculated Max Frequency (THz)'
            }
        )

        scatter_fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='y = x',
            line=dict(color='black', dash='dash')
        ))

        scatter_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black',
            xaxis=dict(range=[0, max_val], showgrid=True, gridcolor='lightgray'),
            yaxis=dict(range=[0, max_val], showgrid=True, gridcolor='lightgray')
        )

        # --- Dash app ---
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H2("Pair Correlation Plot: Max Frequency vs Reference", style={"color": "black"}),

            html.Div([
                dcc.Graph(
                    id='pair-correlation-plot',
                    figure=scatter_fig,
                    style={"width": "45vw", "height": "80vh", "paddingBottom": "40px"}
                ),
                html.Div(
                    id='phonon-plot-display',
                    style={
                        "width": "50vw",
                        "height": "80vh",
                        "marginLeft": "2vw",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "backgroundColor": "#f9f9f9"
                    }
                )
            ], style={
                "display": "flex",
                "flexDirection": "row",
                "alignItems": "center",
                "justifyContent": "space-between"
            })
        ], style={"backgroundColor": "white", "padding": "20px"})

        @app.callback(
            Output('phonon-plot-display', 'children'),
            Input('pair-correlation-plot', 'clickData')
        )
        def display_phonon_plot(clickData):
            if clickData is None:
                return html.Div("Click on a point to view its phonon dispersion plot.")

            point_index = clickData['points'][0]['pointIndex']
            mp_id, model_name = point_index_to_id[point_index]
            img_path = phonon_plot_paths[mp_id][model_name]

            encoded_img = base64.b64encode(open(img_path, 'rb').read()).decode()

            return html.Img(
                src=f'data:image/png;base64,{encoded_img}',
                style={
                    "width": "80%",
                    "height": "80%",
                    "objectFit": "contain",
                    "border": "2px solid black"
                }
            )

        def run_app(app):
            app.run(debug=True, port=8051)

        return run_app(app)
