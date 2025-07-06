import dataclasses
import pathlib

import ase.io
import ase.units
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
import zntrack
from ase.md import Langevin
from ase.io import read, write
import typing as t
import time
import json
from ase.neighborlist import neighbor_list
from joblib import Parallel, delayed
import pandas as pd
import json
from dash import dash_table, dcc, html
import dash
from mlipx.dash_utils import run_app, dash_table_interactive
import pickle
from pathlib import Path



from mlipx.abc import (
    ComparisonResults,
    DynamicsModifier,
    DynamicsObserver,
    NodeWithCalculator,
    NodeWithMolecularDynamics,
)


@dataclasses.dataclass
class LangevinConfig:
    """Configure a Langevin thermostat for molecular dynamics.

    Parameters
    ----------
    timestep : float
        Time step for the molecular dynamics simulation in fs.
    temperature : float
        Temperature of the thermostat.
    friction : float
        Friction coefficient of the thermostat.
    """

    timestep: float
    temperature: float
    friction: float

    def get_molecular_dynamics(self, atoms) -> Langevin:
        return Langevin(
            atoms,
            timestep=self.timestep * ase.units.fs,
            temperature_K=self.temperature,
            friction=self.friction,
        )


class MolecularDynamics(zntrack.Node):
    """Run molecular dynamics simulation.

    Parameters
    ----------
    model : NodeWithCalculator
        Node providing the calculator object for the simulation.
    thermostat : LangevinConfig
        Node providing the thermostat object for the simulation.
    data : list[ase.Atoms]
        Initial configurations for the simulation.
    data_id : int, default=-1
        Index of the initial configuration to use.
    steps : int, default=100
        Number of steps to run the simulation.
    """

    model: NodeWithCalculator = zntrack.deps()
    thermostat: NodeWithMolecularDynamics = zntrack.deps()
    data: list[ase.Atoms] = zntrack.deps(None)
    data_path: pathlib.Path = zntrack.params(None)
    data_id: int = zntrack.params(-1)
    steps: int = zntrack.params(100)
    print_energy_every: int = zntrack.params(1000)
    write_frames_every: int = zntrack.params(10)
    observers: list[DynamicsObserver] = zntrack.deps(None)
    modifiers: list[DynamicsModifier] = zntrack.deps(None)
    external_save_path: pathlib.Path = zntrack.params(None)
    resume_MD: bool = zntrack.params(False)
    resume_trajectory_path: pathlib.Path = zntrack.params(None)

    observer_metrics: dict = zntrack.metrics()
    plots: pd.DataFrame = zntrack.plots(y=["energy", "fmax"], autosave=True)

    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")
    velocities_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "velocities.npy")

    def run(self):
        start_time = time.time()

        # Only check for existing frames and plots if resuming MD
        if self.resume_MD:
            existing_frames = []
            start_idx = 0
            if Path(self.resume_trajectory_path).exists():
                existing_frames = list(ase.io.iread(self.resume_trajectory_path, format="extxyz"))
                start_idx = len(existing_frames) * self.write_frames_every

            plots_path = self.frames_path.with_name("plots.csv")
            if plots_path.exists():
                self.plots = pd.read_csv(plots_path)
            else:
                self.plots = pd.DataFrame(columns=["energy", "fmax", "fnorm"])
        else:
            start_idx = 0
            self.plots = pd.DataFrame(columns=["energy", "fmax", "fnorm"])
            plots_path = self.frames_path.with_name("plots.csv")

        if self.observers is None:
            self.observers = []
        if self.modifiers is None:
            self.modifiers = []
            
        if self.resume_trajectory_path:
            atoms = read(self.resume_trajectory_path, index=-1)
        elif self.data:
            atoms = self.data[self.data_id]
        elif self.data_path:
            atoms = read(self.data_path, self.data_id)

        atoms.calc = self.model.get_calculator()
        dyn = self.thermostat.get_molecular_dynamics(atoms)
        for obs in self.observers:
            obs.initialize(atoms)

        self.observer_metrics = {}

        # --- Collect velocities per frame ---
        from numpy.lib.format import open_memmap
        n_atoms = len(atoms)
        n_frames = self.steps // self.write_frames_every + 1
        velocities_memmap = open_memmap(self.velocities_path, mode='w+', dtype='float64', shape=(n_frames, n_atoms, 3))
        frame_idx = 0

        for idx, _ in enumerate(
            tqdm.tqdm(
                dyn.irun(steps=self.steps),
                total=self.steps,
                initial=start_idx if self.resume_MD else 0
            )
        ):
            if self.resume_MD and idx < start_idx:
                # Don't write the first frame if already present
                continue
            if idx % self.write_frames_every == 0:
                ase.io.write(self.frames_path, atoms, append=True)
                if self.resume_trajectory_path:
                    ase.io.write(self.resume_trajectory_path, atoms, append=True)
                if self.external_save_path:
                    ase.io.write(self.external_save_path, atoms, append=True)

                # Collect velocities for this frame directly to disk
                velocities_memmap[frame_idx] = atoms.get_velocities()
                frame_idx += 1

                plots = {
                    "energy": atoms.get_potential_energy(),
                    "fmax": np.max(np.linalg.norm(atoms.get_forces(), axis=1)),
                    "fnorm": np.linalg.norm(atoms.get_forces()),
                }
                self.plots.loc[len(self.plots)] = plots
                # Save to CSV after each update
                self.plots.to_csv(plots_path, index=False)

            # print every x steps
            if self.print_energy_every is not None and idx % self.print_energy_every == 0:
                epot = atoms.get_potential_energy() / len(atoms)
                ekin = atoms.get_kinetic_energy() / len(atoms)
                elapsed_time = time.time() - start_time
                elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
                tqdm.tqdm.write(
                    f"Step {idx} | Epot = {epot:.3f} eV  Ekin = {ekin:.3f} eV  "
                    f"(T = {ekin / (1.5 * ase.units.kB):.0f} K)  "
                    f"Etot = {epot + ekin:.3f} eV  Elapsed: {int(elapsed_min)}m {elapsed_sec:.1f}s"
                )

            for obs in self.observers:
                if obs.check(atoms):
                    self.observer_metrics[obs.name] = idx

            if len(self.observer_metrics) > 0:
                break

            for mod in self.modifiers:
                mod.modify(dyn, idx)

        # velocities_memmap is already written to disk during the run

        for obs in self.observers:
            # document all attached observers
            self.observer_metrics[obs.name] = self.observer_metrics.get(obs.name, -1)



    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))

    @property
    def velocities(self) -> np.ndarray:
        if not self.velocities_path.exists():
            raise FileNotFoundError(f"Velocities file {self.velocities_path} does not exist.")
        return np.load(self.velocities_path)



    @property
    def figures(self) -> dict[str, go.Figure]:
        return {
            key: px.line(self.plots, x=self.plots.index, y=key, title=key)
            for key in self.plots.columns
        }

    @staticmethod
    def compare(*nodes: "MolecularDynamics") -> ComparisonResults:
        frames = sum([node.frames for node in nodes], [])
        offset = 0
        fig = go.Figure()
        for _, node in enumerate(nodes):
            energies = [atoms.get_potential_energy() for atoms in node.frames]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(energies))),
                    y=energies,
                    mode="lines+markers",
                    name=node.name.replace(f"_{node.__class__.__name__}", ""),
                    customdata=np.stack([np.arange(len(energies)) + offset], axis=1),
                )
            )
            offset += len(energies)

        fig.update_layout(
            title="Energy vs. step",
            xaxis_title="Step",
            yaxis_title="Energy",
        )

        fig.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )

        # Now we set the first energy to zero for better compareability.

        offset = 0
        fig_adjusted = go.Figure()
        for _, node in enumerate(nodes):
            energies = np.array([atoms.get_potential_energy() for atoms in node.frames])
            energies -= energies[0]
            fig_adjusted.add_trace(
                go.Scatter(
                    x=list(range(len(energies))),
                    y=energies,
                    mode="lines+markers",
                    name=node.name.replace(f"_{node.__class__.__name__}", ""),
                    customdata=np.stack([np.arange(len(energies)) + offset], axis=1),
                )
            )
            offset += len(energies)

        fig_adjusted.update_layout(
            title="Adjusted energy vs. step",
            xaxis_title="Step",
            yaxis_title="Adjusted energy",
        )

        fig_adjusted.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        fig_adjusted.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )
        fig_adjusted.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )

        return ComparisonResults(
            frames=frames,
            figures={"energy_vs_steps": fig, "energy_vs_steps_adjusted": fig_adjusted},
        )









    @staticmethod
    def mae_plot_interactive(
        node_dict,
        run_interactive: bool = True,
        ui: str | None = None,
        normalise_to_model: t.Optional[str] = None,
    ):
        import pandas as pd
        import numpy as np
        import tqdm
        from dash import dcc, html
        import dash
        from mlipx.dash_utils import dash_table_interactive, run_app
        import json

        # Define which properties to compute
        properties = [
            'g_r_oo',
            'g_r_oh',
            'g_r_hh',
            'msd_O',
            'vacf',
            'vdos',
        ]

        from mlipx.benchmark_download_utils import get_benchmark_data
        ref_data_path = get_benchmark_data("water_MD_data.zip", force=True) / "water_MD_data"

        properties_dict = {}
        # Add reference data for all RDFs
        with open(ref_data_path / "pbe-d3-330k-g-r_oo.json", "r") as f:
            pbe_D3_330K_oo = json.load(f)
        with open(ref_data_path / "pbe-d3-330k-g-r_hh.json", "r") as f:
            pbe_D3_330K_hh = json.load(f)
        with open(ref_data_path / "pbe-d3-330k-g-r_oh.json", "r") as f:
            pbe_D3_330K_oh = json.load(f)
        with open(ref_data_path / "exp-300k-g-r_oo.json", "r") as f:
            exp_300K = json.load(f)
        # Load VACF reference data (just vacf key)
        with open(ref_data_path / "vcaf_300K_NPT_SPE_water_clean.json", "r") as f:
            vacf_ref_data = json.load(f)
            
        # Insert VACF reference into properties_dict["vacf"] following RDF pattern
        properties_dict["vacf"] = {
            "SPC/E_300K": {
                "time": np.array(vacf_ref_data["x"]) / 10,
                "vaf": vacf_ref_data["y"]
            }
        }

        properties_dict['g_r_oo'] = {
            'pbe_D3_330K_oo': {
                'r': pbe_D3_330K_oo['x'],
                'rdf': pbe_D3_330K_oo['y']
            }
        }

        properties_dict['g_r_oh'] = {
            'pbe_D3_330K_oh': {
                'r': pbe_D3_330K_oh['x'],
                'rdf': pbe_D3_330K_oh['y']
            }
        }
        properties_dict['g_r_hh'] = {
            'pbe_D3_330K_hh': {
                'r': pbe_D3_330K_hh['x'],
                'rdf': pbe_D3_330K_hh['y']
            }
        }
        # Merge in exp_300K data for g_r_oo without overwriting existing dict
        properties_dict['g_r_oo']['exp_300K'] = {
            'r': exp_300K['x'],
            'rdf': exp_300K['y']
        }
        

        # Compute properties for each model
        for model_name, node in tqdm.tqdm(node_dict.items(), desc="Computing properties for models"):
            traj = node.frames
            #traj = traj[::10]
            traj = traj[1000:]
            
            print(f"Processing model: {model_name} with {len(traj)} frames")
            velocities = node.velocities
            velocities = velocities[1000:]
            print("loaded trajectory for model:", model_name)
            o_indices = [atom.index for atom in traj[0] if atom.symbol == 'O']
            h_indices = [atom.index for atom in traj[0] if atom.symbol == 'H']
            for prop in properties:
                if prop not in properties_dict:
                    properties_dict[prop] = {}
                if prop == 'msd_O':
                    time, msd = MolecularDynamics.compute_msd(traj, timestep=1, atom_symbol='O')
                    properties_dict[prop][model_name] = {
                        "time": time.tolist(),
                        "msd": msd.tolist(),
                    }
                    print(f"Computed MSD for {model_name} with {len(time)} time points.")
                    #print(properties_dict[prop][model_name])
                elif prop == 'g_r_oo':
                    # O-O RDF
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj,
                        i_indices=o_indices,
                        j_indices=o_indices,
                        r_max=6.0,
                        bins=100,
                        n_jobs=1,
                    )
                    properties_dict[prop][model_name] = {
                        "r": r,
                        "rdf": rdf,
                    }
                elif prop == 'g_r_oh':
                    # O-H RDF
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj,
                        i_indices=o_indices,
                        j_indices=h_indices,
                        r_max=6.0,
                        bins=100,
                        n_jobs=1,
                    )
                    properties_dict[prop][model_name] = {
                        "r": r,
                        "rdf": rdf,
                    }
                elif prop == 'g_r_hh':
                    # H-H RDF
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj,
                        i_indices=h_indices,
                        j_indices=h_indices,
                        r_max=6.0,
                        bins=100,
                        n_jobs=1,
                    )
                    properties_dict[prop][model_name] = {
                        "r": r,
                        "rdf": rdf,
                    }

                elif prop == 'vacf':
                    # Velocity autocorrelation function
                    vaf = MolecularDynamics.compute_vacf(traj, velocities, timestep=1, atoms_filter=(('O',),))
                    #print(vaf)
                    properties_dict[prop][model_name] = {
                        "time": vaf[0],
                        "vaf": vaf[1],
                    }
                elif prop == 'vdos':
                    # Velocity density of states
                    vdos = MolecularDynamics.compute_vacf(traj, velocities, timestep=1, fft=True, atoms_filter=(('O',),))
                    properties_dict[prop][model_name] = {
                        "frequency": vdos[0],
                        "vdos": vdos[1],
                    }

        # Add msd_dict for later use
        msd_dict = properties_dict["msd_O"]

        # Compute VACF and VDOS MAE tables (dummy placeholders since no ref currently)


        # --- Helper to compute MAE DataFrame for a property ---
        def compute_mae_table(prop, properties_dict, model_names, normalise_to_model=None):
            # Build reference_keys and valid_model_names as per new logic
            reference_keys = []
            valid_model_names = []

            for k in properties_dict[prop].keys():
                data = properties_dict[prop][k]
                if k in model_names:
                    valid_model_names.append(k)
                elif 'rdf' in data or 'vaf' in data:
                    reference_keys.append(k)

            mae_data = []
            for model_name in valid_model_names:
                row = {"Model": model_name}
                # For each reference, compute MAE if both model and reference data are available
                for ref_key in reference_keys:
                    model_rdf_data = properties_dict[prop].get(model_name)
                    if model_rdf_data is None:
                        # fallback: check if model_name is a top-level key (legacy)
                        if model_name in properties_dict and prop in properties_dict[model_name]:
                            model_rdf_data = properties_dict[model_name][prop]
                    if (
                        ref_key in properties_dict[prop]
                        and model_rdf_data is not None
                    ):
                        # Interpolate both to common grid depending on property
                        if 'rdf' in properties_dict[prop][ref_key] and 'rdf' in model_rdf_data:
                            # RDF case
                            r_common = properties_dict[prop][ref_key]['r']
                            rdf_ref = np.interp(r_common, properties_dict[prop][ref_key]['r'], properties_dict[prop][ref_key]['rdf'])
                            rdf_model = np.interp(r_common, model_rdf_data['r'], model_rdf_data['rdf'])
                            mae = np.mean(np.abs(rdf_model - rdf_ref))
                            row[ref_key] = round(mae, 4)
                        elif 'vaf' in properties_dict[prop][ref_key] and 'vaf' in model_rdf_data:
                            # VACF case
                            model_x = np.array(model_rdf_data['time']) / 100
                            model_y = np.array(model_rdf_data['vaf'])
                            ref_x = np.array(properties_dict[prop][ref_key]['time']) / 100
                            ref_y = np.array(properties_dict[prop][ref_key]['vaf'])
                            ref_interp = np.interp(model_x, ref_x, ref_y)
                            mae = np.mean(np.abs(model_y - ref_interp))
                            row[ref_key] = round(mae, 4)
                mae_data.append(row)
            # Construct the dataframe with 'Model' as the first column and one column per reference
            columns = ['Model'] + reference_keys
            mae_df = pd.DataFrame(mae_data, columns=columns)
            # Score and rank logic using reference if present
            ref_col = None
            if prop == "g_r_oo":
                ref_col = "pbe_D3_330K_oo"
            elif prop == "g_r_oh":
                ref_col = "pbe_D3_330K_oh"
            elif prop == "g_r_hh":
                ref_col = "pbe_D3_330K_hh"
            elif prop == "msd_O":
                ref_col = None
            elif prop == "vacf":
                ref_col = "SPC/E_300K"
            elif prop == "vdos":
                ref_col = None
            # Adjust score_col/rank_col for VACF
            if ref_col is not None and ref_col in mae_df.columns:
                score_col = "Score ↓ (PBE)" if prop.startswith("g_r_") else "Score ↓ (SPC/E)"
                rank_col = "Rank (PBE)" if prop.startswith("g_r_") else "Rank (SPC/E)"
                if normalise_to_model is not None and normalise_to_model in mae_df["Model"].values:
                    base_mae = mae_df.loc[mae_df["Model"] == normalise_to_model, ref_col].values[0]
                else:
                    base_mae = mae_df[ref_col].min()
                mae_df[score_col] = mae_df[ref_col] / base_mae
                mae_df[rank_col] = mae_df[score_col].rank(method="min").astype(int)
            return mae_df.round(3)

        model_names = list(node_dict.keys())
        # Compute MAE DataFrames for each property
        mae_df_oo = compute_mae_table('g_r_oo', properties_dict, model_names, normalise_to_model)
        mae_df_oh = compute_mae_table('g_r_oh', properties_dict, model_names, normalise_to_model)
        mae_df_hh = compute_mae_table('g_r_hh', properties_dict, model_names, normalise_to_model)
        vacf_mae_df = compute_mae_table('vacf', properties_dict, model_names, normalise_to_model)

        if ui is None and run_interactive:
            return mae_df_oo, mae_df_oh, mae_df_hh, vacf_mae_df

        # --- Dash Layout ---
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("Water MD Benchmark"),
            html.P("Simulation details: NVT 330K, 64 water molecules, 50,000 steps, 1 fs timestep, initial 10,000 steps discarded."),
            html.H3("O-O RDF MAE Table"),
            dash_table_interactive(
                df=mae_df_oo,
                id="rdf-mae-score-table-oo",
                title=None,
                extra_components=[
                    html.Div(id="rdf-table-details-oo"),
                    dcc.Store(id="rdf-table-last-clicked-oo", data=None),
                ],
            ),
            html.H3("O-H RDF Table"),
            dash_table_interactive(
                df=mae_df_oh,
                id="rdf-mae-score-table-oh",
                title=None,
                extra_components=[
                    html.Div(id="rdf-table-details-oh"),
                    dcc.Store(id="rdf-table-last-clicked-oh", data=None),
                ],
            ),
            html.H3("H-H RDF Table"),
            dash_table_interactive(
                df=mae_df_hh,
                id="rdf-mae-score-table-hh",
                title=None,
                extra_components=[
                    html.Div(id="rdf-table-details-hh"),
                    dcc.Store(id="rdf-table-last-clicked-hh", data=None),
                ],
            ),
            html.H3("VACF Curves"),
            dash_table_interactive(
                df=vacf_mae_df,
                id="vacf-score-table",
                title=None,
                extra_components=[
                    html.Div(id="vacf-table-details"),
                    dcc.Store(id="vacf-table-last-clicked", data=None),
                ],
            ),
            # html.H3("VDOS Curves"),
            # dash_table_interactive(
            #     df=vdos_mae_df,
            #     id="vdos-score-table",
            #     title=None,
            #     extra_components=[
            #         html.Div(id="vdos-table-details"),
            #         dcc.Store(id="vdos-table-last-clicked", data=None),
            #     ],
            # ),
        ],
        style={"backgroundColor": "white"})

        # Register callbacks for all tables at once
        MolecularDynamics.register_callbacks(
            app, [
                ("rdf-mae-score-table-oo", "rdf-table-details-oo", "rdf-table-last-clicked-oo", "g_r_oo"),
                ("rdf-mae-score-table-oh", "rdf-table-details-oh", "rdf-table-last-clicked-oh", "g_r_oh"),
                ("rdf-mae-score-table-hh", "rdf-table-details-hh", "rdf-table-last-clicked-hh", "g_r_hh"),
                ("vacf-score-table", "vacf-table-details", "vacf-table-last-clicked", "vacf"),
                #("vdos-score-table", "vdos-table-details", "vdos-table-last-clicked", "vdos"),
            ],
            mae_df_list=[mae_df_oo, mae_df_oh, mae_df_hh, vacf_mae_df], #, vdos_mae_df],
            properties_dict=properties_dict
        )

        if not run_interactive:
            return app, (mae_df_oo, mae_df_oh, mae_df_hh, vacf_mae_df), properties_dict
        return run_app(app, ui=ui)



    @staticmethod
    def register_callbacks(app, table_configs, mae_df_list, properties_dict):
        from dash.dependencies import Input, Output, State
        from dash.exceptions import PreventUpdate
        from dash import dcc
        import plotly.graph_objs as go

        # Use default arguments in the loop to capture loop variables at definition time (avoid late binding)
        for config in zip(table_configs, mae_df_list):
            table_id, details_id, last_clicked_id, selected_property = config[0]
            mae_df = config[1]
            #print(f"Registering callbacks for {table_id} with property {selected_property}")
            #print(mae_df)
            model_names = list(mae_df["Model"].values)

            def make_callback(selected_property=selected_property, mae_df=mae_df, model_names=model_names):
                def update_rdf_plot(active_cell, last_clicked):
                    # Special handling for VACF and VDOS
                    if selected_property in ["vacf", "vdos"]:
                        if active_cell is None:
                            raise PreventUpdate
                        row = active_cell["row"]
                        model_name = mae_df.loc[row, "Model"]
                        if last_clicked is not None and active_cell == last_clicked:
                            return None, None

                        fig = go.Figure()
                        # Use explicit keys and labels for clarity
                        x_data_key = "time" if selected_property == "vacf" else "frequency"
                        y_data_key = "vaf" if selected_property == "vacf" else "vdos"
                        x_label = "Time (ps)" if selected_property == "vacf" else "Frequency (ps^-1)"
                        y_label = "VACF" if selected_property == "vacf" else "VDOS"
                        for model in model_names:
                            data = properties_dict[selected_property].get(model)
                            if data:
                                import numpy as np
                                if selected_property == "vacf":
                                    x_vals = np.array(data[x_data_key]) / 100  # fs → ps
                                    y_vals = np.array(data[y_data_key])
                                else:
                                    x_vals = np.array(data[x_data_key]) * 1000
                                    y_vals = np.array(data[y_data_key]) / 1000
                                fig.add_trace(go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode='lines',
                                    name=model,
                                    opacity=1.0 if model == model_name else 0.2
                                ))
                        # Add reference VACF trace if plotting VACF
                        if selected_property == "vacf":
                            ref_x = np.array(properties_dict["vacf"]["SPC/E_300K"]["time"]) / 100
                            ref_y = np.array(properties_dict["vacf"]["SPC/E_300K"]["vaf"])
                            fig.add_trace(go.Scatter(
                                x=ref_x,
                                y=ref_y,
                                mode='lines',
                                name="Reference",
                                line=dict(dash='dash', color='black'),
                                opacity=1.0
                            ))
                        # Set layout, including xaxis range for VACF
                        fig.update_layout(
                            title=f"{selected_property.upper()} curves",
                            xaxis_title=x_label,
                            yaxis_title=y_label,
                            xaxis=dict(range=[0, 1]) if selected_property == "vacf" else dict(range=[0, 150])
                        )
                        return dcc.Graph(figure=fig), active_cell

                    # Reference keys are those in properties_dict[selected_property] but not in model_names and must contain 'rdf'
                    reference_keys = [
                        k for k in properties_dict[selected_property]
                        if k not in model_names and 'rdf' in properties_dict[selected_property][k]
                    ]
                    if active_cell is None:
                        raise PreventUpdate
                    row = active_cell["row"]
                    model_name = mae_df.loc[row, "Model"]
                    if last_clicked is not None and active_cell == last_clicked:
                        return None, None

                    col = active_cell["column_id"]
                    # Begin new logic for consistent ordering, labeling, and coloring
                    fig = go.Figure()
                    legend_order = reference_keys + model_names

                    # Assign consistent colors from Plotly's qualitative palette
                    from plotly.express.colors import qualitative
                    color_map = {name: qualitative.Plotly[i % len(qualitative.Plotly)] for i, name in enumerate(legend_order)}

                    # Plot all references
                    for ref_key in reference_keys:
                        trace_name = f"{ref_key} ({selected_property})"
                        r = properties_dict[selected_property][ref_key]['r']
                        rdf = properties_dict[selected_property][ref_key]['rdf']
                        opacity = 1.0 if col == ref_key else 0.3
                        fig.add_trace(go.Scatter(
                            x=r,
                            y=rdf,
                            mode='lines',
                            name=trace_name,
                            opacity=opacity,
                            # Always dashed for reference curves
                            line=dict(color=color_map[ref_key], width=2, dash='dot'),
                        ))

                    # Plot all models
                    for model in model_names:
                        model_rdf = properties_dict[selected_property].get(model)
                        if model_rdf is not None:
                            r = model_rdf['r']
                            rdf = model_rdf['rdf']
                            opacity = 1.0 if model == model_name else 0.2
                            fig.add_trace(go.Scatter(
                                x=r,
                                y=rdf,
                                mode='lines',
                                name=model,
                                opacity=opacity,
                                line=dict(color=color_map[model], width=2),
                            ))

                    # Highlight selected model and reference again for emphasis
                    if model_name in color_map and model_name in properties_dict[selected_property]:
                        r = properties_dict[selected_property][model_name]['r']
                        rdf = properties_dict[selected_property][model_name]['rdf']
                        fig.add_trace(go.Scatter(
                            x=r,
                            y=rdf,
                            mode='lines',
                            name=f"{model_name} (highlight)",
                            line=dict(color=color_map[model_name], width=3),
                            opacity=1.0,
                            showlegend=False,
                        ))

                    if col in reference_keys and col in properties_dict[selected_property]:
                        r = properties_dict[selected_property][col]['r']
                        rdf = properties_dict[selected_property][col]['rdf']
                        fig.add_trace(go.Scatter(
                            x=r,
                            y=rdf,
                            mode='lines',
                            name=f"{col} (highlight)",
                            line=dict(color=color_map[col], width=3, dash='dot'),
                            opacity=1.0,
                            showlegend=False,
                        ))

                    fig.update_layout(
                        title=f"RDF Comparison for {model_name} ({selected_property.replace('_', '-')})",
                        xaxis_title="r (Å)",
                        yaxis_title="g(r)",
                    )
                    return dcc.Graph(figure=fig), active_cell
                return update_rdf_plot

            app.callback(
                Output(details_id, "children"),
                Output(last_clicked_id, "data"),
                Input(table_id, "active_cell"),
                State(last_clicked_id, "data"),
            )(make_callback())
            
    def _compute_partial_rdf(atoms, i_indices, j_indices, r_max):
        i_list, j_list, dists = neighbor_list('ijd', atoms, r_max)

        mask = np.isin(i_list, i_indices) & np.isin(j_list, j_indices)
        valid_distances = dists[mask]
        return valid_distances, len(valid_distances), atoms.get_volume()

    def compute_rdf_optimized_parallel(atoms_list, i_indices, j_indices, r_max=6.0, bins=100, n_jobs=-1):
        i_indices = np.array(i_indices)
        j_indices = np.array(j_indices)
        
        atom_a = atoms_list[0][i_indices][0]
        atom_b = atoms_list[0][j_indices][0]
        pair = f"{atom_a.symbol}-{atom_b.symbol}"


        from tqdm import tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(MolecularDynamics._compute_partial_rdf)(atoms, i_indices, j_indices, r_max)
            for atoms in tqdm(atoms_list, desc=f"Computing {pair} RDF")
        )

        all_distances = []
        total_pairs = 0
        total_volume = 0.0

        for distances, pair_count, volume in results:
            all_distances.append(distances)
            total_pairs += pair_count
            total_volume += volume

        all_distances = np.concatenate(all_distances)
        avg_volume = total_volume / len(atoms_list)
        rho = len(j_indices) / avg_volume  # Average number density

        hist, bin_edges = np.histogram(all_distances, bins=bins, range=(0, r_max))
        r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]

        shell_volumes = 4 * np.pi * r**2 * bin_width
        ideal_gas_distribution = shell_volumes * rho

        rdf = hist / (ideal_gas_distribution * total_pairs)
        
        rdf = rdf / rdf[np.argmax(r)] # liquid rdf should be 1 at large r

        return r, rdf
    
    

    
    @staticmethod
    def benchmark_precompute(
        node_dict,
        cache_dir="app_cache/further_applications_benchmark/molecular_dynamics_cache",
        ui=None,
        run_interactive=False,
        normalise_to_model=None,
    ):
        import os
        os.makedirs(cache_dir, exist_ok=True)
        app, (mae_df_oo, mae_df_oh, mae_df_hh, mae_df_vacf), properties_dict = MolecularDynamics.mae_plot_interactive(
            node_dict=node_dict,
            run_interactive=run_interactive,
            ui=ui,
            normalise_to_model=normalise_to_model,
        )
        mae_df_oo.to_pickle(f"{cache_dir}/mae_df_oo.pkl")
        mae_df_oh.to_pickle(f"{cache_dir}/mae_df_oh.pkl")
        mae_df_hh.to_pickle(f"{cache_dir}/mae_df_hh.pkl")
        mae_df_vacf.to_pickle(f"{cache_dir}/vacf_df.pkl")
        with open(f"{cache_dir}/rdf_data.pkl", "wb") as f:
            pickle.dump(properties_dict, f)
        msd_dict = properties_dict["msd_O"]
        with open(f"{cache_dir}/msd_data.pkl", "wb") as f:
            pickle.dump(msd_dict, f)
        vdos_dict = properties_dict["vdos"]
        with open(f"{cache_dir}/vdos_data.pkl", "wb") as f:
            pickle.dump(vdos_dict, f)
        return app



    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/further_applications_benchmark/molecular_dynamics_cache", 
        ui=None
    ):

        mae_df_oo = pd.read_pickle(f"{cache_dir}/mae_df_oo.pkl")
        mae_df_oh = pd.read_pickle(f"{cache_dir}/mae_df_oh.pkl")
        mae_df_hh = pd.read_pickle(f"{cache_dir}/mae_df_hh.pkl")
        mae_df_vacf = pd.read_pickle(f"{cache_dir}/vacf_df.pkl")
        with open(f"{cache_dir}/rdf_data.pkl", "rb") as f:
            properties_dict = pickle.load(f)
        with open(f"{cache_dir}/msd_data.pkl", "rb") as f:
            msd_dict = pickle.load(f)
        with open(f"{cache_dir}/vdos_data.pkl", "rb") as f:
            vdos_dict = pickle.load(f)
        # Add to properties_dict
        properties_dict["msd_O"] = msd_dict
        properties_dict["vdos"] = vdos_dict

        # Dummy MAE DataFrames for VACF/VDOS
        vdos_df = pd.DataFrame([{"Model": k} for k in vdos_dict.keys()])

        app = dash.Dash(__name__)
        app.layout = MolecularDynamics.build_layout(mae_df_oo, mae_df_oh, mae_df_hh, msd_dict, mae_df_vacf, vdos_df)

        table_configs = [
            ("rdf-mae-score-table-oo", "rdf-table-details-oo", "rdf-table-last-clicked-oo", "g_r_oo"),
            ("rdf-mae-score-table-oh", "rdf-table-details-oh", "rdf-table-last-clicked-oh", "g_r_oh"),
            ("rdf-mae-score-table-hh", "rdf-table-details-hh", "rdf-table-last-clicked-hh", "g_r_hh"),
            ("vacf-score-table", "vacf-table-details", "vacf-table-last-clicked", "vacf"),
            #("vdos-score-table", "vdos-table-details", "vdos-table-last-clicked", "vdos"),
        ]
        mae_df_list = [mae_df_oo, mae_df_oh, mae_df_hh, mae_df_vacf]
        MolecularDynamics.register_callbacks(
            app, table_configs, mae_df_list=mae_df_list, properties_dict=properties_dict
        )

        return run_app(app, ui=ui)
    
    
    
    
    
    @staticmethod
    def build_layout(mae_df_oo, mae_df_oh, mae_df_hh, msd_dict, vacf_df, vdos_df):
        return html.Div([
            html.H1("Water MD Benchmark"),
            html.H3("O-O RDF MAE Table"),
            dash_table_interactive(
                df=mae_df_oo,
                id="rdf-mae-score-table-oo",
                title=None,
                extra_components=[
                    html.Div(id="rdf-table-details-oo"),
                    dcc.Store(id="rdf-table-last-clicked-oo", data=None),
                ],
            ),
            html.H3("O-H RDF Table"),
            dash_table_interactive(
                df=mae_df_oh,
                id="rdf-mae-score-table-oh",
                title=None,
                extra_components=[
                    html.Div(id="rdf-table-details-oh"),
                    dcc.Store(id="rdf-table-last-clicked-oh", data=None),
                ],
            ),
            html.H3("H-H RDF Table"),
            dash_table_interactive(
                df=mae_df_hh,
                id="rdf-mae-score-table-hh",
                title=None,
                extra_components=[
                    html.Div(id="rdf-table-details-hh"),
                    dcc.Store(id="rdf-table-last-clicked-hh", data=None),
                ],
            ),
            # html.H3("Mean Squared Displacement (MSD) for Oxygen"),
            # dcc.Graph(
            #     figure=go.Figure([
            #         go.Scatter(
            #             x=msd_dict[model]["time"],
            #             y=msd_dict[model]["msd"],
            #             mode="lines",
            #             name=model
            #         )
            #         for model in msd_dict
            #     ]).update_layout(
            #         title="MSD vs Time",
            #         xaxis_title="Time (ps)",
            #         yaxis_title="MSD (Å²)"
            #     )
            # ),
            html.H3("Oxygen Velocity Autocorrelation Function (VACF)"),
            dash_table_interactive(
                df=vacf_df,
                id="vacf-score-table",
                title=None,
                extra_components=[
                    html.Div(id="vacf-table-details"),
                    dcc.Store(id="vacf-table-last-clicked", data=None),
                ],
            ),
        #     html.H3("Oxygen VDOS (FT of VACF)"),
        #     dash_table_interactive(
        #         df=vdos_df,
        #         id="vdos-score-table",
        #         title=None,
        #         extra_components=[
        #             html.Div(id="vdos-table-details"),
        #             dcc.Store(id="vdos-table-last-clicked", data=None),
        #         ],
        #     ),
        ], style={"backgroundColor": "white"})
        
        
    @staticmethod
    def compute_msd(traj, timestep=1, atom_symbol="O"):
        """
        Compute the Mean Squared Displacement (MSD) for atoms of the given symbol.

        Parameters
        ----------
        atom_symbol : str
            Element symbol (e.g., "O" or "H") for which to compute MSD.

        Returns
        -------
        time_steps : np.ndarray
            Time steps in picoseconds.
        msd_values : np.ndarray
            MSD values at each time step.
        """
        from tqdm import tqdm
        atom_indices = [atom.index for atom in traj[0] if atom.symbol == atom_symbol]
        num_frames = len(traj)
        initial_positions = traj[0].get_positions()[atom_indices]
        msd_values = np.zeros(num_frames)

        for i, atoms in enumerate(tqdm(traj, desc=f"Computing MSD for {atom_symbol}")):
            current_positions = atoms.get_positions()[atom_indices]
            displacements = current_positions - initial_positions
            squared_displacements = np.sum(displacements**2, axis=1)
            msd_values[i] = np.mean(squared_displacements)

        timestep_fs = timestep
        time_steps = np.arange(num_frames) * timestep_fs / 100
        return time_steps, msd_values
    
    
    
    
    
    
    # def compute_vacf(
    #     traj, 
    #     velocities, 
    #     atoms_filter = ((None,),),
    #     start_index=0, 
    #     timestep=1, 
    #     fft=False
    # ):
        
    #     n_steps = len(velocities)
    #     n_atoms = len(velocities[0])
    
    #     filtered_atoms = []
    #     atom_symbols = traj[0].get_chemical_symbols()
    #     symbols = set(atom_symbols)
    #     for atoms in atoms_filter:
    #         if any(atom is None for atom in atoms):
    #             # If atoms_filter not specified use all atoms.
    #             filtered_atoms.append(range(n_atoms))
    #         elif all(isinstance(a, str) for a in atoms):
    #             # If all symbols, get the matching indices.
    #             atoms = set(atoms)
    #             if atoms.difference(symbols):
    #                 raise ValueError(
    #                     f"{atoms.difference(symbols)} not allowed in VAF"
    #                     f", allowed symbols are {symbols}"
    #                 )
    #             filtered_atoms.append(
    #                 [i for i in range(len(atom_symbols)) if atom_symbols[i] in list(atoms)]
    #             )
    #         elif all(isinstance(a, int) for a in atoms):
    #             filtered_atoms.append(atoms)
    #         else:
    #             raise ValueError(
    #                 "Cannot mix element symbols and indices in vaf atoms_filter"
    #             )

    #     used_atoms = {atom for atoms in filtered_atoms for atom in atoms}
    #     used_atoms = {j: i for i, j in enumerate(used_atoms)}
        
        
    #     vafs = np.sum(
    #         np.asarray(
    #             [
    #                 [
    #                     np.correlate(velocities[:, j, i], velocities[:, j, i], "full")[
    #                         n_steps - 1 :
    #                     ]
    #                     for i in range(3)
    #                 ]
    #                 for j in used_atoms
    #             ]
    #         ),
    #         axis=1,
    #     )

    #     vafs /= n_steps - np.arange(n_steps)

    #     lags = np.arange(n_steps) * timestep

    #     if fft:
    #         vafs = np.fft.fft(vafs, axis=0)
    #         lags = np.fft.fftfreq(n_steps, timestep)

    #     # vafs = (
    #     #     lags,
    #     #     [
    #     #         np.average([vafs[used_atoms[i]] for i in atoms], axis=0)
    #     #         for atoms in filtered_atoms
    #     #     ],
    #     # )
    #     vafs = [
    #             np.average([vafs[used_atoms[i]] for i in atoms], axis=0)
    #             for atoms in filtered_atoms
    #         ]
    #     print(f"Computed VACF for {len(filtered_atoms)} atom groups with {len(lags)} lags.")
    #     print(f"VACF shape: {vafs[0].shape} for {len(vafs)} groups")
        
    #     total_vacf = np.mean(np.stack(vafs), axis=0)
        
    #     return lags, total_vacf
    
    

    def compute_vacf(
        traj,
        velocities,
        atoms_filter=((None,),),
        start_index=0,
        timestep=1,
        fft=False
    ):
        n_steps = len(velocities)
        n_atoms = len(velocities[0])
        filtered_atoms = []
        atom_symbols = traj[0].get_chemical_symbols()
        symbols = set(atom_symbols)
        
        for atoms in atoms_filter:
            if any(atom is None for atom in atoms):
                # If atoms_filter not specified use all atoms.
                filtered_atoms.append(range(n_atoms))
            elif all(isinstance(a, str) for a in atoms):
                # If all symbols, get the matching indices.
                atoms = set(atoms)
                if atoms.difference(symbols):
                    raise ValueError(
                        f"{atoms.difference(symbols)} not allowed in VAF"
                        f", allowed symbols are {symbols}"
                    )
                filtered_atoms.append(
                    [i for i in range(len(atom_symbols)) if atom_symbols[i] in list(atoms)]
                )
            elif all(isinstance(a, int) for a in atoms):
                filtered_atoms.append(atoms)
            else:
                raise ValueError(
                    "Cannot mix element symbols and indices in vaf atoms_filter"
                )
        
        used_atoms = {atom for atoms in filtered_atoms for atom in atoms}
        used_atoms = {j: i for i, j in enumerate(used_atoms)}
        
        # Corrected VACF calculation
        vafs = []
        for j in used_atoms:
            atom_vacf = np.zeros(n_steps)
            for i in range(3):  # x, y, z components
                v = velocities[:, j, i]
                # Proper autocorrelation calculation
                autocorr = np.correlate(v, v, mode='full')
                # Take the second half (positive lags) and normalize
                autocorr = autocorr[len(autocorr)//2:]
                atom_vacf += autocorr
            vafs.append(atom_vacf)
        
        vafs = np.array(vafs)
        
        # Proper normalization: divide by number of overlapping points
        normalization = np.arange(n_steps, 0, -1)
        vafs = vafs / normalization[np.newaxis, :]
        
        lags = np.arange(n_steps) * timestep
        
        if fft:
            # Calculate power spectral density for VDOS
            # Apply window function to reduce spectral leakage
            window = np.hanning(n_steps)
            vafs_windowed = vafs * window[np.newaxis, :]
            
            # Zero-pad for better frequency resolution (optional)
            n_fft = 2 * n_steps
            vafs_fft = np.fft.fft(vafs_windowed, n=n_fft, axis=1)
            
            # Power spectral density (VDOS)
            vafs = np.abs(vafs_fft)**2
            
            # Only take positive frequencies (real signal)
            # For real signals, we only need frequencies from 0 to Nyquist
            vafs = vafs[:, :n_fft//2 + 1]  # Include DC and Nyquist
            lags = np.fft.fftfreq(n_fft, timestep)[:n_fft//2 + 1]
            
            # Convert to proper units (frequencies in Hz or cm^-1)
            # If timestep is in femtoseconds, frequencies will be in THz
            # To convert to cm^-1: freq_cm = freq_THz * 33.356
        
        # Average over atom groups
        vafs_grouped = [
            np.average([vafs[used_atoms[i]] for i in atoms], axis=0)
            for atoms in filtered_atoms
        ]
        
        print(f"Computed VACF for {len(filtered_atoms)} atom groups with {len(lags)} lags.")
        #print(f"VACF shape: {vafs_grouped[0].shape} for {len(vafs_grouped)} groups")
        
        # Calculate total VACF
        total_vacf = np.mean(np.stack(vafs_grouped), axis=0)
        if fft:
            auc = np.trapezoid(total_vacf, lags)
            total_vacf = total_vacf / auc  # Normalize to max value
        else:
            total_vacf = total_vacf / np.max(total_vacf)  # Normalize to max value
        
        return lags, total_vacf