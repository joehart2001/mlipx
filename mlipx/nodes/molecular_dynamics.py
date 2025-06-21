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
    observers: list[DynamicsObserver] = zntrack.deps(None)
    modifiers: list[DynamicsModifier] = zntrack.deps(None)

    observer_metrics: dict = zntrack.metrics()
    plots: pd.DataFrame = zntrack.plots(y=["energy", "fmax"], autosave=True)

    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")

    def run(self):
        start_time = time.time()
        
        if self.observers is None:
            self.observers = []
        if self.modifiers is None:
            self.modifiers = []
        if self.data:
            atoms = self.data[self.data_id]
        elif self.data_path:
            atoms = read(self.data_path, self.data_id)
        atoms.calc = self.model.get_calculator()
        dyn = self.thermostat.get_molecular_dynamics(atoms)
        for obs in self.observers:
            obs.initialize(atoms)

        self.observer_metrics = {}
        self.plots = pd.DataFrame(columns=["energy", "fmax", "fnorm"])

        for idx, _ in enumerate(
            tqdm.tqdm(dyn.irun(steps=self.steps), total=self.steps)
        ):
            if idx % 10 == 0:
                ase.io.write(self.frames_path, atoms, append=True)
                plots = {
                    "energy": atoms.get_potential_energy(),
                    "fmax": np.max(np.linalg.norm(atoms.get_forces(), axis=1)),
                    "fnorm": np.linalg.norm(atoms.get_forces()),
                }
                self.plots.loc[len(self.plots)] = plots
            
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

        for obs in self.observers:
            # document all attached observers
            self.observer_metrics[obs.name] = self.observer_metrics.get(obs.name, -1)



    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))





    @property
    def figures(self) -> dict[str, go.Figure]:
        plots = {}
        for key in self.plots.columns:
            fig = px.line(
                self.plots,
                x=self.plots.index,
                y=key,
                title=key,
            )
            fig.update_traces(
                customdata=np.stack([np.arange(len(self.plots))], axis=1),
            )
            plots[key] = fig
        return plots

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
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        from dash.exceptions import PreventUpdate
        import dash
        from mlipx.dash_utils import dash_table_interactive, run_app
        import json
        
        # Define which properties to compute
        properties = [
            'g_r_oo',
        ]

        from mlipx.benchmark_download_utils import get_benchmark_data
        ref_data_path = get_benchmark_data("water_MD.zip") / "water_MD"

        properties_dict = {}
        # Add reference data
        with open(ref_data_path / "pbe-d3-300k-g(r)_oo.json", "r") as f:
            pbe_D3_330K = json.load(f)
        with open(ref_data_path / "exp-300k-g(r)_oo.json", "r") as f:
            exp_300K = json.load(f)

        properties_dict['pbe_D3_330K'] = {
            'g_r_oo': {
                'r': pbe_D3_330K['x'],
                'rdf': pbe_D3_330K['y']
            }
        }
        properties_dict['exp_300K'] = {
            'g_r_oo': {
                'r': exp_300K['x'],
                'rdf': exp_300K['y']
            }
        }

        # Compute properties for each model
        for model_name, node in node_dict.items():
            properties_dict[model_name] = {}
            traj = node.frames
            o_indices = [atom.index for atom in traj[0] if atom.symbol == 'O']
            for prop in properties:
                if prop == 'g_r_oo':
                    # Compute RDF for the trajectory
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj,
                        i_indices=o_indices,
                        j_indices=o_indices,
                        r_max=6.0,
                        bins=100,
                    )
                    properties_dict[model_name][prop] = {
                        "r": r,
                        "rdf": rdf,
                    }

        # --- Generalized MAE summary table logic ---
        # Prepare MAE data for all properties and references
        mae_data = []
        reference_keys = [k for k in properties_dict.keys() if k not in node_dict]

        for model_name in node_dict.keys():
            for prop in properties:
                for ref_key in reference_keys:
                    if prop not in properties_dict[model_name] or prop not in properties_dict[ref_key]:
                        continue
                    r_common = properties_dict[ref_key][prop]['r']
                    rdf_ref = np.interp(r_common, properties_dict[ref_key][prop]['r'], properties_dict[ref_key][prop]['rdf'])
                    rdf_model = np.interp(r_common, properties_dict[model_name][prop]['r'], properties_dict[model_name][prop]['rdf'])
                    mae = np.mean(np.abs(rdf_model - rdf_ref))
                    mae_data.append({
                        "Model": model_name,
                        "Reference": ref_key,
                        "Property": prop,
                        "MAE": round(mae, 4)
                    })

        mae_df = pd.DataFrame(mae_data)

        # Create Dash App
        if run_interactive:
            app = dash.Dash(__name__)
            app.layout = dash_table_interactive(
                df=mae_df,
                id="rdf-mae-score-table",
                title="RDF MAE Summary Table",
                extra_components=[
                    html.Div(id="rdf-table-details"),
                    dcc.Store(id="rdf-table-last-clicked", data=None),
                ],
            )

            @app.callback(
                Output("rdf-table-details", "children"),
                Output("rdf-table-last-clicked", "data"),
                Input("rdf-mae-score-table", "active_cell"),
                State("rdf-table-last-clicked", "data"),
            )
            def update_rdf_plot(active_cell, last_clicked):
                if active_cell is None:
                    raise PreventUpdate
                row = active_cell["row"]
                model_name = mae_df.loc[row, "Model"]
                ref_model = mae_df.loc[row, "Reference"]
                prop = mae_df.loc[row, "Property"]
                if last_clicked is not None and active_cell == last_clicked:
                    return None, None
                r = properties_dict[ref_model][prop]['r']
                rdf_ref = properties_dict[ref_model][prop]['rdf']
                rdf_model = np.interp(r, properties_dict[model_name][prop]['r'], properties_dict[model_name][prop]['rdf'])

                import plotly.graph_objs as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=r, y=rdf_ref, mode='lines', name=f"{ref_model} ({prop})"))
                fig.add_trace(go.Scatter(x=r, y=rdf_model, mode='lines', name=f"{model_name} ({prop})"))
                fig.update_layout(title=f"{prop} Comparison: {model_name} vs {ref_model}", xaxis_title="r (Å)", yaxis_title="g(r)")
                return dcc.Graph(figure=fig), active_cell

            return run_app(app, ui=ui)
        else:
            return mae_df
            
    def _compute_partial_rdf(atoms, i_indices, j_indices, r_max):
        i_list, j_list, dists = neighbor_list('ijd', atoms, r_max)

        mask = np.isin(i_list, i_indices) & np.isin(j_list, j_indices)
        valid_distances = dists[mask]
        return valid_distances, len(valid_distances), atoms.get_volume()

    def compute_rdf_optimized_parallel(atoms_list, i_indices, j_indices, r_max=6.0, bins=100, n_jobs=-1):
        i_indices = np.array(i_indices)
        j_indices = np.array(j_indices)

        results = Parallel(n_jobs=n_jobs)(
            delayed(MolecularDynamics._compute_partial_rdf)(atoms, i_indices, j_indices, r_max) for atoms in atoms_list
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

        return r, rdf