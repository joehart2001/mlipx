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
        ]

        from mlipx.benchmark_download_utils import get_benchmark_data
        ref_data_path = get_benchmark_data("water_MD.zip") / "water_MD"

        properties_dict = {}
        # Add reference data (only for g_r_oo)
        with open(ref_data_path / "pbe-d3-330k-g-r_oo.json", "r") as f:
            pbe_D3_330K_oo = json.load(f)
        with open(ref_data_path / "pbe-d3-330k-g-r_hh.json", "r") as f:
            pbe_D3_330K_hh = json.load(f)
        with open(ref_data_path / "pbe-d3-330k-g-r_oh.json", "r") as f:
            pbe_D3_330K_oh = json.load(f)
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
        for model_name, node in tqdm.tqdm(node_dict.items(), desc="Computing properties for models"):
            properties_dict[model_name] = {}
            traj = node.frames
            traj = traj[::50]
            print("loaded trajectory for model:", model_name)
            o_indices = [atom.index for atom in traj[0] if atom.symbol == 'O']
            h_indices = [atom.index for atom in traj[0] if atom.symbol == 'H']
            for prop in properties:
                if prop == 'g_r_oo':
                    # O-O RDF
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
                elif prop == 'g_r_oh':
                    # O-H RDF
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj,
                        i_indices=o_indices,
                        j_indices=h_indices,
                        r_max=6.0,
                        bins=100,
                    )
                    properties_dict[model_name][prop] = {
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
                    )
                    properties_dict[model_name][prop] = {
                        "r": r,
                        "rdf": rdf,
                    }

        # --- Create separate MAE DataFrames for each property ---
        reference_keys = [k for k in properties_dict.keys() if k not in node_dict]
        model_names = list(node_dict.keys())

        # O-O: MAE table (models vs references)
        mae_data_oo = []
        for model_name in model_names:
            row = {"Model": model_name}
            prop = 'g_r_oo'
            for ref_key in reference_keys:
                if (
                    prop in properties_dict[model_name]
                    and prop in properties_dict[ref_key]
                ):
                    r_common = properties_dict[ref_key][prop]['r']
                    rdf_ref = np.interp(r_common, properties_dict[ref_key][prop]['r'], properties_dict[ref_key][prop]['rdf'])
                    rdf_model = np.interp(r_common, properties_dict[model_name][prop]['r'], properties_dict[model_name][prop]['rdf'])
                    mae = np.mean(np.abs(rdf_model - rdf_ref))
                    row[ref_key] = round(mae, 4)
            mae_data_oo.append(row)
        mae_df_oo = pd.DataFrame(mae_data_oo)
        # Score and rank logic using normalise_to_model if provided
        if 'pbe_D3_330K' in mae_df_oo.columns:
            if normalise_to_model is not None and normalise_to_model in mae_df_oo["Model"].values:
                base_mae = mae_df_oo.loc[mae_df_oo["Model"] == normalise_to_model, "pbe_D3_330K"].values[0]
            else:
                base_mae = mae_df_oo["pbe_D3_330K"].min()
            mae_df_oo["Score ↓ (PBE)"] = mae_df_oo["pbe_D3_330K"] / base_mae
            mae_df_oo["Rank (PBE)"] = mae_df_oo["Score ↓ (PBE)"].rank(method="min").astype(int)

        # O-H: MAE table (models vs models, no reference)
        mae_data_oh = []
        prop = 'g_r_oh'
        for model_name in model_names:
            row = {"Model": model_name}
            for other_model in model_names:
                if model_name == other_model:
                    continue
                if (
                    prop in properties_dict[model_name]
                    and prop in properties_dict[other_model]
                ):
                    r_common = properties_dict[model_name][prop]['r']
                    rdf_other = np.interp(r_common, properties_dict[other_model][prop]['r'], properties_dict[other_model][prop]['rdf'])
                    rdf_model = np.interp(r_common, properties_dict[model_name][prop]['r'], properties_dict[model_name][prop]['rdf'])
                    mae = np.mean(np.abs(rdf_model - rdf_other))
                    row[other_model] = round(mae, 4)
            mae_data_oh.append(row)
        mae_df_oh = pd.DataFrame(mae_data_oh)

        # H-H: MAE table (models vs models, no reference)
        mae_data_hh = []
        prop = 'g_r_hh'
        for model_name in model_names:
            row = {"Model": model_name}
            for other_model in model_names:
                if model_name == other_model:
                    continue
                if (
                    prop in properties_dict[model_name]
                    and prop in properties_dict[other_model]
                ):
                    r_common = properties_dict[model_name][prop]['r']
                    rdf_other = np.interp(r_common, properties_dict[other_model][prop]['r'], properties_dict[other_model][prop]['rdf'])
                    rdf_model = np.interp(r_common, properties_dict[model_name][prop]['r'], properties_dict[model_name][prop]['rdf'])
                    mae = np.mean(np.abs(rdf_model - rdf_other))
                    row[other_model] = round(mae, 4)
            mae_data_hh.append(row)
        mae_df_hh = pd.DataFrame(mae_data_hh)

        if ui is None and run_interactive:
            return mae_df_oo, mae_df_oh, mae_df_hh

        # --- Dash Layout ---
        app = dash.Dash(__name__)
        app.layout = html.Div([
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
        ])

        # Register callbacks for each table
        MolecularDynamics.register_callbacks(
            app, mae_df_oo, properties_dict,
            table_id="rdf-mae-score-table-oo",
            details_id="rdf-table-details-oo",
            last_clicked_id="rdf-table-last-clicked-oo",
            selected_property="g_r_oo",
        )
        MolecularDynamics.register_callbacks(
            app, mae_df_oh, properties_dict,
            table_id="rdf-mae-score-table-oh",
            details_id="rdf-table-details-oh",
            last_clicked_id="rdf-table-last-clicked-oh",
            selected_property="g_r_oh",
        )
        MolecularDynamics.register_callbacks(
            app, mae_df_hh, properties_dict,
            table_id="rdf-mae-score-table-hh",
            details_id="rdf-table-details-hh",
            last_clicked_id="rdf-table-last-clicked-hh",
            selected_property="g_r_hh",
        )

        if not run_interactive:
            return app, (mae_df_oo, mae_df_oh, mae_df_hh), properties_dict
        return run_app(app, ui=ui)



    @staticmethod
    def register_callbacks(app, mae_df, properties_dict, table_id, details_id, last_clicked_id, selected_property):
        from dash.dependencies import Input, Output, State
        from dash.exceptions import PreventUpdate
        from dash import dcc
        import plotly.graph_objs as go
        import pandas as pd

        # For O-O, show references; for O-H and H-H, only models
        model_names = list(mae_df["Model"].values)
        # All reference keys are those in properties_dict but not in model_names
        reference_keys = [k for k in properties_dict if k not in model_names]

        @app.callback(
            Output(details_id, "children"),
            Output(last_clicked_id, "data"),
            Input(table_id, "active_cell"),
            State(last_clicked_id, "data"),
        )
        def update_rdf_plot(active_cell, last_clicked):
            if active_cell is None:
                raise PreventUpdate
            row = active_cell["row"]
            model_name = mae_df.loc[row, "Model"]
            if last_clicked is not None and active_cell == last_clicked:
                return None, None

            fig = go.Figure()
            col = active_cell["column_id"]
            # For O-O, show references, for O-H and H-H, just models
            if selected_property == "g_r_oo":
                for ref_key in reference_keys:
                    if 'g_r_oo' in properties_dict[ref_key]:
                        r = properties_dict[ref_key]['g_r_oo']['r']
                        rdf = properties_dict[ref_key]['g_r_oo']['rdf']
                        opacity = 1.0 if col == ref_key else 0.3
                        fig.add_trace(go.Scatter(x=r, y=rdf, mode='lines', name=f"{ref_key} (g_r_oo)", opacity=opacity))
            # Plot all model RDFs for this property
            for model in model_names:
                if selected_property in properties_dict[model]:
                    r = properties_dict[model][selected_property]['r']
                    rdf = properties_dict[model][selected_property]['rdf']
                    opacity = 1.0 if model == model_name else 0.4
                    fig.add_trace(go.Scatter(x=r, y=rdf, mode='lines', name=f"{model}", opacity=opacity))

            fig.update_layout(
                title=f"RDF Comparison for {model_name} ({selected_property.replace('_', '-')})",
                xaxis_title="r (Å)",
                yaxis_title="g(r)",
            )
            return dcc.Graph(figure=fig), active_cell
            
    def _compute_partial_rdf(atoms, i_indices, j_indices, r_max):
        i_list, j_list, dists = neighbor_list('ijd', atoms, r_max)

        mask = np.isin(i_list, i_indices) & np.isin(j_list, j_indices)
        valid_distances = dists[mask]
        return valid_distances, len(valid_distances), atoms.get_volume()

    def compute_rdf_optimized_parallel(atoms_list, i_indices, j_indices, r_max=6.0, bins=100, n_jobs=-1):
        i_indices = np.array(i_indices)
        j_indices = np.array(j_indices)

        from tqdm import tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(MolecularDynamics._compute_partial_rdf)(atoms, i_indices, j_indices, r_max)
            for atoms in tqdm(atoms_list, desc="Computing RDF per frame")
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