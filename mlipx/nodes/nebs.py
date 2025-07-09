from ase.io import read, write
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import os
import dash
import pathlib
from copy import copy

import ase.io
from ase.io import read, write
import ase.optimize
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import zntrack
#from ase.mep.neb import NEB, NEBTools, NEBOptimizer
import os
from mlipx.abc import ComparisonResults, NodeWithCalculator, Optimizer
import flask
from ase.io import read, write
from copy import deepcopy
import matplotlib.pyplot as plt
from ase.neb import NEBTools




class NEBinterpolate(zntrack.Node):
    """
    Interpolates between two or three images to create a NEB path.

    Parameters
    ----------
    data : list[ase.Atoms]
        List of atoms objects.
    n_images : int
        Number of images to interpolate.
    mic : bool
        Whether to use the minimum image convention.
    add_constraints : bool
        Whether to copy constraints from initial image to all the interpolated images
    frames_path : pathlib.Path
        Path to save the interpolated frames.

    """

    data: list[ase.Atoms] = zntrack.deps()
    n_images: int = zntrack.params(5)
    mic: bool = zntrack.params(False)
    add_constraints: bool = zntrack.params(True)
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "initial_frames.xyz")

    def run(self):
        frames = []
        if len(self.data) == 2:
            atoms = self.data[0]
            for i in range(self.n_images - 1):
                atoms_copy = atoms.copy()
                frames += [atoms_copy]
            atoms_copy = self.data[1]
            frames += [atoms_copy]
        elif len(self.data) == 3:
            atoms0 = self.data[0]
            atoms1 = self.data[1]
            atoms2 = self.data[2]
            ts_index = self.n_images // 2
            for i in range(ts_index - 1):
                atoms_copy = atoms0.copy()
                frames += [atoms_copy]
            for i in range(ts_index, self.n_images):
                atoms_copy = atoms1.copy()
                frames += [atoms_copy]
            atoms_copy = atoms2.copy()
            frames += [atoms_copy]
        if self.add_constraints is True:
            _constraints = self.data[0].constraints
            for image in frames:
                image.set_constraint(_constraints)

        neb = NEB(frames)
        neb.interpolate(mic=self.mic, apply_constraint=self.add_constraints)
        ase.io.write(self.frames_path, frames)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))


class NEBs(zntrack.Node):
    """
    Runs NEB calculation on a list of images.

    Parameters
    ----------
    data : list[ase.Atoms]
        List of atoms objects.
    model : NodeWithCalculator
        Node with a calculator.
    relax : bool
        Whether to relax the initial and final images.
    optimizer : Optimizer
        ASE optimizer to use.
    fmax : float
        Maximum force allowed.
    n_steps : int
        Maximum number of steps allowed.
    frames_path : pathlib.Path
        Path to save the final frames.
    trajectory_path : pathlib.Path
        Path to save the neb trajectory file.

    Attributes
    ----------
    results : pd.DataFrame
        DataFrame with the data_id and potential energy of the NEB calculation

    """

    data: list[ase.Atoms] = zntrack.deps()
    model: NodeWithCalculator = zntrack.deps()
    k: float = zntrack.params(0.1)
    relax: bool = zntrack.params(True)
    optimizer: Optimizer = zntrack.params(Optimizer.NEBOptimizer.value)
    optimizer_fallback: Optimizer = zntrack.params(Optimizer.FIRE.value)
    fmax: float = zntrack.params(0.04)
    n_steps: int = zntrack.params(300)
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")
    trajectory_path: pathlib.Path = zntrack.outs_path(
        zntrack.nwd / "neb_trajectory.traj"
    )
    results: pd.DataFrame = zntrack.plots(y="potential_energy", x="data_id")

    def run(self):
        frames = []
        # neb_trajectory = []
        calc = self.model.get_calculator()
        
        try:
            optimizer = getattr(ase.optimize, self.optimizer)
        except AttributeError:
            optimizer = getattr(ase.mep.neb, self.optimizer)
            
            
        optimizer_fallback = getattr(ase.optimize, self.optimizer_fallback)
        
        for atoms in self.data:
            atoms_copy = atoms.copy()
            atoms_copy.calc = copy(calc)
            atoms_copy.get_potential_energy()
            frames += [atoms_copy]
            ase.io.write(self.frames_path, atoms_copy, format="extxyz", append=True)
        if self.relax is True:
            for i in [0, -1]:
                dyn = optimizer_fallback(frames[0])
                dyn.run(fmax=self.fmax)
                
        neb = NEB(frames, k = self.k)

        if optimizer == ase.mep.neb.NEBOptimizer:
            dyn = optimizer(neb, trajectory=self.trajectory_path.as_posix(), method='ode')
        else:
            dyn = optimizer(neb, trajectory=self.trajectory_path.as_posix())
        dyn.run(fmax=self.fmax, steps=self.n_steps)
        
        
        forces = neb.get_forces()
        max_force = max(np.linalg.norm(f) for f in forces)

        # Run fallback only if not converged
        if max_force > self.fmax:
            print(f"NEBOptimizer did not converge (fmax = {max_force:.4f}), triggering fallback: {self.optimizer_fallback}")
            dyn_fallback = optimizer_fallback(neb, trajectory=self.trajectory_path.as_posix())
            dyn_fallback.run(fmax=self.fmax, steps=self.n_steps)
        else:
            print(f"NEBOptimizer converged (fmax = {max_force:.4f}), skipping fallback.")


        row_dicts = []
        for i, frame in enumerate(frames):
            row_dicts.append(
                {
                    "data_id": i,
                    "potential_energy": frame.get_potential_energy(),
                    "neb_adjusted_energy": frame.get_potential_energy()
                    - frames[0].get_potential_energy(),
                },
            )
        self.results = pd.DataFrame(row_dicts)
        
        
        
        

class NEB2(zntrack.Node):
    """
    Runs NEB calculation on a list of images.

    Parameters
    ----------
    data : list[ase.Atoms]
        List of atoms objects.
    model : NodeWithCalculator
        Node with a calculator.
    relax : bool
        Whether to relax the initial and final images.
    optimizer : Optimizer
        ASE optimizer to use.
    fmax : float
        Maximum force allowed.
    n_steps : int
        Maximum number of steps allowed.
    frames_path : pathlib.Path
        Path to save the final frames.
    trajectory_path : pathlib.Path
        Path to save the neb trajectory file.

    Attributes
    ----------
    results : pd.DataFrame
        DataFrame with the data_id and potential energy of the NEB calculation

    """

    data_path: str = zntrack.params()
    #all_images: bool = zntrack.params(True)
    model: NodeWithCalculator = zntrack.deps()
    n_images: int = zntrack.params(5)
    k: float = zntrack.params(0.1)
    relax: bool = zntrack.params(True)
    optimizer: Optimizer = zntrack.params(Optimizer.NEBOptimizer.value)
    optimizer_fallback: Optimizer = zntrack.params(Optimizer.FIRE.value)
    fmax: float = zntrack.params(0.04)
    n_steps: int = zntrack.params(300)
    use_janus: bool = zntrack.params(False)
    
    
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "images.xyz")
    # trajectory_path: pathlib.Path = zntrack.outs_path(
    #     zntrack.nwd / "neb_trajectory.traj"
    # )
    results: pd.DataFrame = zntrack.plots(y="potential_energy", x="data_id")

    def run(self):
        calc = self.model.get_calculator()
        
        initial = read(self.data_path, index=0)
        final = read(self.data_path, index=-1)
        initial.calc = calc
        final.calc = calc
        
        
        if self.use_janus:
            from janus_core.calculations.neb import NEB
            interpolator = "pymatgen"
            neb = NEB(
                init_struct=initial,
                final_struct=final,
                n_images=self.n_images,
                interpolator=interpolator,
                minimize=True,
                fmax=self.fmax,
                write_kwargs={
                    "filename": self.frames_path,}
            )
            neb.run()

            ase.io.write(self.frames_path, neb.images)            
        else:
            from ase.mep import NEB


            try:
                optimizer = getattr(ase.optimize, self.optimizer)
            except AttributeError:
                optimizer = getattr(ase.mep.neb, self.optimizer)
                
            optimizer_fallback = getattr(ase.optimize, self.optimizer_fallback)
            
            dyn_initial = optimizer_fallback(initial)
            dyn_initial.run(fmax=self.fmax)
            
            dyn_final = optimizer_fallback(final)
            dyn_final.run(fmax=self.fmax)

            # Initial NEB interpolation
            interpolated = NEB2.add_intermediary_images([initial, final], 1e-10, max_number=self.n_images-2)
            images = interpolated[1]
            for image in images:
                image.set_calculator(deepcopy(calc))
            
            # NEB calculations
            neb = NEB(images=images, climb=False)
            
            if optimizer == ase.mep.neb.NEBOptimizer:
                print("Using NEBOptimizer with ODE method")
                opt = optimizer(neb, method='ode', trajectory=zntrack.nwd / 'neb.traj')
            else:
                opt = optimizer(neb, trajectory=zntrack.nwd / 'neb.traj')
            
            opt.run(fmax=1, steps=500)


                
            neb.climb = True
            print('Climbing NEB:')
            converged = opt.run(fmax=0.05, steps=700)
            write(zntrack.nwd / 'neb_final_climb.xyz', images)

            # Plot NEB band and get barriers
            fig, ax = plt.subplots()
            nt = NEBTools(images)
            nt.plot_band()
            Ef_NEB, deltaE_NEB = nt.get_barrier()
            plt.savefig(zntrack.nwd / 'neb-climb.png')
            
            ase.io.write(self.frames_path, images, format="extxyz")
            

        row_dicts = []
        for i, frame in enumerate(images):
            row_dicts.append(
                {
                    "data_id": i,
                    "potential_energy": frame.get_potential_energy(),
                    "neb_adjusted_energy": frame.get_potential_energy()
                    - images[0].get_potential_energy(),
                    'converged': converged,
                },
            )
        self.results = pd.DataFrame(row_dicts)


    # functions from Lars
    @staticmethod
    def get_distances_between_images(imagesi):
        """Returns distance between each image ie 2norm of d2-d1"""

        spring_lengths = []
        for j in range(len(imagesi) - 1):
            spring_vec = imagesi[j + 1].get_positions() - imagesi[j].get_positions()
            spring_lengths.append(np.linalg.norm(spring_vec))
        return np.array(spring_lengths)

    @staticmethod
    def add_intermediary_images(
        imagesi, dist_cutoff, interpolate_method="idpp", max_number=100, verbose=False,
    ):
        """Add additional images inbetween existing ones, purely based on geometry"""
        # create copy of images
        imagesi = [at.copy() for at in imagesi]
        interp_images = []
        max_dist_images = max(NEB2.get_distances_between_images(imagesi))
        for iter in range(max_number):
            if max_dist_images <= dist_cutoff:
                print(f"Max distance readched after {iter} iterations")
                break
            distances = NEB2.get_distances_between_images(imagesi)
            jmax = np.argmax(distances)

            toInterpolate = [imagesi[jmax]]
            toInterpolate += [toInterpolate[0].copy()]
            toInterpolate += [imagesi[jmax + 1]]

            from ase.mep import NEB
            neb = NEB(toInterpolate)
            neb.interpolate(method=interpolate_method, apply_constraint=True)

            interp_images.append([jmax, toInterpolate[1].copy()])
            # Add images
            imagesi.insert(jmax + 1, toInterpolate[1].copy())
            if verbose:
                print(f"Additional image added at {jmax} with distances {max(distances)}")
            max_dist_images = max(NEB2.get_distances_between_images(imagesi))

        return interp_images, imagesi



    # @property
    # def trajectory_frames(self) -> list[ase.Atoms]:
    #     with self.state.fs.open(self.trajectory_path, "rb") as f:
    #         return list(ase.io.iread(f, format="traj"))

    @property
    def images(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))

    @property
    def figures(self) -> dict[str, go.Figure]:
        # Use Plotly Express to create the scatter plot
        x_values = self.results["data_id"] if "data_id" in self.results else list(range(len(self.results)))
        y_values = self.results["potential_energy"] if "potential_energy" in self.results else [0]*len(x_values)
        system_name = getattr(self, 'name', 'system')
        fig = px.scatter(self.results, x="data_id", y="potential_energy")
        fig.update_layout(title="NEB_path")
        # Add customdata and hovertemplate
        fig.update_traces(
            customdata=[system_name] * len(x_values),
            hovertemplate="Image %{x}, Energy %{y} eV<br>%{customdata}<extra></extra>"
        )
        return {"NEB_path": fig}

    @property
    def traj_plots(self) -> dict[str, go.Figure]:
        trajectory_frames = self.trajectory_frames
        total_iterations = len(trajectory_frames) // len(self.frames)
        neb_length = len(self.frames)
        figure = go.Figure()
        system_name = getattr(self, 'name', 'system')
        for iteration in range(total_iterations):
            images = trajectory_frames[
                iteration * neb_length : (iteration + 1) * neb_length
            ]
            energies = [image.get_potential_energy() for image in images]
            offset = iteration * neb_length
            figure.add_trace(
                go.Scatter(
                    x=list(range(len(energies))),
                    y=energies,
                    mode="lines+markers",
                    name=f"{iteration}",
                    customdata=[system_name] * len(energies),
                )
            )
        figure.update_layout(
            title="Energy vs. NEB image",
            xaxis_title="image number",
            yaxis_title="Energy",
        )
        return {"energy_vs_iteration": figure}

    @staticmethod
    def compare(*nodes: "NEBs") -> ComparisonResults:
        frames = sum([node.frames for node in nodes], [])
        offset = 0
        fig = go.Figure()
        for idx, node in enumerate(nodes):
            energies = [atoms.get_potential_energy() for atoms in node.frames]
            system_name = getattr(node, 'name', 'system')
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(energies))),
                    y=energies,
                    mode="lines+markers",
                    name=node.name.replace(f"_{node.__class__.__name__}", ""),
                    customdata=[system_name] * len(energies),
                )
            )
            offset += len(energies)

        fig.update_layout(
            title="Energy vs. NEB image",
            xaxis_title="image number",
            yaxis_title="Energy",
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

        # Now adjusted

        fig_adjusted = go.Figure()
        offset = 0
        for _, node in enumerate(nodes):
            energies = np.array([atoms.get_potential_energy() for atoms in node.frames])
            energies -= energies[0]
            system_name = getattr(node, 'name', 'system')
            fig_adjusted.add_trace(
                go.Scatter(
                    x=list(range(len(energies))),
                    y=energies,
                    mode="lines+markers",
                    name=node.name.replace(f"_{node.__class__.__name__}", ""),
                    customdata=[system_name] * len(energies),
                )
            )
            offset += len(energies)

        fig_adjusted.update_layout(
            title="Adjusted energy vs. NEB image",
            xaxis_title="Image number",
            yaxis_title="Adjusted energy",
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
            figures={
                "energy_vs_neb_image": fig,
                "adjusted_energy_vs_neb_image": fig_adjusted,
            },
        )




        ref_data = {
            "Si_64": [],
            "Si_216": [],
            "LiFePO4_b": [0.27],
            #"LiFePO4_c": [2.5],
            
        }















    @staticmethod
    def benchmark_precompute(
        node_dict, 
        cache_dir="app_cache/nebs_further_apps/nebs_cache",
        ui=None, 
        run_interactive=True, 
        report=False, 
        normalise_to_model=None
    ):
        """
        Processes NEB nodes, saves summary and full NEB data as pickles.
        - mae_summary.pkl: DataFrame with model names and energy barriers.
        - neb_df.pkl: Dict with model names mapped to energies and reaction coordinates.
        """
        import pandas as pd
        import os
        from collections import defaultdict
        os.makedirs(cache_dir, exist_ok=True)

        ref_data = {
            "Si_64": None,
            "Si_216": None,
            "LiFePO4_start_end_b": 0.27,
            "LiFePO4_start_end_c": 2.5,
        }

        # Define custom column labels for barrier error
        custom_col_names = {
            "Si_64": "Si_64 ΔE (eV)",
            "Si_216": "Si_216 ΔE (eV)",
            "_LiFePO4_start_end_b_": "LiFePO4_b ΔE (eV)",
            "_LiFePO4_start_end_c_": "LiFePO4_c ΔE (eV)",
        }

        #for model, system_dict in node_dict.items():
            #print(f"\nProcessing model: {model}")
            #print(f"Systems: {list(system_dict.keys())}")

        # === Define system groups manually ===
        system_groups = {
            "Si_64": "Si",
            "Si_216": "Si",
            #"LiFePO4_b": "LiFePO4",
            "_LiFePO4_start_end_b_": "LiFePO4",
            "_LiFePO4_start_end_c_": "LiFePO4",
        }

        # === Group nodes by system group ===
        grouped_nodes = defaultdict(lambda: defaultdict(dict))  # group -> model -> system -> node
        for model, system_dict in node_dict.items():
            for system, node in system_dict.items():
                #system = system.strip("_")
                # Always use stripped keys for consistency
                clean_system = system.strip("_")
                normalized_system_groups = {k.strip("_"): v for k, v in system_groups.items()}
                group = normalized_system_groups.get(clean_system, clean_system)

                # Store using clean system name (stripped version)
                grouped_nodes[group][model][clean_system] = node

        # --- Add: all_group_data dict ---
        all_group_data = {}
        
        #print(grouped_nodes)

        for group_name, models in grouped_nodes.items():

            from collections import defaultdict
            barrier_dict = defaultdict(dict)
            neb_data_dict = {}

            for model_name, system_nodes in models.items():
                for system_name, node in system_nodes.items():
                    images = node.images
                    rel_energies = [atoms.get_potential_energy() - images[0].get_potential_energy() for atoms in images]
                    barrier = max(rel_energies) - min(rel_energies)
                    barrier_dict[model_name]["Model"] = model_name
                    barrier_dict[model_name][f"{system_name} Barrier (eV)"] = barrier
                    key = f"{model_name}::{system_name}"
                    neb_data_dict[key] = {
                        "rel_energies": rel_energies,
                        "reaction_coords": list(range(len(rel_energies))),
                    }

                    # save images for WEAS viewer
                    save_dir = os.path.abspath(f"assets/{model_name}/{group_name}/{system_name}")
                    os.makedirs(save_dir, exist_ok=True)
                    write(os.path.join(save_dir, "images.xyz"), images, format='extxyz')

            # Save group summary
            df = pd.DataFrame(list(barrier_dict.values()))

            barrier_cols = [col for col in df.columns if "Barrier" in col and col.endswith("(eV)")]
            if barrier_cols:
                mae_df = df.sort_values(by=barrier_cols[0], ascending=True)
            else:
                mae_df = df

            # Add error columns for all barrier columns, supporting multiple barriers per system,
            # and drop the original "Barrier (eV)" columns, keeping only error columns and Model.
            for col in list(mae_df.columns):
                if "Barrier" in col and col.endswith("(eV)"):
                    base_label = col.split(" Barrier")[0].strip()
                    if base_label in ref_data and ref_data[base_label] is not None:
                        ref_val = ref_data[base_label]
                        error_col = f"{col} Error"
                        mae_df[error_col] = mae_df[col].apply(lambda x: abs(x - ref_val) if pd.notna(x) else None)
                        mae_df.drop(columns=[col], inplace=True)

            # Add score column as average of error columns
            error_cols = [col for col in mae_df.columns if col.endswith("Error")]
            if error_cols:
                mae_df[f"{group_name} Score \u2193"] = mae_df[error_cols].mean(axis=1)
                #print(f"Group {group_name} Score: {mae_df[f'{group_name} Score \u2193'].mean():.3f}")

                # Normalize group score if a reference model is provided
                if normalise_to_model and f"{group_name} Score \u2193" in mae_df.columns:
                    ref_val = mae_df.loc[mae_df["Model"] == normalise_to_model, f"{group_name} Score \u2193"]
                    if not ref_val.empty and ref_val.iloc[0] != 0:
                        mae_df[f"{group_name} Score \u2193"] /= ref_val.iloc[0]
                        
                    
            # Instead of saving individual files, store in all_group_data:
            all_group_data[group_name] = (mae_df.round(3), neb_data_dict, os.path.abspath("assets"))


            
            
        # After all groups, save all_group_data as one pickle
        import pickle
            
        with open(os.path.join(cache_dir, "all_group_data.pkl"), "wb") as f:
            pickle.dump(all_group_data, f)
        with open(os.path.join(cache_dir, "all_group_data.pkl"), "rb") as f:
            all_group_data = pickle.load(f)
        return
    






    @staticmethod
    def launch_dashboard(cache_dir="app_cache/nebs_further_apps/nebs_cache", ui=None):
        import pandas as pd
        from dash import Dash
        from mlipx.dash_utils import run_app
        import dash
        import os
        import pickle

        #print(os.getcwd())

        # Load all groups from single pickle file
        with open(os.path.join(cache_dir, "all_group_data.pkl"), "rb") as f:
            all_group_data = pickle.load(f)

        #for group_name, (mae_df, _, _) in all_group_data.items():
            #print(f"\nGroup: {group_name}")
            #print(mae_df)
            
            
        # Use assets_dir from the first group for Dash assets
        first_assets_dir = next(iter(all_group_data.values()))[2]
        print("Serving assets from:", first_assets_dir)
        app = dash.Dash(__name__, assets_folder=first_assets_dir)
        #app.server.static_folder = 'assets'
        #app.server.static_url_path = '/assets'

        # Set layout using the static method
        app.layout = NEB2.build_layout(all_group_data)
        # Register callbacks using the static method
        NEB2.register_callbacks(app, all_group_data)

        return run_app(app, ui=ui)




    @staticmethod
    def build_layout(all_group_data):
        from dash import html, dcc
        from mlipx.dash_utils import dash_table_interactive
        return html.Div([
            dcc.Tabs([
                dcc.Tab(label=group_name, children=[
                    dash_table_interactive(
                        df=mae_df,
                        id=f"neb-mae-score-table-{group_name}",
                        title=f"{group_name} Path b and c Energy Barriers Error Table",
                        extra_components=[
                            html.Div(id=f"neb-table-{group_name}"),
                            dcc.Store(id=f"neb-table-last-clicked-{group_name}", data=None),
                            dcc.Store(id=f"current-neb-model-{group_name}", data=None),
                            html.Div(
                                dcc.Graph(id=f"neb-plot-{group_name}"),
                                id=f"neb-plot-container-{group_name}",
                                style={"display": "none"}
                            ),
                            html.Div(id=f"weas-viewer-{group_name}", style={'marginTop': '20px'}),
                        ],
                    )
                ]) for group_name, (mae_df, _, _) in all_group_data.items()
            ])
        ])










    @staticmethod
    def register_callbacks(app, all_group_data):
        from dash import Output, Input, State, exceptions
        import plotly.graph_objects as go
        for group_name, (mae_df, neb_df, assets_dir) in all_group_data.items():
            # Prepare custom_col_names mapping for this group
            # Try to infer from mae_df columns if available
            custom_col_names = {}
            for col in mae_df.columns:
                if col == "Model":
                    continue
                # Try to match to system names in neb_df keys
                # neb_df keys are of the form "model_name::system_name"
                for key in neb_df.keys():
                    # key = "model_name::system_name"
                    _model, _system = key.split("::", 1)
                    if col.startswith(_system) or _system in col:
                        custom_col_names[_system] = col
            @app.callback(
                Output(f"neb-plot-{group_name}", "figure"),
                Output(f"neb-plot-container-{group_name}", "style"),
                Output(f"neb-table-last-clicked-{group_name}", "data"),
                Output(f"current-neb-model-{group_name}", "data"),
                Input(f"neb-mae-score-table-{group_name}", "active_cell"),
                State(f"neb-table-last-clicked-{group_name}", "data"),
            )
            def update_neb_plot(active_cell, last_clicked, group_name=group_name, mae_df=mae_df, neb_df=neb_df, custom_col_names=custom_col_names):
                import dash
                if active_cell is None:
                    raise dash.exceptions.PreventUpdate
                row = active_cell["row"]
                col = active_cell["column_id"]
                model_name = mae_df.iloc[row]["Model"]
                if col not in mae_df.columns or col == "Model":
                    return dash.no_update, {"display": "none"}, active_cell, model_name
                if last_clicked is not None and (
                    active_cell["row"] == last_clicked.get("row") and
                    active_cell["column_id"] == last_clicked.get("column_id")
                ):
                    return dash.no_update, {"display": "none"}, None, model_name
                # Reverse-map from custom column names
                reverse_map = {v: k for k, v in custom_col_names.items()}
                system_name = reverse_map.get(col, col)
                key = f"{model_name}::{system_name}"
                neb_data = neb_df[key]
                rel_energies = neb_data["rel_energies"]
                reaction_coords = neb_data["reaction_coords"]
                fig = go.Figure(go.Scatter(
                    x=reaction_coords,
                    y=rel_energies,
                    mode="lines+markers",
                    name="NEB Path",
                    customdata=[system_name] * len(reaction_coords),
                    hovertemplate="Image %{x}, Energy %{y} eV<br>%{customdata}<extra></extra>"
                ))
                fig.update_layout(title=f"{model_name}: NEB Energy Path", xaxis_title="Image Index", yaxis_title="Energy (eV)")
                return fig, {"display": "block"}, active_cell, model_name

            # WEAS viewer callback
            @app.callback(
                Output(f"weas-viewer-{group_name}", "children"),
                Output(f"weas-viewer-{group_name}", "style"),
                Input(f"neb-plot-{group_name}", "clickData"),
                State(f"current-neb-model-{group_name}", "data"),
            )
            def update_weas_viewer(clickData, model_name, group_name=group_name):
                import dash
                if clickData is None or model_name is None:
                    raise dash.exceptions.PreventUpdate
                index = int(clickData["points"][0]["x"])
                point_data = clickData["points"][0]
                system_name = point_data.get("customdata", "unknown")
                system_names = [system_name]
                children = []
                for system_name in system_names:
                    filename = f"/assets/{model_name}/{group_name}/{system_name}/images.xyz"
                    def generate_weas_html(filename, current_frame):
                        return f"""
                        <!doctype html>
                        <html lang="en">
                        <head>
                            <meta charset="utf-8">
                            <title>WEAS Viewer</title>
                        </head>
                        <body>
                            <div id="viewer" style="position: relative; width: 100%; height: 500px; border: 1px solid #ccc;"></div>
                            <div id="debug" style="margin-top: 10px; padding: 10px; background: #f0f0f0; font-family: monospace; display: none;"></div>
                            <script type="module">
                                async function fetchFile(filename) {{
                                    try {{
                                        const response = await fetch(filename);
                                        if (!response.ok) {{
                                            throw new Error(`Failed to load file: ${{filename}} - ${{response.status}}`);
                                        }}
                                        const text = await response.text();
                                        // Debug: show file content
                                        console.log('File content:', text);
                                        document.getElementById("debug").innerHTML = 
                                            `<strong>File content (first 500 chars):</strong><br><pre>${{text.substring(0, 500)}}</pre>`;
                                        document.getElementById("debug").style.display = 'block';
                                        return text;
                                    }} catch (error) {{
                                        console.error('Error fetching file:', error);
                                        throw error;
                                    }}
                                }}
                                function validateXYZ(content) {{
                                    const lines = content.trim().split('\\n');
                                    if (lines.length < 2) {{
                                        throw new Error('XYZ file too short');
                                    }}
                                    const numAtoms = parseInt(lines[0]);
                                    if (isNaN(numAtoms)) {{
                                        throw new Error('First line must be number of atoms');
                                    }}
                                    if (lines.length < numAtoms + 2) {{
                                        throw new Error(`Expected ${{numAtoms + 2}} lines, got ${{lines.length}}`);
                                    }}
                                    // Check coordinate lines
                                    for (let i = 2; i < numAtoms + 2; i++) {{
                                        const parts = lines[i].trim().split(/\\s+/);
                                        if (parts.length < 4) {{
                                            throw new Error(`Line ${{i+1}}: Expected element + 3 coordinates, got ${{parts.length}} parts`);
                                        }}
                                    }}
                                    return true;
                                }}
                                try {{
                                    const {{ WEAS, parseXYZ }} = await import('https://unpkg.com/weas/dist/index.mjs');
                                    const domElement = document.getElementById("viewer");
                                    const editor = new WEAS({{
                                        domElement,
                                        viewerConfig: {{ 
                                            _modelStyle: 2,
                                            backgroundColor: [1, 1, 1, 1]
                                        }},
                                        guiConfig: {{ 
                                            buttons: {{ enabled: false }} 
                                        }}
                                    }});
                                    const structureData = await fetchFile("{filename}");
                                    // Validate XYZ format before parsing
                                    validateXYZ(structureData);
                                    const atoms = parseXYZ(structureData);
                                    editor.avr.atoms = atoms;
                                    editor.avr.modelStyle = 1;
                                    editor.avr.currentFrame = {index};
                                    editor.render();
                                    // Hide debug info if successful
                                    document.getElementById("debug").style.display = 'none';
                                }} catch (error) {{
                                    console.error('Error initializing WEAS:', error);
                                    document.getElementById("viewer").innerHTML = 
                                        `<div style="padding: 20px; color: red;">
                                            <strong>Error loading structure:</strong><br>
                                            ${{error.message}}
                                            <br><br>
                                            <small>Check the browser console for more details.</small>
                                        </div>`;
                                }}
                            </script>
                        </body>
                        </html>
                        """
                    html_content = generate_weas_html(filename, current_frame=index)
                    from dash import html as dash_html
                    children.append(
                        dash_html.Div([
                            dash_html.H4(f"Structure {index}", style={'textAlign': 'center'}),
                            dash_html.Iframe(
                                srcDoc=html_content,
                                style={
                                    "height": "550px",
                                    "width": "100%",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "5px"
                                }
                            )
                        ])
                    )
                return (
                    children if len(children) > 1 else children[0],
                    {"marginTop": "20px"}
                )