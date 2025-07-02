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
from ase.mep import NEB
#from ase.mep.neb import NEB, NEBTools, NEBOptimizer
import os
from mlipx.abc import ComparisonResults, NodeWithCalculator, Optimizer


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
    model: NodeWithCalculator = zntrack.deps()
    n_images: int = zntrack.params(5)
    k: float = zntrack.params(0.1)
    relax: bool = zntrack.params(True)
    optimizer: Optimizer = zntrack.params(Optimizer.NEBOptimizer.value)
    optimizer_fallback: Optimizer = zntrack.params(Optimizer.FIRE.value)
    fmax: float = zntrack.params(0.04)
    n_steps: int = zntrack.params(300)
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "images.xyz")
    trajectory_path: pathlib.Path = zntrack.outs_path(
        zntrack.nwd / "neb_trajectory.traj"
    )
    results: pd.DataFrame = zntrack.plots(y="potential_energy", x="data_id")

    def run(self):
        frames = []
        calc = self.model.get_calculator()
        
        from ase.io import read, write
        initial = read(self.data_path, index=0)  # first image
        final = read(self.data_path, index=-1)  # last image
        
        images = [initial] + [initial.copy() for i in range(self.n_images)] + [final]
        
        neb = NEB(images, k=self.k)
        
        initial.calc = copy(calc)
        final.calc = copy(calc)

        try:
            optimizer = getattr(ase.optimize, self.optimizer)
        except AttributeError:
            optimizer = getattr(ase.mep.neb, self.optimizer)
            
            
        optimizer_fallback = getattr(ase.optimize, self.optimizer_fallback)
        
        dyn_initial = optimizer_fallback(initial)
        dyn_initial.run(fmax=self.fmax)
        
        dyn_final = optimizer_fallback(final)
        dyn_final.run(fmax=self.fmax)
        
        neb.interpolate()
        for image in images[1:len(images) - 1]:
            image.calc = copy(calc)
            image.get_potential_energy()

        

                
        if optimizer == ase.mep.neb.NEBOptimizer:
            dyn = optimizer(neb, trajectory=self.trajectory_path.as_posix(), method='ode')
        else:
            dyn = optimizer(neb, trajectory=self.trajectory_path.as_posix())
            
        dyn.run(fmax=self.fmax, steps=self.n_steps)
        
        for image in neb.images:
            frames += [image]
        
        ase.io.write(self.frames_path, images)
            
        # forces = neb.get_forces()
        # max_force = max(np.linalg.norm(f) for f in forces)

        # # Run fallback only if not converged
        # if max_force > self.fmax:
        #     print(f"NEBOptimizer did not converge (fmax = {max_force:.4f}), triggering fallback: {self.optimizer_fallback}")
        #     dyn_fallback = optimizer_fallback(neb, trajectory=self.trajectory_path.as_posix())
        #     dyn_fallback.run(fmax=self.fmax, steps=self.n_steps)
        # else:
        #     print(f"NEBOptimizer converged (fmax = {max_force:.4f}), skipping fallback.")


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





    @property
    def trajectory_frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.trajectory_path, "rb") as f:
            return list(ase.io.iread(f, format="traj"))

    @property
    def images(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))

    @property
    def figures(self) -> dict[str, go.Figure]:
        fig = px.scatter(self.results, x="data_id", y="potential_energy")
        fig.update_layout(title="NEB_path")
        fig.update_traces(customdata=np.stack([np.arange(len(self.results))], axis=1))
        return {"NEB_path": fig}

    @property
    def traj_plots(self) -> dict[str, go.Figure]:
        trajectory_frames = self.trajectory_frames
        total_iterations = len(trajectory_frames) // len(self.frames)
        neb_length = len(self.frames)
        figure = go.Figure()
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
                    customdata=np.stack([np.arange(len(energies)) + offset], axis=1),
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







    @staticmethod
    def benchmark_precompute(node_dict, cache_dir="app_cache/nebs/nebs_cache", ui=None, run_interactive=False, report=False, normalise_to_model=None):
        """
        Processes NEB nodes, saves summary and full NEB data as pickles.
        - mae_summary.pkl: DataFrame with model names and energy barriers.
        - neb_df.pkl: Dict with model names mapped to energies and reaction coordinates.
        """
        import pandas as pd
        import os
        os.makedirs(cache_dir, exist_ok=True)

        barrier_dict = []
        neb_data_dict = {}

        for model_name, node in node_dict.items():
            images = node.images
            rel_energies = [atoms.get_potential_energy() - images[0].get_potential_energy() for atoms in images]
            barrier = max(rel_energies) - min(rel_energies)
            barrier_dict.append({"Model": model_name, "Barrier (eV)": barrier})
            neb_data_dict[model_name] = {
                "rel_energies": rel_energies,
                "reaction_coords": list(range(len(rel_energies))),
            }
            
            # save images for weas viewer
            #save_dir = f"assets/{model_name}/si_interstitials"
            save_dir = f"assets"
            os.makedirs(save_dir, exist_ok=True)
            for i, atoms in enumerate(node.images):
                write(f"{save_dir}/image_{i}.xyz", atoms, format="xyz")

        # Save MAE-style summary table
        mae_df = pd.DataFrame(barrier_dict).sort_values(by="Barrier (eV)", ascending=True).round(3)
        mae_df.to_pickle(os.path.join(cache_dir, "mae_summary.pkl"))

        # Save full NEB data
        pd.to_pickle(neb_data_dict, os.path.join(cache_dir, "neb_df.pkl"))
    
    
    
    

    # @staticmethod
    # def launch_dashboard(cache_dir="app_cache/nebs/nebs_cache", ui=None):
    #     import pandas as pd
    #     from dash import Dash
    #     from mlipx.dash_utils import run_app, dash_table_interactive
    #     from dash import html, dcc, Input, Output, State
    #     import plotly.graph_objects as go
    #     import dash

    #     # Read cached files
    #     mae_df = pd.read_pickle(f"{cache_dir}/mae_summary.pkl")
    #     neb_df = pd.read_pickle(f"{cache_dir}/neb_df.pkl")

    #     app = Dash(__name__, assets_folder='assets')
    #     app.server.static_folder = "assets"
    #     app.server.static_url_path = "/assets"

    #     app.layout = dash_table_interactive(
    #         df=mae_df,
    #         id="lat-mae-score-table",
    #         title="NEB Energy Barriers Summary Table",
    #         extra_components=[
    #             html.Div(id="lattice-const-table"),
    #             dcc.Store(id="lattice-table-last-clicked", data=None),
    #             dcc.Store(id="current-neb-model", data=None),
    #             html.Div(
    #                 dcc.Graph(id="neb-plot"),
    #                 id="neb-plot-container",
    #                 style={"display": "none"},
    #             ),
    #             html.Div(id="weas-viewer", style={"display": "none"}),
    #         ],
    #     )

    #     @app.callback(
    #         Output("neb-plot", "figure"),
    #         Output("neb-plot-container", "style"),
    #         Output("lattice-table-last-clicked", "data"),
    #         Output("current-neb-model", "data"),
    #         Input("lat-mae-score-table", "active_cell"),
    #         State("lattice-table-last-clicked", "data")
    #     )
    #     def update_neb_plot(active_cell, last_clicked):
    #         if active_cell is None:
    #             raise dash.exceptions.PreventUpdate

    #         row = active_cell["row"]
    #         col = active_cell["column_id"]
    #         model_name = mae_df.iloc[row]["Model"]
    #         if col not in mae_df.columns or col == "Model":
    #             # Hide plot if not a model row
    #             return dash.no_update, {"display": "none"}, active_cell, model_name
    #         if last_clicked is not None and (
    #             active_cell["row"] == last_clicked.get("row") and
    #             active_cell["column_id"] == last_clicked.get("column_id")
    #         ):
    #             # Hide plot if repeated click
    #             return dash.no_update, {"display": "none"}, None, model_name

    #         neb_data = neb_df[model_name]
    #         rel_energies = neb_data["rel_energies"]
    #         reaction_coords = neb_data["reaction_coords"]
    #         fig = go.Figure(go.Scatter(x=reaction_coords, y=rel_energies, mode="lines+markers", name="NEB Path"))
    #         fig.update_layout(title=f"{model_name}: NEB Energy Path", xaxis_title="Image Index", yaxis_title="Energy (eV)")
    #         # Make neb plot visible
    #         return fig, {"display": "block"}, active_cell, model_name

    #     @app.callback(
    #         Output("weas-viewer", "children"),
    #         Output("weas-viewer", "style"),
    #         Input("neb-plot", "clickData"),
    #         State("current-neb-model", "data")
    #     )
    #     def update_weas_viewer(clickData, model_name):
    #         if clickData is None or model_name is None:
    #             # Hide viewer if no click
    #             return dash.no_update, {"display": "none"}

    #         index = int(clickData["points"][0]["x"])
    #         filename = f"assets/{model_name}/si_interstitials/image_{index}.xyz"

    #         def generate_weas_html(fname):
    #             return f"""
    #             <!doctype html>
    #             <html lang="en">
    #             <head>
    #                 <meta charset="utf-8">
    #                 <title>WEAS Viewer</title>
    #             </head>
    #             <body>
    #                 <div id="viewer" style="width: 100%; height: 500px;"></div>
    #                 <script type="module">
    #                     const {{ WEAS, parseXYZ }} = await import('https://unpkg.com/weas/dist/index.mjs');
    #                     const viewer = new WEAS({{
    #                         domElement: document.getElementById('viewer'),
    #                         viewerConfig: {{ backgroundColor: [1,1,1,1], _modelStyle: 1 }},
    #                         guiConfig: {{ buttons: {{ enabled: false }} }}
    #                     }});
    #                     const response = await fetch("/" + fname);
    #                     const text = await response.text();
    #                     const atoms = parseXYZ(text);
    #                     viewer.avr.atoms = atoms;
    #                     viewer.render();
    #                 </script>
    #             </body>
    #             </html>
    #             """

    #         html_content = generate_weas_html(filename)
    #         return html.Iframe(
    #             srcDoc=html_content,
    #             style={"height": "550px", "width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}
    #         ), {"display": "block"}

        
    #     return run_app(app, ui=ui)


    @staticmethod
    def launch_dashboard(cache_dir="app_cache/nebs/nebs_cache", ui=None):
        import pandas as pd
        from dash import Dash
        from mlipx.dash_utils import run_app, dash_table_interactive
        from dash import html, dcc, Input, Output, State
        import plotly.graph_objects as go
        import dash
        import os

        # Read cached files
        mae_df = pd.read_pickle(f"{cache_dir}/mae_summary.pkl")
        neb_df = pd.read_pickle(f"{cache_dir}/neb_df.pkl")

        # Initialize app with proper static configuration
        app = Dash(__name__, assets_folder='assets')
        
        # Alternative static configuration - try one of these:
        # Option A: Configure static serving manually
        app.server.static_folder = os.path.abspath("assets")
        app.server.static_url_path = "/assets"
        
        # Option B: Use Dash's built-in assets serving (comment out Option A if using this)
        # app = Dash(__name__, assets_folder=os.path.abspath('assets'))

        app.layout = dash_table_interactive(
            df=mae_df,
            id="lat-mae-score-table",
            title="NEB Energy Barriers Summary Table",
            extra_components=[
                html.Div(id="lattice-const-table"),
                dcc.Store(id="lattice-table-last-clicked", data=None),
                dcc.Store(id="current-neb-model", data=None),
                html.Div(
                    dcc.Graph(id="neb-plot"),
                    id="neb-plot-container",
                    style={"display": "none"},
                ),
                html.Div(id="weas-viewer", style={"display": "none"}),
            ],
        )

        @app.callback(
            Output("neb-plot", "figure"),
            Output("neb-plot-container", "style"),
            Output("lattice-table-last-clicked", "data"),
            Output("current-neb-model", "data"),
            Input("lat-mae-score-table", "active_cell"),
            State("lattice-table-last-clicked", "data")
        )
        def update_neb_plot(active_cell, last_clicked):
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

            neb_data = neb_df[model_name]
            rel_energies = neb_data["rel_energies"]
            reaction_coords = neb_data["reaction_coords"]
            fig = go.Figure(go.Scatter(x=reaction_coords, y=rel_energies, mode="lines+markers", name="NEB Path"))
            fig.update_layout(title=f"{model_name}: NEB Energy Path", xaxis_title="Image Index", yaxis_title="Energy (eV)")
            return fig, {"display": "block"}, active_cell, model_name

        @app.callback(
            Output("weas-viewer", "children"),
            Output("weas-viewer", "style"),
            Input("neb-plot", "clickData"),
            State("current-neb-model", "data")
        )
        def update_weas_viewer(clickData, model_name):
            if clickData is None or model_name is None:
                return dash.no_update, {"display": "none"}

            index = int(clickData["points"][0]["x"])
            
            # Fix the file path - use app.get_asset_url() or serve differently
            # Option 1: Use Dash's asset serving
            #filename = f"/{model_name}/si_interstitials/image_{index}.xyz"
            filename = f"/image_{index}.xyz"
            asset_url = app.get_asset_url(filename)
            
            # Option 2: Alternative - check if file exists first
            #file_path = os.path.join("assets", model_name, "si_interstitials", f"image_{index}.xyz")
            file_path = os.path.join("assets", f"image_{index}.xyz")
            if not os.path.exists(file_path):
                return html.Div(f"File not found: {file_path}"), {"display": "block"}

            def generate_weas_html(fname):
                return f"""
                <!doctype html>
                <html lang="en">
                <head>
                    <meta charset="utf-8">
                    <title>WEAS Viewer</title>
                </head>
                <body>
                    <div id="viewer" style="width: 100%; height: 500px;"></div>
                    <script type="module">
                        try {{
                            const {{ WEAS, parseXYZ }} = await import('https://unpkg.com/weas/dist/index.mjs');
                            const viewer = new WEAS({{
                                domElement: document.getElementById('viewer'),
                                viewerConfig: {{ backgroundColor: [1,1,1,1], _modelStyle: 1 }},
                                guiConfig: {{ buttons: {{ enabled: false }} }}
                            }});
                            const response = await fetch("{fname}");
                            if (!response.ok) {{
                                throw new Error(`HTTP error! status: ${{response.status}}`);
                            }}
                            const text = await response.text();
                            const atoms = parseXYZ(text);
                            viewer.avr.atoms = atoms;
                            viewer.render();
                        }} catch (error) {{
                            document.getElementById('viewer').innerHTML = `<p>Error loading file: ${{error.message}}</p>`;
                            console.error('Error:', error);
                        }}
                    </script>
                </body>
                </html>
                """

            html_content = generate_weas_html(asset_url)
            return html.Iframe(
                srcDoc=html_content,
                style={"height": "550px", "width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}
            ), {"display": "block"}

        return run_app(app, ui=ui)