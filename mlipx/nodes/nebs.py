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
    use_janus: bool = zntrack.params(False)
    
    
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "images.xyz")
    # trajectory_path: pathlib.Path = zntrack.outs_path(
    #     zntrack.nwd / "neb_trajectory.traj"
    # )
    results: pd.DataFrame = zntrack.plots(y="potential_energy", x="data_id")

    def run(self):
        frames = []
        calc = self.model.get_calculator()
        
        from ase.io import read, write
        initial = read(self.data_path, index=0)  # first image
        final = read(self.data_path, index=-1)  # last image
        
        initial.calc = copy(calc)
        final.calc = copy(calc)
        
        
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

            images = [initial] + [initial.copy() for i in range(self.n_images)] + [final]
            
            neb = NEB(images, k=self.k)
            


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
                print("Using NEBOptimizer with ODE method")
                #dyn = optimizer(neb, trajectory=self.trajectory_path.as_posix(), method='ode')
                dyn = optimizer(neb, method='ode')
            else:
                #dyn = optimizer(neb, trajectory=self.trajectory_path.as_posix())
                dyn = optimizer(neb)
                dyn = optimizer(neb)
                
            dyn.run(fmax=self.fmax, steps=self.n_steps)
            
            #dyn_fallback = optimizer_fallback(neb, trajectory=self.trajectory_path.as_posix())
            # dyn_fallback = optimizer_fallback(neb)
            # dyn_fallback.run(fmax=self.fmax, steps=self.n_steps)
            
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
            save_dir = os.path.abspath(f"assets/{model_name}/si_interstitials")
            os.makedirs(save_dir, exist_ok=True)
            #for i, atoms in enumerate(images):
            fname = f"{save_dir}/images.xyz"
            write(fname, images, format='xyz')
            

        # Save MAE-style summary table
        mae_df = pd.DataFrame(barrier_dict).sort_values(by="Barrier (eV)", ascending=True).round(3)
        mae_df.to_pickle(os.path.join(cache_dir, "mae_summary.pkl"))

        # Save full NEB data
        pd.to_pickle(neb_data_dict, os.path.join(cache_dir, "neb_df.pkl"))

        assets_dir = os.path.abspath("assets")
        with open(f"{cache_dir}/assets_dir.txt", "w") as f:
            f.write(assets_dir)
        
        return


    @staticmethod
    def launch_dashboard(cache_dir="app_cache/nebs/nebs_cache", ui=None):
        import pandas as pd
        from dash import Dash
        from mlipx.dash_utils import run_app, dash_table_interactive
        from dash import html, dcc, Input, Output, State
        import plotly.graph_objects as go
        import dash
        import os
        
        print(os.getcwd())

        # Read cached files
        mae_df = pd.read_pickle(f"{cache_dir}/mae_summary.pkl")
        neb_df = pd.read_pickle(f"{cache_dir}/neb_df.pkl")
        with open(f"{cache_dir}/assets_dir.txt", "r") as f:
            assets_dir = f.read().strip()
        

        # Initialize app with proper static configuration
        app = dash.Dash(__name__, assets_folder= assets_dir)
        
        app.server.static_folder = 'assets'
        app.server.static_url_path = '/assets'
        


        app.layout = dash_table_interactive(
            df=mae_df,
            id="neb-mae-score-table",
            title="NEB Energy Barriers Summary Table",
            extra_components=[
                html.Div(id="neb-table"),
                dcc.Store(id="neb-table-last-clicked", data=None),
                dcc.Store(id="current-neb-model", data=None),
                html.Div(
                    dcc.Graph(id="neb-plot"),
                    id="neb-plot-container",
                    #style={"display": "none"},
                ),
                html.Div(id="weas-viewer", style={'marginTop': '20px'}),
            ],
        )

        @app.callback(
            Output("neb-plot", "figure"),
            Output("neb-plot-container", "style"),
            Output("neb-table-last-clicked", "data"),
            Output("current-neb-model", "data"),
            Input("neb-mae-score-table", "active_cell"),
            State("neb-table-last-clicked", "data")
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
            # print("clickData:", clickData)
            # print("model_name:", model_name)
            # print("filename:", filename)
            if clickData is None or model_name is None:
                raise dash.exceptions.PreventUpdate
                #return dash.no_update, {"display": "none"}

            index = int(clickData["points"][0]["x"])
            

            filename = f"/assets/{model_name}/si_interstitials/images.xyz"
            #asset_url = app.get_asset_url(filename)
            
            
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
                            editor.avr.currentFrame = {current_frame};
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
            return (
                html.Div([
                    html.H4(f"Structure {index}", style={'textAlign': 'center'}),
                    html.Iframe(
                        srcDoc=html_content,
                        style={
                            "height": "550px",
                            "width": "100%",
                            "border": "1px solid #ddd",
                            "borderRadius": "5px"
                        }
                    )
                ]),
                {"marginTop": "20px"}
            )

        return run_app(app, ui=ui)
        
