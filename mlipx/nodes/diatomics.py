import os
import pickle
import functools
import pathlib
import typing as t

import ase
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ase import Atoms
import zntrack
from ase.data import atomic_numbers, covalent_radii, vdw_alvarez
from dash import dcc, html, Input, Output, State, MATCH
import re
from tqdm import tqdm
from mlipx.abc import ComparisonResults, NodeWithCalculator
from mlipx.utils import freeze_copy_atoms
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", message=".*empty or all-NA entries.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered in scalar divide.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight.*", category=UserWarning)


class HomonuclearDiatomics(zntrack.Node):
    """Compute energy-bondlength curves for homonuclear diatomic molecules.

    Parameters
    ----------
    elements : list[str]
        List of elements to consider. For example, ["H", "He", "Li"].
    model : NodeWithCalculator
        Node providing the calculator object for the energy calculations.
    n_points : int, default=100
        Number of points to sample for the bond length between
        min_distance and max_distance.
    min_distance : float, default=0.5
        Minimum bond length to consider in Angstrom.
    max_distance : float, default=2.0
        Maximum bond length to consider in Angstrom.
    data : list[ase.Atoms]|None
        Optional list of ase.Atoms. Diatomics for each element in
        this list will be added to `elements`.
    model_outs:
        Path to store the outputs of the model.
        Some models, like DFT calculators, generate
        files that will be stored in this path.

    Attributes
    ----------
    frames : list[ase.Atoms]
        List of frames with the bond length varied.
    results : pd.DataFrame
        DataFrame with the energy values for each bond length.
    """

    model: NodeWithCalculator = zntrack.deps()
    elements: list[str] = zntrack.params(("H", "He", "Li"))
    data: list[ase.Atoms] | None = zntrack.deps(None)
    het_diatomics: bool = zntrack.params(False)
    completed_traj_dir: pathlib.Path = zntrack.params(None)

    n_points: int = zntrack.params(100)
    min_distance: float = zntrack.params(0.5)
    max_distance: float = zntrack.params(2.0)
    eq_distance: t.Union[t.Literal["covalent-radiuis"], float] = zntrack.params(
        "covalent-radiuis"
    )

    frames: list[ase.Atoms] = zntrack.outs()  # TODO: change to h5md out
    results: pd.DataFrame = zntrack.plots()

    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model_outs")
    trajectory_dir_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "trajectories")

    def build_molecule(self, element, distance) -> ase.Atoms:
        return ase.Atoms([element, element], positions=[(0, 0, 0), (0, 0, distance)])

    def run(self):
        self.frames = []
        self.results = pd.DataFrame()
        self.model_outs.mkdir(exist_ok=True, parents=True)
        self.trajectory_dir_path.mkdir(exist_ok=True, parents=True)
        completed_traj_dir = self.completed_traj_dir
        
        (self.model_outs / "mlipx.txt").write_text("Thank you for using MLIPX!")
        #calc = self.model.get_calculator(directory=self.model_outs)
        calc = self.model.get_calculator()
        e_v = {}

        elements = set(self.elements)
        if self.data is not None:
            for atoms in self.data:
                elements.update(set(atoms.symbols))
                
        # Track skipped elements
        skipped_elements = []

        results_list = []
        for element in tqdm(elements, desc="Homonuclear Elements"):

            # Track already completed elements if provided
            already_done_homo = set()
            if completed_traj_dir is not None:
                completed_traj_dir = Path(completed_traj_dir)
                if completed_traj_dir.exists():
                    for f in completed_traj_dir.glob("*.extxyz"):
                        match = re.match(r"([A-Z][a-z]?)2\.extxyz", f.name)
                        if match:
                            already_done_homo.add(match.group(1))

            for element in tqdm(elements, desc="Homonuclear Elements"):
                if element in already_done_homo:
                    print(f"Skipping {element} — found in completed_traj_dir.")
                    traj = ase.io.read(completed_traj_dir / f"{element}2.extxyz", index=":")
                    self.frames.extend(freeze_copy_atoms(a) for a in traj)
                    continue
            
            # otherwise, proceed with calculations    
            traj_frames = []
            try:
                energies = []
                forces = []
                if self.eq_distance == "covalent-radiuis":
                    # convert element to atomic number
                    # issue: get differnet distance for different elements this way
                    rmin = 0.9 * covalent_radii[atomic_numbers[element]]
                    rvdw = vdw_alvarez.vdw_radii[atomic_numbers[element]] if atomic_numbers[element] < len(vdw_alvarez.vdw_radii) else np.nan
                    rmax = 3.1 * rvdw if not np.isnan(rvdw) else 6
                    rstep = 0.01
                    npts = int((rmax - rmin) / rstep)
                    distances = np.linspace(rmin, rmax, npts)
                    # distances = np.linspace(
                    #     self.min_distance * covalent_radii[atomic_numbers[element]],
                    #     self.max_distance * covalent_radii[atomic_numbers[element]],
                    #     self.n_points,
                    # )

                else:
                    distances = np.linspace(
                        self.min_distance, self.max_distance, self.n_points
                    )
                for distance in distances:
                    molecule = self.build_molecule(element, distance)
                    molecule.calc = calc
                    energies.append(molecule.get_potential_energy())
                    forces.append(molecule.get_forces())
                    self.frames.append(freeze_copy_atoms(molecule))
                    traj_frames.append(freeze_copy_atoms(molecule))
                e_v[element] = pd.DataFrame(energies, index=distances, columns=[element])
                
                ase.io.write(
                    (self.trajectory_dir_path / f"{element}2.extxyz"),
                    traj_frames,
                    append=True,
                )
                
                for distance, energy in zip(distances, energies):
                    results_list.append({
                        "element": element,
                        "distance": distance,
                        "energy": energy,
                    })

                
            except Exception as e:
                print(f"Skipping element {element}: {e}")
                skipped_elements.append(element)
                continue

        # After all elements, print skipped elements if any
        if skipped_elements:
            print(f"Skipped {len(skipped_elements)} elements: {', '.join(skipped_elements)} — likely unsupported by the model.")
        
        # ---- heteronuclear diatomics ----       
        hetero_pairs = []
        if self.het_diatomics:
            for i, elem1 in enumerate(self.elements):
                for elem2 in self.elements[i+1:]:
                    hetero_pairs.append((elem1, elem2))

        # Track already completed heteronuclear pairs if provided
        already_done_hetero = set()
        if completed_traj_dir is not None:
            completed_traj_dir = Path(completed_traj_dir)
            if completed_traj_dir.exists():
                for f in completed_traj_dir.glob("*.extxyz"):
                    match = re.match(r"([A-Z][a-z]?)([A-Z][a-z]?)\.extxyz", f.name)
                    if match and match.group(1) != match.group(2):
                        already_done_hetero.add((match.group(1), match.group(2)))
                        already_done_hetero.add((match.group(2), match.group(1)))  # to cover both orderings
                        
        for elem1, elem2 in tqdm(hetero_pairs, desc="Heteronuclear Pairs"):
            if (elem1, elem2) in already_done_hetero:
                print(f"Skipping {elem1}-{elem2} — found in completed_traj_dir.")
                traj = ase.io.read(completed_traj_dir / f"{elem1}{elem2}.extxyz", index=":")
                self.frames.extend(freeze_copy_atoms(a) for a in traj)
                continue
            traj_frames = []
            try:
                energies = []
                if self.eq_distance == "covalent-radiuis":
                    rmin = 0.9 * (covalent_radii[atomic_numbers[elem1]] + covalent_radii[atomic_numbers[elem2]]) / 2
                    rvdw1 = vdw_alvarez.vdw_radii[atomic_numbers[elem1]] if atomic_numbers[elem1] < len(vdw_alvarez.vdw_radii) else np.nan
                    rvdw2 = vdw_alvarez.vdw_radii[atomic_numbers[elem2]] if atomic_numbers[elem2] < len(vdw_alvarez.vdw_radii) else np.nan
                    rvdw = (rvdw1 + rvdw2) / 2 if not (np.isnan(rvdw1) or np.isnan(rvdw2)) else 6
                    rmax = 3.1 * rvdw
                    rstep = 0.01
                    npts = int((rmax - rmin) / rstep)
                    distances = np.linspace(rmin, rmax, npts)
                else:
                    distances = np.linspace(self.min_distance, self.max_distance, self.n_points)
                for distance in distances:
                    molecule = ase.Atoms([elem1, elem2], positions=[(0, 0, 0), (0, 0, distance)])
                    molecule.calc = calc
                    energies.append(molecule.get_potential_energy())
                    self.frames.append(freeze_copy_atoms(molecule))
                    traj_frames.append(freeze_copy_atoms(molecule))
                df = pd.DataFrame(energies, index=distances, columns=[f"{elem1}-{elem2}"])
                e_v[f"{elem1}-{elem2}"] = df
                ase.io.write((self.trajectory_dir_path / f"{elem1}{elem2}.extxyz"), traj_frames, append=True)
            except Exception as e:
                #print(f"Skipping hetero pair {elem1}-{elem2}: {e}")
                continue
        
        #self.results = pd.DataFrame(results_list)

        self.results = functools.reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
            e_v.values(),
        )

    def get_traj(self, element) -> list[ase.Atoms]:

        return ase.io.read(self.trajectory_dir_path / f"{element}2.extxyz", index=":")

    @property
    def figures(self) -> dict:
        # return a plot for each element
        plots = {}
        for element in self.results.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.results[element].dropna().index,
                    y=self.results[element].dropna(),
                    mode="lines",
                )
            )
            offset = 0
            for prev_element in self.results.columns:
                if prev_element == element:
                    break
                offset += self.n_points

            fig.update_traces(
                customdata=np.stack([np.arange(self.n_points) + offset], axis=1),
            )
            plots[f"{element}-{element} bond"] = fig
        return plots

    @classmethod
    def compare(cls, *nodes: "HomonuclearDiatomics") -> ComparisonResults:
        """Compare the energy-bondlength curves for homonuclear diatomic molecules.

        Parameters
        ----------
        nodes : HomonuclearDiatomics
            Nodes to compare.

        Returns
        -------
        ComparisonResults
            Comparison results.
        """
        figures = {}
        for node in nodes:
            for element in node.results.columns:
                # check if a figure for this element already exists
                if f"{element}-{element} bond" not in figures:
                    # create a line plot and label it with node.name
                    fig = go.Figure()
                    fig.update_layout(title=f"{element}-{element} bond")
                    fig.update_xaxes(title="Distance / Å")
                    fig.update_yaxes(title="Energy / eV")
                else:
                    fig = figures[f"{element}-{element} bond"]

                # add a line plot node.results[element] vs node.results.index
                fig.add_trace(
                    go.Scatter(
                        x=node.results[element].dropna().index,
                        y=node.results[element].dropna(),
                        mode="lines",
                        name=node.name.replace(f"_{cls.__name__}", ""),
                    )
                )
                offset = 0
                for prev_element in node.results.columns:
                    if prev_element == element:
                        break
                    offset += node.n_points
                fig.update_traces(
                    customdata=np.stack([np.arange(node.n_points) + offset], axis=1),
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
                figures[f"{element}-{element} bond"] = fig

                # Now with adjusted

                # check if a figure for this element already exists
                if f"{element}-{element} bond (adjusted)" not in figures:
                    # create a line plot and label it with node.name
                    fig = go.Figure()
                    fig.update_layout(title=f"{element}-{element} bond")
                    fig.update_xaxes(title="Distance / Å")
                    fig.update_yaxes(title="Adjusted energy / eV")
                else:
                    fig = figures[f"{element}-{element} bond (adjusted)"]

                # find the closest to the cov. dist. index to set the energy to zero
                one_idx = np.abs(
                    node.results[element].dropna().index
                    - covalent_radii[atomic_numbers[element]]
                ).argmin()

                # add a line plot node.results[element] vs node.results.index
                fig.add_trace(
                    go.Scatter(
                        x=node.results[element].dropna().index,
                        y=node.results[element].dropna()
                        - node.results[element].dropna().iloc[one_idx],
                        mode="lines",
                        name=node.name.replace(f"_{cls.__name__}", ""),
                    )
                )
                offset = 0
                for prev_element in node.results.columns:
                    if prev_element == element:
                        break
                    offset += node.n_points
                fig.update_traces(
                    customdata=np.stack([np.arange(node.n_points) + offset], axis=1),
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
                figures[f"{element}-{element} bond (adjusted)"] = fig

        return {"frames": nodes[0].frames, "figures": figures}





# above code built into mlipx, my code is below:


    # ------ interactive plotting ------



    @staticmethod
    def mae_plot_interactive(
        node_dict,
        run_interactive: bool = True,
        ui: str | None = None,
        normalise_to_model: t.Optional[str] = None,
    ):
        results_dict = {}

        for model_name, node in node_dict.items():
            results = node.results
            results.index.name = None
            results.reset_index(inplace=True)
            results.rename(columns={"index": "distance"}, inplace=True)
            results_dict[model_name] = results
            

            from mlipx.mlip_arena_utils import get_homonuclear_diatomic_properties, get_homonuclear_diatomic_stats
            get_homonuclear_diatomic_properties(model_name, node_dict[model_name])
            HomonuclearDiatomics.p_table_diatomic_plot(
                diatomics_df=results_dict[model_name],
                model_name=model_name,
            )

        stats_df = get_homonuclear_diatomic_stats(list(node_dict.keys()))
        stats_df = HomonuclearDiatomics.score_diatomics(stats_df, normalise_to_model=normalise_to_model)
        stats_df["Rank"] = stats_df["Score"].rank(ascending=True, method="min"
                                                  ).fillna(-1).astype(int)
        
        # stats_df["Rank"] = (
        #     stats_df["Score"]
        #     .rank(ascending=True, method="min")
        #     .fillna(-1)
        #     .astype(int)
        # )
        path = Path("benchmark_stats/molecular_benchmark/homonuclear_diatomics/stats")
        path.mkdir(exist_ok=True, parents=True)
        stats_df.to_csv(path / "stats.csv", index=False)
        #if normalise_to_model is not None:
            #stats_df['Score']

        # ---- Dash app ----
        import dash
        app = dash.Dash(__name__, suppress_callback_exceptions=True)


        from mlipx.mlip_arena_utils import get_homonuclear_diatomic_stats
        from mlipx.dash_utils import dash_table_interactive


        app.layout = html.Div([
            dash_table_interactive(
                df=stats_df,
                id='diatomics-stats-table',
                title="Homonuclear Diatomics Statistics",
                info= "This table is not interactive.",
            ),
            html.H2("Homonuclear Diatomic Explorer"),
            html.Label("Select Model:"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": name, "value": name} for name in results_dict],
                value=list(results_dict.keys())[0],
                clearable=False,
                style={"width": "300px", "marginBottom": "20px"}
            ),
            html.Label("Select Element:"),
            dcc.Dropdown(
                id="element-dropdown",
                options=[{"label": "All", "value": "All"}],  # this will be updated in callback
                value="All",
                clearable=False,
                style={"width": "300px", "marginBottom": "20px"}
            ),
            dcc.Graph(id="diatom-element-or-ptable-plot", style={"height": "700px", "width": "100%", "marginTop": "20px"}),
        ], style={"backgroundColor": "white"})
        
        
        
        # Register callbacks before running the app (+ make the tables etc)
        HomonuclearDiatomics.register_callbacks(app, results_dict)
        
        from mlipx.dash_utils import run_app
        if not run_interactive:
            return app, results_dict, stats_df
        
        return run_app(app, ui=ui)



    # ----------- helper functins -----------


    def process_model(model_name, results_dict):
        df = results_dict[model_name]
        data = []
        element_columns = {col: col for col in df.columns if col != "distance"}
        for col in element_columns:
            match = re.match(r"^([A-Z][a-z]?)", col)
            if not match:
                continue
            el = match.group(1)
            shift = df[col].iloc[-1]
            data.append({"Element": el, "Column": col, "Shift": round(shift, 4)})
        df_table = pd.DataFrame(data).drop_duplicates(subset="Element")
        return df, df_table



    @staticmethod
    def register_callbacks(app, results_dict):
        """Register Dash callbacks for the MAE interactive plot."""


        # Callback to update element dropdown options based on model
        @app.callback(
            Output("element-dropdown", "options"),
            Input("model-dropdown", "value")
        )
        def update_element_dropdown(model_name):
            df, _ = HomonuclearDiatomics.process_model(model_name, results_dict)
            options = [{"label": "All", "value": "All"}]
            for col in df.columns:
                if col != "distance":
                    element = re.match(r"^([A-Z][a-z]?)", col).group(1)
                    options.append({"label": element, "value": element})
            return options

        # Callback to render periodic table or dimer curve plot
        @app.callback(
            Output("diatom-element-or-ptable-plot", "figure"),
            Input("model-dropdown", "value"),
            Input("element-dropdown", "value")
        )
        def update_element_or_ptable_plot(model_name, element_value):
            df, _ = HomonuclearDiatomics.process_model(model_name, results_dict)
            if element_value == "All":
                fig_path = Path("benchmark_stats/molecular_benchmark/homonuclear_diatomics/plots") / f"{model_name}_ptable_diatomics.svg"

                svg_bytes = fig_path.read_bytes()
                import base64
                encoded = base64.b64encode(svg_bytes).decode()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="markers", marker=dict(opacity=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig.add_layout_image(
                    dict(
                        source=f"data:image/svg+xml;base64,{encoded}",
                        xref="x", yref="y",
                        x=0, y=0,
                        sizex=18,  # match columns of periodic table
                        sizey=10,  # match rows
                        xanchor="left", yanchor="bottom",
                        layer="below"
                    )
                )

                fig.update_layout(
                    xaxis=dict(
                        visible=False,
                        range=[0, 18],
                        constrain="domain"
                    ),
                    yaxis=dict(
                        visible=False,
                        range=[0, 10],
                        scaleanchor="x",
                        scaleratio=1
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=None,
                    autosize=True,
                    dragmode="pan",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                return fig
            else:
                df, _ = HomonuclearDiatomics.process_model(model_name, results_dict)

                import io, base64
                buf = io.BytesIO()

                HomonuclearDiatomics.p_table_diatomic_plot(
                    diatomics_df=df,
                    model_name=model_name,
                    selected_element=element_value,
                    out_path=buf
                )
                buf.seek(0)
                encoded_image = base64.b64encode(buf.read()).decode("utf-8")

                fig = go.Figure()
                fig.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{encoded_image}",
                        xref="x",
                        yref="y",
                        x=0,
                        y=0,
                        sizex=18,  # match columns of periodic table
                        sizey=10,  # match rows
                        xanchor="left",
                        yanchor="bottom",
                        layer="below"
                    )
                )
                fig.update_layout(
                    xaxis=dict(
                        visible=False,
                        range=[0, 18],
                        constrain="domain"
                    ),
                    yaxis=dict(
                        visible=False,
                        range=[0, 10],
                        scaleanchor="x",
                        scaleratio=1
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=None,
                    autosize=True,
                    dragmode="pan",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                return fig

    
    
    
    # insprired by: https://github.com/stenczelt/MACE-MP-work/tree/ba9ac8c2a98d93077ea231c7c4f48f0ad00a8d62/B6_pair-repulsion
    import io
    @staticmethod
    def p_table_diatomic_plot(
        diatomics_df: pd.DataFrame,
        model_name: str,
        selected_element: str | None = None,
        out_path: t.Optional[io.BytesIO] = None
    ):
        
        homonuclear_element_columns = {
            col: col
            for col in diatomics_df.columns
            if col != "distance" and "-" not in col
        }

        if selected_element:
            # take selected element's homonuclear + all its heteronuclear combinations (A-B or B-A)
            element_cols_plot = {
                col: col
                for col in diatomics_df.columns
                if col != "distance"
                and (
                    (col == selected_element)  # Homonuclear
                    or ("-" in col and selected_element in col.split("-"))  # A-B or B-A
                )
            }
        else:
            element_cols_plot = homonuclear_element_columns


        
        ptable_positions = {
            # First row
            "H": (0, 0), "He": (0, 17),
            # Second row
            "Li": (1, 0), "Be": (1, 1),
            "B": (1, 12), "C": (1, 13), "N": (1, 14), "O": (1, 15), "F": (1, 16), "Ne": (1, 17),
            # Third row
            "Na": (2, 0), "Mg": (2, 1),
            "Al": (2, 12), "Si": (2, 13), "P": (2, 14), "S": (2, 15), "Cl": (2, 16), "Ar": (2, 17),
            # Fourth row
            "K": (3, 0), "Ca": (3, 1), "Sc": (3, 2), "Ti": (3, 3), "V": (3, 4), "Cr": (3, 5), "Mn": (3, 6),
            "Fe": (3, 7), "Co": (3, 8), "Ni": (3, 9), "Cu": (3, 10), "Zn": (3, 11),
            "Ga": (3, 12), "Ge": (3, 13), "As": (3, 14), "Se": (3, 15), "Br": (3, 16), "Kr": (3, 17),
            # Fifth row
            "Rb": (4, 0), "Sr": (4, 1), "Y": (4, 2), "Zr": (4, 3), "Nb": (4, 4), "Mo": (4, 5), "Tc": (4, 6),
            "Ru": (4, 7), "Rh": (4, 8), "Pd": (4, 9), "Ag": (4, 10), "Cd": (4, 11),
            "In": (4, 12), "Sn": (4, 13), "Sb": (4, 14), "Te": (4, 15), "I": (4, 16), "Xe": (4, 17),
            # Sixth row
            "Cs": (5, 0), "Ba": (5, 1), "La": (8, 3), "Hf": (5, 3), "Ta": (5, 4), "W": (5, 5), "Re": (5, 6),
            "Os": (5, 7), "Ir": (5, 8), "Pt": (5, 9), "Au": (5, 10), "Hg": (5, 11),
            "Tl": (5, 12), "Pb": (5, 13), "Bi": (5, 14), "Po": (5, 15), "At": (5, 16), "Rn": (5, 17),
            # Seventh row
            "Fr": (6, 0), "Ra": (6, 1), "Ac": (9, 3), "Rf": (6, 3), "Db": (6, 4), "Sg": (6, 5), "Bh": (6, 6),
            "Hs": (6, 7), "Mt": (6, 8), "Ds": (6, 9), "Rg": (6, 10), "Cn": (6, 11),
            "Nh": (6, 12), "Fl": (6, 13), "Mc": (6, 14), "Lv": (6, 15), "Ts": (6, 16), "Og": (6, 17),
            # Lanthanides (row 8)
            "Ce": (8, 4), "Pr": (8, 5), "Nd": (8, 6), "Pm": (8, 7), "Sm": (8, 8), "Eu": (8, 9), "Gd": (8, 10),
            "Tb": (8, 11), "Dy": (8, 12), "Ho": (8, 13), "Er": (8, 14), "Tm": (8, 15), "Yb": (8, 16), "Lu": (8, 17),
            # Actinides (row 9)
            "Th": (9, 4), "Pa": (9, 5), "U": (9, 6), "Np": (9, 7), "Pu": (9, 8), "Am": (9, 9), "Cm": (9, 10),
            "Bk": (9, 11), "Cf": (9, 12), "Es": (9, 13), "Fm": (9, 14), "Md": (9, 15), "No": (9, 16), "Lr": (9, 17),
        }

        from matplotlib import pyplot as plt
        n_rows = 10
        n_cols = 18
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 15), constrained_layout=True)
        axes = axes.reshape(n_rows, n_cols)

        
        # Plot each element
        for col in element_cols_plot:
            if "-" in col:
                a, b = col.split("-")
                if selected_element == a:
                    other = b
                elif selected_element == b:
                    other = a
                else:
                    continue
            else:
                other = col

            if other not in ptable_positions:
                continue

            row, col_idx = ptable_positions[other]
            ax = axes[row, col_idx]

            x = diatomics_df["distance"]
            y = diatomics_df[col]
            
            # shift to zero at xmax
            shift = y.iloc[-1]
            y = y - shift

            ax.plot(x, y, linewidth=1, zorder = 1)
            ax.set_title(f"{col}, shift: {shift:.4f}", fontsize=10)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_yticks([-20, -10, 0, 10, 20])
            ax.set_xlim(0, 6)
            ax.set_ylim(-20, 20)
            
            ax.axhline(0, color='grey', linewidth=0.5, zorder = 0)

            # Draw a red box around the homonuclear subplot if selected
            if selected_element and col == selected_element:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)

        # Turn off unused axes
        for r in range(n_rows):
            for c in range(n_cols):
                if not axes[r, c].has_data():
                    axes[r, c].axis("off")

        if selected_element:
            plt.suptitle(f"Heteronuclear Diatomics for {selected_element}: {model_name}", fontsize=30)
        else:
            plt.suptitle(f"Homonuclear Diatomics: {model_name}", fontsize=30)


        
        if out_path is not None:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(out_path, format="png")
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            path = Path("benchmark_stats/molecular_benchmark/homonuclear_diatomics/plots/")
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(path / f"{model_name}_ptable_diatomics.svg")
            plt.savefig(path / f"{model_name}_ptable_diatomics.pdf")
        plt.close(fig)


        
    @staticmethod
    def score_diatomics(
        stats_df: pd.DataFrame,
        normalise_to_model: t.Optional[str] = None,
    ) -> pd.DataFrame:

        ideal_scores = {
            "Force flips": 1,
            "Tortuosity": 1,
            "Energy minima": 1,
            "Energy inflections": 1,
            "Spearman's coeff. (E: repulsion)": -1,
            "Spearman's coeff. (F: descending)": -1,
            "Spearman's coeff. (E: attraction)": 1,
            "Spearman's coeff. (F: ascending)": 1,
        }
        
        stats_df["Score"] = 0.0
        stats_df.set_index("Model", inplace=True)
        
        for model in stats_df.index:
            for col, ideal in ideal_scores.items():
                if col in stats_df.columns:
                    diff = stats_df.loc[model, col] - ideal
                    stats_df.loc[model, "Score"] += abs(diff)

        
        if normalise_to_model:
            stats_df["Score"] = stats_df["Score"] / stats_df.loc[normalise_to_model, "Score"]
        stats_df = stats_df.reset_index()
        
        return stats_df.round(3)
    
    
    
    @staticmethod
    def benchmark_precompute(
        node_dict, 
        cache_dir="app_cache/molecular_benchmark/diatomics_cache", 
        ui=None, 
        run_interactive=False, 
        normalise_to_model=None
    ):
        
        os.makedirs(cache_dir, exist_ok=True)
        app, results_dict, stats_df = HomonuclearDiatomics.mae_plot_interactive(
            node_dict=node_dict,
            run_interactive=run_interactive,
            ui=ui,
            normalise_to_model=normalise_to_model,
        )
        with open(f"{cache_dir}/results_dict.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        stats_df.to_pickle(f"{cache_dir}/stats_df.pkl")
        return app

    @staticmethod
    def launch_dashboard(cache_dir="app_cache/molecular_benchmark/diatomics_cache", ui=None):
        from mlipx.dash_utils import run_app
        import pandas as pd
        import dash
        app = dash.Dash(__name__)
        with open(f"{cache_dir}/results_dict.pkl", "rb") as f:
            results_dict = pickle.load(f)
        stats_df = pd.read_pickle(f"{cache_dir}/stats_df.pkl")

        app.layout = HomonuclearDiatomics.build_layout(stats_df, results_dict)
        HomonuclearDiatomics.register_callbacks(app, results_dict)
        return run_app(app, ui=ui)

    @staticmethod
    def build_layout(stats_df, results_dict):
        from mlipx.dash_utils import dash_table_interactive
        return html.Div([
            dash_table_interactive(
                df=stats_df,
                id='diatomics-stats-table',
                title="Homo- and heteronuclear Diatomics Statistics",
                info="This table is not interactive.",
            ),
            html.H2("Homo- and heteronuclear Diatomic Explorer"),
            html.Label("Select Model:"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": name, "value": name} for name in results_dict],
                value=list(results_dict.keys())[0],
                clearable=False,
                style={"width": "300px", "marginBottom": "20px"}
            ),
            html.Label("Select Element:"),
            dcc.Dropdown(
                id="element-dropdown",
                options=[{"label": "All", "value": "All"}],
                value="All",
                clearable=False,
                style={"width": "300px", "marginBottom": "20px"}
            ),
            dcc.Loading(
                id="loading-graph",
                type="circle",  # or "default", "dot"
                children=dcc.Graph(
                    id="diatom-element-or-ptable-plot",
                    style={"height": "700px", "width": "100%", "marginTop": "20px"}
                ),
            ),
            #dcc.Graph(id="diatom-element-or-ptable-plot", style={"height": "700px", "width": "100%", "marginTop": "20px"}),
        ], style={"backgroundColor": "white"})