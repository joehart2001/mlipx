import functools
import pathlib
import typing as t

import ase
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ase import Atoms
import tqdm
import zntrack
from ase.data import atomic_numbers, covalent_radii, vdw_alvarez
from dash import dcc, html, Input, Output, State, MATCH
import re

from mlipx.abc import ComparisonResults, NodeWithCalculator
from mlipx.utils import freeze_copy_atoms
from pathlib import Path

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
        
        (self.model_outs / "mlipx.txt").write_text("Thank you for using MLIPX!")
        calc = self.model.get_calculator(directory=self.model_outs)
        e_v = {}

        elements = set(self.elements)
        if self.data is not None:
            for atoms in self.data:
                elements.update(set(atoms.symbols))

        results_list = []
        for element in elements:
            
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
                tbar = tqdm.tqdm(
                    distances, desc=f"{element}-{element} bond ({distances[0]:.2f} Å)"
                )
                for distance in tbar:
                    tbar.set_description(f"{element}-{element} bond ({distance:.2f} Å)")
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
            results.index.name = "distance"
            results.reset_index(inplace=True)
            results_dict[model_name] = results

            from mlipx.mlip_arena_utils import get_homonuclear_diatomic_properties, get_homonuclear_diatomic_stats
            get_homonuclear_diatomic_properties(model_name, node_dict[model_name])
            HomonuclearDiatomics.p_table_diatomic_plot(
                diatomics_df=results_dict[model_name],
                model_name=model_name,
            )

        stats_df = get_homonuclear_diatomic_stats(list(node_dict.keys()))
        stats_df = HomonuclearDiatomics.score_diatomics(stats_df, normalise_to_model=normalise_to_model)
        stats_df["Rank"] = stats_df["Score"].rank(ascending=True, method="min").astype(int)
        
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
        HomonuclearDiatomics.register_callbacks(app, node_dict, results_dict)
        
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
    def register_callbacks(app, node_dict, results_dict):
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
                col = element_value
                if col not in df.columns:
                    return go.Figure()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["distance"], y=df[col], mode="lines", name=element_value))
                fig.add_hline(y=0, line=dict(dash="dash", width=1, color="gray"))
                fig.update_layout(
                    title=f"{element_value} Bond Curve",
                    xaxis_title="Distance [Å]",
                    yaxis_title="Energy [meV]",
                    xaxis=dict(range=[0, 6]),
                    yaxis=dict(range=[-20, 20])
                )
                return fig
    
    
    
    # insprired by: https://github.com/stenczelt/MACE-MP-work/tree/ba9ac8c2a98d93077ea231c7c4f48f0ad00a8d62/B6_pair-repulsion
    
    @staticmethod
    def p_table_diatomic_plot(
        diatomics_df: pd.DataFrame,
        model_name: str,
    ):
        
        element_columns = {col: col for col in diatomics_df.columns if col != "distance"}

        
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
        for col, el in element_columns.items():
            if el not in ptable_positions:
                continue

            row, col_idx = ptable_positions[el]
            ax = axes[row, col_idx]

            x = diatomics_df["distance"]
            y = diatomics_df[col]
            
            # shift to zero at xmax
            shift = y.iloc[-1]
            y = y - shift

            ax.plot(x, y, linewidth=1, zorder = 1)
            ax.set_title(f"{el}, shift: {shift:.4f}", fontsize=10)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_yticks([-20, -10, 0, 10, 20])
            ax.set_xlim(0, 6)
            ax.set_ylim(-20, 20)
            
            ax.axhline(0, color='grey', linewidth=0.5, zorder = 0)

        # Turn off unused axes
        for r in range(n_rows):
            for c in range(n_cols):
                if not axes[r, c].has_data():
                    axes[r, c].axis("off")

        plt.suptitle(f"Homonuclear Diatomics: {model_name}", fontsize=18)
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
    