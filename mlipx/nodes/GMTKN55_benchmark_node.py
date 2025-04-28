import pathlib
import typing as t
import json

import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative
import zntrack

import warnings
from pathlib import Path
from typing import Any, Callable
from ase.calculators.calculator import Calculator
from tqdm import tqdm
from phonopy.api_phonopy import Phonopy
import yaml

import pandas as pd
from dash.exceptions import PreventUpdate
from dash import dash_table
import socket
import time
from typing import List, Dict, Any, Optional
import cctk
from ase.io.trajectory import Trajectory
from plotly.io import write_image

from scipy.stats import gaussian_kde

from mlipx.abc import ComparisonResults, NodeWithCalculator

import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from mlipx.benchmark_download_utils import get_benchmark_data




class GMTKN55Benchmark(zntrack.Node):
    """Benchmark model against GMTKN55
    """
    # inputs
    #GMTKN55_yaml: pathlib.Path = zntrack.params()
    #subsets_csv: pathlib.Path = zntrack.params()
    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()
    
    subsets: Optional[List[str]] = zntrack.params(None)
    skip_subsets: Optional[List[str]] = zntrack.params(None)
    allowed_multiplicity: Optional[List[int]] = zntrack.params(None)
    allowed_charge: Optional[List[int]] = zntrack.params(None)
    allowed_elements: Optional[List[int]] = zntrack.params(None)


    # outputs
    # nwd: ZnTrack's node working directory for saving files
    model_benchmark_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "benchmark.csv")
    reference_values_ouptut: pathlib.Path = zntrack.outs_path(zntrack.nwd / "reference_values.json")
    predicted_values_output: pathlib.Path = zntrack.outs_path(zntrack.nwd / "predicted_values.json")
    



    def run(self):
        
        calc = self.model.get_calculator()
        
        # download GMTKN55_yaml and subsets.csv
        GMTKN55_dir = get_benchmark_data("GMTKN55.zip") / "GMTKN55"
        
        with open(GMTKN55_dir / "GMTKN55.yaml", "r") as file:
            structure_dict = yaml.safe_load(file)

            
        ref_values = {}
        pred_values = {}
            
        results_summary = []
        
        print(f"\nEvaluating with model: {self.model_name}")
        overall_errors = []
        overall_weights = []
        
        with open(self.model_benchmark_output, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Subset", "MAE", "Completed"])

            for subset_name, subset in structure_dict.items():
                # e.g. subset Amino20x4, BHROT27
                if self.subsets and subset_name not in self.subsets:
                    continue
                if self.skip_subsets and subset_name in self.skip_subsets:
                    continue
                
                subset_name = subset_name.lower()
                

                
                subset_errors = []
                weights = []
                
                ref_values[subset_name] = {}
                pred_values[subset_name] = {}

                for system_name, system in subset.items():
                    # sytem 1,2,3...

                    
                    ref_value = system["Energy"]
                    weight = system["Weight"]
                    
                    def _should_run_system(
                    system: Dict[str, Any],
                    allowed_elements: Optional[List[int]],
                    allowed_multiplicity: Optional[List[int]],
                    allowed_charge: Optional[List[int]],
                    ) -> bool:
                        for species in system["Species"].values():
                            elements = [cctk.helper_functions.get_number(e) for e in species["Elements"]]
                            multiplicity = species["UHF"] + 1
                            charge = species["Charge"]

                            if allowed_elements and any(el not in allowed_elements for el in elements):
                                return False
                            if allowed_multiplicity and multiplicity not in allowed_multiplicity:
                                return False
                            if allowed_charge and charge not in allowed_charge:
                                return False

                        return True

                    if not _should_run_system(
                        system, self.allowed_elements, self.allowed_multiplicity, self.allowed_charge
                    ):
                        continue

                    try:
                        comp_value = 0
                        species_involved = []
                        for species_name, species in system["Species"].items():
                            #e.g. species name ALA_xac, ALA_xag (for Amino20x4)
                            
                            species_involved.append(species_name)
                            
                            atoms = ase.Atoms(
                                species["Elements"],
                                positions=np.array(species["Positions"])
                            )
                            atoms.info['head'] = 'mp_pbe'
                            atoms.cell = None
                            atoms.pbc = False

                            atoms.calc = calc
                            result = atoms.get_potential_energy()
                            comp_value += result * species["Count"] * 23.0609  # eV to kcal/mol
                            # the sign of count is defined so the correct relative energy is calculated
                            
                        overall_label = "-".join(str(s) for s in species_involved)

                        error = ref_value - comp_value
                        ref_values[subset_name][overall_label] = ref_value
                        pred_values[subset_name][overall_label] = comp_value
                        weights.append(weight)
                        subset_errors.append(error)
                        

                    except Exception as e:
                        print(f"Error in system {system_name}, skipping. Exception: {e}")

                mae = np.mean(np.abs(subset_errors)) if subset_errors else None
                completed = len(subset_errors) == len(subset.items())
                csv_writer.writerow([subset_name, mae, completed])
                overall_errors.extend(subset_errors)
                overall_weights.extend(weights)


        
        
        with open(self.reference_values_ouptut, "w") as f:
            json.dump(ref_values, f)
        with open(self.predicted_values_output, "w") as f:
            json.dump(pred_values, f)
       
            
            
    @property
    def reference_dict(self):
        """Get the reference values dictionary."""
        with open(self.reference_values_ouptut, "r") as f:
            ref_values = json.load(f)
        return ref_values
    
    @property
    def predicted_dict(self):
        """Get the predicted values dictionary."""
        with open(self.predicted_values_output, "r") as f:
            pred_values = json.load(f)
        return pred_values
    
    @property
    def benchmark_results(self):
        """Get the benchmark results."""
        benchmark_results = pd.read_csv(self.model_benchmark_output)
        return benchmark_results
    




    @staticmethod
    def calculate_weighted_mae(benchmark_df, subsets_df, category="All"):
        total_weighted_mae = 0
        total_weight = 0
        filtered_subsets = subsets_df[subsets_df["excluded"].str.lower() != "true"]
        if category != "All":
            filtered_subsets = filtered_subsets[filtered_subsets["category"] == category]
        
        for _, subset_row in filtered_subsets.iterrows():
            subset = subset_row["subset"]
            weight = float(subset_row["weight"])
            num = float(subset_row["num"])

            match = benchmark_df[
                (benchmark_df["subset"] == subset) &
                (benchmark_df["completed"].astype(str).str.lower() == "true")
            ]
            if not match.empty:
                mae = float(match["mae"].values[0])
                total_weighted_mae += mae * weight * num
                total_weight += num
                #total_weight += weight * num

        return total_weighted_mae / total_weight if total_weight > 0 else None
            
            
            
            
            

    @staticmethod
    def mae_plot_interactive(benchmark_node_dict, ui = None):
        
        subsets_path = get_benchmark_data("GMTKN55.zip") / "GMTKN55/subsets.csv"
        
        subsets_df = pd.read_csv(subsets_path)
        subsets_df.columns = subsets_df.columns.str.lower()
        subsets_df["subset"] = subsets_df["subset"].str.lower()
        subsets_df["excluded"] = subsets_df["excluded"].astype(str)
        
        categories = list(subsets_df["category"].unique()) + ["All"]
        
        
        # subset mae
        mae_data = []
        benchmark_tables = []
        wtmad_rows = []
        
        #for file in benchmark_files:
        for model_name, node in benchmark_node_dict.items():

            
            df = node.benchmark_results.copy()
            df.columns = df.columns.str.lower()
            df = df[df["completed"].astype(str).str.lower().str.lower() == "true"]
            df["subset"] = df["subset"].str.lower()
            
            # calculate MAE for each category
            row = {"Model": model_name}
            for cat in categories:
                row[cat] = GMTKN55Benchmark.calculate_weighted_mae(df, subsets_df, category=cat)
            benchmark_tables.append(row)
            
            # WTMAD for each model
            # wtmad = node.wtmad
            # wtmad_table = pd.DataFrame([
            #     {"Model": model_name, "WTMAD": node.wtmad}
            #     for model_name, node in benchmark_node_dict.items()
            # ])
            # wtmad_table = wtmad_table.sort_values(by="WTMAD")
            
            # weighted total MAE
            
            wtmad = row["All"]
            wtmad_rows.append({"Model": model_name, "WTMAD": wtmad})

            
            
            
            
            
                        
            # merge descriptions
            df = df.merge(subsets_df[["subset", "description", "category"]], on="subset", how="left")

            category_abbreviations = {
                "Basic properties and reaction energies for small systems": "Small systems",
                "Reaction energies for large systems and isomerisation reactions": "Large systems",
                "Reaction barrier heights": "Barrier heights",
                "Intramolecular noncovalent interactions": "Intramolecular NCIs",
                "Intermolecular noncovalent interactions": "Intermolecular NCIs",
                "All": "All"
            }
        
            # mae for each subset
            for _, row in df.iterrows():
                mae_data.append({
                    "model": model_name,
                    "subset": row["subset"],
                    "mae": row["mae"],
                    "description": row["description"],
                    "category": category_abbreviations.get(row["category"], row["category"])  # fallback to original if missing
                })

                


        
        benchmark_df = pd.DataFrame(benchmark_tables)
        benchmark_df = benchmark_df.sort_values(by="All", ascending=True)
        benchmark_df = benchmark_df.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        benchmark_df = benchmark_df.rename(columns={
            col: category_abbreviations.get(col, col) for col in benchmark_df.columns
        })

        mae_df = pd.DataFrame(mae_data)
        wtmad_table = pd.DataFrame(wtmad_rows).sort_values(by="WTMAD")
        # 2 d.p.
        wtmad_table = wtmad_table.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        
        
        #--------------- saving plots and tables ---------------
        
        
        results_dir = Path("GMTKN55-benchmark-stats")
        results_dir.mkdir(exist_ok=True)
        
        wtmad_table.to_csv(os.path.join(results_dir, "wtmad.csv"), index=False)
        benchmark_df.to_csv(os.path.join(results_dir, "category_mae.csv"), index=False)
        
        # subset mae
        mae_df.to_csv(os.path.join(results_dir, "mae_per_subset.csv"), index=False)

        fig = px.scatter(mae_df, x="subset", y="mae", color="model", hover_data={"description": True}, custom_data=["model", "subset", "description"], title="MAE Per-subset", labels={"subset": "Subset", "mae": "MAE", "model": "Model"})
        fig.update_traces(hovertemplate="<br>".join(["Model: %{customdata[0]}", "Subset: %{customdata[1]}", "Description: %{customdata[2]}", "MAE: %{y:.2f} kcal/mol", "<extra></extra>"]))
        fig.update_layout(paper_bgcolor='white', font_color='black', title_font=dict(size=20), margin=dict(t=50, r=30, b=50, l=50), width=800, height=500, xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=45, tickmode="linear", dtick=1), yaxis=dict(showgrid=True, gridcolor='lightgray'))
        fig.write_image(os.path.join(results_dir, "mae_per_subset.png"))
        
        # scatter plots
        path = Path("GMTKN55-benchmark-stats/scatter_plots")
        path.mkdir(parents=True, exist_ok=True)
        for model_name in mae_df["model"].unique():
            path_model = path / model_name
            path_model.mkdir(parents=True, exist_ok=True)

            
            model_df = mae_df[mae_df["model"] == model_name]
            for _, row in model_df.iterrows():
                subset_name, description = row["subset"], row["description"]
                try:
                    preds = list(benchmark_node_dict[model_name].predicted_dict[subset_name].values()); refs = list(benchmark_node_dict[model_name].reference_dict[subset_name].values()); species = list(benchmark_node_dict[model_name].predicted_dict[subset_name].keys())
                except KeyError: continue
                mae = np.mean(np.abs(np.array(refs) - np.array(preds)))
                scatter_fig = px.scatter(x=refs, y=preds, custom_data=[species], labels={"x": "Reference Energy (kcal/mol)", "y": "Predicted Energy (kcal/mol)"}, title=f"{model_name} — {subset_name}: Predicted vs Reference")
                scatter_fig.update_traces(hovertemplate="<br>".join(["Species: %{customdata[0]}", "Reference: %{x:.2f} kcal/mol", "Predicted: %{y:.2f} kcal/mol", "<extra></extra>"]))
                scatter_fig.add_trace(go.Scatter(x=[min(refs), max(refs)], y=[min(refs), max(refs)], mode="lines", line=dict(dash="dot", color="gray"), name="y = x"))
                scatter_fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, text=f"MAE: {mae:.2f}", showarrow=False, align="left", font=dict(size=12, color="black"), bordercolor="black", borderwidth=1, borderpad=4, bgcolor="white", opacity=0.8)
                scatter_fig.update_layout(height=500, plot_bgcolor='white', paper_bgcolor='white', font_color='black', margin=dict(t=50, r=30, b=50, l=50), xaxis=dict(showgrid=True, gridcolor='lightgray', scaleanchor="y", scaleratio=1), yaxis=dict(showgrid=True, gridcolor='lightgray'))
                scatter_fig.write_image(os.path.join(path_model, f"{subset_name}.png"))
                pd.DataFrame({"species": species, "reference": refs, "predicted": preds}).to_csv(os.path.join(path_model, f"{subset_name}.csv"), index=False)

            
            
        
        
        # -------------------- Dash app ------------------------

        if ui is None:
            return

        # --- Dash app ---
        app = dash.Dash(__name__)
        app.title = "GMTKN55 Dashboard"



        


        app.layout = html.Div([
            html.H1("GMTKN55 Benchmarking Dashboard", style={"color": "black"}),

            html.H2("WTMAD (weighted total mean absolute deviation)", style={"color": "black", "marginTop": "20px"}),
            dash_table.DataTable(
                data=wtmad_table.round(3).to_dict("records"),
                columns=[{"name": i, "id": i} for i in wtmad_table.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "minWidth": "100px", "border": "1px solid black"},
                style_header={"backgroundColor": "lightgray", "fontWeight": "bold"},
            ),

            html.H2("MAD/MAE per Category (kcal/mol)", style={"color": "black", "marginTop": "30px"}),
            dash_table.DataTable(
                data=benchmark_df.round(3).to_dict("records"),
                columns=[{"name": i, "id": i} for i in benchmark_df.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "minWidth": "100px", "border": "1px solid black"},
                style_header={"backgroundColor": "lightgray", "fontWeight": "bold"},
            ),
            
            html.Hr(style={"marginTop": "40px", "marginBottom": "30px", "borderTop": "2px solid #bbb"}),


            html.H2("MAD/MAE per Subset", style={"color": "black", "marginTop": "30px"}),
            html.Div([
                html.Label("Color by:", style={"marginRight": "10px"}),
                dcc.RadioItems(
                    id="color-toggle",
                    options=[
                        {"label": "Category", "value": "category"},
                        {"label": "Model", "value": "model"},
                    ],
                    value="category",
                    labelStyle={"display": "inline-block", "marginRight": "15px"}
                )
            ], style={"marginBottom": "20px"}),
            dcc.Graph(id="mae-plot"),

            html.H2("Predicted vs Reference Energies", style={"color": "black", "marginTop": "30px"}),
            dcc.Graph(id="pred-vs-ref-plot")

        ], style={"backgroundColor": "white", "padding": "20px"})


        @app.callback(
            Output("mae-plot", "figure"),
            Input("color-toggle", "value")
        )
        
        def update_mae_plot(color_by):
            symbol_col = "model" if color_by == "category" else "category"
            
            fig = px.scatter(
                mae_df,
                x="subset",
                y="mae",
                color=color_by,
                symbol=symbol_col,
                hover_data={"description": True},
                custom_data=["model", "subset", "description", "category"],
                title="MAE Per-subset",
                labels={
                    "subset": "Subset",
                    "mae": "MAE (kcal/mol)",
                    "model": "Model",
                }
            )

            fig.update_traces(
                hovertemplate="<br>".join([
                    "Model: %{customdata[0]}",
                    "Subset: %{customdata[1]}",
                    "Description: %{customdata[2]}",
                    "Category: %{customdata[3]}",
                    "MAE: %{y:.2f} kcal/mol",
                    "<extra></extra>"
                ])
            )

            fig.update_layout(
                paper_bgcolor='white',
                font_color='black',
                title_font=dict(size=20),
                margin=dict(t=50, r=30, b=50, l=50),
                width=800,
                height=400,
                xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=45, tickmode="linear", dtick=1),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )

            return fig
        

        @app.callback(
            Output("pred-vs-ref-plot", "figure"),
            Input("mae-plot", "clickData"),
        )

        def update_scatter(click_data):
            if click_data is None:
                raise dash.exceptions.PreventUpdate

            model_name, subset_name, *_ = click_data["points"][0]["customdata"]
            subset_name = subset_name.lower()
            print(f"Selected: {model_name} | {subset_name}")

            try:
                pred_list = [benchmark_node_dict[model_name].predicted_dict[subset_name][label] for label in benchmark_node_dict[model_name].predicted_dict[subset_name]]
                ref_list = [benchmark_node_dict[model_name].reference_dict[subset_name][label] for label in benchmark_node_dict[model_name].reference_dict[subset_name]]
                species_list = [label for label in benchmark_node_dict[model_name].predicted_dict[subset_name]]
            except KeyError:
                print(f"Model {model_name} or subset {subset_name} not found in data.")
                return go.Figure()
            
            

            preds = np.array([i for i in pred_list])
            refs = np.array([i for i in ref_list])
            #mae_val = mae_df[(mae_df["model"] == model_name) & (mae_df["subset"] == subset_name)]["mae"].values[0]
            mae = np.mean(np.abs(refs - preds))

            min_val = min(refs.min(), preds.min(), 0)
            max_val = max(refs.max(), preds.max())
            pad = 0.05 * (max_val - min_val)
            x_range = [min_val - pad, max_val + pad]
            y_range = x_range

            fig = px.scatter(
                x=refs,
                y=preds,
                custom_data=[species_list],
                labels={"x": "Reference Energy (kcal/mol)", "y": "Predicted Energy (kcal/mol)"},
                title=f"{model_name} — {subset_name}: Predicted vs Reference"
            )

            fig.update_traces(
                hovertemplate="<br>".join([
                    "Species: %{customdata[0]}",
                    "Reference: %{x:.2f} kcal/mol",
                    "Predicted: %{y:.2f} kcal/mol",
                    "<extra></extra>"
                ])
            )
        

            fig.add_trace(go.Scatter(
                x=np.linspace(min_val - 100, max_val + 100, 100),
                y=np.linspace(min_val - 100, max_val + 100, 100),
                mode="lines", name="y = x",
                line=dict(dash="dot", color="gray")
            ))

            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=f"MAE (kcal/mol): {mae:.2f}",
                showarrow=False,
                align="left",
                font=dict(size=12, color="black"),
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )
            



            fig.update_layout(
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='black',
                margin=dict(t=50, r=30, b=50, l=50),
                xaxis=dict(range=x_range, showgrid=True, gridcolor='lightgray', scaleanchor='y', scaleratio=1),
                yaxis=dict(range=y_range, showgrid=True, gridcolor='lightgray')
            )

            return fig



        def get_free_port():
            """Find an unused local port."""
            s = socket.socket()
            s.bind(('', 0))  # let OS pick a free port
            port = s.getsockname()[1]
            s.close()
            return port


        def run_app(app, ui):
            port = get_free_port()
            url = f"http://localhost:{port}"

            def _run_server():
                app.run(debug=True, use_reloader=False, port=port)
                
            if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
                import threading
                print(f"\n Detected SSH session — skipping browser launch.")
                #threading.Thread(target=_run_server, daemon=True).start()
                return
            elif ui == "browser":
                import webbrowser
                import threading
                #threading.Thread(target=_run_server, daemon=True).start()
                time.sleep(1.5)
                _run_server()
                #webbrowser.open(url)
            elif ui == "notebook":
                _run_server()
            
            else:
                print(f"Unknown UI option: {ui}. Please use, 'browser', or 'notebook'.")
                return

            print(f"Dash app running at {url}")
            
        return run_app(app, ui=ui)
    
    
    