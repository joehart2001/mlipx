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
from mlipx.dash_utils import dash_table_interactive
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
    def calculate_weighted_mae(benchmark_df, subsets_df, category="All (WTMAD)"):
        total_weighted_mae = 0
        total_weight = 0
        filtered_subsets = subsets_df[subsets_df["excluded"].str.lower() != "true"]
        if category != "All (WTMAD)":
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
    def mae_plot_interactive(
        node_dict, 
        ui = None, 
        run_interactive = True,
        normalise_to_model: Optional[str] = None,
        ):
        
        subsets_path = get_benchmark_data("GMTKN55.zip") / "GMTKN55/subsets.csv"
        
        subsets_df = pd.read_csv(subsets_path)
        subsets_df.columns = subsets_df.columns.str.lower()
        subsets_df["subset"] = subsets_df["subset"].str.lower()
        subsets_df["excluded"] = subsets_df["excluded"].astype(str)
        
        categories = list(subsets_df["category"].unique()) + ["All (WTMAD)"]
        
        
        # subset mae
        mae_data = []
        benchmark_tables = []
        wtmad_rows = []
        
        #for file in benchmark_files:
        for model_name, node in node_dict.items():

            
            df = node.benchmark_results.copy()
            df.columns = df.columns.str.lower()
            df = df[df["completed"].astype(str).str.lower().str.lower() == "true"]
            df["subset"] = df["subset"].str.lower()
            
            # calculate MAE for each category
            row = {"Model": model_name}
            for cat in categories:
                row[cat] = GMTKN55Benchmark.calculate_weighted_mae(df, subsets_df, category=cat)
            benchmark_tables.append(row)
            
            
            # weighted total MAE
            
            wtmad = row["All (WTMAD)"]
            wtmad_rows.append({"Model": model_name, "WTMAD": wtmad})

            
            
            
            
            
                        
            # merge descriptions
            df = df.merge(subsets_df[["subset", "description", "category"]], on="subset", how="left")

            category_abbreviations = {
                "Basic properties and reaction energies for small systems": "Small systems",
                "Reaction energies for large systems and isomerisation reactions": "Large systems",
                "Reaction barrier heights": "Barrier heights",
                "Intramolecular noncovalent interactions": "Intramolecular NCIs",
                "Intermolecular noncovalent interactions": "Intermolecular NCIs",
                "All (WTMAD)": "All (WTMAD)"
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
        benchmark_df = benchmark_df.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        benchmark_df = benchmark_df.rename(columns={
            col: category_abbreviations.get(col, col) for col in benchmark_df.columns
        })

        mae_df = pd.DataFrame(mae_data)
        wtmad_table = pd.DataFrame(wtmad_rows)
        wtmad_table = wtmad_table.rename(columns={"WTMAD": "Weighted Total MAD/MAE [kcal/mol] \u2193"})
        
        if normalise_to_model:
            wtmad_table['Score'] = wtmad_table['Weighted Total MAD/MAE [kcal/mol] \u2193'] / wtmad_table[wtmad_table['Model'] == normalise_to_model]['Weighted Total MAD/MAE [kcal/mol] \u2193'].values[0]
        else:
            wtmad_table['Score'] = wtmad_table['Weighted Total MAD/MAE [kcal/mol] \u2193']
            
        wtmad_table = wtmad_table.round(3)
        wtmad_table['Rank'] = wtmad_table['Score'].rank(ascending=True)
        # 2 d.p.
        #wtmad_table = wtmad_table.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        
        
        #--------------- saving plots and tables ---------------
        
        
        results_dir = Path("benchmark_stats/molecular_benchmark/GMTKN55/")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
                    
        wtmad_table.to_csv(os.path.join(results_dir, "wtmad.csv"), index=False)
        benchmark_df.to_csv(os.path.join(results_dir, "category_mae.csv"), index=False)
        
        # subset mae
        mae_df.to_csv(os.path.join(results_dir, "mae_per_subset.csv"), index=False)

        fig = px.scatter(mae_df, x="subset", y="mae", color="model", hover_data={"description": True}, custom_data=["model", "subset", "description"], title="MAE Per-subset", labels={"subset": "Subset", "mae": "MAE", "model": "Model"})
        fig.update_traces(hovertemplate="<br>".join(["Model: %{customdata[0]}", "Subset: %{customdata[1]}", "Description: %{customdata[2]}", "MAE: %{y:.2f} kcal/mol", "<extra></extra>"]))
        fig.update_layout(paper_bgcolor='white', font_color='black', title_font=dict(size=20), margin=dict(t=50, r=30, b=50, l=50), width=800, height=500, xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=45, tickmode="linear", dtick=1), yaxis=dict(showgrid=True, gridcolor='lightgray'))
        fig.write_image(os.path.join(results_dir, "mae_per_subset.png"))
        
        # scatter plots
        scatter_path = results_dir / "scatter_plots"
            
            
        for model_name in mae_df["model"].unique():
            path_model = scatter_path / model_name
            
            if not os.path.exists(path_model):
                os.makedirs(path_model)
            
            model_df = mae_df[mae_df["model"] == model_name]
            for _, row in model_df.iterrows():
                subset_name, description = row["subset"], row["description"]
                try:
                    preds = list(node_dict[model_name].predicted_dict[subset_name].values())
                    refs = list(node_dict[model_name].reference_dict[subset_name].values())
                    species = list(node_dict[model_name].predicted_dict[subset_name].keys())
                except KeyError: continue
                
                mae = np.mean(np.abs(np.array(refs) - np.array(preds)))

                from mlipx.dash_utils import create_scatter_plot
                scatter_fig = create_scatter_plot(
                    ref_vals = refs, 
                    pred_vals = preds, 
                    model_name = model_name, 
                    mae = mae, 
                    metric_label = ("Energy", "kcal/mol"),
                    hover_data=species,
                    hovertemplate="<br>".join([
                        "Species: %{customdata[0]}",
                        "Reference: %{x:.2f} kcal/mol",
                        "Predicted: %{y:.2f} kcal/mol",
                        "<extra></extra>"
                    ])
                )
                
                scatter_fig.write_image(os.path.join(path_model, f"{subset_name}.png"))
                pd.DataFrame({"species": species, "reference": refs, "predicted": preds}).to_csv(os.path.join(path_model, f"{subset_name}.csv"), index=False)

            
            
        
        

        if ui is None and run_interactive:
            return 



        # --- Dash app ---
        app = dash.Dash(__name__)
        app.title = "GMTKN55 Dashboard"



        


        app.layout = html.Div([
            html.H1("GMTKN55 Benchmarking Dashboard", style={"color": "black"}),

            dash_table_interactive(
                df=wtmad_table,
                id="GMTKN55-wtmad-table",
                title="WTMAD (weighted total mean absolute deviation)",
                info= "This table is not interactive.",
            ),

            dash_table_interactive(
                df=benchmark_df.round(3),
                id="GMTKN55-category-table",
                title="MAD/MAE per Category (kcal/mol)",
                info= "This table is not interactive.",
            ),

            html.H2("MAD/MAE per Subset", style={"color": "black", "marginTop": "30px"}),
            html.P("Click on a point to see the scatter plot of predicted vs reference energies used for the MAE calculation."),
            html.Div([
                html.Label("Color by:", style={"marginRight": "10px"}),
                dcc.RadioItems(
                    id="GMTKN55-color-toggle",
                    options=[
                        {"label": "Category", "value": "category"},
                        {"label": "Model", "value": "model"},
                    ],
                    value="category",
                    labelStyle={"display": "inline-block", "marginRight": "15px"}
                )
            ], style={"marginBottom": "20px"}),
            html.Div([
                dcc.Graph(id="GMTKN55-mae-plot", style={"width": "100%"})
            ]),

            html.Div(
                id="pred-vs-ref-container",
                children=[],
                # children=[
                #     html.H2("Predicted vs Reference Energies", style={"color": "black", "marginTop": "30px"}),
                #     dcc.Graph(id="GMTKN55-pred-vs-ref-plot")
                # ]
            ),

        ], style={"backgroundColor": "white", "padding": "20px"})


        # Register callbacks
        GMTKN55Benchmark.register_callbacks(app, node_dict, mae_df)


        from mlipx.dash_utils import run_app

        if not run_interactive:
            return app, wtmad_table, mae_df

        return run_app(app, ui=ui)
    
    
    
    # ----------- helper fuctions -----------
    
    
    
    
    @staticmethod
    def register_callbacks(
        app: dash.Dash,
        node_dict: Dict[str, NodeWithCalculator],
        mae_df: pd.DataFrame
    ):

        @app.callback(
            Output("GMTKN55-mae-plot", "figure"),
            Input("GMTKN55-color-toggle", "value")
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
                autosize=True,
                height=400,
                xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=45, tickmode="linear", dtick=1),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )

            return fig

        @app.callback(
            Output("pred-vs-ref-container", "children"),
            Input("GMTKN55-mae-plot", "clickData"),
        )
        def update_scatter(click_data):
            if click_data is None:
                raise PreventUpdate 

            model_name, subset_name, *_ = click_data["points"][0]["customdata"]
            subset_name = subset_name.lower()
            print(f"Selected: {model_name} | {subset_name}")

            try:
                pred_list = [node_dict[model_name].predicted_dict[subset_name][label] for label in node_dict[model_name].predicted_dict[subset_name]]
                ref_list = [node_dict[model_name].reference_dict[subset_name][label] for label in node_dict[model_name].reference_dict[subset_name]]
                species_list = [label for label in node_dict[model_name].predicted_dict[subset_name]]
            except KeyError:
                print(f"Model {model_name} or subset {subset_name} not found in data.")
                # Return the default children (section header and empty graph)
                return [
                    html.H2("Predicted vs Reference Energies", style={"color": "black", "marginTop": "30px"}),
                    dcc.Graph(id="GMTKN55-pred-vs-ref-plot")
                ]

            preds = np.array([i for i in pred_list])
            refs = np.array([i for i in ref_list])
            mae = np.mean(np.abs(refs - preds))

            from mlipx.dash_utils import create_scatter_plot
            scatter_fig = create_scatter_plot(
                ref_vals = refs,
                pred_vals = preds,
                model_name = model_name,
                mae = mae,
                metric_label = ("Energy", "kcal/mol"),
                hover_data=(species_list, "Species"),
                hovertemplate="<br>".join([
                    "Species: %{customdata[0]}",
                    "Reference: %{x:.2f} kcal/mol",
                    "Predicted: %{y:.2f} kcal/mol",
                    "<extra></extra>"
                ])
            )

            return [
                html.H2("Predicted vs Reference Energies", style={"color": "black", "marginTop": "30px"}),
                dcc.Graph(id="GMTKN55-pred-vs-ref-plot", figure=scatter_fig)
            ]
    