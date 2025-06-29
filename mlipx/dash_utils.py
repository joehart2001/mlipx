import dash
from dash import dash_table
import pandas as pd
from dash import dcc, html
from typing import List, Dict, Any, Optional
from dash.exceptions import PreventUpdate
import socket
import os
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, MATCH
import base64
import csv
import warnings
import matplotlib
import matplotlib.cm
from dash import Dash, html
from dash.development.base_component import Component
from typing import List, Union

# ----

def reserve_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    return s, port

def run_app(app, ui, port: Optional[int] = None):
    if port:
        port = int(port)
        sock, _ = reserve_free_port()
    else:
        sock, port = reserve_free_port()
    url = f"http://localhost:{port}"
    sock.close()

    def _run_server():
        app.run(debug=True, use_reloader=False, port=port)

    if ui == "browser":
        _run_server()
    elif ui == "notebook":
        _run_server()
    else:
        print(f"Unknown UI option: {ui}. Please use 'browser' or 'notebook'.")
        return

    print(f"Dash app running at {url}")
            
            
# ------

def dash_table_interactive(
                df: pd.DataFrame, 
                id: str, 
                title: str,
                info: str = "Info: click on an interactive cell to show plots, click on the models column to collapse",
                extra_components: list = None,
                interactive: bool = True,
                static_coloured_table: bool = False,
                
) -> html.Div:
    
    """
    df : pd.DataFrame
        DataFrame to display in the table.
    id : str
        ID of the DataTable component.
    title : str
        Title to display above the table.
    extra_components : list, optional
        Additional Dash components (e.g., dcc.Store, html.Div) to include below the table.
        These are useful when the table is part of an interactive workflow and needs to:
            - Store metadata or internal state (e.g., `dcc.Store(id="last-clicked")`)
            - Display dynamic content triggered by user interaction, such as:
                - A plot or table shown in response to a cell click
                - A tabbed layout (e.g., scatter plot + Δ table in the lattice constant example)

    """
    
    
    
    
    return html.Div([
        html.H2(title, style={"color": "black"}),
        html.P(info, style={"fontSize": "14px", "color": "#555"}) if interactive else None,

        dash_table.DataTable(
            id=id,
            columns=[{"name": col, "id": col} for col in df.columns],
            data=df.to_dict('records'),
            style_cell={'textAlign': 'center', 'fontSize': '14px'},
            style_header={'fontWeight': 'bold',"whiteSpace": "normal"},
            cell_selectable=interactive,
            style_data_conditional=colour_table(df, all_cols=True) if static_coloured_table else None,
            # col sorting, doesn't move the selected cell
            # sort_action="native", # enable column header sort
            # sort_mode="single", # allow sorting by one column at a time
            
            # persistence=True,
            # persistence_type='memory'
        ),

        html.Br(),
        
        *(extra_components if extra_components else [])
        # dcc.Store(id='lattice-table-last-clicked'),
        # html.Div(id="lattice-const-table"),
        
    ], style={"backgroundColor": "white", "padding": "20px"})
    
    

# --------- combining benchmark utils ------------

def process_data(data, key_extractor, value_extractor):
    if isinstance(data, list):
        result = {}
        for node in data:
            key = key_extractor(node)
            value = value_extractor(node)

            if isinstance(value, dict):
                # Merge nested dictionaries
                if key not in result:
                    result[key] = {}
                result[key].update(value)
            else:
                result[key] = value
        return result

    elif isinstance(data, dict):
        return data

    else:
        raise ValueError(f"{data} should be a list or dict")



def combine_mae_tables(*mae_dfs):
    """ combine mae tables from different nodes for a summary table
    """
    combined_parts = []

    for df in mae_dfs:
        df = df.copy()
        df_cols = df.columns.tolist()
        if "Model" not in df_cols:
            raise ValueError("Each input dataframe must contain a 'Model' column.")
        other_cols = [col for col in df.columns if col != "Model"]
        df = df.set_index("Model")
        df.columns = other_cols
        combined_parts.append(df)

    combined = pd.concat(combined_parts, axis=1)

    combined.reset_index(inplace=True)
    return combined



def colour_table(
    benchmark_score_df: pd.DataFrame,
    all_cols: bool = False,
    col_name: str | None = None,
    
) -> List[Dict[str, Any]]:
    """ Viridis-style colormap for Dash DataTable
    """
    
    cmap = matplotlib.cm.get_cmap("viridis_r")

    def rgba_from_val(val, vmin, vmax, cmap):
        norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0
        rgba = cmap(norm)
        r, g, b = [int(255 * x) for x in rgba[:3]]
        return f'rgb({r}, {g}, {b})'

    style_data_conditional = []


    if all_cols:
        cols_to_color = benchmark_score_df.select_dtypes(include="number").columns
    elif col_name is not None:
        if col_name not in benchmark_score_df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame.")
        cols_to_color = [col_name]
    else:
        raise ValueError("Specify either all_cols=True or provide col_name.")

    for col in cols_to_color:
        vmin = benchmark_score_df[col].min()
        vmax = benchmark_score_df[col].max()

        for val in benchmark_score_df[col]:
            style_data_conditional.append({
                'if': {'filter_query': f'{{{col}}} = {val}', 'column_id': col},
                'backgroundColor': rgba_from_val(val, vmin, vmax, cmap),
                'color': 'white' if val > (vmin + vmax) / 2 else 'black'
            })

    return style_data_conditional



def combine_apps(
    benchmark_score_df: pd.DataFrame,
    benchmark_title: str,
    apps_or_layouts_list: List[Union[Dash, Component]],
    id: str = "benchmark-score-table",
    benchmark_table_info: str = "",
    extra_components: list = None,
    static_coloured_table: bool = False,
    weights_components: list = None,
):
    """Combines multiple Dash apps or layouts into a single layout.

    Accepts a list of Dash apps (`dash.Dash`) or Dash layout components (`html.Div`, `dcc.Graph`, etc.).

    Args:
        benchmark_score_df: Benchmark score table data.
        benchmark_title: Title at the top of the app.
        apps_or_layouts_list: List of Dash apps or layout components.
        id: HTML ID of the score table.
        benchmark_table_info: Info box content for the benchmark table.
        extra_components: Optional additional components for the table.
        static_coloured_table: If True, disables dynamic coloring.
        weights_components: Optional list of Dash components (e.g., sliders and input boxes) for editing benchmark weights, shown below the benchmark score table.
    """
    benchmark_score_table = dash_table_interactive(
        df=benchmark_score_df,
        id=id,
        title="Benchmark Score Table",
        info=benchmark_table_info,
        extra_components=extra_components,
        interactive=True,
        static_coloured_table=static_coloured_table,
    )

    children = [
        html.H1(f"{benchmark_title}", style={"color": "black"}),
        html.Div([
            benchmark_score_table,
            html.Br(),
            *(weights_components if weights_components else [])
        ], style={
            "backgroundColor": "white",
            "padding": "20px",
            "border": "2px solid black",
        })
    ]

    # Extract and wrap all layout components
    for idx, entry in enumerate(apps_or_layouts_list):
        layout = entry.layout if isinstance(entry, Dash) else entry
        children.append(
            html.Div(layout.children if hasattr(layout, "children") else layout, style={
                "backgroundColor": "white",
                "padding": "20px",
                "border": "2px solid black",
            })
        )

    return html.Div(
        children,
        style={"backgroundColor": "#f8f8f8"}
    )
    


# def combine_apps(
#     benchmark_score_df: pd.DataFrame,
#     benchmark_title: str,
#     apps_list: List[dash.Dash],
#     id: str = "benchmark-score-table",
#     benchmark_table_info: str = "",
#     #style_data_conditional: List[Dict[str, Any]] | None = None,
#     extra_components: list = None,
#     static_coloured_table: bool = False,
# ):
#     """ combines multiple Dash apps into a single app, where the first app is the main app
#          e.g. used in bulk_crystal_benchmark
         
#          TODO: potential issue: id='benchmark-score-table' will be duplicated in each benchmark
         
#          static_coloured_table: use false when you want no colours or when adding a dynamically coloured table
#     """
    
#     benchmark_score_table = dash_table_interactive(
#         df=benchmark_score_df,
#         id=id,
#         title="Benchmark Score Table",
#         info=benchmark_table_info,
#         extra_components=extra_components,
#         interactive=True,
#         static_coloured_table=static_coloured_table,
#     )



#     app_layout_dict = {
#         f"app_{i}": app.layout if hasattr(app, "layout") else app
#         for i, app in enumerate(apps_list)
#     }
    
#     children = [
#         html.H1(f"{benchmark_title}", style={"color": "black"}),
#         html.Div(benchmark_score_table, style={
#             "backgroundColor": "white",
#             "padding": "20px",
#             "border": "2px solid black",
#         })
#     ]

#     # add all other app layouts except app_1
#     for idx in range(len(apps_list)):
#         children.append(
#             html.Div(app_layout_dict[f"app_{idx}"].children, style={
#                 "backgroundColor": "white",
#                 "padding": "20px",
#                 "border": "2px solid black",
#             })
#         )

#     # assign layout to app_1 (main app)
#     # apps_list[0].layout = html.Div(
#     #     children,
#     #     style={"backgroundColor": "#f8f8f8"}
#     # )
    
#     return html.Div(
#         children,
#         style={"backgroundColor": "#f8f8f8"}
#     )
    
#     #return apps_list[0]





# ------- registering callbacks -------


# TODO this
def register_callbacks_table_scatter_table(
    app, 
    mae_df, 
    lat_const_df
):
    """ register callbacks for interactive table -> scatter plot and table tabs
    - mae-score-table is the interactive table
    - 
    """
    # decorator tells dash the function below is a callback
    @app.callback(
        Output("lattice-const-table", "children"),
        Output("lattice-table-last-clicked", "data"),
        Input("lat-mae-score-table", "active_cell"),
        State("lattice-table-last-clicked", "data")
    )
    def update_lattice_const_plot(active_cell, last_clicked):
        if active_cell is None:
            raise PreventUpdate

        row = active_cell["row"]
        col = active_cell["column_id"]
        model_name = mae_df.loc[row, "Model"]
        # vital for closing plot
        if col not in mae_df.columns or col == "Model":
            return None, active_cell

        # Toggle behavior: if the same model is clicked again, collapse
        if last_clicked is not None and (
            active_cell["row"] == last_clicked.get("row") and
            active_cell["column_id"] == last_clicked.get("column_id")
        ):
            return None, None

        # Else, render the plot + table
        mae_val = (lat_const_df[model_name] - lat_const_df["ref"]).abs().mean()

        ref_vals = lat_const_df["ref"]
        pred_vals = lat_const_df[model_name]
        formulas = lat_const_df.index.tolist()

        fig = LatticeConstant.create_scatter_plot(ref_vals, pred_vals, model_name, mae_val, formulas)

        abs_diff = pred_vals - ref_vals
        pct_diff = 100 * abs_diff / ref_vals

        table_df = pd.DataFrame({
            "Element": formulas,
            "DFT (Å)": ref_vals,
            f"{model_name} (Å)": pred_vals,
            "Δ": abs_diff.round(3),
            "Δ/%": pct_diff.round(2)
        }).round(3)


        summary_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in table_df.columns],
            data=table_df.reset_index(drop=True).to_dict('records'),
            style_cell={'textAlign': 'center', 'fontSize': '14px'},
            style_header={'fontWeight': 'bold'},
            style_table={'overflowX': 'auto'},
        )

        return html.Div([
            dcc.Tabs([
                dcc.Tab(label="Scatter Plot", children=[dcc.Graph(figure=fig)]),
                dcc.Tab(label="Δ Table", children=[html.Div(summary_table, style={"padding": "20px"})])
            ])
        ]), active_cell
        


# -------- plotting functions --------
        

def create_scatter_plot(
    ref_vals: List[float], 
    pred_vals: List[float], 
    model_name: str, 
    mae: float,
    metric_label: tuple[str] = ("metric", "units"),
    hovertemplate: str | None = None,
    hover_data: tuple[List[str], str] | List[str] | None = None,
) -> px.scatter:
    """Create a scatter plot comparing ref vs predicted + mae in legend."""
    
    x_col = "Reference"
    y_col = "Predicted"
    hover_col = "CustomData"

    data = {
        x_col: ref_vals,
        y_col: pred_vals,
    }

    if isinstance(hover_data, tuple) and hover_data[0] is not None:
        data[hover_col] = hover_data[0]
        hover_label = hover_data[1]
        custom_data_cols = [hover_col]
    else:
        custom_data_cols = None
        hover_label = None

    df = pd.DataFrame(data)

    combined_min = min(min(ref_vals), min(pred_vals))
    combined_max = max(max(ref_vals), max(pred_vals))

    fig = px.scatter(
        data_frame=df,
        x=x_col,
        y=y_col,
        custom_data=custom_data_cols,
        labels={
            x_col: f"Reference {metric_label[0]} [{metric_label[1]}]",
            y_col: f"{model_name} {metric_label[0]} [{metric_label[1]}]",
        },
        title=f"{model_name} — {metric_label[0]}",
    )
    
    
    combined_min = min(min(ref_vals), min(pred_vals))
    combined_max = max(max(ref_vals), max(pred_vals))

    fig.add_shape(
        type="line",
        x0=combined_min, y0=combined_min,
        x1=combined_max, y1=combined_max,
        xref='x', yref='y',
        line=dict(color="black", dash="dash")
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
        xaxis=dict(showgrid=True, gridcolor="lightgray", scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
    )
    
    if hovertemplate:
        fig.update_traces(hovertemplate=hovertemplate)
    elif custom_data_cols and hover_label:
        fig.update_traces(hovertemplate="<br>".join([
            f"{hover_label}: %{{customdata[0]}}",
            f"Reference: %{{x:.3f}} {metric_label[1]}",
            f"Predicted: %{{y:.3f}} {metric_label[1]}",
            "<extra></extra>"
        ]))

    

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=f"MAE (Å): {mae:.3f}",
        showarrow=False,
        align="left",
        font=dict(size=12, color="black"),
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )

    return fig