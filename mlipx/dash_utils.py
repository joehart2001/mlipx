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

# ----

def reserve_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    return s, port

def run_app(app, ui):
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
                extra_components: list = None,
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

        dash_table.DataTable(
            id=id,
            columns=[{"name": col, "id": col} for col in df.columns],
            data=df.to_dict('records'),
            style_cell={'textAlign': 'center', 'fontSize': '14px'},
            style_header={'fontWeight': 'bold'},
            cell_selectable=True,
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
):
    """ Viridis-style colormap for Dash DataTable
    """
    
    score_min = benchmark_score_df['Avg MAE \u2193'].min()
    score_max = benchmark_score_df['Avg MAE \u2193'].max()
    
    import matplotlib
    import matplotlib.cm

    # For lower-is-better columns (like MAE), use viridis_r (reversed)
    cmap = matplotlib.cm.get_cmap("viridis_r")
    
    def rgba_from_val(val, vmin, vmax, cmap):
        norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0
        rgba = cmap(norm)
        r, g, b = [int(255 * x) for x in rgba[:3]]
        return f'rgb({r}, {g}, {b})'

    vmin, vmax = score_min, score_max
    style_data_conditional = [
        {
            'if': {'filter_query': f'{{Avg MAE \u2193}} = {score}', 'column_id': 'Avg MAE \u2193'},
            'backgroundColor': rgba_from_val(score, vmin, vmax, cmap),
            'color': 'white' if score > (score_min + score_max) / 2 else 'black'
        }
        for score in benchmark_score_df['Avg MAE \u2193']
    ]
    return style_data_conditional





def combine_apps(
    benchmark_score_df: pd.DataFrame,
    benchmark_title: str,
    apps_list: List[dash.Dash],
    style_data_conditional: List[Dict[str, Any]] | None = None,
):
    """ combines multiple Dash apps into a single app, where the first app is the main app
         e.g. used in bulk_crystal_benchmark
    """
    
    benchmark_score_table = html.Div([
        html.H2("Benchmark Score Table", style={'color': 'Black', 'padding': '1rem'}),
        dash_table.DataTable(
            id='benchmark-score-table',
            columns=[{"name": col, "id": col} for col in benchmark_score_df.columns],
            data=benchmark_score_df.to_dict('records'),
            style_cell={'textAlign': 'center', 'fontSize': '14px'},
            style_header={'fontWeight': 'bold'},
            style_data_conditional=style_data_conditional if style_data_conditional else None,
        ),
    ])
    
    app_layout_dict = {f"app_{i}": app.layout for i, app in enumerate(apps_list)}
    
    children = [
        html.H1(f"{benchmark_title}", style={"color": "black"}),
        html.Div(benchmark_score_table, style={
            "backgroundColor": "white",
            "padding": "20px",
            "border": "2px solid black",
        })
    ]

    # add all other app layouts except app_1
    for idx in range(len(apps_list)):
        children.append(
            html.Div(app_layout_dict[f"app_{idx}"].children, style={
                "backgroundColor": "white",
                "padding": "20px",
                "border": "2px solid black",
            })
        )

    # assign layout to app_1 (main app)
    # apps_list[0].layout = html.Div(
    #     children,
    #     style={"backgroundColor": "#f8f8f8"}
    # )
    
    return html.Div(
        children,
        style={"backgroundColor": "#f8f8f8"}
    )
    
    #return apps_list[0]





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
    hover: List[str] | None = None
) -> px.scatter:
    """Create a scatter plot comparing ref vs predicted + mae in legend."""
    
    combined_min = min(min(ref_vals), min(pred_vals))
    combined_max = max(max(ref_vals), max(pred_vals))

    fig = px.scatter(
        x=ref_vals,
        y=pred_vals,
        hover_name=hover,
        labels={
            "x": f"Reference {metric_label[0]} [{metric_label[1]}]",
            "y": f"{model_name} {metric_label[0]} [{metric_label[1]}]",
        },
        title=f"{model_name} - {metric_label}"
    )

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