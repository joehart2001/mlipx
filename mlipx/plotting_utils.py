import io
import pathlib
import typing as t
import json

import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative



def generate_density_scatter_figure(
    x, 
    y, 
    label=None, 
    model=None, 
    mae=None, 
    max_points=500, 
    num_cells=100,
    mode="elasticity"
):
    
    
    x, y = np.array(x), np.array(y)
    x_min, x_max = x.min() - 0.05, x.max() + 0.05 # add a small margin otherwise the egde points are cut off
    y_min, y_max = y.min() - 0.05, y.max() + 0.05
    cell_w = (x_max - x_min) / num_cells
    cell_h = (y_max - y_min) / num_cells

    cell_counts = np.zeros((num_cells, num_cells))
    cell_points = { (i, j): [] for i in range(num_cells) for j in range(num_cells) }

    for idx, (xi, yi) in enumerate(zip(x, y)):
        cx = int((xi - x_min) // cell_w)
        cy = int((yi - y_min) // cell_h)
        if 0 <= cx < num_cells and 0 <= cy < num_cells:
            cell_counts[cx, cy] += 1
            cell_points[(cx, cy)].append(idx)  # store index, not coordinates

    plot_x, plot_y, plot_colors = [], [], []
    point_cells = []

    # plotting points logic
    # cell has more than 5 points: randomly select one
    # cell has less than 5 points: plot all points
    # helps for large datasets but keeps outliers visiable
    for (cx, cy), indices in cell_points.items():
        if len(indices) > 5:
            idx = np.random.choice(indices)
            if len(plot_x) < max_points:
                plot_x.append(x[idx])
                plot_y.append(y[idx])
                plot_colors.append(cell_counts[cx, cy])
                point_cells.append((cx, cy))
        else:
            indices = indices
        for idx in indices:
            plot_x.append(x[idx])
            plot_y.append(y[idx])
            plot_colors.append(cell_counts[cx, cy])
            point_cells.append((cx, cy))

    fig = go.Figure(go.Scatter(
        x=plot_x, y=plot_y, mode='markers',
        marker=dict(color=plot_colors, size=5, colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Density")),
        text=[f"Density: {int(c)}" for c in plot_colors],
        customdata=point_cells
    ))

    if mode == "elasticity":

        fig.update_layout(
            title=f'{label} Scatter Plot - {model}',
            xaxis_title=f'{label} DFT [GPa]',
            yaxis_title=f'{label} Predicted [GPa]',
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(size=16, color='black'),
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True),
            hovermode='closest'
        )

        combined_min, combined_max = min(x_min, y_min), max(x_max, y_max)
        fig.add_shape(type='line', x0=combined_min, y0=combined_min, x1=combined_max, y1=combined_max,
                        line=dict(dash='dash'))

        if mae is not None:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.98,
                text=f"{label} MAE: {mae} [GPa]<br>Total points: {len(x)}",
                showarrow=False,
                font=dict(size=14, color="black"),
                bordercolor="black", borderwidth=1,
                borderpad=4, bgcolor="white", opacity=0.8
            )
    
    # temportary before generalising
    elif mode == "QMOF":
        fig.update_layout(
            title=f'{label} Scatter Plot - {model}',
            xaxis_title="DFT Energy [eV/atom]",
            yaxis_title="Predicted Energy [eV/atom]",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(size=16, color='black'),
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True),
            hovermode='closest'
        )
        
        combined_min, combined_max = min(x_min, y_min), max(x_max, y_max)
        fig.add_shape(type='line', x0=combined_min, y0=combined_min, x1=combined_max, y1=combined_max,
                        line=dict(dash='dash'))

        if mae is not None:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.98,
                text=f"{label} MAE: {mae} [eV/atom]<br>Total points: {len(x)}",
                showarrow=False,
                font=dict(size=14, color="black"),
                bordercolor="black", borderwidth=1,
                borderpad=4, bgcolor="white", opacity=0.8
            )

    return fig, cell_points, cell_w, cell_h, x_min, y_min
