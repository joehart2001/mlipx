import fnmatch
import importlib.metadata
import json
import pathlib
import uuid
import webbrowser
import re

import dvc.api
import plotly.io as pio
import typer
import zntrack
from tqdm import tqdm
from typing_extensions import Annotated
from zndraw import ZnDraw
import os

from mlipx import benchmark, recipes

app = typer.Typer()
app.add_typer(recipes.app, name="recipes")
app.add_typer(benchmark.app, name="benchmark")

# Load plugins

entry_points = importlib.metadata.entry_points(group="mlipx.recipes")
for entry_point in entry_points:
    entry_point.load()


@app.command()
def main():
    typer.echo("Hello World")


@app.command()
def compare(  # noqa C901
    nodes: Annotated[list[str], typer.Argument(help="Path to the node to compare")],
    zndraw_url: Annotated[
        str,
        typer.Option(
            envvar="ZNDRAW_URL",
            help="URL of the ZnDraw server to visualize the results",
        ),
    ],
    kwarg: Annotated[list[str], typer.Option("--kwarg", "-k")] = None,
    token: Annotated[str, typer.Option("--token")] = None,
    glob: Annotated[
        bool, typer.Option("--glob", help="Allow glob patterns to select nodes.")
    ] = False,
    convert_nan: Annotated[bool, typer.Option()] = False,
    browser: Annotated[
        bool,
        typer.Option(
            help="""Whether to open the ZnDraw GUI in the default web browser."""
        ),
    ] = True,
    figures_path: Annotated[
        str | None,
        typer.Option(
            help="Provide a path to save the figures to."
            "No figures will be saved by default."
        ),
    ] = None,
):
    """Compare mlipx nodes and visualize the results using ZnDraw."""
    # TODO: allow for glob patterns
    if kwarg is None:
        kwarg = []
    node_names, revs, remotes = [], [], []
    if glob:
        fs = dvc.api.DVCFileSystem()
        with fs.open("zntrack.json", mode="r") as f:
            all_nodes = list(json.load(f).keys())

    for node in nodes:
        # can be name or name@rev or name@remote@rev
        parts = node.split("@")
        if glob:
            filtered_nodes = [x for x in all_nodes if fnmatch.fnmatch(x, parts[0])]
        else:
            filtered_nodes = [parts[0]]
        for x in filtered_nodes:
            node_names.append(x)
            if len(parts) == 1:
                revs.append(None)
                remotes.append(None)
            elif len(parts) == 2:
                revs.append(parts[1])
                remotes.append(None)
            elif len(parts) == 3:
                remotes.append(parts[1])
                revs.append(parts[2])
            else:
                raise ValueError(f"Invalid node format: {node}")

    node_instances = {}
    for node_name, rev, remote in tqdm(
        zip(node_names, revs, remotes), desc="Loading nodes"
    ):
        node_instances[node_name] = zntrack.from_rev(node_name, remote=remote, rev=rev)

    if len(node_instances) == 0:
        typer.echo("No nodes to compare")
        return

    typer.echo(f"Comparing {len(node_instances)} nodes")

    kwargs = {}
    for arg in kwarg:
        key, value = arg.split("=", 1)
        kwargs[key] = value
    result = node_instances[node_names[0]].compare(*node_instances.values(), **kwargs)

    token = token or str(uuid.uuid4())
    typer.echo(f"View the results at {zndraw_url}/token/{token}")
    vis = ZnDraw(zndraw_url, token=token, convert_nan=convert_nan)
    length = len(vis)
    vis.extend(result["frames"])
    del vis[:length]  # temporary fix
    vis.figures = result["figures"]
    if browser:
        webbrowser.open(f"{zndraw_url}/token/{token}")
    if figures_path:
        for desc, fig in result["figures"].items():
            pio.write_json(fig, pathlib.Path(figures_path) / f"{desc}.json")

    vis.socket.sleep(5)



from typer import Option, Argument

@app.command()
def phonon_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to phonon nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,

    ):
    """Launch interactive benchmark for phonon dispersion."""
    import fnmatch
    import dvc.api
    import json

    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())

    selected_nodes = []
    if glob:
        for pattern in nodes:
            matched = fnmatch.filter(all_nodes, pattern)
            for name in matched:
                model = name.split("_phonons-dispersion")[0]
                if ("reference" in name or "ref" in name) or (not models or model in models):
                    selected_nodes.append(name)
    else:
        for name in nodes:
            model = name.split("_phonons-dispersion")[0]
            if ("reference" in name or "ref" in name) or (not models or model in models):
                selected_nodes.append(name)

    if not selected_nodes:
        typer.echo("No matching nodes found.")
        raise typer.Exit()

    # Instantiate nodes
    node_objects = {}
    for name in selected_nodes:
        node_objects[name] = zntrack.from_rev(name)

    # Group nodes by mp-id and by model
    pred_node_dict = {}
    ref_node_dict = {}

    for name, node in node_objects.items():
        mp_id = name.split("_")[-1]
        model = name.split("_phonons-dispersion")[0]
        if "reference" in name or "ref" in name:
            ref_node_dict[mp_id] = node
        else:
            pred_node_dict.setdefault(mp_id, {})[model] = node

    if models:
        # filter to selected models
        pred_node_dict = {
            mp: {m: node for m, node in models_dict.items() if m in models}
            for mp, models_dict in pred_node_dict.items()
        }


    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)

    print('\n UI = ', ui)

    from mlipx import PhononDispersion
    PhononDispersion.benchmark_interactive(
        pred_node_dict=pred_node_dict,
        ref_node_dict=ref_node_dict,
        ui = ui,
    )
    



@app.command()
def gmtkn55_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to phonon nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    subsets: Annotated[str, typer.Option("--subsets", "-s", help="Subsets")] = "subsets.csv",
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,

    ):
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())

    selected_nodes = []
    if glob:
        for pattern in nodes:
            matched = fnmatch.filter(all_nodes, pattern)
            for name in matched:
                model = name.split("_GMTKN55Benchmark")[0]
                if not models or model in models:
                    selected_nodes.append(name)
    else:
        for name in nodes:
            model = name.split("_GMTKN55Benchmark")[0]
            if not models or model in models:
                selected_nodes.append(name)

    if not selected_nodes:
        typer.echo("No matching nodes found.")
        raise typer.Exit()

    # Instantiate nodes
    node_objects = {}
    for name in selected_nodes:
        node_objects[name] = zntrack.from_rev(name)

    benchmark_node_dict = {}

    for name, node in node_objects.items():
        model = name.split("_GMTKN55Benchmark")[0]
        benchmark_node_dict[model] = node

    if models:
        # filter to selected models
        benchmark_node_dict = {
            m: node for m, node in benchmark_node_dict.items() if m in models
        }

    if not os.path.exists(subsets):
        typer.echo("subsets.csv not found in the current directory, please provide the subsets.csv path using --subsets.")
        raise typer.Exit(1)
    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)

    print('\n UI = ', ui)

    from mlipx import GMTKN55Benchmark
    GMTKN55Benchmark.mae_plot_interactive(
        benchmark_node_dict = benchmark_node_dict,
        subsets_path=subsets,
        ui = ui
    )
    

