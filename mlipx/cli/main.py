import fnmatch
import importlib.metadata
import json
import pathlib
import sys
import uuid
import webbrowser
import re

import dvc.api
import plotly.io as pio
import typer
import zntrack
from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from typing_extensions import Annotated
#from zndraw import ZnDraw
import os

from mlipx import benchmark, recipes

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



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
def info():
    """Print the version of mlipx and the available models."""
    from mlipx.models import AVAILABLE_MODELS  # slow import

    console = Console()
    # Get Python environment info
    python_version = sys.version.split()[0]
    python_executable = sys.executable
    python_platform = sys.platform

    py_table = Table(title="🐍 Python Environment", box=box.ROUNDED)
    py_table.add_column("Version", style="cyan", no_wrap=True)
    py_table.add_column("Executable", style="magenta")
    py_table.add_column("Platform", style="green")
    py_table.add_row(python_version, python_executable, python_platform)

    # Get model availability
    mlip_table = Table(title="🧠 MLIP Codes", box=box.ROUNDED)
    mlip_table.add_column("Model", style="bold")
    mlip_table.add_column("Available", style="bold")

    for model in sorted(AVAILABLE_MODELS):
        status = AVAILABLE_MODELS[model]
        if status is True:
            mlip_table.add_row(model, "[green]:heavy_check_mark: Yes[/green]")
        elif status is False:
            mlip_table.add_row(model, "[red]:x: No[/red]")
        elif status is None:
            mlip_table.add_row(model, "[yellow]:warning: Unknown[/yellow]")
        else:
            mlip_table.add_row(model, "[red]:boom: Error[/red]")

    # Get versions of key packages
    mlipx_table = Table(title="📦 mlipx Ecosystem", box=box.ROUNDED)
    mlipx_table.add_column("Package", style="bold")
    mlipx_table.add_column("Version", style="cyan")

    for package in ["mlipx", "zntrack", "zndraw"]:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "[red]Not installed[/red]"
        mlipx_table.add_row(package, version)

    # Display all
    console.print(mlipx_table)
    console.print(py_table)
    console.print(mlip_table)


# @app.command()
# def compare(  # noqa C901
#     nodes: Annotated[list[str], typer.Argument(help="Path to the node to compare")],
#     zndraw_url: Annotated[
#         str,
#         typer.Option(
#             envvar="ZNDRAW_URL",
#             help="URL of the ZnDraw server to visualize the results",
#         ),
#     ],
#     kwarg: Annotated[list[str], typer.Option("--kwarg", "-k")] = None,
#     token: Annotated[str, typer.Option("--token")] = None,
#     glob: Annotated[
#         bool, typer.Option("--glob", help="Allow glob patterns to select nodes.")
#     ] = False,
#     convert_nan: Annotated[bool, typer.Option()] = False,
#     browser: Annotated[
#         bool,
#         typer.Option(
#             help="""Whether to open the ZnDraw GUI in the default web browser."""
#         ),
#     ] = True,
#     figures_path: Annotated[
#         str | None,
#         typer.Option(
#             help="Provide a path to save the figures to."
#             "No figures will be saved by default."
#         ),
#     ] = None,
# ):
#     """Compare mlipx nodes and visualize the results using ZnDraw."""
#     # TODO: allow for glob patterns
#     if kwarg is None:
#         kwarg = []
#     node_names, revs, remotes = [], [], []
#     if glob:
#         fs = dvc.api.DVCFileSystem()
#         with fs.open("zntrack.json", mode="r") as f:
#             all_nodes = list(json.load(f).keys())

#     for node in nodes:
#         # can be name or name@rev or name@remote@rev
#         parts = node.split("@")
#         if glob:
#             filtered_nodes = [x for x in all_nodes if fnmatch.fnmatch(x, parts[0])]
#         else:
#             filtered_nodes = [parts[0]]
#         for x in filtered_nodes:
#             node_names.append(x)
#             if len(parts) == 1:
#                 revs.append(None)
#                 remotes.append(None)
#             elif len(parts) == 2:
#                 revs.append(parts[1])
#                 remotes.append(None)
#             elif len(parts) == 3:
#                 remotes.append(parts[1])
#                 revs.append(parts[2])
#             else:
#                 raise ValueError(f"Invalid node format: {node}")

#     node_instances = {}
#     for node_name, rev, remote in tqdm(
#         zip(node_names, revs, remotes), desc="Loading nodes"
#     ):
#         node_instances[node_name] = zntrack.from_rev(node_name, remote=remote, rev=rev)

#     if len(node_instances) == 0:
#         typer.echo("No nodes to compare")
#         return

#     typer.echo(f"Comparing {len(node_instances)} nodes")

#     kwargs = {}
#     for arg in kwarg:
#         key, value = arg.split("=", 1)
#         kwargs[key] = value
#     result = node_instances[node_names[0]].compare(*node_instances.values(), **kwargs)

#     token = token or str(uuid.uuid4())
#     typer.echo(f"View the results at {zndraw_url}/token/{token}")
#     vis = ZnDraw(zndraw_url, token=token, convert_nan=convert_nan)
#     length = len(vis)
#     vis.extend(result["frames"])
#     del vis[:length]  # temporary fix
#     vis.figures = result["figures"]
#     if browser:
#         webbrowser.open(f"{zndraw_url}/token/{token}")
#     if figures_path:
#         for desc, fig in result["figures"].items():
#             pio.write_json(fig, pathlib.Path(figures_path) / f"{desc}.json")

#     vis.socket.sleep(5)




# original mlipx functions above, new functions below:


from typer import Option, Argument

@app.command()
def phonon_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to phonon nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    normalise_to_model: Annotated[str, Option("--normalise_to_model", help="Model to normalise to")] = "mace_mp_0a_D3",
    batched: Annotated[bool, Option("--batched", help="does one node contain many materials")]=True,
    no_plots: Annotated[bool, Option("--no_plots", help="Disable plots")] = False,

    ):
    """Launch interactive benchmark for phonon dispersion."""
    import fnmatch
    import dvc.api
    import json
    # import torch

    # if hasattr(torch.fx._symbolic_trace, "TRACED_MODULES"):
    #     torch.fx._symbolic_trace.TRACED_MODULES.clear()

    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())

    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_phonons-dispersion")

    if batched:
        pred_node_dict, ref_node = load_nodes_phonon_batch(node_objects, models, split_str="_phonons-dispersion")
        
    
    else: 
        pred_node_dict, ref_node_dict = load_nodes_mpid_model(node_objects, models, split_str="_phonons-dispersion")
    


    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)

    print('\n UI = ', ui)
    
    if batched:
        from mlipx import PhononAllBatch
        PhononAllBatch.benchmark_interactive(
            pred_node_dict=pred_node_dict,
            ref_phonon_node=ref_node,
            ui=ui,
            no_plots=no_plots,
        )
            
    else:
        from mlipx import PhononDispersion
        PhononDispersion.benchmark_interactive(
            pred_node_dict=pred_node_dict,
            ref_node_dict=ref_node_dict,
            ui = ui,
            normalise_to_model=normalise_to_model,
            no_plots=no_plots,
        )
    
# ------ helper funcitons -------



def load_node_objects(
    nodes: list[str],
    glob: bool,
    models: list[str] | None,
    all_nodes: list[str],
    split_str: str,
    #exp_data: bool = False,
) -> dict[str, zntrack.Node]:
    
    selected_nodes = []
    if glob:
        for pattern in nodes:
            matched = fnmatch.filter(all_nodes, pattern)
            for name in matched:
                model = name.split(split_str)[0]
                if not models or model in models:
                    selected_nodes.append(name)
                elif "ref" in name or "reference" in name:
                    selected_nodes.append(name)
    else:
        for name in nodes:
            model = name.split(split_str)[0]
            if not models or model in models:
                selected_nodes.append(name)

    if not selected_nodes:
        typer.echo("No matching nodes found.")
        raise typer.Exit()

    # Instantiate nodes
    node_objects = {}
    # def load_node(name):
    #     return name, zntrack.from_rev(name)

    # with ThreadPoolExecutor() as executor:
    #     futures = {executor.submit(load_node, name): name for name in selected_nodes}
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Loading ZnTrack nodes"):
    #         name, obj = future.result()
    #         node_objects[name] = obj
        
    for name in selected_nodes:
        node_objects[name] = zntrack.from_rev(name)

    return node_objects
    
    
def load_nodes_phonon_batch(node_objects, models, split_str):
    
    pred_node_dict = {}
    ref_phonon_node = None
    
    for name, node in node_objects.items():
        if "reference" in name or "ref" in name or "Ref" in name:
            ref_phonon_node = node
            continue
        
        model = name.split(split_str)[0]
        
        if models and model not in models:
            continue
        
        pred_node_dict[model] = node
        
    return pred_node_dict, ref_phonon_node
    
    
def load_nodes_mpid_model(node_objects, models, split_str):
    """Load nodes which are structured: 
        - ref: Dict["mp_id": zntrack.Node]]
        - pred: Dict["mp_id": Dict["model_name", zntrack.Node]]
    """
    pred_node_dict = {}
    ref_node_dict = {}

    for name, node in node_objects.items():
        mp_id = name.split("_")[-1]
        model = name.split(split_str)[0]
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
    
    return pred_node_dict, ref_node_dict

# ------- end of helper functions -------



@app.command()
def gmtkn55_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to phonon nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,

    ):
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())

    
    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_GMTKN55Benchmark")

    benchmark_node_dict = {}

    for name, node in node_objects.items():
        model = name.split("_GMTKN55Benchmark")[0]
        benchmark_node_dict[model] = node

    if models:
        # filter to selected models
        benchmark_node_dict = {
            m: node for m, node in benchmark_node_dict.items() if m in models
        }

    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)

    print('\n UI = ', ui)

    from mlipx import GMTKN55Benchmark
    GMTKN55Benchmark.mae_plot_interactive(
        node_dict = benchmark_node_dict,
        ui = ui
    )
    

@app.command()
def cohesive_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to cohesive nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,

    ):
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())

    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_CohesiveEnergies")

    benchmark_node_dict = {}

    for name, node in node_objects.items():
        model = name.split("_CohesiveEnergies")[0]
        benchmark_node_dict[model] = node

    if models:
        # filter to selected models
        benchmark_node_dict = {
            m: node for m, node in benchmark_node_dict.items() if m in models
        }

    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)

    print('\n UI = ', ui)

    from mlipx import CohesiveEnergies
    CohesiveEnergies.mae_plot_interactive(
        node_dict=benchmark_node_dict,
        ui=ui
    )


@app.command()
def elasticity_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to elasticity nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,

    ):
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
    
    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_Elasticity")
    
    benchmark_node_dict = load_nodes_model(node_objects, models, split_str="_Elasticity")
    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    
    from mlipx import Elasticity
    Elasticity.mae_plot_interactive(
        node_dict=benchmark_node_dict,
        ui=ui
    )
    

def load_nodes_model(node_objects, models, split_str):
    """Load nodes which are structured: Dict["model_name": zntrack.Node]
    """
    benchmark_node_dict = {}
    
    for name, node in node_objects.items():
        model = name.split(split_str)[0]
        benchmark_node_dict[model] = node
    if models:
        # filter to selected models
        benchmark_node_dict = {
            m: node for m, node in benchmark_node_dict.items() if m in models
        }
    return benchmark_node_dict



@app.command()
def bulk_crystal_benchmark(
    #nodes: Annotated[list[str], typer.Argument(help="Path(s) to cohesive nodes")],
    #glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    normalise_to_model: Annotated[str, Option("--normalise_to_model", help="Model to normalise to")] = "mace_mp_0a_D3",

    ):
    
    nodes = [
        "*Elasticity*",
        "*Phonon*",
        "*LatticeConst*",
    ]
    glob = True
    
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
    
    
    phonon_pred_node_dict, phonon_ref_node_dict, elasticity_dict, lattice_const_dict, lattice_const_ref_node_dict = get_bulk_crystal_benchmark_node_dicts(
        nodes,
        glob,
        models,
        all_nodes,
        split_str_phonons="_phonons-dispersion",
        split_str_elasticity="_Elasticity",
        split_str_lattice_const="_lattice-constant",
    )
    
    
    from mlipx import BulkCrystalBenchmark
    BulkCrystalBenchmark.benchmark_interactive(
        phonon_ref_data=phonon_ref_node_dict,
        phonon_pred_data=phonon_pred_node_dict,
        elasticity_data=elasticity_dict,
        lattice_const_data=lattice_const_dict,
        lattice_const_ref_node_dict=lattice_const_ref_node_dict,
        ui = ui,
        normalise_to_model=normalise_to_model,
    )
    

def get_bulk_crystal_benchmark_node_dicts(
    nodes: list[str],
    glob: bool,
    models: list[str] | None,
    all_nodes: list[str],
    split_str_phonons: str,
    split_str_elasticity: str,
    split_str_lattice_const: str,
) -> dict[str, zntrack.Node]:
    
    phonon_nodes = [node for node in all_nodes if "phonon" in node]
    elasticity_nodes = [node for node in all_nodes if "Elasticity" in node]
    lattice_const_nodes = [node for node in all_nodes if "LatticeConst" in node]
        
    phonon_node_objects = load_node_objects(nodes, glob, models, phonon_nodes, split_str=split_str_phonons)
    elasticity_node_objects = load_node_objects(nodes, glob, models, elasticity_nodes, split_str=split_str_elasticity)
    lattice_const_node_objects = load_node_objects(nodes, glob, models, lattice_const_nodes, split_str=split_str_lattice_const)
    
    phonon_pred_node_dict, phonon_ref_node_dict = load_nodes_mpid_model(phonon_node_objects, models, split_str=split_str_phonons)
    elasticity_dict = load_nodes_model(elasticity_node_objects, models, split_str=split_str_elasticity)
    lattice_const_dict, lattice_const_ref_node_dict = load_nodes_and_ref_node_lat(lattice_const_node_objects, models, split_str=split_str_lattice_const)
    
    return phonon_pred_node_dict, phonon_ref_node_dict, elasticity_dict, lattice_const_dict, lattice_const_ref_node_dict
    
    
@app.command()
def lattice_constants_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to lattice constant nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,

    ):
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_lattice-constant")
    
    benchmark_node_dict, lattice_const_ref_node_dict = load_nodes_and_ref_node_lat(node_objects, models, split_str="_lattice-constant")
    
    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    from mlipx import LatticeConstant
    LatticeConstant.mae_plot_interactive(
        node_dict=benchmark_node_dict,
        ref_node_dict = lattice_const_ref_node_dict,
        ui=ui
    )
    
    
def load_nodes_and_ref_node_lat(node_objects, models, split_str):
    """Load nodes which are structured: Dict["model_name": zntrack.Node]
    and a single reference node
    """
    benchmark_node_dict = {}
    ref_node_dict = {}
    
    for name, node in node_objects.items():
        model = name.split(split_str)[0]
        
        if "ref" in model:
            print(f"Found reference node: {name}, {node}")
            if "exp" in name:
                ref_node_dict["exp"] = node
            elif "dft" in name:
                ref_node_dict["dft"] = node

        else:
            formula = name.split("LatticeConst-")[-1]
            benchmark_node_dict.setdefault(formula, {})[model] = node
    if models:
        # filter to selected models
        benchmark_node_dict = {
            ele: {m: node for m, node in models_dict.items() if m in models}
            for ele, models_dict in benchmark_node_dict.items()
        }
    return benchmark_node_dict, ref_node_dict



@app.command()
def X23_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to X23 nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    ):
    
    
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_X23Benchmark")
    benchmark_node_dict = {}
    for name, node in node_objects.items():
        model = name.split("_X23Benchmark")[0]
        benchmark_node_dict[model] = node
    if models:
        # filter to selected models
        benchmark_node_dict = {
            m: node for m, node in benchmark_node_dict.items() if m in models
        }
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    from mlipx import X23Benchmark
    X23Benchmark.mae_plot_interactive(
        node_dict=benchmark_node_dict,
        ui=ui
    )
    
@app.command()
def DMC_ICE_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to DMC ICE nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    ):
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_DMCICE13Benchmark")
    benchmark_node_dict = {}
    for name, node in node_objects.items():
        model = name.split("_DMCICE13Benchmark")[0]
        benchmark_node_dict[model] = node
    if models:
        # filter to selected models
        benchmark_node_dict = {
            m: node for m, node in benchmark_node_dict.items() if m in models
        }
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    from mlipx import DMCICE13Benchmark
    DMCICE13Benchmark.mae_plot_interactive(
        node_dict=benchmark_node_dict,
        ui=ui
    )


@app.command()
def mol_crystal_benchmark(
    #nodes: Annotated[list[str], typer.Argument(help="Path(s) to molecular crystal nodes")],
    #glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    ):
    
    nodes = [
        "*X23Benchmark*",
        "*DMCICE13Benchmark*",
    ]
    glob = True
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())

 
    X23_dict, ICE_DMC_dict = get_mol_crystal_benchmark_node_dicts(
        nodes,
        glob,
        models,
        all_nodes,
        split_str_X23="_X23Benchmark",
        split_str_ICE="_DMCICE13Benchmark",
    )
    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    from mlipx import MolecularCrystalBenchmark
    MolecularCrystalBenchmark.benchmark_interactive(
        X23_data=X23_dict,
        DMC_ICE_data=ICE_DMC_dict,
        ui=ui
    )
    

def get_mol_crystal_benchmark_node_dicts(
    nodes: list[str],
    glob: bool,
    models: list[str] | None,
    all_nodes: list[str],
    split_str_X23: str,
    split_str_ICE: str,
) -> dict[str, zntrack.Node]:
    
    X23_nodes = [node for node in all_nodes if "X23Benchmark" in node]
    ICE_DMC_nodes = [node for node in all_nodes if "DMCICE13Benchmark" in node]
        
    X23_node_objects = load_node_objects(nodes, glob, models, X23_nodes, split_str=split_str_X23)
    ICE_DMC_node_objects = load_node_objects(nodes, glob, models, ICE_DMC_nodes, split_str=split_str_ICE)
    
    X23_dict = load_nodes_model(X23_node_objects, models, split_str=split_str_X23)
    ICE_DMC_dict = load_nodes_model(ICE_DMC_node_objects, models, split_str=split_str_ICE)
    
    return X23_dict, ICE_DMC_dict
    
    
    
    
@app.command()
def full_benchmark_compare(
    #nodes: Annotated[list[str], typer.Argument(help="Path(s) to full benchmark nodes")],
    #glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    return_app: Annotated[bool, Option("--return_app", help="Return the app instance")] = False,
    report: Annotated[bool, Option("--report", help="Generate a report")] = True,
    normalise_to_model: Annotated[str, Option("--normalise_to_model", help="Model to normalise to")] = "mace_mp_0a_D3",
    ):
    
    nodes = [
        "*Elasticity*",
        "*Phonon*",
        "*LatticeConst*",
        "*X23Benchmark*",
        "*DMCICE13Benchmark*",
        "*GMTKN55Benchmark*",
        "*HomonuclearDiatomics*",
    ]
    glob = True
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
        
    # bulk crystal benchmark
    phonon_pred_node_dict, phonon_ref_node_dict, elasticity_dict, lattice_const_dict, lattice_const_ref_node_dict = get_bulk_crystal_benchmark_node_dicts(
        nodes,
        glob,
        models,
        all_nodes,
        split_str_phonons="_phonons-dispersion",
        split_str_elasticity="_Elasticity",
        split_str_lattice_const="_lattice-constant",
    )
    # molecular crystal benchmark
    X23_dict, ICE_DMC_dict = get_mol_crystal_benchmark_node_dicts(
        nodes,
        glob,
        models,
        all_nodes,
        split_str_X23="_X23Benchmark",
        split_str_ICE="_DMCICE13Benchmark",
    )
    # molecular benchmark
    GMTKN55_dict, HD_dict = get_mol_benchmark_node_dicts(
        nodes,
        glob,
        models,
        all_nodes,
        split_str_GMTKN55="_GMTKN55Benchmark",
        split_str_HD="_homonuclear-diatomics",
    )
    

    
    
    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    
    from mlipx import FullBenchmark
    if return_app:
        return FullBenchmark.benchmark_interactive(
            phonon_ref_data=phonon_ref_node_dict,
            phonon_pred_data=phonon_pred_node_dict,
            elasticity_data=elasticity_dict,
            lattice_const_data=lattice_const_dict,
            lattice_const_ref_node_dict=lattice_const_ref_node_dict,
            X23_data=X23_dict,
            DMC_ICE_data=ICE_DMC_dict,
            GMTKN55_data=GMTKN55_dict,
            HD_data=HD_dict,
            ui=ui,
            return_app = return_app,
            report=report,
            normalise_to_model=normalise_to_model,
        )
    else:
        FullBenchmark.benchmark_interactive(
            phonon_ref_data=phonon_ref_node_dict,
            phonon_pred_data=phonon_pred_node_dict,
            elasticity_data=elasticity_dict,
            lattice_const_data=lattice_const_dict,
            lattice_const_ref_node_dict=lattice_const_ref_node_dict,
            X23_data=X23_dict,
            DMC_ICE_data=ICE_DMC_dict,
            GMTKN55_data=GMTKN55_dict,
            HD_data=HD_dict,
            ui=ui,
            report=report,
            normalise_to_model=normalise_to_model,
        )
    
    
    
    
@app.command()
def molecular_benchmark(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to molecular benchmark nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    normalise_to_model: Annotated[str, Option("--normalise_to_model", help="Model to normalise to")] = "mace_mp_0a_D3",
    ):
    
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
    
    
    GMTKN55_dict, HD_dict = get_mol_benchmark_node_dicts(
        nodes,
        glob,
        models,
        all_nodes,
        split_str_GMTKN55="_GMTKN55Benchmark",
        split_str_HD="_homonuclear-diatomics",
    )
    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    from mlipx import MolecularBenchmark
    MolecularBenchmark.benchmark_interactive(
        GMTKN55_data=GMTKN55_dict,
        HD_data=HD_dict,
        ui=ui,
        normalise_to_model=normalise_to_model, # default model for normalisation
    )
    

def get_mol_benchmark_node_dicts(
    nodes: list[str],
    glob: bool,
    models: list[str] | None,
    all_nodes: list[str],
    split_str_GMTKN55: str = "_GMTKN55Benchmark",
    split_str_HD: str = "_homonuclear-diatomics",
) -> dict[str, zntrack.Node]:
    
    GMTKN55_nodes = [node for node in all_nodes if "GMTKN55Benchmark" in node]
    HD_nodes = [node for node in all_nodes if "HomonuclearDiatomics" in node]
    
    GMTKN55_node_objects = load_node_objects(nodes, glob, models, GMTKN55_nodes, split_str=split_str_GMTKN55)
    HD_node_objects = load_node_objects(nodes, glob, models, HD_nodes, split_str=split_str_HD)
    
    GMTKN55_dict = load_nodes_model(GMTKN55_node_objects, models, split_str=split_str_GMTKN55)
    HD_dict = load_nodes_model(HD_node_objects, models, split_str=split_str_HD)
    
    return GMTKN55_dict, HD_dict




@app.command()
def diatomics_compare(
    nodes: Annotated[list[str], typer.Argument(help="Path(s) to diatomics nodes")],
    glob: Annotated[bool, typer.Option("--glob", help="Enable glob patterns")] = False,
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model names to filter")] = None,
    ui: Annotated[str, Option("--ui", help="Select UI mode", show_choices=True)] = None,
    normalise_to_model: Annotated[str, Option("--normalise_to_model", help="Model to normalise to")] = "mace_mp_0a_D3",
):
    """Compare diatomic molecules using zntrack nodes."""
    # Load all node names from zntrack.json
    fs = dvc.api.DVCFileSystem()
    with fs.open("zntrack.json", mode="r") as f:
        all_nodes = list(json.load(f).keys())
    
    node_objects = load_node_objects(nodes, glob, models, all_nodes, split_str="_homonuclear-diatomics")
    
    benchmark_node_dict = load_nodes_model(node_objects, models, split_str="_homonuclear-diatomics")
    
    if ui not in {None, "browser"}:
        typer.echo("Invalid UI mode. Choose from: none or browser.")
        raise typer.Exit(1)
    print('\n UI = ', ui)
    
    from mlipx import HomonuclearDiatomics
    HomonuclearDiatomics.mae_plot_interactive(
        node_dict=benchmark_node_dict,
        ui=ui,
        normalise_to_model=normalise_to_model,
    )


