import json
import os
import pathlib
import subprocess
import typing as t

import jinja2
import typer

from jinja2 import Environment, FileSystemLoader

CWD = pathlib.Path(__file__).parent

app = typer.Typer()


def initialize_directory():
    """Initialize a Git and DVC repository."""
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["dvc", "init"], check=True)


def render_template(template_name: str, output_name: str, **context):
    """Render a Jinja2 template and write it to a file."""
    template_path = CWD / template_name
    template = jinja2.Template(template_path.read_text())
    with open(output_name, "w") as f:
        f.write(template.render(**context))


def render_templateception(template_name: str, output_name: str, **context):
    """Render a Jinja2 template and write it to a file, supporting includes.
        Robust to be able to render templates inside of templates e.g. for bulk_crystal_benchmark.py.jinja2
        which contrains a call to phonons.py.jinja2
    """
    template_path = CWD / template_name
    # set up the environment with a loader
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    # get template by name
    template = env.get_template(template_path.name)

    with open(output_name, "w") as f:
        f.write(template.render(**context))


def repro_if_requested(repro: bool):
    """Run the repro pipeline if requested."""
    if repro:
        subprocess.run(["python", "main.py"], check=True)
        subprocess.run(["dvc", "repro"], check=True)


def render_models(models: str | None):
    """Render the models.py file if models are specified."""
    if models:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))


def parse_inputs(datapath: str | None, material_ids: str | None, smiles: str | None, subsets: str | None = None):
    """Parse and validate input arguments."""
    if not any([datapath, material_ids, smiles, ]):
        #raise ValueError(
        #    "Provide at least one of `datapath`, `material_ids`, or `smiles`."
        #)
        print(
            "Provide at least one of `datapath`, `material_ids`, or `smiles`, unless running a predefined benchmark."
        )

    return {
        "datapath": datapath.split(",") if datapath else None,
        "material_ids": material_ids.split(",") if material_ids else None,
        "smiles": smiles.split(",") if smiles else None,
        "subsets": subsets.split(",") if subsets else None,
    }


def handle_recipe(
    template_name: str,
    initialize: bool,
    repro: bool,
    datapath: str | None,
    material_ids: str | None,
    smiles: str | None,
    **additional_context,
):
    """Common logic for handling recipes."""
    if initialize:
        initialize_directory()

    inputs = parse_inputs(datapath, material_ids, smiles)
    render_template(template_name, "main.py", **inputs, **additional_context)
    repro_if_requested(repro)
    
    
def handle_recipeception(
    template_name: str,
    initialize: bool,
    repro: bool,
    datapath: str | None,
    material_ids: str | None,
    smiles: str | None,
    **additional_context,
):
    """Common logic for handling recipes.
        Allows for rendering templates inside of templates e.g. for bulk_crystal_benchmark.py.jinja2
    """
    if initialize:
        initialize_directory()

    inputs = parse_inputs(datapath, material_ids, smiles)
    #inputs.update(additional_context)
    render_templateception(template_name, "main.py", **inputs, **additional_context)
    repro_if_requested(repro)




@app.command()
def relax(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Perform a relaxation task."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        CWD / "relax.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )


# @app.command()
# def neb(
#     initialize: bool = False,
#     datapath: str = "...",
#     repro: bool = False,
#     models: str | None = None,
# ):
#     """Build a NEB recipe."""
#     if models is not None:
#         render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
#     if initialize:
#         initialize_directory()
#     template = jinja2.Template((CWD / "neb.py").read_text())
#     with open("main.py", "w") as f:
#         f.write(template.render(datapath=datapath))
#     repro_if_requested(repro)


@app.command()
def neb(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
    use_janus: bool = False,
    all_images: bool = False,
):

    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "neb.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        use_janus=use_janus,
        all_images=all_images,
    )
    


    
    

@app.command()
def vibrational_analysis(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run vibrational analysis."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "vibrational_analysis.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )


@app.command()
def phase_diagram(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Build a phase diagram."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "phase_diagram.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )


@app.command()
def pourbaix_diagram(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Build a Pourbaix diagram."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "pourbaix_diagram.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=None,
    )


@app.command()
def md(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
    steps: int = 1000,
    temperature: int = 300,
    resume_MD: bool = False,
    resume_trajectory_path: str | None = None,
):
    """Build an MD recipe."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "md.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        steps=steps,
        temperature=temperature,
        resume_MD=resume_MD,
    )


@app.command()
def homonuclear_diatomics(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
    het_diatomics: bool = True,
):
    """Run homonuclear diatomics calculations."""
    if models is not None:
        models_lst = models.split(",")
        orcashell = ""
        if "orca" in models_lst:
            if "MLIPX_ORCA" not in os.environ:
                orcashell = typer.prompt("Enter the path to the Orca executable")
            else:
                orcashell = None

        render_template(
            CWD / "models.py.jinja2",
            "models.py",
            models=models_lst,
            orcashell=orcashell,
        )

    handle_recipe(
        "homonuclear_diatomics.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        het_diatomics=het_diatomics,
    )


@app.command()
def ev(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Compute Energy-Volume curves."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "energy_volume.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )


@app.command()
def metrics(
    initialize: bool = False,
    datapath: str = "...",
    isolated_atom_energies: bool = False,
    repro: bool = False,
    models: str | None = None,
):
    """Compute Energy and Force Metrics.

    Parameters
    ----------
    initialize : bool
        Initialize a git and dvc repository.
    datapath : str
        Path to the data directory.
    isolated_atom_energies: bool
        Compute metrics based on isolated atom energies.
    """
    if initialize:
        initialize_directory()
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    template = jinja2.Template((CWD / "metrics.py").read_text())
    with open("main.py", "w") as f:
        f.write(
            template.render(
                datapath=datapath, isolated_atom_energies=isolated_atom_energies
            )
        )
    repro_if_requested(repro)


@app.command()
def invariances(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Test rotational, permutational, and translational invariance."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "invariances.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )


@app.command()
def adsorption(
    initialize: bool = False,
    repro: bool = False,
    slab_config: str | None = None,
    slab_material_id: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Test rotational, permutational, and translational invariance."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    if slab_config is not None:
        slab_config = json.loads(slab_config)
    handle_recipe(
        "adsorption.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=None,
        material_ids=None,
        smiles=smiles,
        slab_config=slab_config,
    )


@app.command()
def phonons(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    n_materials: int | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run phonon calculations."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "phonons.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        n_materials=n_materials,
    )
    
    
@app.command()
def phonons_all(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    n_phonons: int | None = None,
    n_phonons_start: int | None = None,
    small: bool = False,
    medium: bool = False,
    n_jobs: int | None = -1,
    check_completed: bool | None = None,
    generate_displacements: bool = False,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
    ref: bool = False,
):
    """Run phonon calculations."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "phonons_all.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        n_phonons=n_phonons,
        n_phonons_start=n_phonons_start,
        small=small,
        medium=medium,
        check_completed=check_completed,
        generate_displacements=generate_displacements,
        n_jobs=n_jobs,
        ref=ref,
    )
    
    
    
@app.command()
def gmtkn55(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run GMTKN55 benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "GMTKN55_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )
    
@app.command()
def wiggle150(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run WIGGLE150 benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "wiggle150.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )
    

@app.command()
def cohesive_energies(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run cohesive energies benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "cohesive_energies.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
    )
    

@app.command()
def elasticity(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    small: bool = False,
    medium: bool = False,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
    n_jobs: int | None = -1,
):
    """Run elasticity benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "elasticity.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        n_materials=n_materials,
        smiles=smiles,
        n_jobs=n_jobs,
    )


@app.command()
def bulk_crystal(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    n_phonons: int | None = None,
    small: bool = False,
    medium: bool = False,
    n_jobs: int | None = -1,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run bulk crystal benchmark."""
    if models is not None:
        render_templateception(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipeception(
        "bulk_crystal_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        n_materials=n_materials,
        n_phonons=n_phonons,
        small=small,
        medium=medium,
        n_jobs=n_jobs,
    )
    

@app.command()
def lattice_constants(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run lattice constants benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "lattice_constants.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        n_materials=n_materials,
        smiles=smiles,
    )
    
@app.command()
def atomisation_energy(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run atomisation energy benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "atomisation_energy.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        n_materials=n_materials,
        smiles=smiles,
    )
    

@app.command()
def X23(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run X23 benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "X23_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        n_materials=n_materials,
        smiles=smiles,
    )
    
@app.command()
def DMC_ICE(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run DMC ICE benchmark."""
    if models is not None:
        render_template(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipe(
        "DMC_ICE_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        n_materials=n_materials,
        smiles=smiles,
    )
    

@app.command()
def molecular_crystal(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run molecular crystal benchmark."""
    if models is not None:
        render_templateception(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipeception(
        "molecular_crystal_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        n_materials=n_materials,
    )
    
@app.command()
def full_benchmark(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    n_materials: int | None = None,
    n_phonons: int | None = None,
    small: bool = False,
    medium: bool = False,
    steps: int = 1000,
    temperature: int = 300,
    n_jobs: int | None = -1,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run full benchmark."""
    if models is not None:
        render_templateception(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipeception(
        "full_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        n_materials=n_materials,
        n_phonons=n_phonons,
        small=small,
        medium=medium,
        steps=steps,
        temperature=temperature,
        n_jobs=n_jobs,
    )
    
    

@app.command()
def homonuclear_diatomics_benchmark(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    het_diatomics: bool = False,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
):
    """Run homonuclear diatomics calculations."""
    if models is not None:
        models_lst = models.split(",")
        orcashell = ""
        if "orca" in models_lst:
            if "MLIPX_ORCA" not in os.environ:
                orcashell = typer.prompt("Enter the path to the Orca executable")
            else:
                orcashell = None

        render_template(
            CWD / "models.py.jinja2",
            "models.py",
            models=models_lst,
            orcashell=orcashell,
        )

    handle_recipe(
        "homonuclear_diatomics_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        het_diatomics=het_diatomics,
    )
    
@app.command()
def further_applications_benchmark(
    initialize: bool = False,
    repro: bool = False,
    datapath: str | None = None,
    material_ids: str | None = None,
    smiles: str | None = None,
    models: t.Annotated[str | None, typer.Option()] = None,
    steps: int = 1000,
    temperature: int = 300,
):
    """Run further applications benchmark."""
    if models is not None:
        render_templateception(CWD / "models.py.jinja2", "models.py", models=models.split(","))
    handle_recipeception(
        "further_applications_benchmark.py.jinja2",
        initialize=initialize,
        repro=repro,
        datapath=datapath,
        material_ids=material_ids,
        smiles=smiles,
        steps=steps,
        temperature=temperature,
    )