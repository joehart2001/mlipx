# assume we already have force constants and phonon frequencies (input file)

# we can use seekpath to get the band paths in the Brillouin zone

# we can use phonopy to get the phonon band structure

from pathlib import Path
from typing import Any, Callable
from functools import partial

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from tqdm import tqdm
from phonopy.api_phonopy import Phonopy


#from seekpath import get_path, get_cell
from phonopy.structure.atoms import PhonopyAtoms

#---------------code adapted from Balazs--------------


def calculate_fc2_phonopy_set(
    phonons: Phonopy,
    calculator: Calculator,
    log: bool = True,
    pbar_kwargs: dict[str, Any] = {},
) -> np.ndarray:
    # calculate FC2 force set

    #log_message(f"Computing FC2 force set in {get_chemical_formula(phonons)}.", output=log)

    forces = []
    nat = len(phonons.supercell)

    for sc in tqdm(
        phonons.supercells_with_displacements,
        desc=f"FC2 calculation: {get_chemical_formula(phonons)}",
        **pbar_kwargs,
    ):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc = calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    phonons.forces = force_set
    return force_set



def init_phonopy(
    atoms: Atoms,
    fc2_supercell: np.ndarray | None = None,  
    primitive_matrix: Any = "auto",
    log: str | Path | bool = True,
    symprec: float = 1e-5,
    displacement_distance: float = 0.03,
    **kwargs: Any,
) -> tuple[Phonopy, list[Any]]:
    """Calculate fc2 and fc3 force lists from phonopy.

    """
    if not log:
        log_level = 0
    elif log is not None:
        log_level = 1

    if fc2_supercell is not None :
        _fc2_supercell = fc2_supercell
    else:
        if "fc2_supercell" in atoms.info.keys() :
            _fc2_supercell = atoms.info["fc2_supercell"]
        else:
            raise ValueError(f'{atoms.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.')


    # Initialise Phonopy object
    phonons = aseatoms2phonopy(
        atoms,
        fc2_supercell=_fc2_supercell,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        log_level=log_level,
        **kwargs,
    )

    phonons.generate_displacements(distance=displacement_distance)

    return phonons



def init_phonopy_from_ref(
    atoms: Atoms,
    fc2_supercell: np.ndarray | None = None,  
    primitive_matrix: Any = "auto",
    log: str | Path | bool = True,
    symprec: float = 1e-5,
    displacement_dataset: dict | None = None,
    **kwargs: Any,
) -> tuple[Phonopy, list[Any]]:
    """Calculate fc2 and fc3 force lists from phonopy.

    """
    if not log:
        log_level = 0
    elif log is not None:
        log_level = 1

    if fc2_supercell is not None :
        _fc2_supercell = fc2_supercell
    else:
        if "fc2_supercell" in atoms.info.keys() :
            _fc2_supercell = atoms.info["fc2_supercell"]
        else:
            raise ValueError(f'{atoms.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.')


    # Initialise Phonopy object
    phonons = aseatoms2phonopy(
        atoms,
        fc2_supercell=_fc2_supercell,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        log_level=log_level,
        **kwargs,
    )
    
    # set the displacement dataset
    phonons.dataset = displacement_dataset


    return phonons


def get_fc2_and_freqs(
    phonons: Phonopy,
    calculator: Calculator | None = None,
    q_mesh : np.ndarray | None = None,
    symmetrize_fc2 = True,
    log: str | Path | bool = True,
    pbar_kwargs: dict[str, Any] = {"leave": False},
    **kwargs: Any,
) -> tuple[Phonopy, np.ndarray, np.ndarray]:
   

    if calculator is None:
        raise ValueError(
            f'{get_chemical_formula(phonons)} "calculator" was provided when calculating fc2 force sets.'
        )

    fc2_set = calculate_fc2_phonopy_set(phonons, calculator, log=log, pbar_kwargs=pbar_kwargs)

    phonons.produce_force_constants(show_drift=False)
    
    if symmetrize_fc2:
        phonons.symmetrize_force_constants(show_drift=False)
    
    if q_mesh is not None:
        phonons.run_mesh(q_mesh, **kwargs)
        freqs= phonons.get_mesh_dict()["frequencies"]
    else:
        freqs=[]

    return phonons, fc2_set, freqs




def load_force_sets(
    phonons: Phonopy, fc2_set: np.ndarray
) -> Phonopy:
    phonons.forces = fc2_set
    phonons.produce_force_constants(symmetrize_fc2=True)
    return phonons


# some helper functions:

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.api_phonopy import Phonopy
from ase import Atoms

def aseatoms2phonoatoms(atoms):
    phonoatoms = PhonopyAtoms(
        atoms.symbols, cell=atoms.cell, positions=atoms.positions, pbc=True
    )
    return phonoatoms

def phonoatoms2aseatoms(phonoatoms):
    atoms = Atoms(
        phonoatoms.symbols, cell=phonoatoms.cell, positions=phonoatoms.positions, pbc=True
    )
    return atoms


def aseatoms2phonopy(
    atoms, fc2_supercell, primitive_matrix=None, nac_params=None, symprec=1e-5,**kwargs
) -> Phonopy:
    unitcell = aseatoms2phonoatoms(atoms)
    return Phonopy(
        unitcell=unitcell,
        supercell_matrix=fc2_supercell,
        primitive_matrix=primitive_matrix,
        nac_params = nac_params,
        symprec=symprec,
        **kwargs,
    )

def phonopy2aseatoms(phonons: Phonopy) -> Atoms:
    phonopy_atoms = phonons.unitcell
    atoms = Atoms(
        phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        positions=phonopy_atoms.positions,
        pbc=True,
    )

    if phonons.supercell_matrix is not None:
        atoms.info["fc2_supercell"] = phonons.supercell_matrix

    if phonons.primitive_matrix is not None:
        atoms.info["primitive_matrix"] = phonons.primitive_matrix

    if phonons.mesh_numbers is not None:
        atoms.info["q_mesh"] = phonons.mesh_numbers

    return atoms


def get_chemical_formula(phonons: Phonopy, mode="metal", **kwargs):
    unitcell = phonons.unitcell
    atoms = Atoms(
        unitcell.symbols, cell=unitcell.cell, positions=unitcell.positions, pbc=True
    )
    return atoms.get_chemical_formula(mode=mode, **kwargs)

def load_phonopy(yaml_file, **kwargs):
    from phonopy.cui.load import load
    return load(yaml_file, **kwargs)