import pathlib

import ase.io
import zntrack
from ase.build import bulk
from ase import Atoms


class Crystal(zntrack.Node):
    """Generate a bulk material structure.

    Parameters
    ----------
    element: str
        The chemical symbol of the element.
    crystal_structure: str
        The type of crystal structure (e.g., "bcc", "fcc", "hcp").
    a: float
        The lattice constant.
    c: float, optional
        The c/a ratio for hexagonal structures.

    Example
    -------
    >>> Crystal(element="W", crystal_structure="bcc", a=3.16)

    """

    element: str = zntrack.params()
    crystal_structure: str = zntrack.params()
    a: float = zntrack.params()
    c: float = zntrack.params(default=None)  # Only needed for hcp
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")
    
    def run(self):
        if self.crystal_structure == "hcp":
            if self.c is None:
                raise ValueError("hcp structure requires a c/a ratio (c).")
            atoms = bulk(self.element, self.crystal_structure, a=self.a, c=self.c)
        else:
            atoms = bulk(self.element, self.crystal_structure, a=self.a)

        ase.io.write(self.frames_path, atoms)


    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))