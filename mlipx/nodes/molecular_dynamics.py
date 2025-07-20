import dataclasses
import pathlib

from scipy.ndimage import gaussian_filter1d

import ase.io
import ase.units
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
import zntrack
from ase.md import Langevin
from ase.md.npt import NPT
from ase.io import read, write
import typing as t
import time
import json
from ase.neighborlist import neighbor_list
from joblib import Parallel, delayed
import pandas as pd
import json
from dash import dash_table, dcc, html
import dash
from mlipx.dash_utils import run_app, dash_table_interactive
import pickle
from pathlib import Path
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import IsotropicMTKNPT


from mlipx.abc import (
    ComparisonResults,
    DynamicsModifier,
    DynamicsObserver,
    NodeWithCalculator,
    NodeWithMolecularDynamics,
)



@dataclasses.dataclass
class LangevinConfig:
    """Configure a Langevin thermostat for molecular dynamics.

    Parameters
    ----------
    timestep : float
        Time step for the molecular dynamics simulation in fs.
    temperature : float
        Temperature of the thermostat.
    friction : float
        Friction coefficient of the thermostat.
    """
    timestep: float
    temperature: float
    friction: float

    def get_molecular_dynamics(self, atoms) -> Langevin:
        return Langevin(
            atoms,
            timestep=self.timestep * ase.units.fs,
            temperature_K=self.temperature,
            friction=self.friction,
        )


# NPTConfig dataclass for NPT ensemble
@dataclasses.dataclass
class NPTConfig:
    """Configure an NPT barostat for molecular dynamics.

    Parameters
    ----------
    timestep : float
        Time step for the MD simulation in fs.
    temperature : float
        Target temperature in Kelvin.
    externalstress : float
        External pressure/stress (default: 0.0).
    ttime : float
        Thermostat time constant in fs.
    pfactor : float
        Barostat mass factor (dimensionless or in ASE units).
    """
    timestep: float
    temperature: float
    externalstress: float = 0.0
    thermostat_time: float = 20.0  # thermostat time in fs
    barostat_time: float = 75.0  # barostat time in fs
    bulk_modulus: float = 2.0  # Bulk modulus in GPa
    #ttime: float = 20.0    # thermostat time in fs
    #pfactor: float = 2.0   # Barostat parameter in GPa


    def get_molecular_dynamics(self, atoms):
        return NPT(
            atoms,
            timestep=self.timestep * ase.units.fs,
            temperature_K=self.temperature,
            externalstress=self.externalstress,
            ttime=self.thermostat_time * ase.units.fs,
            pfactor=(self.barostat_time**2 * self.bulk_modulus)*ase.units.GPa*(ase.units.fs**2),
        )


@dataclasses.dataclass
class NPT_MTK_Config:
    """Configure an NPT barostat using the Nose-Hoover MTK integrator.

    Parameters
    ----------
    timestep : float
        The time step in fs.
    temperature : float
        The target temperature in Kelvin.
    pressure : float
        External pressure in GPa (default: 0.0).
    tdamp : float
        Thermostat time constant in fs (typically 100x timestep).
    pdamp : float
        Barostat time constant in fs (typically 1000x timestep).
    tchain : int
        Number of thermostat chain variables.
    pchain : int
        Number of barostat chain variables.
    tloop : int
        Number of thermostat substeps.
    ploop : int
        Number of barostat substeps.
    extra_kwargs : dict
        Extra arguments passed to the integrator.
    """
    timestep: float
    temperature: float
    pressure: float = 1
    tdamp: float = 100
    pdamp: float = 1000
    tchain: int = 3
    pchain: int = 3
    tloop: int = 1
    ploop: int = 1
    extra_kwargs: dict = dataclasses.field(default_factory=dict)

    def get_molecular_dynamics(self, atoms):
        return IsotropicMTKNPT(
            atoms,
            timestep=self.timestep * ase.units.fs,
            temperature_K=self.temperature,
            pressure_au=self.pressure * ase.units.GPa,
            tdamp=self.tdamp * ase.units.fs,
            pdamp=self.pdamp * ase.units.fs,
            tchain=self.tchain,
            pchain=self.pchain,
            tloop=self.tloop,
            ploop=self.ploop,
            **self.extra_kwargs,
        )

        
class MolecularDynamics(zntrack.Node):
    """Run molecular dynamics simulation.

    Parameters
    ----------
    model : NodeWithCalculator
        Node providing the calculator object for the simulation.
    thermostat : LangevinConfig
        Node providing the thermostat object for the simulation.
    data : list[ase.Atoms]
        Initial configurations for the simulation.
    data_id : int, default=-1
        Index of the initial configuration to use.
    steps : int, default=100
        Number of steps to run the simulation.
    ensemble : str, default="NVT"
        Ensemble to use for molecular dynamics ("NVT" or "NPT").
    """

    model: NodeWithCalculator = zntrack.deps()
    thermostat_barostat_config: t.Union[LangevinConfig, NPTConfig, NPT_MTK_Config] = zntrack.deps()
    data: list[ase.Atoms] = zntrack.deps(None)
    data_path: pathlib.Path = zntrack.params(None)
    data_id: int = zntrack.params(-1)
    steps: int = zntrack.params(100)
    print_energy_every: int = zntrack.params(1000)
    write_frames_every: int = zntrack.params(10)
    observers: list[DynamicsObserver] = zntrack.deps(None)
    modifiers: list[DynamicsModifier] = zntrack.deps(None)
    external_save_path: pathlib.Path = zntrack.params(None)
    resume_trajectory_path: pathlib.Path = zntrack.params(None)
    ensemble: str = zntrack.params("NVT")  # Options: "NVT", "NPT"

    observer_metrics: dict = zntrack.metrics()
    plots: pd.DataFrame = zntrack.plots(y=["energy", "fmax"], autosave=True)

    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")
    traj_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "trajectory.traj")


    def run(self):
        start_time = time.time()

        # Only check for existing frames and plots if resuming MD
        # if self.resume_trajectory_path is not None:
        #     if Path(self.resume_trajectory_path).exists():
        #         atoms = read(self.resume_trajectory_path, index=-1)
            
                #existing_frames = list(ase.io.iread(self.resume_trajectory_path, format="extxyz"))
                #start_idx = len(existing_frames) * self.write_frames_every

        #     plots_path = self.frames_path.with_name("plots.csv")
        #     if plots_path.exists():
        #         self.plots = pd.read_csv(plots_path)
        #     else:
        # self.plots = pd.DataFrame(columns=["energy", "fmax", "fnorm"])
        # else:
            
        plots_path = self.frames_path.with_name("plots.csv")

        if self.observers is None:
            self.observers = []
        if self.modifiers is None:
            self.modifiers = []

        if self.resume_trajectory_path and Path(self.resume_trajectory_path).exists():
            # Resume from last frame
            atoms = read(self.resume_trajectory_path, index=-1)
            atoms.calc = self.model.get_calculator()

            # Load full trajectory to count previous frames
            full = read(self.resume_trajectory_path, index=":")
            start_idx = len(full) * self.write_frames_every
            # populate the traj file with the existing frames
            with Trajectory(self.traj_path, mode="w") as traj:
                for atoms_i in full:
                    traj.write(atoms_i)

            if start_idx >= self.steps:
                print(f"Already completed {start_idx} steps, stopping run.")
                # repopulate the node directory with the existing data
                write(self.traj_path, full, format="extxyz")
                
                self.plots = pd.DataFrame(columns=["energy", "fmax", "fnorm"])
                
                for atoms_i in full:
                    ase.io.write(self.frames_path, atoms_i, append=True)
                    if self.external_save_path:
                        ase.io.write(self.external_save_path, atoms_i, append=True)
                    plots = {
                        "energy": atoms_i.get_potential_energy(),
                        "fmax": np.max(np.linalg.norm(atoms_i.get_forces(), axis=1)),
                        "fnorm": np.linalg.norm(atoms_i.get_forces()),
                    }
                    self.plots.loc[len(self.plots)] = plots
                self.plots.to_csv(plots_path, index=False)
                self.observer_metrics = {}
                return
            

            print(f"Resuming from trajectory: {self.resume_trajectory_path} at step {start_idx}")

            traj_mode = 'a'  # Append to both files

            # Try to resume previous plot data
            plots_path = self.frames_path.with_name("plots.csv")
            if plots_path.exists():
                self.plots = pd.read_csv(plots_path)
            else:
                self.plots = pd.DataFrame(columns=["energy", "fmax", "fnorm"])
        else:
            # Fresh start
            if self.data:
                atoms = self.data[self.data_id]
            elif self.data_path:
                atoms = read(self.data_path, self.data_id)
            atoms.calc = self.model.get_calculator()
            start_idx = 0
            traj_mode = 'w'
            self.plots = pd.DataFrame(columns=["energy", "fmax", "fnorm"])

        # Select MD integrator based on ensemble
        # if self.ensemble.upper() == "NVT":
        #     dyn = self.thermostat_barostat_config.get_molecular_dynamics(atoms)
        # elif self.ensemble.upper() == "NPT":
        #     from ase.md.npt import NPT
        #     dyn = NPT(
        #         atoms,
        #         timestep=self.thermostat_barostat_config.timestep * ase.units.fs,
        #         temperature_K=self.thermostat_barostat_config.temperature,
        #         externalstress=0.0,
        #         ttime=25.0 * ase.units.fs,
        #         pfactor=1.0,
        #     )
        # elif self.ensemble.upper() == "NPT_MTK":
        #     dyn = IsotropicMTKNPT
        # else:
        #     raise ValueError(f"Unknown ensemble '{self.ensemble}'. Choose 'NVT' or 'NPT'.")
        
        dyn = self.thermostat_barostat_config.get_molecular_dynamics(atoms)
        
        for obs in self.observers:
            obs.initialize(atoms)

        self.observer_metrics = {}


        
        if self.resume_trajectory_path:
            if Path(self.resume_trajectory_path).exists():
                resume_traj = Trajectory(self.resume_trajectory_path, mode="a", atoms=atoms)
                node_traj = Trajectory(self.traj_path, mode=traj_mode, atoms=atoms)
                dyn.attach(resume_traj.write, interval=self.write_frames_every)
                dyn.attach(node_traj.write, interval=self.write_frames_every)
        else:
            trajectory = Trajectory(self.traj_path, mode=traj_mode, atoms=atoms)
            dyn.attach(trajectory.write, interval=self.write_frames_every)
            
        remaining_steps = max(0, self.steps - start_idx)

        for local_idx, _ in enumerate(
            tqdm.tqdm(
                dyn.irun(steps=remaining_steps),
                total=self.steps,
                initial=start_idx,
            )
        ):
            idx = start_idx + local_idx

            if idx % self.write_frames_every == 0:
                ase.io.write(self.frames_path, atoms, append=True)


                if self.external_save_path:
                    ase.io.write(self.external_save_path, atoms, append=True)

                plots = {
                    "energy": atoms.get_potential_energy(),
                    "fmax": np.max(np.linalg.norm(atoms.get_forces(), axis=1)),
                    "fnorm": np.linalg.norm(atoms.get_forces()),
                }
                self.plots.loc[len(self.plots)] = plots
                # Save to CSV after each update
                self.plots.to_csv(plots_path, index=False)

            # print every x steps
            if self.print_energy_every is not None and idx % self.print_energy_every == 0:
                epot = atoms.get_potential_energy() / len(atoms)
                ekin = atoms.get_kinetic_energy() / len(atoms)
                elapsed_time = time.time() - start_time
                elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
                tqdm.tqdm.write(
                    f"Step {idx} | Epot = {epot:.3f} eV  Ekin = {ekin:.3f} eV  "
                    f"(T = {ekin / (1.5 * ase.units.kB):.0f} K)  "
                    f"Etot = {epot + ekin:.3f} eV  Elapsed: {int(elapsed_min)}m {elapsed_sec:.1f}s"
                )

            for obs in self.observers:
                if obs.check(atoms):
                    self.observer_metrics[obs.name] = idx

            if len(self.observer_metrics) > 0:
                break

            for mod in self.modifiers:
                mod.modify(dyn, idx)

        for obs in self.observers:
            # document all attached observers
            self.observer_metrics[obs.name] = self.observer_metrics.get(obs.name, -1)



    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, format="extxyz"))

    # @property
    # def traj(self) -> list[ase.Atoms]:
    #     """Return the trajectory as a list of Atoms objects."""
    #     with self.state.fs.open(self.traj_path, "rb") as f:
    #         return list(Trajectory(f))

    @property
    def traj(self) -> list[ase.Atoms]:
        """Return the trajectory as a list of Atoms objects."""

        return read(self.traj_path, index=":")

    @property
    def figures(self) -> dict[str, go.Figure]:
        return {
            key: px.line(self.plots, x=self.plots.index, y=key, title=key)
            for key in self.plots.columns
        }

    @staticmethod
    def compare(*nodes: "MolecularDynamics") -> ComparisonResults:
        frames = sum([node.frames for node in nodes], [])
        offset = 0
        fig = go.Figure()
        for _, node in enumerate(nodes):
            energies = [atoms.get_potential_energy() for atoms in node.frames]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(energies))),
                    y=energies,
                    mode="lines+markers",
                    name=node.name.replace(f"_{node.__class__.__name__}", ""),
                    customdata=np.stack([np.arange(len(energies)) + offset], axis=1),
                )
            )
            offset += len(energies)

        fig.update_layout(
            title="Energy vs. step",
            xaxis_title="Step",
            yaxis_title="Energy",
        )

        fig.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )

        # Now we set the first energy to zero for better compareability.

        offset = 0
        fig_adjusted = go.Figure()
        for _, node in enumerate(nodes):
            energies = np.array([atoms.get_potential_energy() for atoms in node.frames])
            energies -= energies[0]
            fig_adjusted.add_trace(
                go.Scatter(
                    x=list(range(len(energies))),
                    y=energies,
                    mode="lines+markers",
                    name=node.name.replace(f"_{node.__class__.__name__}", ""),
                    customdata=np.stack([np.arange(len(energies)) + offset], axis=1),
                )
            )
            offset += len(energies)

        fig_adjusted.update_layout(
            title="Adjusted energy vs. step",
            xaxis_title="Step",
            yaxis_title="Adjusted energy",
        )

        fig_adjusted.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        fig_adjusted.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )
        fig_adjusted.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(120, 120, 120, 0.3)",
            zeroline=False,
        )

        return ComparisonResults(
            frames=frames,
            figures={"energy_vs_steps": fig, "energy_vs_steps_adjusted": fig_adjusted},
        )









    @staticmethod
    def mae_plot_interactive(
        node_dict_NVT,
        node_dict_NPT: t.Optional[dict[str, NodeWithMolecularDynamics]] = None,
        node_dict_stability: t.Optional[dict[str, NodeWithMolecularDynamics]] = None,        
        run_interactive: bool = True,
        ui: str | None = None,
        normalise_to_model: t.Optional[str] = None,
    ):
        import pandas as pd
        import numpy as np
        import tqdm
        from dash import dcc, html
        import dash
        from mlipx.dash_utils import dash_table_interactive, run_app
        import json

        # Define which properties to compute
        NVT_properties = [
            'g_r_oo',
            'g_r_oh',
            'g_r_hh',
            'msd_O',
            'vacf',
            'vdos',
        ]
        NPT_properties = [
            'density',
        ]
        
        # ------------ ref data ------------

        from mlipx.benchmark_download_utils import get_benchmark_data
        ref_data_path = get_benchmark_data("water_MD_data.zip", force=True) / "water_MD_data"

        NVT_properties_dict = {}
        NPT_properties_dict = {}
        # Add reference data for all RDFs
        with open(ref_data_path / "pbe-d3-330k-g-r_oo.json", "r") as f:
            pbe_D3_330K_oo = json.load(f)
        with open(ref_data_path / "pbe-d3-330k-g-r_hh.json", "r") as f:
            pbe_D3_330K_hh = json.load(f)
        with open(ref_data_path / "pbe-d3-330k-g-r_oh.json", "r") as f:
            pbe_D3_330K_oh = json.load(f)
        with open(ref_data_path / "exp-300k-g-r_oo.json", "r") as f:
            exp_300K = json.load(f)
        # Load VACF reference data (just vacf key)
        with open(ref_data_path / "vcaf_300K_NPT_SPE_water_clean.json", "r") as f:
            vacf_ref_data = json.load(f)
        with open(ref_data_path / "vdos_300K_PBE_D3.json", "r") as f:
            vdos_ref_data = json.load(f)
    
        with open(ref_data_path / "water-NPT-PBE-330K-density.json", "r") as f:
            NPT_pbe_330K_density = json.load(f)
            
            
        NPT_properties_dict['density'] = {
            'PBE_330K': {
                'time': NPT_pbe_330K_density['x'],
                'density': NPT_pbe_330K_density['y'],
            }
        }
        
            
        # Insert VACF reference into NVT_properties_dict["vacf"] following RDF pattern
        NVT_properties_dict["vacf"] = {
            "SPC/E_300K": {
                "time": np.array(vacf_ref_data["x"]) / 10,
                "vaf": vacf_ref_data["y"]
            }
        }
        
        # Normalize the reference VDOS
        ref_freq = np.array(vdos_ref_data["x"]) / 10
        ref_vdos = np.array(vdos_ref_data["y"])
        ref_vdos /= np.trapz(ref_vdos, x=ref_freq) 
        ref_vdos = ref_vdos * 1000
        NVT_properties_dict["vdos"] = {
            "PBE_D3_300K": {
                "frequency": ref_freq.tolist(),
                "vdos": ref_vdos.tolist()
            }
        }

        NVT_properties_dict['g_r_oo'] = {
            'PBE_D3_330K': {
                'r': pbe_D3_330K_oo['x'],
                'rdf': pbe_D3_330K_oo['y']
            }
        }

        NVT_properties_dict['g_r_oh'] = {
            'PBE_D3_330K': {
                'r': pbe_D3_330K_oh['x'],
                'rdf': pbe_D3_330K_oh['y']
            }
        }
        NVT_properties_dict['g_r_hh'] = {
            'PBE_D3_330K': {
                'r': pbe_D3_330K_hh['x'],
                'rdf': pbe_D3_330K_hh['y']
            }
        }
        # Merge in exp_300K data for g_r_oo without overwriting existing dict
        NVT_properties_dict['g_r_oo']['EXP_300K'] = {
            'r': exp_300K['x'],
            'rdf': exp_300K['y']
        }
        
        NVT_properties_dict['msd_O'] = {
            "PBE_TS_vdW(SC)_300K": {
                "time": [], # to include
                "msd": [],
                "D": 0.044  # Å^2/ps
            }
        }
        
        groups = {
            "water_rdfs": {
                "NVT_properties": ['g_r_oo', 'g_r_oh', 'g_r_hh'],
                "table_id": "rdf-mae-score-table",
                "details_id": "rdf-table-details",
                "last_clicked_id": "rdf-table-last-clicked",
                "title": "RDFs: O-O, O-H, H-H",
            },
            "water_dynamic_properties": {
                "NVT_properties": ['msd_O', 'vacf', 'vdos'],
                "table_id": "dynamic-score-table",
                "details_id": "dynamic-table-details",
                "last_clicked_id": "dynamic-table-last-clicked",
                "title": "Dynamic Properties: MSD, VACF, VDOS",
            },
        }
        
        
        # ----------- predicted data -----------
        

        # Compute properties for each model
        for model_name, node in tqdm.tqdm(node_dict_NVT.items(), desc="Computing NVT properties for models"):
            traj = node.traj
            #traj = traj[::10]
            traj = traj[1000:]
            
            print(f"Processing model: {model_name} with {len(traj)} frames")
            velocities = [atoms.get_velocities() for atoms in traj]
            print("loaded trajectory for model:", model_name)
            o_indices = [atom.index for atom in traj[0] if atom.symbol == 'O']
            h_indices = [atom.index for atom in traj[0] if atom.symbol == 'H']
            for prop in NVT_properties:
                if prop not in NVT_properties_dict:
                    NVT_properties_dict[prop] = {}
                if prop == 'msd_O':
                    time, msd = MolecularDynamics.compute_msd(traj, timestep=1, atom_symbol='O')
                    NVT_properties_dict[prop][model_name] = {
                        "time": time.tolist(),
                        "msd": msd.tolist(),
                    }
                    print(f"Computed MSD for {model_name} with {len(time)} time points.")
                    #print(properties_dict[prop][model_name])
                elif prop == 'g_r_oo':
                    # O-O RDF
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj[::100],
                        i_indices=o_indices,
                        j_indices=o_indices,
                        r_max=6.0,
                        bins=100,
                        n_jobs=1,
                    )
                    NVT_properties_dict[prop][model_name] = {
                        "r": r,
                        "rdf": rdf,
                    }
                elif prop == 'g_r_oh':
                    # O-H RDF
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj[::100],
                        i_indices=o_indices,
                        j_indices=h_indices,
                        r_max=6.0,
                        bins=100,
                        n_jobs=1,
                    )
                    NVT_properties_dict[prop][model_name] = {
                        "r": r,
                        "rdf": rdf,
                    }
                elif prop == 'g_r_hh':
                    # H-H RDF
                    r, rdf = MolecularDynamics.compute_rdf_optimized_parallel(
                        traj[::100],
                        i_indices=h_indices,
                        j_indices=h_indices,
                        r_max=6.0,
                        bins=100,
                        n_jobs=1,
                    )
                    NVT_properties_dict[prop][model_name] = {
                        "r": r,
                        "rdf": rdf,
                    }

                elif prop == 'vacf':
                    # Velocity autocorrelation function
                    vaf = MolecularDynamics.compute_vacf(traj, velocities, timestep=1, atoms_filter=(('O',),))
                    #print(vaf)
                    NVT_properties_dict[prop][model_name] = {
                        "time": vaf[0],
                        "vaf": vaf[1],
                    }
                elif prop == 'vdos':
                    # Velocity density of states
                    freq, vdos_values = MolecularDynamics.compute_vacf(traj, velocities, timestep=1, fft=True, atoms_filter=(('O',),))
                    #vdos_values /= np.trapz(vdos_values, x=freq)  # Normalize VDOS
                    freq *= 1000  # Convert to desired units (ps^-1)
                    NVT_properties_dict[prop][model_name] = {
                        "frequency": freq.tolist(),
                        "vdos": vdos_values.tolist(),
                    }

        # Add msd_dict for later use
        msd_dict = NVT_properties_dict["msd_O"]
        
        # print(properties_dict.keys())
        # print(properties_dict['vdos']['mace_mp_0a_D3'].keys())

        # Compute VACF and VDOS MAE tables (dummy placeholders since no ref currently)


        # --- Helper to compute MAE DataFrame for a property ---
        def compute_mae_table(prop, NVT_properties_dict, model_names, normalise_to_model=None):
            import numpy as np
            import pandas as pd

            # Dynamically determine reference keys and valid models
            reference_keys = []
            valid_model_names = []

            for k, data in NVT_properties_dict[prop].items():
                if k in model_names:
                    valid_model_names.append(k)
                elif isinstance(data, dict) and any(key in data for key in ["rdf", "vaf", "vdos"]):
                    reference_keys.append(k)

            mae_data = []

            for model_name in valid_model_names:
                row = {"Model": model_name}
                model_data = NVT_properties_dict[prop].get(model_name)
                for ref_key in reference_keys:
                    ref_data = NVT_properties_dict[prop].get(ref_key)
                    # Guard against missing or invalid data
                    if (
                        model_data is None or ref_data is None
                        or not isinstance(model_data, dict)
                        or not isinstance(ref_data, dict)
                    ):
                        continue

                    # RDF MAE computation
                    if "rdf" in ref_data and "rdf" in model_data:
                        r_ref = np.array(ref_data["r"])
                        rdf_ref = np.array(ref_data["rdf"])
                        r_model = np.array(model_data["r"])
                        rdf_model = np.array(model_data["rdf"])
                        if (
                            len(r_ref) == 0 or len(rdf_ref) == 0
                            or len(r_model) == 0 or len(rdf_model) == 0
                        ):
                            continue
                        rdf_model_interp = np.interp(r_ref, r_model, rdf_model)
                        mae = np.mean(np.abs(rdf_model_interp - rdf_ref))
                        row[f"{prop} ({ref_key})"] = round(mae, 4)

                    # VACF MAE computation
                    elif "vaf" in ref_data and "vaf" in model_data:
                        t_ref = np.array(ref_data["time"]) / 100
                        vaf_ref = np.array(ref_data["vaf"])
                        t_model = np.array(model_data["time"]) / 100
                        vaf_model = np.array(model_data["vaf"])
                        if (
                            len(t_ref) == 0 or len(vaf_ref) == 0
                            or len(t_model) == 0 or len(vaf_model) == 0
                        ):
                            continue
                        mask = (t_model >= t_ref[0]) & (t_model <= t_ref[-1])
                        t_model_masked = t_model[mask]
                        vaf_model_masked = vaf_model[mask]
                        if len(t_model_masked) == 0 or len(vaf_model_masked) == 0:
                            continue
                        vaf_ref_interp = np.interp(t_model_masked, t_ref, vaf_ref)
                        mae = np.mean(np.abs(vaf_model_masked - vaf_ref_interp))
                        row[f"{prop} ({ref_key})"] = round(mae, 4)

                    # VDOS MAE computation
                    elif "vdos" in ref_data and "vdos" in model_data:
                        freq_ref = np.array(ref_data["frequency"])
                        vdos_ref = np.array(ref_data["vdos"])
                        freq_model = np.array(model_data["frequency"])
                        vdos_model = np.array(model_data["vdos"])
                        if (
                            len(freq_ref) == 0 or len(vdos_ref) == 0
                            or len(freq_model) == 0 or len(vdos_model) == 0
                        ):
                            continue
                        mask = (freq_model >= freq_ref[0]) & (freq_model <= freq_ref[-1])
                        freq_model_masked = freq_model[mask]
                        vdos_model_masked = vdos_model[mask]
                        if len(freq_model_masked) == 0 or len(vdos_model_masked) == 0:
                            continue
                        vdos_ref_interp = np.interp(freq_model_masked, freq_ref, vdos_ref)
                        mae = np.mean(np.abs(vdos_model_masked - vdos_ref_interp))
                        row[f"{prop} ({ref_key})"] = round(mae, 4)

                mae_data.append(row)

            # Columns are Model plus all reference-property pairs (now using new header format)
            columns = ["Model"] + [f"{prop} ({ref_key})" for ref_key in reference_keys]
            return pd.DataFrame(mae_data, columns=columns).round(3)

        model_names = list(node_dict_NVT.keys())

        # --- Helper to build group MAE tables ---
        def build_group_mae_table(group, NVT_properties_dict, model_names, normalise_to_model=None):
            import pandas as pd
            dfs = []
            scores = {}
            for prop in group["NVT_properties"]:
                df = compute_mae_table(prop, NVT_properties_dict, model_names)
                if df.empty or len(df.columns) <= 1:
                    print(f"No data for property '{prop}' in group '{group['title']}'")
                    continue
                print(prop)
                print(df)
                mae_cols = [col for col in df.columns if col != "Model"]
                ref_col = mae_cols[0]  # Assume first column is the reference

                if normalise_to_model is not None:
                    match = df[df["Model"] == normalise_to_model]
                    ref_mae = match[ref_col].values[0]
                    scores[prop] = df[ref_col] / ref_mae

                else:
                    scores[prop] = df[ref_col]
                    
                dfs.append(df.set_index("Model"))

            merged = pd.concat(dfs, axis=1)
            merged["Score ↓"] = np.average(list(scores.values()))
            merged["Rank"] = merged["Score ↓"].rank(method="min").astype(int)
            merged.reset_index(inplace=True)
            return merged

        # Build group MAE tables
        group_mae_tables = {}
        for group_name, group in groups.items():
            group_mae_tables[group_name] = build_group_mae_table(group, NVT_properties_dict, model_names, normalise_to_model)

        if ui is None and run_interactive:
            return group_mae_tables, NVT_properties_dict, groups

        if not run_interactive:
            return group_mae_tables, NVT_properties_dict, groups
        

        return run_app(app, ui=ui)



    @staticmethod
    def register_callbacks(app, groups, group_mae_tables, NVT_properties_dict):
        from dash import Input, Output, State, no_update, html, dcc
        import plotly.graph_objs as go
        import numpy as np
        import re

        # Plot configuration dictionary
        plot_config = {
            "g_r_oo": {
                "xlim": (2.0, 5.5),
                "ylim": (None, 4),
                "xaxis_title": "r (Å)",
                "yaxis_title": "g(r)",
                "title": "O-O RDF",
            },
            "g_r_oh": {
                "xlim": (0, 4.5),
                "ylim": (None, None),
                "xaxis_title": "r (Å)",
                "yaxis_title": "g(r)",
                "title": "O-H RDF",
            },
            "g_r_hh": {
                "xlim": (0, 5),
                "ylim": (None, None),
                "xaxis_title": "r (Å)",
                "yaxis_title": "g(r)",
                "title": "H-H RDF",
            },
            "vacf": {
                "xlim": (0, 1),
                "ylim": (None, None),
                "xaxis_title": "Time (ps)",
                "yaxis_title": "VACF",
                "title": "Velocity Auto-correlation Function (Oxygen)",
            },
            "vdos": {
                "xlim": (0, 100),
                "ylim": (None, None),
                "xaxis_title": "Frequency (ps⁻¹)",
                "yaxis_title": "VDOS",
                "title": "Vibrational Density of States",
            },
        }

        for group_name, group in groups.items():
            table_id = group["table_id"]
            details_id = group["details_id"]
            last_clicked_id = group["last_clicked_id"]
            props = group["NVT_properties"]
            df = group_mae_tables[group_name]

            @app.callback(
                Output(details_id, "children"),
                Output(last_clicked_id, "data"),
                Input(table_id, "active_cell"),
                State(table_id, "data"),
                State(last_clicked_id, "data"),
                prevent_initial_call=True,
            )
            def update_property_plot(active_cell, table_data, last_clicked, table_id=table_id, props=props, details_id=details_id):
                import plotly.graph_objs as go
                from dash import html, dcc
                import numpy as np
                import re

                if active_cell is None:
                    raise dash.exceptions.PreventUpdate

                row_idx = active_cell.get("row")
                col_id = active_cell.get("column_id")
                if row_idx is None or col_id is None:
                    raise dash.exceptions.PreventUpdate

                model_name = table_data[row_idx]["Model"].strip()
                ref_name = col_id.strip()
                # New parsing: parse "prop (ref_key)" header
                match = re.match(r"(.*?) \((.*?)\)", ref_name)
                if match:
                    prop, ref_name = match.group(1), match.group(2)

                # Collapse plot if the "Model" column is clicked, consistent with lattice constant plots
                if col_id == "Model":
                    return html.Div(), None

                # if last_clicked == [model_name, ref_name]:
                #     return html.Div(), None

                tabs = []

                for _prop in props:
                    if _prop not in NVT_properties_dict:
                        continue
                    prop_data = NVT_properties_dict[_prop]

                    # Only show plot if _prop matches prop (from header)
                    if match and _prop != prop:
                        continue

                    if ref_name not in prop_data or model_name not in prop_data:
                        continue

                    fig = go.Figure()

                    # Plot reference
                    ref = prop_data[ref_name]
                    if _prop == "vacf":
                        x, y = np.array(ref["time"]) / 100, ref["vaf"]
                    elif _prop == "vdos":
                        x, y = ref["frequency"], ref["vdos"]
                    elif _prop == "msd_O":
                        x, y = ref["time"], ref["msd"]
                    else:  # RDF
                        x, y = ref["r"], ref["rdf"]

                    fig.add_trace(go.Scatter(x=x, y=y, name=f"{ref_name} (Ref)", line=dict(dash="dash", color="black")))

                    # Plot model
                    model = prop_data[model_name]
                    if _prop == "vacf":
                        x, y = np.array(model["time"]) / 100, model["vaf"]
                    elif _prop == "vdos":
                        x, y = model["frequency"], model["vdos"]
                    elif _prop == "msd_O":
                        x, y = model["time"], model["msd"]
                    else:  # RDF
                        x, y = model["r"], model["rdf"]

                    fig.add_trace(go.Scatter(x=x, y=y, name=model_name, line=dict(width=3)))

                    # Lookup plot configuration for this property
                    config = plot_config.get(_prop, {})
                    fig.update_layout(
                        title=config.get("title", f"{_prop} — {model_name} vs {ref_name}"),
                        xaxis_title=config.get("xaxis_title", "x"),
                        yaxis_title=config.get("yaxis_title", "y"),
                        xaxis_range=config.get("xlim"),
                        yaxis_range=config.get("ylim"),
                        margin=dict(l=20, r=20, t=40, b=20),
                    )

                return dcc.Graph(figure=fig), [model_name, ref_name]
            

    
    @staticmethod
    def benchmark_precompute(
        node_dict_NVT,
        node_dict_NPT=None,
        node_dict_stability=None,
        cache_dir="app_cache/further_applications_benchmark/molecular_dynamics_cache",
        ui=None,
        run_interactive=False,
        normalise_to_model=None,
    ):
        import os
        os.makedirs(cache_dir, exist_ok=True)
        group_mae_tables, NVT_properties_dict, groups = MolecularDynamics.mae_plot_interactive(
            node_dict_NVT=node_dict_NVT,
            node_dict_NPT=node_dict_NPT,
            node_dict_stability=node_dict_stability,
            run_interactive=run_interactive,
            ui=ui,
            normalise_to_model=normalise_to_model,
        )
        # Save each group mae table separately
        for group_name, group_df in group_mae_tables.items():
            group_df.to_pickle(f"{cache_dir}/mae_df_{group_name}.pkl")
        with open(f"{cache_dir}/rdf_data.pkl", "wb") as f:
            pickle.dump(NVT_properties_dict, f)
        msd_dict = NVT_properties_dict["msd_O"]
        with open(f"{cache_dir}/msd_data.pkl", "wb") as f:
            pickle.dump(msd_dict, f)
        vdos_dict = NVT_properties_dict["vdos"]
        with open(f"{cache_dir}/vdos_data.pkl", "wb") as f:
            pickle.dump(vdos_dict, f)
        # Save groups structure
        with open(f"{cache_dir}/groups.pkl", "wb") as f:
            pickle.dump(groups, f)
        return



    @staticmethod
    def launch_dashboard(
        cache_dir="app_cache/further_applications_benchmark/molecular_dynamics_cache",
        ui=None
    ):
        import pickle
        import pandas as pd
        import dash
        # Load groups structure
        with open(f"{cache_dir}/groups.pkl", "rb") as f:
            groups = pickle.load(f)
        # Load group mae tables
        group_mae_tables = {}
        for group_name in groups:
            group_mae_tables[group_name] = pd.read_pickle(f"{cache_dir}/mae_df_{group_name}.pkl")
        with open(f"{cache_dir}/rdf_data.pkl", "rb") as f:
            NVT_properties_dict = pickle.load(f)
        with open(f"{cache_dir}/msd_data.pkl", "rb") as f:
            msd_dict = pickle.load(f)
        NVT_properties_dict["msd_O"] = msd_dict
        #properties_dict["vdos"] = vdos_dict

        app = dash.Dash(__name__)
        app.layout = MolecularDynamics.build_layout(groups=groups, group_mae_tables=group_mae_tables)
        MolecularDynamics.register_callbacks(
            app,
            groups=groups,
            group_mae_tables=group_mae_tables,
            NVT_properties_dict=NVT_properties_dict
        )
        return run_app(app, ui=ui)
    
    
    
    
    
    @staticmethod
    def build_layout(groups, group_mae_tables):
        from dash import html, dcc
        from mlipx.dash_utils import dash_table_interactive
        layout_children = [
            html.H1("Water MD Benchmark"),
            html.P("Simulation details: NVT 330K, D3 dispersion, 64 water molecules, 10k steps equilibration + 40k steps production run, 1 fs timestep."),
        ]
        for group_name, group in groups.items():
            group_df = group_mae_tables[group_name]
            layout_children.append(html.H3(group["title"]))
            layout_children.append(
                dash_table_interactive(
                    df=group_df,
                    id=group["table_id"],
                    title=None,
                    extra_components=[
                        html.Div(id=group["details_id"]),
                        dcc.Store(id=group["last_clicked_id"], data=None),
                    ],
                )
            )
            
        return html.Div(layout_children, style={"backgroundColor": "white"})
        
        
        
        
        
        
    @staticmethod
    def compute_msd(traj, timestep=1, atom_symbol="O"):
        """
        Compute the Mean Squared Displacement (MSD) for atoms of the given symbol.

        Parameters
        ----------
        atom_symbol : str
            Element symbol (e.g., "O" or "H") for which to compute MSD.

        Returns
        -------
        time_steps : np.ndarray
            Time steps in picoseconds.
        msd_values : np.ndarray
            MSD values at each time step.
        """
        from tqdm import tqdm
        atom_indices = [atom.index for atom in traj[0] if atom.symbol == atom_symbol]
        num_frames = len(traj)
        initial_positions = traj[0].get_positions()[atom_indices]
        msd_values = np.zeros(num_frames)

        for i, atoms in enumerate(tqdm(traj, desc=f"Computing MSD for {atom_symbol}")):
            current_positions = atoms.get_positions()[atom_indices]
            displacements = current_positions - initial_positions
            squared_displacements = np.sum(displacements**2, axis=1)
            msd_values[i] = np.mean(squared_displacements)

        timestep_fs = timestep
        time_steps = np.arange(num_frames) * timestep_fs / 100
        return time_steps, msd_values
    
    
    
    

    


    # ------------ computing properties ------------

    def compute_vacf(
        traj,
        velocities,
        atoms_filter=((None,),),
        start_index=0,
        timestep=1,
        fft=False
    ):
        n_steps = len(velocities)
        n_atoms = len(velocities[0])
        filtered_atoms = []
        atom_symbols = traj[0].get_chemical_symbols()
        symbols = set(atom_symbols)
        
        for atoms in atoms_filter:
            if any(atom is None for atom in atoms):
                # If atoms_filter not specified use all atoms.
                filtered_atoms.append(range(n_atoms))
            elif all(isinstance(a, str) for a in atoms):
                # If all symbols, get the matching indices.
                atoms = set(atoms)
                if atoms.difference(symbols):
                    raise ValueError(
                        f"{atoms.difference(symbols)} not allowed in VAF"
                        f", allowed symbols are {symbols}"
                    )
                filtered_atoms.append(
                    [i for i in range(len(atom_symbols)) if atom_symbols[i] in list(atoms)]
                )
            elif all(isinstance(a, int) for a in atoms):
                filtered_atoms.append(atoms)
            else:
                raise ValueError(
                    "Cannot mix element symbols and indices in vaf atoms_filter"
                )
        
        used_atoms = {atom for atoms in filtered_atoms for atom in atoms}
        used_atoms = {j: i for i, j in enumerate(used_atoms)}
        
        # Corrected VACF calculation
        vafs = []
        for j in used_atoms:
            atom_vacf = np.zeros(n_steps)
            velocities = np.array(velocities)
            for i in range(3):  # x, y, z components
                v = velocities[:, j, i]
                # Proper autocorrelation calculation
                autocorr = np.correlate(v, v, mode='full')
                # Take the second half (positive lags) and normalize
                autocorr = autocorr[len(autocorr)//2:]
                atom_vacf += autocorr
            vafs.append(atom_vacf)
        
        vafs = np.array(vafs)
        
        # Proper normalization: divide by number of overlapping points
        normalization = np.arange(n_steps, 0, -1)
        vafs = vafs / normalization[np.newaxis, :]
        
        lags = np.arange(n_steps) * timestep
        
        if fft:
            # Calculate power spectral density for VDOS
            # Apply window function to reduce spectral leakage
            window = np.hanning(n_steps)
            vafs_windowed = vafs * window[np.newaxis, :]
            
            # Zero-pad for better frequency resolution (optional)
            n_fft = 2 * n_steps
            vafs_fft = np.fft.fft(vafs_windowed, n=n_fft, axis=1)
            
            # Power spectral density (VDOS)
            vafs = np.abs(vafs_fft)**2
            
            # Only keep strictly positive frequencies
            freqs = np.fft.fftfreq(n_fft, timestep)
            mask = freqs > 0  # only keep strictly positive frequencies
            freqs = freqs[mask]
            vafs = vafs[:, mask]
            lags = freqs
        
        # Average over atom groups
        vafs_grouped = [
            np.average([vafs[used_atoms[i]] for i in atoms], axis=0)
            for atoms in filtered_atoms
        ]
        
        print(f"Computed VACF for {len(filtered_atoms)} atom groups with {len(lags)} lags.")
        #print(f"VACF shape: {vafs_grouped[0].shape} for {len(vafs_grouped)} groups")
        
        # Calculate total VACF
        total_vacf = np.mean(np.stack(vafs_grouped), axis=0)
        if fft:
            try:
                if hasattr(np, "trapezoid"):
                    auc = np.trapezoid(total_vacf, lags)
                else:
                    auc = np.trapz(total_vacf, lags)
                total_vacf = total_vacf / auc  # Normalize to unit area
            except Exception as e:
                print(f"Normalization failed: {e}")
            # Gaussian smoothing for VDOS
            sigma = 5  # adjust smoothing width as needed
            total_vacf = gaussian_filter1d(total_vacf, sigma)
        else:
            total_vacf = total_vacf / np.max(total_vacf)  # Normalize to max value
        
        return lags, total_vacf
    
    

    def _compute_partial_rdf(atoms, i_indices, j_indices, r_max):
        i_list, j_list, dists = neighbor_list('ijd', atoms, r_max)

        mask = np.isin(i_list, i_indices) & np.isin(j_list, j_indices)
        valid_distances = dists[mask]
        return valid_distances, len(valid_distances), atoms.get_volume()


    def compute_rdf_optimized_parallel(atoms_list, i_indices, j_indices, r_max=6.0, bins=100, n_jobs=-1):
        i_indices = np.array(i_indices)
        j_indices = np.array(j_indices)
        
        atom_a = atoms_list[0][i_indices][0]
        atom_b = atoms_list[0][j_indices][0]
        pair = f"{atom_a.symbol}-{atom_b.symbol}"


        from tqdm import tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(MolecularDynamics._compute_partial_rdf)(atoms, i_indices, j_indices, r_max)
            for atoms in tqdm(atoms_list, desc=f"Computing {pair} RDF")
        )

        all_distances = []
        total_pairs = 0
        total_volume = 0.0

        for distances, pair_count, volume in results:
            all_distances.append(distances)
            total_pairs += pair_count
            total_volume += volume

        all_distances = np.concatenate(all_distances)
        avg_volume = total_volume / len(atoms_list)
        rho = len(j_indices) / avg_volume  # Average number density

        hist, bin_edges = np.histogram(all_distances, bins=bins, range=(0, r_max))
        r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]

        shell_volumes = 4 * np.pi * r**2 * bin_width
        ideal_gas_distribution = shell_volumes * rho

        rdf = hist / (ideal_gas_distribution * total_pairs)
        
        rdf = rdf / rdf[np.argmax(r)] # liquid rdf should be 1 at large r

        return r, rdf
    
    