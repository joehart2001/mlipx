from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from ase.io import read
from scipy import stats
from scipy.interpolate import UnivariateSpline
from tqdm.auto import tqdm
import json


from mlipx.benchmark_download_utils import get_benchmark_data




def get_homonuclear_diatomic_properties(model, node, pbe_ref = False):
    

    df = pd.DataFrame(
        columns=[
            "name",
            "method",
            "R",
            "E",
            "F",
            "S^2",
            "force-flip-times",
            "force-total-variation",
            "force-jump",
            "energy-diff-flip-times",
            "energy-grad-norm-max",
            "energy-jump",
            "energy-total-variation",
            "tortuosity",
            "conservation-deviation",
            "spearman-descending-force",
            "spearman-ascending-force",
            "spearman-repulsion-energy",
            "spearman-attraction-energy",
            "num-energy-minima",
            "num-energy-inflections",
            #"pbe-energy-mae",
            #"pbe-force-mae",
        ]
    )

    for symbol in tqdm(chemical_symbols[1:]):
        da = symbol + symbol
        
        

        traj_fpath = Path(node.trajectory_dir_path) / f"{str(symbol)}2.extxyz"

        if traj_fpath.exists():
            traj = read(traj_fpath, index=":")
        else:
            continue

        Rs, Es, Fs, S2s = [], [], [], []
        for atoms in traj:
            vec = atoms.positions[1] - atoms.positions[0]
            r = np.linalg.norm(vec)
            e = atoms.get_potential_energy()
            f = np.inner(vec / r, atoms.get_forces()[1])
            # s2 = np.mean(np.power(atoms.get_magnetic_moments(), 2))

            Rs.append(r)
            Es.append(e)
            Fs.append(f)
            # S2s.append(s2)

        rs = np.array(Rs)
        es = np.array(Es)
        fs = np.array(Fs)

        # sort interatomic distances and align to zero at far field
        indices = np.argsort(rs)[::-1]
        rs = rs[indices]
        es = es[indices]
        eshift = es[0]
        es -= eshift
        fs = fs[indices]

        iminf = np.argmin(fs)
        imine = np.argmin(es)

        de_dr = np.gradient(es, rs)
        d2e_dr2 = np.gradient(de_dr, rs)
        
        # no. of minima
        # finite diff to approximate deriv, convert to signs of slope, diff again to see how many times it crosses zero
        num_minima = np.sum((np.diff(np.sign(np.diff(es))) > 0))
        # no. of inflections
        # sign changes of 2nd derivative
        num_inflections = np.sum(np.diff(np.sign(d2e_dr2)) != 0)

        # avoid numerical sensitity close to zero
        rounded_fs = np.copy(fs)
        rounded_fs[np.abs(rounded_fs) < 1e-2] = 0  # 10meV/A
        fs_sign = np.sign(rounded_fs)
        mask = fs_sign != 0
        rounded_fs = rounded_fs[mask]
        fs_sign = fs_sign[mask]
        f_flip = np.diff(fs_sign) != 0

        fdiff = np.diff(fs)
        fdiff_sign = np.sign(fdiff)
        mask = fdiff_sign != 0
        fdiff = fdiff[mask]
        fdiff_sign = fdiff_sign[mask]
        fdiff_flip = np.diff(fdiff_sign) != 0
        fjump = (
            np.abs(fdiff[:-1][fdiff_flip]).sum() + np.abs(fdiff[1:][fdiff_flip]).sum()
        )

        ediff = np.diff(es)
        ediff[np.abs(ediff) < 1e-3] = 0  # 1meV
        ediff_sign = np.sign(ediff)
        mask = ediff_sign != 0
        ediff = ediff[mask]
        ediff_sign = ediff_sign[mask]
        ediff_flip = np.diff(ediff_sign) != 0
        ejump = (
            np.abs(ediff[:-1][ediff_flip]).sum() + np.abs(ediff[1:][ediff_flip]).sum()
        )
        
        if pbe_ref:
            try:
                pbe_traj = read(f"./vasp/{da}/PBE.extxyz", index=":")

                pbe_rs, pbe_es, pbe_fs = [], [], []

                for atoms in pbe_traj:
                    vec = atoms.positions[1] - atoms.positions[0]
                    r = np.linalg.norm(vec)
                    pbe_rs.append(r)
                    pbe_es.append(atoms.get_potential_energy())
                    pbe_fs.append(np.inner(vec / r, atoms.get_forces()[1]))

                pbe_rs = np.array(pbe_rs)
                pbe_es = np.array(pbe_es)
                pbe_fs = np.array(pbe_fs)

                indices = np.argsort(pbe_rs)
                pbe_rs = pbe_rs[indices]
                pbe_es = pbe_es[indices]
                pbe_fs = pbe_fs[indices]

                pbe_es -= pbe_es[-1]

                xs = np.linspace(pbe_rs.min(), pbe_rs.max(), int(1e3))

                cs = UnivariateSpline(pbe_rs, pbe_es, s=0)
                pbe_energy_mae = np.mean(np.abs(es - cs(rs)))

                cs = UnivariateSpline(pbe_rs, pbe_fs, s=0)
                pbe_force_mae = np.mean(np.abs(fs - cs(rs)))
            except Exception as e:
                print(e)
                pbe_energy_mae = None
                pbe_force_mae = None

        conservation_deviation = np.mean(np.abs(fs + de_dr))

        etv = np.sum(np.abs(np.diff(es)))

        data = {
            "name": da,
            "method": model,
            "R": rs,
            "E": es + eshift,
            "F": fs,
            "S^2": S2s,
            "force-flip-times": np.sum(f_flip),
            "force-total-variation": np.sum(np.abs(np.diff(fs))),
            "force-jump": fjump,
            "energy-diff-flip-times": np.sum(ediff_flip),
            "energy-grad-norm-max": np.max(np.abs(de_dr)),
            "energy-jump": ejump,
            # "energy-grad-norm-mean": np.mean(de_dr_abs),
            "energy-total-variation": etv,
            "tortuosity": etv / (abs(es[0] - es.min()) + (es[-1] - es.min())),
            "conservation-deviation": conservation_deviation,
            "spearman-descending-force": stats.spearmanr(
                rs[iminf:], fs[iminf:]
            ).statistic,
            "spearman-ascending-force": stats.spearmanr(
                rs[:iminf], fs[:iminf]
            ).statistic,
            "spearman-repulsion-energy": stats.spearmanr(
                rs[imine:], es[imine:]
            ).statistic,
            "spearman-attraction-energy": stats.spearmanr(
                rs[:imine], es[:imine]
            ).statistic,
            "num-energy-minima": num_minima,
            "num-energy-inflections": num_inflections,
            #"pbe-energy-mae": pbe_energy_mae,
            #"pbe-force-mae": pbe_force_mae,
        }

        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        
        

    json_fpath = Path(f"HD-stats/{model}/homonuclear-diatomics.json")

    # if json_fpath.exists():
    #     df0 = pd.read_json(json_fpath)
    #     df = pd.concat([df0, df], ignore_index=True)
    #     df.drop_duplicates(inplace=True, subset=["name", "method"], keep="last")

    # mkdir 
    json_fpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(json_fpath, orient="records")

    
    return df




def get_homonuclear_diatomic_stats(models):




    DATA_DIR = Path("HD-stats")

    dfs = [
        pd.read_json(DATA_DIR / f"{model}/homonuclear-diatomics.json")
        for model in models
    ]
    df = pd.concat(dfs, ignore_index=True)

    table = pd.DataFrame()

    for model in models:
        rows = df[df["method"] == model]
        #metadata = MODELS.get(model, {})

        new_row = {
            "Model": model,
            "Conservation deviation [eV/Å]": rows["conservation-deviation"].mean(),
            "Tortuosity": rows["tortuosity"].mean(),
            "Energy jump [eV]": rows["energy-jump"].mean(),
            "Force flips": rows["force-flip-times"].mean(),
            "No. energy minima": rows["num-energy-minima"].mean(),
            "No. energy inflections": rows["num-energy-inflections"].mean(),
            "Spearman's coeff. (E: repulsion)": rows[
                "spearman-repulsion-energy"
            ].mean(),
            "Spearman's coeff. (F: descending)": rows[
                "spearman-descending-force"
            ].mean(),
            "Spearman's coeff. (E: attraction)": rows[
                "spearman-attraction-energy"
            ].mean(),
            "Spearman's coeff. (F: ascending)": rows[
                "spearman-ascending-force"
            ].mean(),
            #"PBE energy MAE [eV]": rows["pbe-energy-mae"].mean(),
            #"PBE force MAE [eV/Å]": rows["pbe-force-mae"].mean(),
        }

        table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

    table.set_index("Model", inplace=True)

    table.sort_values("Conservation deviation [eV/Å]", ascending=True, inplace=True)
    table["Rank"] = np.argsort(table["Conservation deviation [eV/Å]"].to_numpy())

    table.sort_values(
        "Spearman's coeff. (E: repulsion)", ascending=True, inplace=True
    )
    table["Rank"] += np.argsort(table["Spearman's coeff. (E: repulsion)"].to_numpy())

    table.sort_values(
        "Spearman's coeff. (F: descending)", ascending=True, inplace=True
    )
    table["Rank"] += np.argsort(table["Spearman's coeff. (F: descending)"].to_numpy())

    # NOTE: it's not fair to models trained on different level of theory
    # table.sort_values("PBE energy MAE [eV]", ascending=True, inplace=True)
    # table["Rank"] += np.argsort(table["PBE energy MAE [eV]"].to_numpy())

    # table.sort_values("PBE force MAE [eV/Å]", ascending=True, inplace=True)
    # table["Rank"] += np.argsort(table["PBE force MAE [eV/Å]"].to_numpy())

    table.sort_values("Tortuosity", ascending=True, inplace=True)
    table["Rank"] += np.argsort(table["Tortuosity"].to_numpy())

    table.sort_values("Energy jump [eV]", ascending=True, inplace=True)
    table["Rank"] += np.argsort(table["Energy jump [eV]"].to_numpy())

    table.sort_values("Force flips", ascending=True, inplace=True)
    table["Rank"] += np.argsort(np.abs(table["Force flips"].to_numpy() - 1))
    
    table.sort_values("No. energy minima", ascending=True, inplace=True)
    table["Rank"] += np.argsort(table["No. energy minima"].to_numpy())
    
    table.sort_values("No. energy inflections", ascending=True, inplace=True)
    table["Rank"] += np.argsort(table["No. energy inflections"].to_numpy())

    table["Rank"] += 1

    table.sort_values(["Rank", "Conservation deviation [eV/Å]"], ascending=True, inplace=True)

    table["Rank aggr."] = table["Rank"]
    table["Rank"] = table["Rank aggr."].rank(method='min').astype(int)

    table = table.reindex(
        columns=[
            "Rank",
            "Rank aggr.",
            "Conservation deviation [eV/Å]",
            #"PBE energy MAE [eV]",
            #"PBE force MAE [eV/Å]",
            "Energy jump [eV]",
            "Force flips",
            "Tortuosity",
            "No. energy minima",
            "No. energy inflections",
            "Spearman's coeff. (E: repulsion)",
            "Spearman's coeff. (F: descending)",
            "Spearman's coeff. (E: attraction)",
            "Spearman's coeff. (F: ascending)",
        ]
    )
    
    return table

    # subset=[
    #     "Conservation deviation [eV/Å]",
    #     "Spearman's coeff. (E: repulsion)",
    #     "Spearman's coeff. (F: descending)",
    #     "Tortuosity",
    #     "Energy jump [eV]",
    #     "Force flips",
    #     "Spearman's coeff. (E: attraction)",
    #     "Spearman's coeff. (F: ascending)",
    #     "PBE energy MAE [eV]",
    #     "PBE force MAE [eV/Å]",
    # ]



# def render():
#     st.dataframe(
#         s,
#         use_container_width=True,
#     )
#     with st.expander("Explanation", icon=":material/info:"):
#         st.caption(
#             r"""
#             - **Conservation deviation**: The average deviation of force from negative energy gradient along the diatomic curves. 
            
#             $$
#             \text{Conservation deviation} = \left\langle\left| \mathbf{F}(\mathbf{r})\cdot\frac{\mathbf{r}}{\|\mathbf{r}\|} +  \nabla_rE\right|\right\rangle_{r = \|\mathbf{r}\|}
#             $$

#             - **Spearman's coeff. (E: repulsion)**: Spearman's correlation coefficient of energy prediction within equilibrium distance $r \in (r_{min}, r_o = \argmin_{r} E(r))$.
#             - **Spearman's coeff. (F: descending)**: Spearman's correlation coefficient of force prediction before maximum attraction $r \in (r_{min}, r_a = \argmin_{r} F(r))$.
#             - **Tortuosity**: The ratio between total variation in energy and sum of absolute energy differences between $r_{min}$, $r_o$, and $r_{max}$.
#             - **Energy jump**: The sum of energy discontinuity between sampled points. 

#             $$
#             \text{Energy jump} = \sum_{r_i \in [r_\text{min}, r_\text{max}]} \left| \text{sign}{\left[ E(r_{i+1}) - E(r_i)\right]} - \text{sign}{\left[E(r_i) - E(r_{i-1})\right]}\right| \times \\ \left( \left|E(r_{i+1}) - E(r_i)\right| + \left|E(r_i) - E(r_{i-1})\right|\right)
#             $$
#             - **Force flips**: The number of force direction changes.
#             """
#         )
#         st.info('PBE energies and forces are provided __only__ for reference. Due to the known convergence issue of plane-wave DFT with diatomic molecules and different dataset the models might be trained on, comparing models with PBE is not rigorous and thus these metrics are excluded from rank aggregation.', icon=":material/warning:")
        
        