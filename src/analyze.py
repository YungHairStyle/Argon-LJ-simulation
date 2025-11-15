#!/usr/bin/env python3
"""
Analyze MD outputs produced by main_merged.py / md_core_merged.py.

Features
- Load thermo CSV (time, E_pot, E_kin, T) and make clean time-series plots
- Parse minimal .gro snapshot to get positions and box
- Compute radial distribution function g(r)
- Compute static structure factor S(k) and shell-average S(|k|)
- Block-average utilities and table
- Save plots as both .png and .svg (scalable)

Usage
-----
python analyze.py \
  --mode bulk \
  --data_dir "/path/to/data/" \
  --out_dir  "/output/path" \
  --rc 2.5 \
  --nbins 120 \
  --dr 0.02 \
  --maxk 6

This will look for:
- thermo_bulk.csv (or thermo_slab.csv for --mode slab)
- argon_bulk.gro   (or argon_slab.gro for --mode slab)

Notes
- For slab, distances use 3D coordinates; if you want strictly in-plane g(r), set --inplane and it will use only x,y distances.
- All plots use plain matplotlib (no styles, default colors) and one chart per figure.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
#------------------------------------------------------------------
#------------------------------------------------------------------
# Global settings
slab = True  # Global flag for slab vs bulk mode
#------------------------------------------------------------------
#------------------------------------------------------------------


# ---- Core helpers (mirrors md_core_merged) ----

def _as_box(L):
    try:
        Lx, Ly, Lz = L
        return float(Lx), float(Ly), float(Lz)
    except Exception:
        return float(L), float(L), float(L)


def minimum_image_disp(drij: np.ndarray, L, mode: str = "bulk") -> np.ndarray:
    Lx, Ly, Lz = _as_box(L)
    d = np.array(drij, dtype=float, copy=True)
    d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    if mode == "bulk":
        d[..., 2] -= Lz * np.round(d[..., 2] / Lz)
    return d


def displacement_table(coordinates: np.ndarray, L, mode: str = "bulk") -> np.ndarray:
    r = np.asarray(coordinates, dtype=float)
    table = r[:, np.newaxis, :] - r[np.newaxis, :, :]
    return minimum_image_disp(table, L, mode)


def distance_table(disp: np.ndarray) -> np.ndarray:
    return np.linalg.norm(disp, axis=-1)


def pair_correlation(dists_1d: np.ndarray, natom: int, nbins: int, dr: float, L) -> Tuple[np.ndarray, np.ndarray]:
    Lx, Ly, Lz = _as_box(L)
    Omega = Lx * Ly * Lz
    hist, edges = np.histogram(dists_1d, bins=nbins, range=(0.0, nbins * dr))
    r = 0.5 * (edges[:-1] + edges[1:])
    dOmega = (4.0 * np.pi / 3.0) * ((r + 0.5 * dr) ** 3 - (r - 0.5 * dr) ** 3)
    ideal = ((natom - 1) / 2.0) * (natom / Omega) * dOmega
    with np.errstate(divide="ignore", invalid="ignore"):
        g = hist / ideal
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g, r


def legal_kvecs(maxn: int, L) -> np.ndarray:
    Lx, Ly, Lz = _as_box(L)
    grid = np.arange(-maxn, maxn + 1)
    k = np.array([(i, j, k) for i in grid for j in grid for k in grid], dtype=float)
    k[:, 0] *= 2.0 * np.pi / Lx
    k[:, 1] *= 2.0 * np.pi / Ly
    k[:, 2] *= 2.0 * np.pi / Lz
    return k


def calc_rhok(kvecs: np.ndarray, pos: np.ndarray) -> np.ndarray:
    arg = kvecs @ pos.T
    return np.exp(-1j * arg).sum(axis=1)


def calc_sk(kvecs: np.ndarray, pos: np.ndarray) -> np.ndarray:
    rho_k = calc_rhok(kvecs, pos)
    rho_mk = calc_rhok(-kvecs, pos)
    N = pos.shape[0]
    return (rho_k * rho_mk) / N


def calc_av_sk(kvecs: np.ndarray, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sk = np.real(calc_sk(kvecs, pos))
    kmod = np.linalg.norm(kvecs, axis=1)
    uniq, inv = np.unique(np.round(kmod, 12), return_inverse=True)
    av = np.zeros(len(uniq))
    for i in range(len(uniq)):
        av[i] = np.mean(sk[inv == i])
    return uniq, av

# ---- I/O helpers ----

def read_thermo_csv(path: str):
    data = np.genfromtxt(path, delimiter=",", names=True)
    # Expect columns: time, E_pot, E_kin, T
    return data


def read_gro(path: str):
    """Minimal .gro reader (positions + box)."""
    with open(path, "r") as f:
        title = f.readline().rstrip("\n")
        n = int(f.readline().strip())
        pos = np.zeros((n, 3), float)
        for i in range(n):
            line = f.readline()
            # Columns: resid(5) resname(5) atom(5) idx(5) x(8.3) y(8.3) z(8.3) [vx vy vz]
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            pos[i] = [x, y, z]
        # box line: Lx Ly Lz
        last = f.readline().split()
        if len(last) >= 3:
            Lx, Ly, Lz = map(float, last[:3])
        else:
            raise ValueError(".gro box line malformed")
    return title, pos, (Lx, Ly, Lz)

# ---- Plot save helper ----

def savefig(out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{stem}.png")
    svg = os.path.join(out_dir, f"{stem}.svg")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.savefig(svg, bbox_inches="tight")
    print(f"[save] {png}\n[save] {svg}")

# ---- Main analysis pipeline ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default = "slab" if slab else "bulk")
    ap.add_argument("--out_dir", type=str, default="C:/Users/Alex/OneDrive - Concordia University - Canada/phys440/project/Argon-LJ-simulation/figures/analyze", help="Output directory for plots and summary")
    ap.add_argument("--rc", type=float, default=2.5)
    ap.add_argument("--nbins", type=int, default=120)
    ap.add_argument("--dr", type=float, default=0.02)
    ap.add_argument("--maxk", type=int, default=6)
    ap.add_argument("--inplane", action="store_true", help="Use in-plane (x,y) distances for slab g(r)")
    args = ap.parse_args()



    thermo_path = "C:/Users/Alex/OneDrive - Concordia University - Canada/phys440/project/Argon-LJ-simulation/data/thermo_slab.csv" if slab else "C:/Users/Alex/OneDrive - Concordia University - Canada/phys440/project/Argon-LJ-simulation/data/thermo_bulk.csv"
    gro_path    = "C:/Users/Alex/OneDrive - Concordia University - Canada/phys440/project/Argon-LJ-simulation/data/argon_slab.gro" if slab else "C:/Users/Alex/OneDrive - Concordia University - Canada/phys440/project/Argon-LJ-simulation/data/argon_bulk.gro"

    # --- Load thermo and plot ---
    th = read_thermo_csv(thermo_path)

    # Temperature vs time
    plt.figure()
    plt.plot(th["time"], th["T"])  # default style, one plot per fig
    plt.xlabel("time")
    plt.ylabel("Temperature")
    plt.title("Instantaneous temperature vs time")
    savefig(args.out_dir, f"T_vs_time_{args.mode}")
    plt.close()

    # Energies vs time
    plt.figure()
    plt.plot(th["time"], th["E_pot"], label="E_pot")
    plt.plot(th["time"], th["E_kin"], label="E_kin")
    Etot = th["E_pot"] + th["E_kin"]
    plt.plot(th["time"], Etot, label="E_tot")
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Energies vs time")
    plt.legend()
    savefig(args.out_dir, f"Energies_vs_time_{args.mode}")
    plt.close()

    # --- Load coordinates from .gro ---
    title, pos, box = read_gro(gro_path)

    # --- g(r) ---
    if slab and args.inplane:
        # In-plane distances only
        rij = displacement_table(pos[:, :2], (box[0], box[1], 1.0), mode="bulk")  # 2D trick with Lz=1
        dij = np.linalg.norm(rij, axis=-1)
    else:
        disp = displacement_table(pos, box, mode=("slab" if args.mode == "slab" else "bulk"))
        dij = distance_table(disp)
    # Upper triangle to 1D list (i<j)
    N = dij.shape[0]
    dlist = dij[np.triu_indices(N, k=1)]
    g, r = pair_correlation(dlist, N, args.nbins, args.dr, box)

    plt.figure()
    plt.plot(r, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Radial distribution function")
    savefig(args.out_dir, f"gr_{args.mode}")
    plt.close()

    # --- S(k) ---
    kvecs = legal_kvecs(args.maxk, box)
    kmod, avsk = calc_av_sk(kvecs, pos)
    plt.figure()
    plt.plot(kmod, avsk)
    plt.xlabel("|k|")
    plt.xlim(0,10)
    plt.ylabel("S(|k|)")
    plt.title("Shell-averaged static structure factor")
    savefig(args.out_dir, f"Sk_{args.mode}")
    plt.close()

    # --- Block averages for T, E ---
    def block_average(series: np.ndarray, nblocks=5):
        series = np.asarray(series)
        series = series.reshape(-1, 1)
        Tn = series.shape[0]
        L = Tn // nblocks
        if L < 1:
            return np.array([series.mean()]), np.array([0.0])
        means = []
        for i in range(nblocks - 1):
            means.append(series[i * L : (i + 1) * L].mean())
        means.append(series[(nblocks - 1) * L :].mean())
        means = np.array(means)
        return means.mean(), means.std(ddof=1) / np.sqrt(nblocks)

    T_mean, T_err = block_average(th["T"], nblocks=8)
    E_mean, E_err = block_average(Etot, nblocks=8)

    # Save a small summary txt
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"summary_{args.mode}.txt"), "w") as f:
        f.write(f"Title: {title}\n")
        f.write(f"N: {N}\n")
        f.write(f"Box: {box}\n")
        f.write(f"T_mean: {T_mean:.6f} +- {T_err:.6f}\n")
        f.write(f"E_tot mean: {E_mean:.6f} +- {E_err:.6f}\n")
    print("[save] summary written")

if __name__ == "__main__":
    main()

