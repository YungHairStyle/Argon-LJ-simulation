#!/usr/bin/env python3
"""
Analysis module for MD outputs.

Key entry point (for other scripts):
    analyze_trajectory(mode, data_dir, out_dir, rc, nbins, dr, maxk, inplane=False)

This expects files:
    data_dir / thermo_{mode}.csv
    data_dir / argon_{mode}.gro

and writes plots + summary into out_dir.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ---- Core helpers (mirrors MD core) ----

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


def pair_correlation(
    dists_1d: np.ndarray, natom: int, nbins: int, dr: float, L
) -> Tuple[np.ndarray, np.ndarray]:
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

# ---- Block average helper ----

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

# ---- Main analysis pipeline (callable from main.py) ----

def analyze_trajectory(
    mode: str,
    data_dir: str,
    out_dir: str,
    rc: float = 2.5,
    nbins: int = 120,
    dr: float = 0.02,
    maxk: int = 15,
    inplane: bool = False,
):
    """
    High-level analysis entry point. Loads CSV + .gro, makes:
      - T vs time
      - Energies vs time
      - g(r)
      - S(k)
      - summary_{mode}.txt
    """
    mode = mode.lower()
    is_slab = (mode == "slab")

    thermo_path = os.path.join(data_dir, f"thermo_{mode}.csv")
    gro_path = os.path.join(data_dir, f"argon_{mode}.gro")

    # --- Load thermo and plot ---
    th = read_thermo_csv(thermo_path)

    # Temperature vs time
    plt.figure()
    plt.plot(th["time"], th["T"])
    plt.xlabel("time")
    plt.ylabel("Temperature")
    plt.title("Instantaneous temperature vs time")
    savefig(out_dir, f"T_vs_time_{mode}")
    plt.close()

    # Energies vs time
    Etot = th["E_pot"] + th["E_kin"]
    plt.figure()
    plt.plot(th["time"], th["E_pot"], label="E_pot")
    plt.plot(th["time"], th["E_kin"], label="E_kin")
    plt.plot(th["time"], Etot, label="E_tot")
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Energies vs time")
    plt.legend()
    savefig(out_dir, f"Energies_vs_time_{mode}")
    plt.close()

    # --- Load coordinates from .gro ---
    title, pos, box = read_gro(gro_path)

    # --- g(r) ---
    if is_slab and inplane:
        # Use in-plane distances only: set z=0 but keep periodicity in x,y
        pos_xy = pos.copy()
        pos_xy[:, 2] = 0.0
        disp = displacement_table(pos_xy, (box[0], box[1], 1.0), mode="bulk")
    else:
        disp = displacement_table(pos, box, mode=("slab" if is_slab else "bulk"))

    dij = distance_table(disp)

    # Upper triangle to 1D list (i<j)
    N = dij.shape[0]
    dlist = dij[np.triu_indices(N, k=1)]
    g, r = pair_correlation(dlist, N, nbins, dr, box)

    plt.figure()
    plt.plot(r, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Radial distribution function")
    savefig(out_dir, f"gr_{mode}")
    plt.close()

    # --- S(k) ---
    kvecs = legal_kvecs(maxk, box)
    kmod, avsk = calc_av_sk(kvecs, pos)
    # Remove k = 0 (and extremely small k)
    mask = kmod > 1e-4   # cutoff = 0.000001
    kmod = kmod[mask]
    avsk = avsk[mask]
    plt.figure()
    plt.plot(kmod, avsk)
    plt.xlabel("|k|")
    plt.xlim(0, maxk)
    plt.ylim(0,max(avsk)*1.1)
    plt.ylabel("S(|k|)")
    plt.title("Shell-averaged static structure factor")
    savefig(out_dir, f"Sk_{mode}")
    plt.close()

    # --- Block averages for T, E ---
    T_mean, T_err = block_average(th["T"], nblocks=8)
    E_mean, E_err = block_average(Etot, nblocks=8)

    os.makedirs(out_dir, exist_ok=True)

