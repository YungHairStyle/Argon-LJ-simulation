#!/usr/bin/env python3
"""

DO NOT USE THIS FILE IT IS DEPRECATED. USE ANALYZE.PY INSTEAD.
analyze_results.py
==================
Post-process LJ MD outputs for SLAB or BULK runs.

- Reads:
    HW/Final project/data/argon_{MODE}.gro
    HW/Final project/data/thermo_{MODE}.csv
- Writes figures (SVG) & metadata JSON to:
    HW/Final project/figures/{MODE}/

Figures produced:
  • temperature.svg              (T vs time)
  • energies.svg                 (E_pot, E_kin, E_tot vs time)
  • energy_smooth.svg            (E_tot + rolling mean)
  • temp_hist.svg                (temperature histogram, post burn-in)
  • density_z.svg                (SLAB only)
  • rdf.svg                      (g∥(r) for SLAB, g(r) 3D for BULK)
  • analysis_meta.json           (run info, RDF params, etc.)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util, sys
from typing import Optional, Tuple
import LJ

# -----------------------
# Configuration defaults
# -----------------------
MODE = "slab"  # "slab" or "bulk"
DATA_DIR_DEFAULT = Path("HW/Final project/data")        #change this to your input data path
OUT_ROOT_DEFAULT = Path("HW/Final project/figures/analyze_results/")     #change this to your desired output path

# -----------------------
# Load local LJ.py (same folder as this script)
# -----------------------
_THIS = Path(__file__).resolve()
LJ_PATH = _THIS.with_name("LJ.py")
spec = importlib.util.spec_from_file_location("LJ", LJ_PATH)
LJ = importlib.util.module_from_spec(spec)
sys.modules["LJ"] = LJ
spec.loader.exec_module(LJ)


# -----------------------
# Utilities
# -----------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

import re

def read_gro(path: Path) -> Tuple[np.ndarray, LJ.Box]: 
    """
    Robust .gro reader: extracts x,y,z from the first three floats on each atom line.
    Works whether velocities are present or not.
    """
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {path}")

    try:
        N = int(lines[1].strip())
    except Exception as e:
        raise ValueError(f"Cannot read atom count from '{path}': {e}")

    # Helper: regex for floats like  -12.345, 1.23e-02, 0.500
    float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

    pos = []
    for i in range(2, 2 + N):
        line = lines[i]
        nums = float_re.findall(line)
        if len(nums) < 3:
            raise ValueError(f"Atom line {i-1} has fewer than 3 floats: '{line}'")
        x, y, z = map(float, nums[:3])  # first three floats are x,y,z
        pos.append((x, y, z))

    # Box line: first 3 floats are Lx, Ly, Lz
    box_line = lines[2 + N]
    nums = float_re.findall(box_line)
    if len(nums) < 3:
        raise ValueError(f"Box line missing floats: '{box_line}'")

    Lx, Ly, Lz = map(float, nums[:3])

    r = np.asarray(pos, dtype=float)
    box = LJ.Box(Lx, Ly, Lz)
    return r, box



# -----------------------
# Time-series plots
# -----------------------
def plot_time_series(thermo_csv: Path, out_dir: Path) -> dict:
    out = {}
    df = pd.read_csv(thermo_csv)
    if not {"time", "E_pot", "E_kin", "T"}.issubset(df.columns):
        raise ValueError(f"{thermo_csv} missing required columns; found {df.columns.tolist()}")
    df["E_tot"] = df["E_pot"] + df["E_kin"]

    # T vs t
    plt.figure()
    plt.plot(df["time"], df["T"])
    plt.xlabel("time (reduced)")
    plt.ylabel("Temperature T")
    plt.title("Instantaneous Temperature")
    plt.tight_layout()
    fT = out_dir / "temperature.svg"
    plt.savefig(fT); plt.close()
    out["temperature"] = str(fT)

    # Energies vs t
    plt.figure()
    plt.plot(df["time"], df["E_pot"], label="E_pot")
    plt.plot(df["time"], df["E_kin"], label="E_kin")
    plt.plot(df["time"], df["E_tot"], label="E_tot")
    plt.xlabel("time (reduced)")
    plt.ylabel("Energy (reduced)")
    plt.title("Energies vs Time")
    plt.legend()
    plt.tight_layout()
    fE = out_dir / "energies.svg"
    plt.savefig(fE); plt.close()
    out["energies"] = str(fE)

    # E_tot with rolling mean
    plt.figure()
    win = max(5, len(df)//100)
    df["E_tot_smooth"] = df["E_tot"].rolling(window=win, min_periods=1, center=True).mean()
    plt.plot(df["time"], df["E_tot"], label="E_tot")
    plt.plot(df["time"], df["E_tot_smooth"], label=f"rolling mean (w={win})")
    plt.xlabel("time (reduced)")
    plt.ylabel("Total Energy (reduced)")
    plt.title("Total Energy with Rolling Average")
    plt.legend()
    plt.tight_layout()
    fS = out_dir / "energy_smooth.svg"
    plt.savefig(fS); plt.close()
    out["energy_smooth"] = str(fS)

    # Temperature histogram (post burn-in, drop first 20%)
    burn = int(0.2 * len(df))
    eq = df.iloc[burn:] if burn < len(df) else df
    plt.figure()
    plt.hist(eq["T"].values, bins=30)
    plt.xlabel("Temperature T")
    plt.ylabel("Count")
    plt.title("Temperature Distribution (post burn-in)")
    plt.tight_layout()
    fH = out_dir / "temp_hist.svg"
    plt.savefig(fH); plt.close()
    out["temp_hist"] = str(fH)

    # Quick stats for captions
    out["stats"] = {
        "samples_total": int(len(df)),
        "burn_in_dropped": int(burn),
        "T_mean_eq": float(eq["T"].mean()) if len(eq) else None,
        "T_std_eq": float(eq["T"].std()) if len(eq) else None,
        "E_pot_mean_eq": float(eq["E_pot"].mean()) if len(eq) else None,
        "E_kin_mean_eq": float(eq["E_kin"].mean()) if len(eq) else None,
        "E_tot_mean_eq": float(eq["E_tot"].mean()) if len(eq) else None,
    }
    return out


# -----------------------
# Structural plots
# -----------------------
def plot_density_z_slab(r: np.ndarray, box: LJ.Box, out_dir: Path, nbins: int = 100) -> str:
    zc, rhoz = LJ.density_profile_z(r, box.Lz, nbins=nbins)
    plt.figure()
    plt.plot(zc, rhoz)
    plt.xlabel("z")
    plt.ylabel("ρ(z)")
    plt.title("Density Profile ρ(z) (slab)")
    plt.tight_layout()
    f = out_dir / "density_z.svg"
    plt.savefig(f); plt.close()
    return str(f)

def auto_liquid_window_z(zc: np.ndarray, rhoz: np.ndarray, frac: float = 0.5) -> Optional[Tuple[float, float]]:
    if rhoz.size == 0:
        return None
    thr = float(frac) * float(np.max(rhoz))
    mask = rhoz >= thr
    if not np.any(mask):
        return None
    idx = np.where(mask)[0]
    starts, ends = [idx[0]], []
    for k in range(1, len(idx)):
        if idx[k] != idx[k-1] + 1:
            ends.append(idx[k-1]); starts.append(idx[k])
    ends.append(idx[-1])
    lengths = np.array(ends) - np.array(starts)
    kmax = int(np.argmax(lengths))
    i0, i1 = starts[kmax], ends[kmax]
    dz = (zc[1] - zc[0]) if len(zc) > 1 else 0.0
    return (float(zc[i0] - 0.5*dz), float(zc[i1] + 0.5*dz))

def plot_rdf_slab(r: np.ndarray, box: LJ.Box, out_dir: Path, rc: float = 2.5, bins: int = 120,
                  z_window: Optional[Tuple[float, float]] = None) -> str:
    # In-plane RDF (x–y only)
    r_all, g_all = LJ.rdf_inplane(r, box, rc=rc, nbins=bins, z_window=None)
    plt.figure()
    plt.plot(r_all, g_all, label="all z")

    if z_window is not None:
        r_liq, g_liq = LJ.rdf_inplane(r, box, rc=rc, nbins=bins, z_window=z_window)
        plt.plot(r_liq, g_liq, label=f"z in [{z_window[0]:.2f},{z_window[1]:.2f}]")

    plt.xlabel("r (in-plane)")
    plt.ylabel("g∥(r)")
    plt.title("In-plane RDF g∥(r) (slab)")
    plt.legend()
    plt.tight_layout()
    f = out_dir / "rdf.svg"
    plt.savefig(f); plt.close()
    return str(f)

def plot_rdf_3d_bulk(r: np.ndarray, box: LJ.Box, out_dir: Path, rc: float = 2.5, bins: int = 160) -> str:
    """
    Simple 3D RDF g(r) using minimum-image distances and ideal-gas normalization.
    """
    # Pairwise distances with broadcasting (N^2 memory for simplicity; OK for modest N)
    N = len(r)
    dr = r[None, :, :] - r[:, None, :]
    dr = LJ.minimum_image_3d(dr.reshape(-1, 3), box.L).reshape(N, N, 3)
    d = np.linalg.norm(dr, axis=-1)

    # Keep upper triangle, exclude self
    iu = np.triu_indices(N, k=1)
    dist = d[iu]
    dist = dist[(dist > 1e-8) & (dist < rc)]

    hist, edges = np.histogram(dist, bins=bins, range=(0.0, rc))
    r_centers = 0.5 * (edges[1:] + edges[:-1])
    dr_bin = edges[1] - edges[0]

    rho = N / box.volume
    shell_vol = 4.0 * np.pi * (r_centers**2) * dr_bin
    # Number of ideal-gas pairs per bin: rho * shell_vol * N  (each particle sees rho*shell_vol neighbors)
    g = hist / (rho * shell_vol * N)

    plt.figure()
    plt.plot(r_centers, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function g(r) (bulk)")
    plt.tight_layout()
    f = out_dir / "rdf.svg"
    plt.savefig(f); plt.close()
    return str(f)


# -----------------------
# Main driver
# -----------------------
def main(
    mode: str = MODE,
    data_dir: Path = DATA_DIR_DEFAULT,
    out_root: Path = OUT_ROOT_DEFAULT,
    rc: float = 2.5
) -> None:
    mode = mode.lower().strip()
    if mode not in {"slab", "bulk"}:
        raise ValueError("mode must be 'slab' or 'bulk'")

    # Resolve paths
    data_dir = Path(data_dir)
    out_dir = Path(out_root) / mode
    ensure_dir(out_dir)

    thermo_csv = data_dir / f"thermo_{mode}.csv"
    gro_path   = data_dir / f"argon_{mode}.gro"

    outputs = {"mode": mode, "inputs": {"thermo_csv": str(thermo_csv), "gro": str(gro_path)}}

    # Time series
    if thermo_csv.exists():
        outputs["time_series"] = plot_time_series(thermo_csv, out_dir)
    else:
        print(f"[warn] Missing thermo CSV: {thermo_csv}")

    # Structural (from GRO)
    if gro_path.exists():
        r, box = read_gro(gro_path)
        if mode == "slab":
            # Density profile
            outputs["density_z"] = plot_density_z_slab(r, box, out_dir)

            # Auto-pick liquid window for nicer in-plane RDF
            zc, rhoz = LJ.density_profile_z(r, box.Lz, nbins=120)
            zwin = auto_liquid_window_z(zc, rhoz, frac=0.5)
            outputs["rdf"] = plot_rdf_slab(r, box, out_dir, rc=rc, z_window=zwin)
            outputs["rdf_window"] = None if zwin is None else [float(zwin[0]), float(zwin[1])]
        else:
            outputs["rdf"] = plot_rdf_3d_bulk(r, box, out_dir, rc=rc)
    else:
        print(f"[warn] Missing GRO file: {gro_path}")

    # Save metadata
    meta_path = out_dir / "analysis_meta.json"
    with meta_path.open("w") as f:
        json.dump(outputs, f, indent=2)
    print("[analysis] Wrote:", meta_path)
    # Pretty-print summary
    for k, v in outputs.items():
        if k == "time_series":
            print(f"  {k}:")
            for kk, vv in v.items():
                if kk == "stats":
                    print(f"    stats: {vv}")
                else:
                    print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze LJ MD outputs for slab or bulk.")
    ap.add_argument("--mode", choices=["slab", "bulk"], default=MODE, help="Which dataset to analyze")
    ap.add_argument("--data_dir", default=str(DATA_DIR_DEFAULT), help="Directory containing argon_*.gro and thermo_*.csv")
    ap.add_argument("--out_root", default=str(OUT_ROOT_DEFAULT), help="Root directory for figures/{mode}")
    ap.add_argument("--rc", type=float, default=2.5, help="Cutoff for RDF")
    args = ap.parse_args()

    main(mode=args.mode, data_dir=Path(args.data_dir), out_root=Path(args.out_root), rc=args.rc)
