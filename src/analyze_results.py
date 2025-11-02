
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util, sys

# Load local LJ.py
LJ_PATH = Path(__file__).with_name('LJ.py')
spec = importlib.util.spec_from_file_location("LJ", LJ_PATH)
LJ = importlib.util.module_from_spec(spec)
sys.modules["LJ"] = LJ
spec.loader.exec_module(LJ)

def read_gro(path):
    # Minimal .gro reader for coordinates and box -> returns (r, box)
    with open(path, 'r') as f:
        lines = f.read().strip().splitlines()
    if len(lines) < 3:
        raise ValueError("Invalid .gro file")
    N = int(lines[1].strip())
    pos = []
    for i in range(2, 2+N):
        line = lines[i]
        try:
            x = float(line[20:28]); y = float(line[28:36]); z = float(line[36:44])
        except Exception:
            parts = line.split()
            x, y, z = map(float, parts[-3:])
        pos.append([x, y, z])
    bx = lines[2+N].split()
    if len(bx) >= 3:
        Lx, Ly, Lz = map(float, bx[:3])
    else:
        raise ValueError("Box line missing or malformed in .gro")
    r = np.array(pos, dtype=float)
    box = LJ.Box(Lx, Ly, Lz)
    return r, box

def auto_liquid_window_z(z_centers, rho_z, frac=0.5):
    # Pick contiguous z-window where rho(z) exceeds 'frac' of its max.
    if rho_z.size == 0:
        return None
    thr = frac * np.max(rho_z)
    mask = rho_z >= thr
    if not np.any(mask):
        return None
    idx = np.where(mask)[0]
    starts = [idx[0]]
    ends = []
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            ends.append(idx[i-1]); starts.append(idx[i])
    ends.append(idx[-1])
    lengths = np.array(ends) - np.array(starts)
    k = int(np.argmax(lengths))
    i0, i1 = starts[k], ends[k]
    dz = (z_centers[1] - z_centers[0]) if len(z_centers) > 1 else 0.0
    zmin = z_centers[i0] - 0.5*dz
    zmax = z_centers[i1] + 0.5*dz
    return (float(zmin), float(zmax))

def plot_time_series(thermo_csv, out_dir, title_suffix=""):
    df = pd.read_csv(thermo_csv)
    df["E_tot"] = df["E_pot"] + df["E_kin"]
    plt.figure()
    plt.plot(df["time"], df["T"])
    plt.xlabel("time (reduced)")
    plt.ylabel("Temperature T")
    plt.title(f"Instantaneous Temperature{title_suffix}")
    plt.tight_layout()
    fT = Path(out_dir, "temperature.svg")
    plt.savefig(fT)
    plt.close()

    plt.figure()
    plt.plot(df["time"], df["E_pot"], label="E_pot")
    plt.plot(df["time"], df["E_kin"], label="E_kin")
    plt.plot(df["time"], df["E_tot"], label="E_tot")
    plt.xlabel("time (reduced)")
    plt.ylabel("Energy (reduced)")
    plt.title(f"Energies vs Time{title_suffix}")
    plt.legend()
    plt.tight_layout()
    fE = Path(out_dir, "energies.svg")
    plt.savefig(fE)
    plt.close()
    return str(fT), str(fE)

def plot_density_and_rdf(gro_path, out_dir, rc=2.5, nbins=100, rdf_bins=120, window_frac=0.5):
    r, box = read_gro(gro_path)
    zc, rhoz = LJ.density_profile_z(r, box.Lz, nbins=nbins)
    plt.figure()
    plt.plot(zc, rhoz)
    plt.xlabel("z")
    plt.ylabel("rho(z)")
    plt.title("Density Profile rho(z)")
    plt.tight_layout()
    fRho = Path(out_dir, "density_z.svg")
    plt.savefig(fRho)
    plt.close()

    win = auto_liquid_window_z(zc, rhoz, frac=window_frac)
    rvals_all, g_all = LJ.rdf_inplane(r, box, rc=rc, nbins=rdf_bins, z_window=None)
    if win is not None:
        rvals_liq, g_liq = LJ.rdf_inplane(r, box, rc=rc, nbins=rdf_bins, z_window=win)
    else:
        rvals_liq, g_liq = None, None

    plt.figure()
    plt.plot(rvals_all, g_all, label="all z")
    if rvals_liq is not None:
        plt.plot(rvals_liq, g_liq, label=f"z in [{win[0]:.2f},{win[1]:.2f}]")
    plt.xlabel("r (in-plane)")
    plt.ylabel("g_parallel(r)")
    plt.title("In-plane RDF g_parallel(r)")
    plt.legend()
    plt.tight_layout()
    fRDF = Path(out_dir, "rdf_inplane.svg")
    plt.savefig(fRDF)
    plt.close()

    meta = {
        "N": int(len(r)),
        "box": {"Lx": float(box.Lx), "Ly": float(box.Ly), "Lz": float(box.Lz)},
        "rdf_rc": float(rc),
        "nbins_density": int(nbins),
        "nbins_rdf": int(rdf_bins),
        "liquid_window": None if win is None else [float(win[0]), float(win[1])]
    }
    with open(Path(out_dir, "analysis_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    return str(fRho), str(fRDF), meta

def main(thermo_csv="HW/Final project/data/thermo_slab.csv", gro_path="HW/Final project/data/argon_slab.gro", out_dir="HW/Final project/figures", rc=2.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fT = fE = None
    if Path(thermo_csv).exists():
        fT, fE = plot_time_series(thermo_csv, out_dir, title_suffix="")
    else:
        print(f"[warn] thermo CSV not found at {thermo_csv}")
    fRho = fRDF = None
    meta = {}
    if Path(gro_path).exists():
        fRho, fRDF, meta = plot_density_and_rdf(gro_path, out_dir, rc=rc)
    else:
        print(f"[warn] GRO file not found at {gro_path}")
    print('[analysis] Outputs:')
    if fT: print(' - Temperature:', fT)
    if fE: print(' - Energies:', fE)
    if fRho: print(' - Density profile:', fRho)
    if fRDF: print(' - RDF (in-plane):', fRDF)
    if meta:
        print(' - Meta:', meta)

if __name__ == "__main__":
    main()
