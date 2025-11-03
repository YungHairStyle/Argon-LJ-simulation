"""
Main driver using md_core_merged with MODE = "bulk" or "slab".
- Simple O(N^2) all-pairs force evaluation (no neighbor list)
- FCC initializers for bulk and slab (with vacuum in z for slab)
- Velocity–Verlet + optional Andersen thermostat
- Time-series logging (E_pot, E_kin, T) and optional CSV/.gro output

Copy this next to md_core_merged.py and run:  python main_merged.py
"""

from __future__ import annotations
import csv, time
import numpy as np
import LJ as core

# =============================
# Configuration (edit these)
# =============================
MODE = "slab"      # "slab" or "bulk"

# --- Slab parameters ---
CELLS_X = 4        # FCC cells along x (slab)
CELLS_Y = 4        # FCC cells along y (slab)
LAYERS_Z = 4       # FCC unit layers stacked in z (slab)
LZ = 12.0          # total box height including vacuum (slab)

# --- Bulk parameters ---
CELLS = 6          # FCC cells per axis (bulk)

# --- Common LJ/MD parameters ---
A = 1.78           # FCC lattice parameter
T = 1.0            # temperature (reduced)
MASS = 1.0         # particle mass
RC = 2.5           # LJ cutoff
DT = 0.002         # time step
STEPS = 3000       # total MD steps
EQUIL_STEPS = 1000 # discard this many steps for averages
THERMO_NU = 0.2    # Andersen collision frequency; 0 disables thermostat
SEED = None        # RNG seed (or None for random)

# --- Sampling & output ---
SAMPLE_EVERY = 5
SAVE_DIR = "HW/Final project/data/"         # Output directory 
SAVE_GRO = SAVE_DIR + ("argon_slab.gro" if MODE == "slab" else "argon_bulk.gro")
SAVE_THERMO = SAVE_DIR + ("thermo_slab.csv" if MODE == "slab" else "thermo_bulk.csv")
TITLE = f"Argon-{MODE}"

# =============================
# Helpers
# =============================

FCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
], dtype=float)

def fcc_bulk_positions(n_cells: int, a: float) -> tuple[np.ndarray, tuple[float,float,float]]:
    """Return (positions, (Lx,Ly,Lz)) for a bulk FCC cube of n_cells per axis."""
    cells = np.arange(n_cells)
    R = []
    for i in cells:
        for j in cells:
            for k in cells:
                cell_origin = np.array([i, j, k], float)
                for b in FCC_BASIS:
                    R.append((cell_origin + b) * a)
    L = n_cells * a
    return np.array(R, float), (L, L, L)


def fcc_slab_positions(n_x: int, n_y: int, n_layers_z: int, a: float, Lz: float) -> tuple[np.ndarray, tuple[float,float,float]]:
    """Return (positions, (Lx,Ly,Lz)) for an FCC slab with vacuum along z.
    n_layers_z counts FCC cells stacked along z; box height is Lz (>= n_layers_z*a).
    """
    cells_x = np.arange(n_x)
    cells_y = np.arange(n_y)
    cells_z = np.arange(n_layers_z)
    R = []
    for i in cells_x:
        for j in cells_y:
            for k in cells_z:
                cell_origin = np.array([i, j, k], float)
                for b in FCC_BASIS:
                    R.append((cell_origin + b) * a)
    Lx = n_x * a
    Ly = n_y * a
    # Leave Lz as provided to include vacuum
    return np.array(R, float), (Lx, Ly, float(Lz))


def write_gro(path: str, pos: np.ndarray, box: tuple[float,float,float], title: str="frame"):
    """Minimal .gro writer without velocities."""
    Lx, Ly, Lz = box
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{len(pos):5d}\n")
        for i, (x,y,z) in enumerate(pos, start=1):
            # Fake residue/atom labels
            f.write(f"{1:5d}{'AR':>5s}{'Ar':>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"   {Lx:8.5f} {Ly:8.5f} {Lz:8.5f}\n")

# =============================
# Main
# =============================

def run():
    rng = np.random.default_rng(SEED)

    if MODE.lower() == "bulk":
        r, box = fcc_bulk_positions(CELLS, A)
        slab_mode = "bulk"
    elif MODE.lower() == "slab":
        r, box = fcc_slab_positions(CELLS_X, CELLS_Y, LAYERS_Z, A, LZ)
        slab_mode = "slab"
    else:
        raise ValueError("MODE must be 'bulk' or 'slab'")

    # Slight jitter to avoid overlapping pairs
    r += 1e-6 * rng.normal(size=r.shape)

    # Wrap initial positions
    r = core.wrap_positions(r, box, slab_mode)

    # Velocities
    v = core.initial_velocities(len(r), MASS, T)

    # Precompute tables
    disp = core.displacement_table(r, box, slab_mode)
    dist = core.distance_table(disp)

    # Time series containers
    times, epots, ekins, temps = [], [], [], []

    # Optional thermostat function
    def apply_thermostat(vv):
        if THERMO_NU and THERMO_NU > 0.0:
            return core.thermostat_andersen(vv, MASS, T, DT, nu=THERMO_NU)
        return vv

    t0 = time.time()
    next_progress = max(1000, STEPS // 20)

    N = len(r)
    Lx, Ly, Lz = core._as_box(box)
    print(f"[info] MODE={MODE}  N={N}  box=({Lx:.3f}, {Ly:.3f}, {Lz:.3f})")

    for step in range(STEPS):
        # Integrate one step
        r, v, disp, dist = core.advance(r, v, MASS, DT, disp, dist, RC, box, slab_mode)
        # Optional thermostat (acts after full step)
        v = apply_thermostat(v)

        if step % SAMPLE_EVERY == 0:
            U = core.potential(dist, RC)
            K = core.kinetic(MASS, v)
            times.append(step * DT)
            epots.append(U)
            ekins.append(K)
            temps.append( (2.0 * K) / (3.0 * N) )

        if (step + 1) % next_progress == 0:
            elapsed = time.time() - t0
            Tinst = temps[-1] if temps else float('nan')
            En = (epots[-1] + ekins[-1]) / N if epots else float('nan')
            print(f"[{step+1:>7d}/{STEPS}] T={Tinst:.3f}  E/N={En:.3f}  elapsed={elapsed:.1f}s")

    print("[info] Run complete.")

    # Post-run summaries
    if temps:
        eq_idx = max(0, EQUIL_STEPS // SAMPLE_EVERY)
        if eq_idx < len(temps):
            T_mean = float(np.mean(temps[eq_idx:]))
            Epot_mean = float(np.mean(epots[eq_idx:]))
            Ekin_mean = float(np.mean(ekins[eq_idx:]))
            print(f"[summary] <T> = {T_mean:.4f}, <E_pot> = {Epot_mean:.4f}, <E_kin> = {Ekin_mean:.4f}")
        else:
            print("[summary] Not enough sampled points to compute post-equil averages.")

    # Save outputs
    try:
        if SAVE_GRO:
            write_gro(SAVE_GRO, r, box, title=TITLE)
            print(f"[save] Wrote {SAVE_GRO}")
        if SAVE_THERMO:
            with open(SAVE_THERMO, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "E_pot", "E_kin", "T"])
                for row in zip(times, epots, ekins, temps):
                    w.writerow(row)
            print(f"[save] Wrote {SAVE_THERMO}")
    except FileNotFoundError:
        print("[warn] SAVE_DIR not found; skipping file outputs.")


if __name__ == "__main__":
    run()
