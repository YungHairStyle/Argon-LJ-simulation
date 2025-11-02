"""
main.py (fully featured, no CLI)
--------------------------------
Edit the configuration block below, then run:
    python main.py

Features
========
- MODE = "bulk" or "slab"
- FCC initializers (bulk or slab)
- 3D PBC (bulk) vs 2D PBC with open z (slab)
- Verlet neighbor list with skin
- LJ forces (shifted at rc)
- Velocity-Verlet + optional Andersen thermostat
- Time-series logging (E_pot, E_kin, T), CSV export
- Pressure tensor and (for slabs) surface tension estimate
- Optional .gro snapshot

Dependencies: NumPy, and LJ.py in the same folder.
"""

from __future__ import annotations

import time
import csv
import numpy as np
import LJ


# =============================
# Configuration (edit these)
# =============================

# Geometry / mode
MODE = "slab"      # "slab" or "bulk"

# --- Slab parameters ---
CELLS_X = 4        # FCC cells along x (slab)
CELLS_Y = 4        # FCC cells along y (slab)
LAYERS_Z = 4       # FCC unit layers stacked in z (slab)
LZ = 12.0          # total box height with vacuum (slab)

# --- Bulk parameters ---
CELLS = 6          # FCC cells per axis (bulk)

# --- Common LJ/MD parameters ---
A = 1.62           # FCC lattice parameter (near LJ minimum ~1.122462)
T = 1.0            # temperature (reduced)
MASS = 1.0         # particle mass
RC = 2.5           # LJ cutoff
DT = 0.002         # time step
STEPS = 3000      # total MD steps
EQUIL_STEPS = 1000 # discard this many steps for averages
THERMO_NU = 0.2    # Andersen collision frequency; 0 disables thermostat
SEED = None        # RNG seed (or None for random)

# --- Sampling & output ---
SAMPLE_EVERY = 5                # stride for time-series sampling
SAVE_DIR = "HW/Final project/data"  # output directory
SAVE_GRO = SAVE_DIR + "argon_slab.gro" if MODE == "slab" else SAVE_DIR + "argon_bulk.gro"  # or None
SAVE_THERMO = SAVE_DIR + "thermo_slab.csv" if MODE == "slab" else SAVE_DIR + "thermo_bulk.csv"  # or None
TITLE = f"Argon-{MODE}"

# --- Neighbor list ---
SKIN = 0.3         # Verlet skin distance


# =============================
# End of configuration
# =============================


def run():
    rng = np.random.default_rng(SEED)

    if MODE.lower() == "bulk":
        r, v, box = LJ.init_fcc_bulk(n_cells=CELLS, a=A, T=T, mass=MASS, rng=rng)
        slab_mode = False
    elif MODE.lower() == "slab":
        r, v, box = LJ.init_fcc_slab(
            n_cells_x=CELLS_X, n_cells_y=CELLS_Y, n_layers_z=LAYERS_Z,
            a=A, Lz=LZ, T=T, mass=MASS, rng=rng
        )
        slab_mode = True
    else:
        raise ValueError("MODE must be 'bulk' or 'slab'")

    N = len(r)
    print(f"[info] MODE={MODE}  N={N}  box=({box.Lx:.3f}, {box.Ly:.3f}, {box.Lz:.3f})")

    # Neighbor list
    nlist = LJ.NeighborList(rcut=RC, skin=SKIN, slab_mode=slab_mode)
    nlist.build(r, box)

    # Optional thermostat
    if THERMO_NU and THERMO_NU > 0.0:
        def thermo(vv):
            return LJ.andersen_thermostat(vv, T, MASS, DT, nu=THERMO_NU, rng=rng)
    else:
        thermo = None

    # Time series containers
    times, epots, ekins, temps = [], [], [], []

    t0 = time.time()
    next_progress = max(1000, STEPS // 20)

    for step in range(STEPS):
        r, v, U, K = LJ.step_velocity_verlet(
            r, v, box, MASS, DT, nlist, RC, slab_mode, thermostat=thermo
        )

        if step % SAMPLE_EVERY == 0:
            times.append(step * DT)
            epots.append(U)
            ekins.append(K)
            temps.append(LJ.instantaneous_temperature(v, MASS))

        if (step + 1) % next_progress == 0:
            elapsed = time.time() - t0
            Tinst = temps[-1] if temps else float('nan')
            En = (U + K) / N
            print(f"[{step+1:>7d}/{STEPS}] T={Tinst:.3f}  E/N={En:.3f}  elapsed={elapsed:.1f}s")

    print("[info] Run complete.")

    # Post-run summaries (production region only)
    if temps:
        eq_idx = max(0, EQUIL_STEPS // SAMPLE_EVERY)
        if eq_idx < len(temps):
            T_mean = float(np.mean(temps[eq_idx:]))
            Epot_mean = float(np.mean(epots[eq_idx:]))
            Ekin_mean = float(np.mean(ekins[eq_idx:]))
            print(f"[summary] <T> = {T_mean:.4f}, <E_pot> = {Epot_mean:.4f}, <E_kin> = {Ekin_mean:.4f}")
        else:
            print("[summary] Not enough sampled points to compute post-equil averages.")

    # Pressure tensor and (for slabs) surface tension at final frame
    P = LJ.pressure_tensor_LJ(r, v, box, MASS, nlist, RC)
    print(f"[final] Pressure tensor (instantaneous):\n{P}")
    if slab_mode:
        gamma = LJ.surface_tension_gamma(P, box.Lz)
        print(f"[final] Surface tension estimate gamma = {gamma:.6f} (reduced)")

    # Bulk tail corrections (informational)
    if not slab_mode:
        rho = N / box.volume
        Utail_per_particle, Ptail = LJ.bulk_tail_corrections(rho, RC)
        print(f"[bulk tails] U_tail/N = {Utail_per_particle:.6f},  P_tail = {Ptail:.6f}")

    # Save outputs
    if SAVE_GRO:
        LJ.write_gro(SAVE_GRO, r, box, title=TITLE, write_velocities=False)
        print(f"[save] Wrote {SAVE_GRO}")

    if SAVE_THERMO:
        with open(SAVE_THERMO, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "E_pot", "E_kin", "T"])
            for row in zip(times, epots, ekins, temps):
                w.writerow(row)
        print(f"[save] Wrote {SAVE_THERMO}")


if __name__ == "__main__":
    run()