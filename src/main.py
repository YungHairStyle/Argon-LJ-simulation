"""
Main script to run LJ MD simulation + analysis.

Edit the parameters in this file, then run:

    python3 main.py
"""

import os
from pathlib import Path
import LJ
import analyze

# =============================
# Global settings (edit me)
# =============================

# --- Simulation mode ---
#UNCOMMENT ONE OF THE FOLLOWING TWO TO SELECT THE  MODE ---
MODE = "slab"
#MODE = "bulk"

# --- Slab parameters ---
CELLS_X   = 4          # FCC cells along x (slab)
CELLS_Y   = 4          # FCC cells along y (slab)
LAYERS_Z  = 4          # FCC cells stacked in z (slab)
LZ        = 12.0       # total box height including vacuum (slab)

# --- Bulk parameters ---
CELLS_BULK = 6         # FCC cells per axis (bulk)

# --- Common LJ/MD parameters ---
A         = 1.78       # FCC lattice parameter
T         = 1.0        # temperature (reduced units)
MASS      = 1.0        # particle mass
RC        = 2.5        # LJ cutoff
DT        = 0.004      # time step
STEPS     = 20000       # total MD steps
EQUIL_STEPS = 1000     # steps considered "equilibration" for averages
PROB      = 0.02        # Andersen collision probability (0 disables thermostat)
SEED      = None       # RNG seed (None = random)

# --- Sampling ---
SAMPLE_EVERY = 5       # sample every N steps for thermo output

# --- Paths (where to read/write) ---
## DO NOT TOUCH THIS WILL PUT THE FILES IN THE RIGHT PLACE DONT WORRY ABOUT IT ##
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "figures"/ MODE.lower()


# --- Analysis parameters ---
NBINS    = 200        # number of bins for g(r)
DR       = 0.02       # bin width for g(r)
MAXK     = 15        # max integer for |k| grid (S(k))
INPLANE  = True       # for slab: use in-plane g(r) if True

# Create directories if needed
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

mode = MODE.lower()
title = f"Argon-{mode}"

# 1) Run the MD simulation
if True:
    md_result = LJ.run_md(
        mode        = mode,
        cells_x     = CELLS_X,
        cells_y     = CELLS_Y,
        layers_z    = LAYERS_Z,
        cells_bulk  = CELLS_BULK,
        a           = A,
        T           = T,
        mass        = MASS,
        rc          = RC,
        dt          = DT,
        steps       = STEPS,
        equil_steps = EQUIL_STEPS,
        prob        = PROB,
        seed        = SEED,
        sample_every= SAMPLE_EVERY,
        data_dir    = DATA_DIR,
        Lz          = LZ,
        title       = title,
    )

# 2) Analyze the output (make plots + summary)
if True:
    analyze.analyze_trajectory(
        mode    = mode,
        data_dir= DATA_DIR,
        out_dir = FIG_DIR,
        rc      = RC,
        nbins   = NBINS,
        dr      = DR,
        maxk    = MAXK,
        inplane = (INPLANE and mode == "slab"),
    )

