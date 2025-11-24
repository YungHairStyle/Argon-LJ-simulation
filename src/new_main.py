"""
Main script to run LJ MD simulation + analysis.
Edit the parameters in this file, then run:
    python3 main.py
"""
import os
from pathlib import Path
import LJ
import analyze
import structure_analysis
import slab_structure_analysis  # NEW: Import the slab-specific analysis

# =============================
# Global settings (edit me)
# =============================
# --- Simulation mode ---
#UNCOMMENT ONE OF THE FOLLOWING TWO TO SELECT THE  MODE ---
#MODE = "slab"
MODE = "bulk"

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
DT        = 0.02      # time step
STEPS     = 6000      # total MD steps
EQUIL_STEPS = 1000     # steps considered "equilibration" for averages
PROB      = 0.02       # Andersen collision probability (0 disables thermostat)
SEED      = None       # RNG seed (None = random)

# --- Sampling ---
SAMPLE_EVERY = 10       # sample every N steps for thermo output

# --- Paths (where to read/write) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "figures"/ MODE.lower()

# --- Analysis parameters ---
NBINS    = 200         # number of bins for g(r)
DR       = 0.02        # bin width for g(r)
MAXK     = 15          # max integer for |k| grid (S(k))
INPLANE  = True        # for slab: use in-plane g(r) if True

# --- NEW: Slab slicing parameters ---
SLICE_THICKNESS = 1.0   # Thickness of each slice (in sigma)
SLICE_OFFSETS   = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] # Offsets from the center to scan
                        # 0.0 = Exact Center
                        # 4.5 = Likely approaching the interface/vapor

# Create directories if needed
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

mode = MODE.lower()
title = f"Argon-{mode}"

# 1) Run the MD simulation
if False:
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
    # --- Standard thermo analysis (T vs time, E vs time) ---
    analyze.analyze_trajectory(
        mode    = mode,
        data_dir= DATA_DIR,
        out_dir = FIG_DIR,
        rc      = RC,
        nbins   = NBINS,
        dr      = DR,
        maxk    = MAXK,
        inplane = False, 
    )
    
    # --- Structural analysis: Choose method based on mode ---
    
    # Calculate N and L based on the current mode settings
    if mode == "bulk":
        current_n_atoms = 4 * CELLS_BULK**3
        current_l_ref   = CELLS_BULK * A
        
        # --- A) Existing Standard Bulk Analysis ---
        print("\n" + "="*60)
        print("RUNNING BULK STRUCTURAL ANALYSIS (Standard 3D)")
        print("="*60 + "\n")
        
        structure_analysis.analyze_structural_only(
            mode    = mode,
            data_dir= DATA_DIR,
            out_dir = FIG_DIR,
            rc      = RC,
            nbins   = NBINS,
            dr      = DR,
            maxk    = MAXK,
            inplane = False,
            n_atoms = current_n_atoms,
            l_ref   = current_l_ref
        )

        # --- B) NEW: Run Slab Slicing on Bulk (Method Comparison) ---
        # We create a sub-folder so these plots don't overwrite the standard ones
        SLICE_CHECK_DIR = FIG_DIR / "slicing_method_check"
        os.makedirs(SLICE_CHECK_DIR, exist_ok=True)

        print("\n" + "-"*60)
        print("RUNNING SLICING CHECK ON BULK (2D Slicing vs 3D Standard)")
        print("-"*60 + "\n")

        slab_structure_analysis.analyze_slab_with_slicing(
            mode            = mode,   # distinct name for plot titles
            data_dir        = DATA_DIR,
            out_dir         = SLICE_CHECK_DIR, # save in separate folder
            n_atoms         = current_n_atoms,
            l_ref           = current_l_ref,   # Box length acts as Lz here
            rc              = RC,
            nbins           = NBINS,
            dr              = DR,
            maxk            = MAXK,
            slice_thickness = SLICE_THICKNESS,
            # Check center (0.0) and one random offset (2.0) to prove isotropy
            offsets         = [0.0, 2.0]       
        )
        
    else:  # mode == "slab"
        current_n_atoms = 4 * CELLS_X * CELLS_Y * LAYERS_Z
        current_l_ref   = CELLS_X * A
        
        # For slab: use the NEW slab-specific analysis with scanning
        print("\n" + "="*60)
        print("RUNNING SLAB STRUCTURAL SCAN")
        print("="*60 + "\n")
        
        slab_structure_analysis.analyze_slab_with_slicing(
            mode            = mode,
            data_dir        = DATA_DIR,
            out_dir         = FIG_DIR,
            n_atoms         = current_n_atoms,
            l_ref           = current_l_ref,
            rc              = RC,
            nbins           = NBINS,
            dr              = DR,
            maxk            = MAXK,
            slice_thickness = SLICE_THICKNESS,
            offsets         = SLICE_OFFSETS 
        )