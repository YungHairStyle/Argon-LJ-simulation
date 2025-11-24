#!/usr/bin/env python3
"""
Slab-specific structural analysis with "Slab Scanning".

This module computes g_xy(r) and S_xy(k) at various z-offsets
from the slab's center of mass.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List
import analyze  # For utility functions

# =============================
# Z-SLICING UTILITIES
# =============================

def find_slab_center(pos: np.ndarray, Lz: float) -> float:
    """
    Calculates the Center of Mass (COM) of the slab in the z-direction.
    
    This handles the case where the slab might drift slightly from Lz/2.
    It assumes the slab is a single contiguous block and hasn't crossed 
    the PBC boundary at z=0/Lz.
    """
    # Simple average of Z coordinates
    # (Robust enough for a slab surrounded by vacuum)
    z_com = np.mean(pos[:, 2])
    return z_com

def slice_slab_atoms(pos: np.ndarray, box: Tuple[float, float, float], 
                     center_z: float, slice_thickness: float) -> Tuple[np.ndarray, float]:
    """
    Extract atoms from a specific slice centered at 'center_z'.
    
    Returns
    -------
    sliced_pos : Atoms in the slice
    rho_slice : Number density of this specific slice
    """
    Lx, Ly, Lz = box
    
    z_min = center_z - slice_thickness / 2.0
    z_max = center_z + slice_thickness / 2.0
    
    # Create mask for atoms within the slice
    z_coords = pos[:, 2]
    mask = (z_coords >= z_min) & (z_coords <= z_max)
    
    sliced_pos = pos[mask]
    
    # Calculate local density (important to see if we are in vapor)
    vol_slice = Lx * Ly * slice_thickness
    rho_slice = len(sliced_pos) / vol_slice
    
    return sliced_pos, rho_slice

# =============================
# 2D STRUCTURAL ANALYSIS
# =============================

def compute_2d_distances(pos: np.ndarray, box: Tuple[float, float, float]) -> np.ndarray:
    """Compute all pairwise 2D in-plane distances."""
    Lx, Ly, Lz = box
    N = pos.shape[0]
    dists = []
    # Note: This is O(N^2). For >2000 atoms, consider scipy.spatial.cKDTree
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dists.append(np.sqrt(dx**2 + dy**2))
    return np.array(dists)

def pair_correlation_2d(dists_2d: np.ndarray, N_atoms: int, 
                        nbins: int, dr: float, 
                        box: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate 2D in-plane g_xy(r)."""
    Lx, Ly, _ = box
    Area = Lx * Ly
    r_max = min(Lx, Ly) / 2.0
    nbins_actual = int(min(nbins * dr, r_max) / dr)
    
    hist, edges = np.histogram(dists_2d, bins=nbins_actual, range=(0.0, nbins_actual * dr))
    r = 0.5 * (edges[:-1] + edges[1:])
    dArea = np.pi * ((r + 0.5 * dr)**2 - (r - 0.5 * dr)**2)
    rho_2D = N_atoms / Area
    
    ideal_hist = ((N_atoms - 1) / 2.0) * rho_2D * dArea
    ideal_hist = np.maximum(ideal_hist, 1e-12)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        g_2d = hist / ideal_hist
        g_2d = np.nan_to_num(g_2d)
        
    return g_2d, r

def compute_2d_structure_factor(pos: np.ndarray, box: Tuple[float, float, float], maxk: int):
    """Calculate 2D in-plane S_xy(k)."""
    Lx, Ly, _ = box
    N = pos.shape[0]
    if N == 0: return np.array([]), np.array([])

    grid = np.arange(-maxk, maxk + 1)
    kx_grid = grid * (2.0 * np.pi / Lx)
    ky_grid = grid * (2.0 * np.pi / Ly)
    
    kvecs = []
    for kx in kx_grid:
        for ky in ky_grid:
            if kx == 0 and ky == 0: continue
            kvecs.append([kx, ky, 0.0]) # kz = 0
    kvecs = np.array(kvecs)
    
    arg = kvecs @ pos.T
    rho_k = np.exp(-1j * arg).sum(axis=1)
    S_k_raw = (np.abs(rho_k)**2) / N
    
    k_mod = np.linalg.norm(kvecs, axis=1)
    unique_k, inverse = np.unique(np.round(k_mod, 8), return_inverse=True)
    S_avg = np.zeros(len(unique_k))
    for i in range(len(unique_k)):
        S_avg[i] = np.mean(S_k_raw[inverse == i])
        
    return unique_k, S_avg

# =============================
# PLOTTING
# =============================

def plot_scan_slice(g_2d, r, k_mod, S_k, offset, rho_local, out_dir, time_label):
    """Generate plot for a specific slice offset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: g(r)
    ax = axes[0]
    ax.plot(r, g_2d, lw=2, color='#2E86AB')
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel(r'$r$ ($\sigma$)')
    ax.set_ylabel(r'$g_{xy}(r)$')
    ax.set_title(f'g(r) @ Offset {offset:.1f} (Density: {rho_local:.3f})')
    
    # Panel 2: S(k)
    ax = axes[1]
    if len(k_mod) > 0:
        ax.plot(k_mod, S_k, lw=2, color='#A23B72')
        ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel(r'$k$ ($\sigma^{-1}$)')
    ax.set_ylabel(r'$S_{xy}(k)$')
    ax.set_title(f'S(k) @ Offset {offset:.1f}')
    
    plt.suptitle(f"Structure Scan | {time_label} | Z-Offset = {offset:.1f}", fontweight='bold')
    plt.tight_layout()
    
    # Filename includes offset so you can compare them
    fname = f"scan_offset_{offset:.1f}_{time_label.replace(' ', '_')}"
    analyze.savefig(out_dir, fname)
    plt.close()

# =============================
# IO & DRIVER
# =============================

def read_gro_last_frame(path: str, N_atoms: int):
    """Reads only the last frame from trajectory."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    lines_per_frame = 2 + N_atoms + 1
    num_frames = len(lines) // lines_per_frame
    if num_frames == 0: return None

    # Extract last frame
    start = (num_frames - 1) * lines_per_frame
    coord_lines = lines[start+2 : start+2+N_atoms]
    box_line = lines[start+2+N_atoms]
    
    coords = []
    for line in coord_lines:
        coords.append([float(line[20:28]), float(line[28:36]), float(line[36:44])])
    
    box = tuple(map(float, box_line.split()[:3]))
    return np.array(coords), box

def analyze_slab_with_slicing(mode, data_dir, out_dir, n_atoms, l_ref, 
                              rc, nbins, dr, maxk, 
                              slice_thickness=None, offsets=[0.0]):
    """
    Main driver for slab scanning.
    
    offsets: list of floats
        Distances from the slab Center of Mass to take slices.
        e.g. [0.0, 2.0, 4.0]
    """
    gro_path = os.path.join(data_dir, f"argon_{mode}_traj.gro")
    result = read_gro_last_frame(gro_path, n_atoms)
    if not result:
        print("Error: Could not read trajectory.")
        return
        
    pos, box = result
    print(f"\n--- Analyzing Last Frame (Equilibrated) ---")
    
    # 1. Auto-detect slab center (COM)
    z_com = find_slab_center(pos, box[2])
    print(f"Detected Slab Center of Mass at z = {z_com:.3f}")
    
    if slice_thickness is None:
        slice_thickness = 2.0 # Default 2.0 sigma thickness
        
    # 2. Loop through offsets
    for offset in offsets:
        target_z = z_com + offset
        
        # Check if we are outside the box
        if target_z > box[2] or target_z < 0:
            print(f"Skipping offset {offset}: Outside simulation box.")
            continue

        print(f"-> Slicing at Offset {offset:.1f} (z = {target_z:.3f} +/- {slice_thickness/2:.2f})")
        
        # Slice
        sliced_pos, rho_local = slice_slab_atoms(pos, box, target_z, slice_thickness)
        N_slice = len(sliced_pos)
        
        # Safety check: If density is too low, we are in vapor
        if N_slice < 10:
            print(f"   [Warning] Only {N_slice} atoms found. Likely in vapor. Skipping plot.")
            continue
            
        # Compute
        dists = compute_2d_distances(sliced_pos, box)
        g_2d, r = pair_correlation_2d(dists, N_slice, nbins, dr, box)
        k_mod, S_k = compute_2d_structure_factor(sliced_pos, box, maxk)
        
        # Plot
        plot_scan_slice(g_2d, r, k_mod, S_k, offset, rho_local, out_dir, "Final_Frame")

    print("\nScanning complete. Check 'figures' folder.")