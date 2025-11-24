import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List

# --- IMPORT CORE FUNCTIONS FROM EXISTING MODULES ---
import analyze      # To get pair_correlation, calc_av_sk, etc.

# --- I/O AND FRAME SELECTION ---

def read_gro_trajectory(path: str, N_atoms: int, L_guess: float) -> List[Tuple[str, np.ndarray, Tuple[float, float, float], int]]:
    """Reads coordinates and box dimensions for the first and last frames."""
    lines_per_frame = 2 + N_atoms + 1
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Trajectory file not found at {path}")
        return []
        
    num_total_frames = len(lines) // lines_per_frame
    if num_total_frames < 2:
        frame_indices = [0]
    else:
        frame_indices = [0, num_total_frames - 1]

    results = []
    
    for idx in frame_indices:
        start_line = idx * lines_per_frame
        title_line = lines[start_line]
        coord_lines = lines[start_line + 2 : start_line + 2 + N_atoms]
        box_line = lines[start_line + 2 + N_atoms]
        
        coords = []
        for line in coord_lines:
            x = float(line[20:28].strip())
            y = float(line[28:36].strip())
            z = float(line[36:44].strip())
            coords.append([x, y, z])
                
        box_dims = tuple(map(float, box_line.split()[:3]))
        results.append((title_line.strip(), np.array(coords), box_dims, idx))

    return results

# --- 2D STRUCTURAL ANALYSIS (FOR SLAB) ---

def get_dists_in_plane(pos: np.ndarray, L):
    N = pos.shape[0]
    dists_2d = []
    for i in range(N):
        for j in range(i + 1, N):
            dr = pos[i] - pos[j]
            dr = analyze.minimum_image_disp(dr, L, mode="slab")
            r_xy = np.linalg.norm(dr[:2])
            dists_2d.append(r_xy)
    return np.array(dists_2d)

def pair_correlation_2d(dists_2d: np.ndarray, N_atoms: int, nbins: int, dr: float, L):
    Lx, Ly, _ = analyze._as_box(L)
    Area = Lx * Ly
    r_max = min(Lx, Ly) / 2.0
    nbins_actual = int(min(nbins * dr, r_max) / dr)
    hist, edges = np.histogram(dists_2d, bins=nbins_actual, range=(0.0, nbins_actual * dr))
    r = 0.5 * (edges[:-1] + edges[1:])
    dArea = np.pi * ((r + 0.5 * dr) ** 2 - (r - 0.5 * dr) ** 2)
    rho_2D = N_atoms / Area
    ideal_hist = ((N_atoms - 1) / 2) * rho_2D * dArea 
    ideal_hist = np.maximum(ideal_hist, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_2d = hist / ideal_hist
        g_2d = np.nan_to_num(g_2d, nan=0.0, posinf=0.0, neginf=0.0)
    return g_2d, r

# --- STRUCTURAL ANALYSIS DRIVER ---

def run_structural_analysis(pos: np.ndarray, L, N: int, mode: str, is_initial: bool, out_dir: str, nbins: int, dr: float, maxk: int, inplane: bool):
    time_label = "Initial (t=0)" if is_initial else "Equilibrated (t_final)"
    g_dim_label = "3D (Volume)"
    
    if mode == "slab" and inplane:
        dists = get_dists_in_plane(pos, L)
        g, r = pair_correlation_2d(dists, N, nbins, dr, L)
        g_dim_label = "2D (In-Plane Area)"
    else:
        disp_table = analyze.displacement_table(pos, L, mode=mode)
        dij = analyze.distance_table(disp_table)
        N_particles = dij.shape[0]
        dlist = dij[np.triu_indices(N_particles, k=1)] 
        g, r = analyze.pair_correlation(dlist, N, nbins, dr, L)
        if mode == "slab":
             g_dim_label = "3D (Full Box)"

    kvecs = analyze.legal_kvecs(maxk, L)
    kmod, avsk = analyze.calc_av_sk(kvecs, pos)
    mask = kmod > 1e-4
    kmod = kmod[mask]
    avsk = avsk[mask]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(r, g, linewidth=2)
    ax[0].axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax[0].set_xlabel(r'Distance $r$ ($\sigma$)')
    ax[0].set_ylabel(r'$g(r)$')
    ax[0].set_title(f'{mode.upper()} - $g(r)$ ({g_dim_label}) {time_label}')
    ax[0].set_ylim(0, max(2.5, np.max(g) * 1.1)) 
    
    ax[1].plot(kmod, avsk, linewidth=2, color='r')
    ax[1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax[1].set_xlabel(r'Wavevector $k$ ($\sigma^{-1}$)')
    ax[1].set_ylabel(r'$S(k)$')
    ax[1].set_title(f'{mode.upper()} - $S(k)$ {time_label}')
    ax[1].set_xlim(0, maxk)
    ax[1].set_ylim(0, max(2.5, np.max(avsk) * 1.1)) 
    plt.tight_layout()
    
    filename = f"structure_{mode}_{time_label.replace(' ', '_').replace('=', '')}"
    analyze.savefig(out_dir, filename)

def analyze_structural_only(mode: str, data_dir: str, out_dir: str, rc: float, nbins: int, dr: float, maxk: int, inplane: bool, n_atoms: int, l_ref: float):
    """
    Wrapper function called by main.py.
    NOW ACCEPTS n_atoms and l_ref as ARGUMENTS.
    """
    mode = mode.lower()
    gro_path = os.path.join(data_dir, f"argon_{mode}_traj.gro")
    frames = read_gro_trajectory(gro_path, n_atoms, l_ref)

    if not frames:
        print(f"ERROR: No frames loaded from {gro_path}. Cannot perform structural analysis.")
        return

    for title, pos, L, idx in frames:
        is_initial = (idx == 0)
        run_structural_analysis(pos, L, n_atoms, mode, is_initial, out_dir, nbins, dr, maxk, inplane)