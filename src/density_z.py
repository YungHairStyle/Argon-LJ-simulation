import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_density_profile(gro_file, bin_width=0.2):
    """
    Reads a .gro file and calculates density profiles for start, middle, and end frames.
    """
    # --- ADDED: Constant based on your main.py setting ---
    SAMPLE_EVERY = 5 
    
    # specific frames to capture
    frames_data = []
    
    with open(gro_file, 'r') as f:
        lines = f.readlines()

    # Parse the .gro file structure
    # A frame consists of: Title line, Atom count line, Atom lines, Box line
    
    # Get number of atoms from second line
    try:
        natoms = int(lines[1].strip())
    except ValueError:
        print("Error reading number of atoms. Check .gro format.")
        return

    lines_per_frame = natoms + 3
    total_lines = len(lines)
    total_frames = total_lines // lines_per_frame
    
    print(f"Found {total_frames} frames in trajectory.")

    # Indices of frames we want: Start (0), Middle, End
    target_indices = [0, total_frames // 2, total_frames - 1]
    labels = ["Start", "Middle", "End"] # Simplified labels for cleaner output
    colors = ["red", "orange", "blue"]

    plt.figure(figsize=(8, 6))

    # Loop through targets and process
    for i, frame_idx in enumerate(target_indices):
        start_line = frame_idx * lines_per_frame
        
        # Calculate step number for the legend
        step_number = frame_idx * SAMPLE_EVERY 

        # Extract Z coordinates (column 6 in standard .gro, index 5 in 0-based split)
        # .gro format is fixed width, but split() usually works for simple files
        z_coords_raw = []
        
        # Read atom lines
        for j in range(natoms):
            line = lines[start_line + 2 + j]
            parts = line.split()
            # Z is usually the last coordinate. 
            # parts: [resid, resname, atomname, atomnum, x, y, z] (sometimes vx, vy, vz)
            # We assume standard position:
            z = float(parts[6]) 
            z_coords_raw.append(z)
            
        # Read box dimensions to get Area (Lx * Ly)
        box_line = lines[start_line + natoms + 2]
        box_dims = [float(x) for x in box_line.split()]
        Lx, Ly, Lz = box_dims[0], box_dims[1], box_dims[2]
        area = Lx * Ly
        
        # ------------------------------------------------------------
        # MODIFICATION: Apply Constant Visual Shift to center the plot.
        # Observation: Slab peaks around z=4, but Lz/2 is 6. 
        # Shift entire distribution by 6 - 4 = 2 sigma units.
        z_coords = np.array(z_coords_raw)
        VISUAL_SHIFT = 2.0 
        z_coords_shifted = (z_coords + VISUAL_SHIFT) % Lz
        # ------------------------------------------------------------

        # Create Histogram (Density Calculation)
        bins = np.arange(0, Lz + bin_width, bin_width)
        hist, bin_edges = np.histogram(z_coords_shifted, bins=bins)
        
        # Convert count to number density: rho = N / (Area * bin_width)
        volume_slice = area * bin_width
        rho = hist / volume_slice
        
        # Center the bins for plotting
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # --- MODIFIED: Legend now includes step number ---
        legend_label = f"Step {step_number:,} ({labels[i]})"
        plt.plot(bin_centers, rho, label=legend_label, color=colors[i], linewidth=2)

    plt.xlabel("Z-position ($\sigma$)")
    plt.ylabel("Number Density $\\rho(z)$")
    plt.title("Density Profile along Z-axis (Slab Evolution)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    save_path = Path(gro_file).parent.parent / "figures" / "slab" / "density_profile.png"
    plt.savefig(save_path)
    print(f"Density profile saved to: {save_path}")
    plt.show()

# --- Run the function ---
# Update path if necessary to match your actual data location
gro_path = r"..\data\argon_slab_traj.gro" 
calculate_density_profile(gro_path)