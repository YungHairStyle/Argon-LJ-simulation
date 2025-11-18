import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_density_profile(gro_file, bin_width=0.2):
    """
    Reads a .gro file and calculates density profiles for start, middle, and end frames.
    """
    # Constant based on your main.py setting
    SAMPLE_EVERY = 5 
    
    # specific frames to capture
    frames_data = []
    
    with open(gro_file, 'r') as f:
        lines = f.readlines()

    # Parse the .gro file structure
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
    labels = ["Start", "Middle", "End"]
    colors = ["red", "orange", "blue"]

    plt.figure(figsize=(8, 6))

    # Loop through targets and process
    for i, frame_idx in enumerate(target_indices):
        start_line = frame_idx * lines_per_frame
        
        # Calculate step number for the legend
        step_number = frame_idx * SAMPLE_EVERY 

        # Extract Z coordinates
        z_coords_raw = []
        
        # Read atom lines
        for j in range(natoms):
            line = lines[start_line + 2 + j]
            parts = line.split()
            z = float(parts[6]) 
            z_coords_raw.append(z)
            
        # Read box dimensions
        box_line = lines[start_line + natoms + 2]
        box_dims = [float(x) for x in box_line.split()]
        Lx, Ly, Lz = box_dims[0], box_dims[1], box_dims[2]
        area = Lx * Ly
        
        # ------------------------------------------------------------
        # BULK MODIFICATION: NO VISUAL SHIFT APPLIED.
        # The z_coords are used as-is, assuming full periodic wrapping.
        # ------------------------------------------------------------
        z_coords = np.array(z_coords_raw)

        # Create Histogram (Density Calculation)
        bins = np.arange(0, Lz + bin_width, bin_width)
        hist, bin_edges = np.histogram(z_coords, bins=bins)
        
        # Convert count to number density: rho = N / (Area * bin_width)
        volume_slice = area * bin_width
        rho = hist / volume_slice
        
        # Center the bins for plotting
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Plot
        legend_label = f"Step {step_number:,} ({labels[i]})"
        plt.plot(bin_centers, rho, label=legend_label, color=colors[i], linewidth=2)

    plt.xlabel("Z-position ($\sigma$)")
    plt.ylabel("Number Density $\\rho(z)$")
    plt.title("Density Profile along Z-axis (Bulk Evolution)") # Title changed
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot (Updated filename/path for bulk)
    save_path = Path(gro_file).parent.parent / "figures" / "bulk" / "density_profile_bulk.png" 
    plt.savefig(save_path)
    print(f"Density profile saved to: {save_path}")
    plt.show()

# --- Run the function ---
# Update path to bulk trajectory file
gro_path = r"..\data\argon_bulk_traj.gro" 
calculate_density_profile(gro_path)