import numpy as np
import csv, time, os

#############################
# Box & minimum-image tools #
#############################

def _as_box(L):
    """Return (Lx, Ly, Lz) given scalar or iterable L."""
    try:
        Lx, Ly, Lz = L  # iterable
        return float(Lx), float(Ly), float(Lz)
    except Exception:
        return float(L), float(L), float(L)


def wrap_positions(pos, L, mode="bulk"):
    """
    Set periodic boundary conditions. Wrap particle positions into the simulation cell.
    - bulk: wrap x,y,z into [0, L)
    - slab: wrap x,y into [0, Lx/Ly); leave z unchanged
    pos: (N,3)
    L: side length
    """
    Lx, Ly, Lz = _as_box(L)
    p = np.array(pos, dtype=float, copy=True)
    # Wrap x,y
    p[:, 0] -= Lx * np.floor(p[:, 0] / Lx)
    p[:, 1] -= Ly * np.floor(p[:, 1] / Ly)
    if mode == "bulk":
        p[:, 2] -= Lz * np.floor(p[:, 2] / Lz)  # wrap z
    return p


def minimum_image_disp(drij, L, mode="bulk"):
    """
    Apply minimum image to displacement vectors.
    - bulk: componentwise minimum-image in x,y,z
    - slab: minimum-image only in x,y; z left as is
    drij: (...,3)
    L: side length
    """
    Lx, Ly, Lz = _as_box(L)
    d = np.array(drij, dtype=float, copy=True)
    d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    if mode == "bulk":
        d[..., 2] -= Lz * np.round(d[..., 2] / Lz)
    return d


#############################
# FCC geometry & I/O        #
#############################

FCC_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=float,
)


def fcc_bulk_positions(n_cells, a):
    """
    Return (positions, (Lx, Ly, Lz)) for a bulk FCC cube of n_cells per axis.
    """
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


def fcc_slab_positions(n_x, n_y, n_layers_z, a, Lz):
    """
    Return (positions, (Lx, Ly, Lz)) for an FCC slab with vacuum along z.
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
    return np.array(R, float), (Lx, Ly, float(Lz))


def write_gro(path, pos, box, title="frame"):
    """Minimal .gro writer without velocities."""
    Lx, Ly, Lz = _as_box(box)
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{len(pos):5d}\n")
        for i, (x, y, z) in enumerate(pos, start=1):
            # Fake residue/atom labels
            f.write(f"{1:5d}{'AR':>5s}{'Ar':>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"   {Lx:8.5f} {Ly:8.5f} {Lz:8.5f}\n")

def write_gro_frame(f, pos, box, title="frame"):
    """Write a single frame to an already-open .gro trajectory file."""
    Lx, Ly, Lz = _as_box(box)
    f.write(f"{title}\n")
    f.write(f"{len(pos):5d}\n")
    for i, (x, y, z) in enumerate(pos, start=1):
        f.write(f"{1:5d}{'AR':>5s}{'Ar':>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
    f.write(f"   {Lx:8.5f} {Ly:8.5f} {Lz:8.5f}\n")


#############################
# State construction         #
#############################

def cubic_lattice(tiling, L):
    """
    required for: initialization

    args:
        tiling (int): determines number of coordinates,
        by tiling^3
        L (float): side length of simulation box
    returns:
        array of shape (tiling**3, 3): coordinates on a cubic lattice,
        all between -0.5L and 0.5L
    """
    coords = []
    for x in range(tiling):
        for y in range(tiling):
            for z in range(tiling):
                coords.append([x, y, z])
    coord = np.array(coords, dtype=float) / tiling
    coord -= 0.5
    return coord * float(L)


def initial_velocities(N, m, T):
    """
    initialize velocities at a desired temperature
    required for: initialization
    """
    velocities = np.random.rand(N, 3)
    # center velocities
    new_v = velocities - 0.5
    # zero the total velocity
    total_v = np.sum(new_v, axis=0)
    new_v -= total_v / N
    # get the right temperature
    current_temp = get_temperature(m, new_v)
    factor = np.sqrt(T / current_temp)
    new_v *= factor
    return new_v


def get_temperature(mass, velocities):
    """
    calculates the instantaneous temperature
    required for: initial_velocities()
    """
    N = len(velocities)
    dof = 3 * N
    total_vsq = np.einsum("ij,ij", velocities, velocities)
    return mass * total_vsq / dof


#############################
# Tables & observables       #
#############################

def displacement_table(coordinates, L, mode="bulk"):
    """
    required for: force(), advance()
    """
    table = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    return minimum_image_disp(table, L, mode)


def distance_table(disp):
    return np.linalg.norm(disp, axis=-1)


def kinetic(m, v):
    """
    required for measurement
    """
    total_vsq = np.einsum("ij,ij", v, v)
    return 0.5 * m * total_vsq


def potential(dist, rc):
    """
    LJ 12-6 with energy shift to zero at rc. All-pairs O(N^2).
    """
    r = np.copy(dist)
    r[np.diag_indices(len(r))] = np.inf
    v = 4 * np.power(r, -6) * (np.power(r, -6) - 1)
    vc = 4 * np.power(rc, -6) * (np.power(rc, -6) - 1)
    v[r < rc] -= vc  # shift
    v[r >= rc] = 0   # cut
    return 0.5 * np.sum(v)


def force(disp, dist, rc):
    """
    Compute forces from LJ potential.
    required for: advance()
    """
    r = np.array(dist, dtype=float, copy=True)
    n = r.shape[0]
    r[np.diag_indices(n)] = np.inf
    r = np.maximum(r, 1e-12)
    # |F| = 24*(2/r^14 - 1/r^8)
    mag = 24.0 * (2.0 / r ** 14 - 1.0 / r ** 8)
    mag[r >= rc] = 0.0
    f = np.sum(mag[:, :, None] * disp, axis=1)
    return f


def advance(pos, vel, mass, dt, disp, dist, rc, L, mode="bulk"):
    """
    Velocity-Verlet step with variable box style (bulk/slab).
    """
    acc = force(disp, dist, rc) / mass
    v_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * v_half
    pos_new = wrap_positions(pos_new, L, mode)
    disp_new = displacement_table(pos_new, L, mode)
    dist_new = distance_table(disp_new)
    dist_new = np.maximum(dist_new, 1e-12)
    acc_new = force(disp_new, dist_new, rc) / mass
    v_new = v_half + 0.5 * dt * acc_new
    return pos_new, v_new, disp_new, dist_new


#############################
# g(r), S(k), k-vectors     #
#############################

def pair_correlation(dists, natom, nbins, dr, L):
    """Calculate the pair correlation function g(r)."""
    Lx, Ly, Lz = _as_box(L)
    Omega = Lx * Ly * Lz
    histogram = np.histogram(dists, bins=nbins, range=(0, nbins * dr))
    r = (histogram[1] + dr / 2)[:-1]  # centers of the bins
    dOmega = ((4 * np.pi) / 3) * ((r + (dr / 2)) ** 3 - (r - (dr / 2)) ** 3)
    idealhist = ((natom - 1) / 2) * (natom / Omega) * dOmega
    g = histogram[0] / idealhist
    return g, r


def calc_rhok(kvecs, pos):
    """Fourier transform of particle density."""
    arg = np.dot(kvecs, pos.T)
    return np.exp(-1j * arg).sum(axis=1)


def calc_sk(kvecs, pos):
    """
    Calculate the structure factor S(k).
    """
    rho_k = calc_rhok(kvecs, pos)
    rho_mk = calc_rhok(-kvecs, pos)
    N = pos.shape[0]
    return (rho_k * rho_mk) / N


def calc_av_sk(kvecs, pos):
    """
    Average structure factor over shells of |k|.
    """
    sk = np.real(calc_sk(kvecs, pos))
    nk = np.linalg.norm(kvecs, axis=1)
    uniq, inv = np.unique(np.round(nk, 12), return_inverse=True)
    av = np.zeros(len(uniq))
    for i in range(len(uniq)):
        av[i] = np.mean(sk[inv == i])
    return uniq, av


def legal_kvecs(maxn, L):
    """Calculate k vectors commensurate with a rectangular box."""
    Lx, Ly, Lz = _as_box(L)
    grid = np.arange(-maxn, maxn + 1)
    k = np.array([(i, j, k) for i in grid for j in grid for k in grid], dtype=float)
    k[:, 0] *= 2.0 * np.pi / Lx
    k[:, 1] *= 2.0 * np.pi / Ly
    k[:, 2] *= 2.0 * np.pi / Lz
    return k


#############################
# Thermostat                #
#############################

def thermostat_andersen(v, m, T, prob):
    """
    Apply Andersen thermostat.
    """
    N = v.shape[0]
    v_new = np.copy(v)
    sigma = np.sqrt(T / m)
    for i in range(N):
        if np.random.rand() < prob:
            v_new[i, :] = np.random.normal(loc=0.0, scale=sigma, size=3)
    return v_new


#############################
# Distances utility (O(N^2)) #
#############################

def my_disp_in_box(drij, L, mode="bulk"):
    """Impose minimum image condition on displacement vector."""
    return minimum_image_disp(drij, L, mode)


def all_dists(pos, L, mode="bulk"):
    """
    get all the pairwise distances between a list of positions
    """
    N = pos.shape[0]
    dists = np.zeros(N * (N - 1) // 2, dtype=float)
    cur = 0
    for i in range(N):
        for j in range(i + 1, N):
            dr = pos[i] - pos[j]
            dr = minimum_image_disp(dr, L, mode)
            dists[cur] = np.linalg.norm(dr)
            cur += 1
    return dists


#############################
# Statistics                 #
#############################

def block_average(tseries, nblocks=5):
    """
    calculate the block average of a time series 
    """
    tseries = np.asarray(tseries)
    if tseries.ndim == 1:
        tseries = tseries[:, None]
    T, M = tseries.shape
    blocklen = int(T / nblocks)
    if blocklen < 1:
        raise ValueError("Not enough samples for the requested number of blocks")
    means = np.zeros((nblocks, M))
    for i in range(nblocks - 1):
        means[i, :] = tseries[i * blocklen : (i + 1) * blocklen, :].mean(axis=0)
    means[nblocks - 1, :] = tseries[(nblocks - 1) * blocklen :, :].mean(axis=0)
    mean = means.mean(axis=0)
    err = means.std(axis=0, ddof=1) / np.sqrt(nblocks)
    return mean, err


#############################
# High-level MD driver      #
#############################

def run_md(
    mode,
    cells_x,
    cells_y,
    layers_z,
    cells_bulk,
    a,
    T,
    mass,
    rc,
    dt,
    steps,
    equil_steps,
    prob,
    seed,
    sample_every,
    data_dir,
    Lz=None,
    title=None,
):
    """
    High-level MD driver. Writes:
      - thermo_{mode}.csv
      - argon_{mode}.gro
    into data_dir.
    """
    rng = np.random.default_rng(seed)
    mode = mode.lower()
    if mode == "bulk":
        r, box = fcc_bulk_positions(cells_bulk, a)
        slab_mode = "bulk"
    elif mode == "slab":
        if Lz is None:
            raise ValueError("Lz must be provided for slab simulations")
        r, box = fcc_slab_positions(cells_x, cells_y, layers_z, a, Lz)
        slab_mode = "slab"
    else:
        raise ValueError("mode must be 'bulk' or 'slab'")

    if title is None:
        title = f"Argon-{mode}"

    # Slight jitter
    r += 1e-6 * rng.normal(size=r.shape)

    # Wrap initial positions
    r = wrap_positions(r, box, slab_mode)

    # Velocities
    v = initial_velocities(len(r), mass, T)

    # Precompute tables
    disp = displacement_table(r, box, slab_mode)
    dist = distance_table(disp)

    times, epots, ekins, temps = [], [], [], []

    def apply_thermostat(vv):
        if prob > 0.0:
            return thermostat_andersen(vv, mass, T, prob=prob)
        return vv

    t0 = time.time()
    next_progress = max(1000, steps // 20)

    N = len(r)
    Lx, Ly, Lz_box = _as_box(box)
    print(f"[info] MODE={mode}  N={N}  box=({Lx:.3f}, {Ly:.3f}, {Lz_box:.3f})")

    # open trajectory file for multiple frames
    traj_path = os.path.join(data_dir, f"argon_{mode}_traj.gro")
    with open(traj_path, "w") as traj_file:

        for step in range(steps):
            r, v, disp, dist = advance(r, v, mass, dt, disp, dist, rc, box, slab_mode)
            v = apply_thermostat(v)

            if step % sample_every == 0:
                U = potential(dist, rc)
                K = kinetic(mass, v)
                times.append(step * dt)
                epots.append(U)
                ekins.append(K)
                temps.append((2.0 * K) / (3.0 * N))

                # write one frame to trajectory
                write_gro_frame(traj_file, r, box, title=f"{title} step {step}")

            if (step + 1) % next_progress == 0:
                elapsed = time.time() - t0
                Tinst = temps[-1] if temps else float("nan")
                En = (epots[-1] + ekins[-1]) / N if epots else float("nan")
                print(f"[{step+1:>7d}/{steps}] T={Tinst:.3f}  E/N={En:.3f}  elapsed={elapsed:.1f}s")

    print("[info] Run complete.")

    # Post-run summaries (printed only)
    if temps:
        eq_idx = max(0, equil_steps // sample_every)
        if eq_idx < len(temps):
            T_mean = float(np.mean(temps[eq_idx:]))
            Epot_mean = float(np.mean(epots[eq_idx:]))
            Ekin_mean = float(np.mean(ekins[eq_idx:]))
            print(f"[summary] <T> = {T_mean:.4f}, <E_pot> = {Epot_mean:.4f}, <E_kin> = {Ekin_mean:.4f}")
        else:
            print("[summary] Not enough sampled points to compute post-equil averages.")

    # Save outputs
    os.makedirs(data_dir, exist_ok=True)
    save_gro = os.path.join(data_dir, f"argon_{mode}.gro")
    save_thermo = os.path.join(data_dir, f"thermo_{mode}.csv")

    write_gro(save_gro, r, box, title=title)
    print(f"[save] Wrote {save_gro}")

    with open(save_thermo, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "E_pot", "E_kin", "T"])
        for row in zip(times, epots, ekins, temps):
            w.writerow(row)
    print(f"[save] Wrote {save_thermo}")

    return {
        "mode": mode,
        "N": N,
        "box": box,
        "thermo_path": save_thermo,
        "gro_path": save_gro,
    }
