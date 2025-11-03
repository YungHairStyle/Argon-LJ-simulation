import numpy as np
from numpy.random import default_rng

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
    Wrap particle positions into the simulation cell.
    - bulk: wrap x,y,z into [0, L)
    - slab: wrap x,y into [0, Lx/Ly); leave z unchanged
    pos: (N,3)
    L: scalar or (Lx, Ly, Lz)
    """
    Lx, Ly, Lz = _as_box(L)
    p = np.array(pos, dtype=float, copy=True)
    # Wrap x,y
    p[:, 0] -= Lx * np.floor(p[:, 0] / Lx)
    p[:, 1] -= Ly * np.floor(p[:, 1] / Ly)
    if mode == "bulk":
        p[:, 2] -= Lz * np.floor(p[:, 2] / Lz)
    return p


def minimum_image_disp(drij, L, mode="bulk"):
    """
    Apply minimum image to displacement vectors.
    - bulk: componentwise minimum-image in x,y,z
    - slab: minimum-image only in x,y; z left as is
    drij: (...,3)
    L: scalar or (Lx, Ly, Lz)
    """
    Lx, Ly, Lz = _as_box(L)
    d = np.array(drij, dtype=float, copy=True)
    d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    if mode == "bulk":
        d[..., 2] -= Lz * np.round(d[..., 2] / Lz)
    return d


#############################
# State construction         #
#############################

def cubic_lattice(tiling, L):
    """
    Return coordinates on a cubic lattice centered at cell middle.
    - L: scalar box length; for slab you typically still seed in a cube.
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
    """Maxwell-Boltzmann draw with zero COM and exact temperature."""
    v = np.random.normal(0.0, 1.0, size=(N, 3))
    v -= v.mean(axis=0, keepdims=True)
    K = 0.5 * m * np.einsum("ij,ij->", v, v)
    Tcur = (2.0 / (3.0 * N)) * K
    if Tcur > 0:
        v *= np.sqrt(T / Tcur)
    return v


def get_temperature(mass, velocities):
    N = len(velocities)
    dof = 3 * N
    total_vsq = np.einsum("ij,ij", velocities, velocities)
    return mass * total_vsq / dof


#############################
# Tables & observables       #
#############################

def displacement_table(coordinates, L, mode="bulk"):
    r = np.asarray(coordinates, dtype=float)
    table = r[:, np.newaxis, :] - r[np.newaxis, :, :]
    return minimum_image_disp(table, L, mode)


def distance_table(disp):
    return np.linalg.norm(disp, axis=-1)


def kinetic(m, v):
    total_vsq = np.einsum("ij,ij", v, v)
    return 0.5 * m * total_vsq


def potential(dist, rc):
    """LJ 12-6 with energy shift to zero at rc. All-pairs O(N^2)."""
    r = np.array(dist, dtype=float, copy=True)
    n = r.shape[0]
    r[np.diag_indices(n)] = np.inf
    # guard tiny
    r = np.maximum(r, 1e-12)
    v = 4.0 * (r ** -12 - r ** -6)
    vc = 4.0 * (rc ** -12 - rc ** -6)
    v[r < rc] -= vc  # shift
    v[r >= rc] = 0.0
    return 0.5 * np.sum(v)


def force(disp, dist, rc):
    """
    Compute LJ forces from displacement & distance tables.
    Returns (N,3).
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
    """Velocity-Verlet step with variable box style (bulk/slab)."""
    acc = force(disp, dist, rc) / mass
    v_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * v_half
    pos_new = wrap_positions(pos_new, L, mode)
    disp_new = displacement_table(pos_new, L, mode)
    dist_new = distance_table(disp_new)
    # avoid zero distances in next force call
    dist_new = np.maximum(dist_new, 1e-12)
    acc_new = force(disp_new, dist_new, rc) / mass
    v_new = v_half + 0.5 * dt * acc_new
    return pos_new, v_new, disp_new, dist_new


#############################
# g(r), S(k), k-vectors     #
#############################

def pair_correlation(dists, natom, nbins, dr, L):
    """Pair correlation g(r) using ideal-gas normalization.
    L can be scalar or (Lx, Ly, Lz) for volume.
    """
    Lx, Ly, Lz = _as_box(L)
    Omega = Lx * Ly * Lz
    hist, edges = np.histogram(dists, bins=nbins, range=(0.0, nbins * dr))
    r = (edges[:-1] + edges[1:]) * 0.5
    dOmega = (4.0 * np.pi / 3.0) * ((r + 0.5 * dr) ** 3 - (r - 0.5 * dr) ** 3)
    ideal = ((natom - 1) / 2.0) * (natom / Omega) * dOmega
    with np.errstate(divide="ignore", invalid="ignore"):
        g = hist / ideal
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g, r


def calc_rhok(kvecs, pos):
    arg = kvecs @ pos.T
    return np.exp(-1j * arg).sum(axis=1)


def calc_sk(kvecs, pos):
    rho_k = calc_rhok(kvecs, pos)
    rho_mk = calc_rhok(-kvecs, pos)
    N = pos.shape[0]
    return (rho_k * rho_mk) / N


def calc_av_sk(kvecs, pos):
    sk = np.real(calc_sk(kvecs, pos))
    nk = np.linalg.norm(kvecs, axis=1)
    uniq, inv = np.unique(np.round(nk, 12), return_inverse=True)
    av = np.zeros(len(uniq))
    for i in range(len(uniq)):
        av[i] = np.mean(sk[inv == i])
    return uniq, av


def legal_kvecs(maxn, L):
    Lx, Ly, Lz = _as_box(L)
    grid = np.arange(-maxn, maxn + 1)
    k = np.array([(i, j, k) for i in grid for j in grid for k in grid], dtype=float)
    k[:, 0] *= 2.0 * np.pi / Lx
    k[:, 1] *= 2.0 * np.pi / Ly
    k[:, 2] *= 2.0 * np.pi / Lz
    return k


#############################
# Thermostats                #
#############################

def thermostat_andersen(v, m, T, dt, nu):
    """Andersen thermostat: resample with prob p = 1-exp(-nu*dt)."""
    rng = default_rng()
    p = 1.0 - np.exp(-nu * dt)
    N, ndim = v.shape
    mask = rng.random(N) < p
    v_new = v.copy()
    v_new[mask, :] = rng.normal(0.0, np.sqrt(T / m), size=(mask.sum(), ndim))
    # remove COM drift
    v_new -= v_new.mean(axis=0, keepdims=True)
    return v_new


def thermostat_stochastic(v, m, T, prob):
    """Simple per-particle resampling used in earlier versions."""
    rng = default_rng()
    N, ndim = v.shape
    v_new = v.copy()
    mask = rng.random(N) < prob
    v_new[mask, :] = rng.normal(0.0, np.sqrt(T / m), size=(mask.sum(), ndim))
    return v_new


#############################
# Distances utility (O(N^2)) #
#############################

def my_disp_in_box(drij, L, mode="bulk"):
    """Compatibility helper: same behavior as minimum_image_disp."""
    return minimum_image_disp(drij, L, mode)


def all_dists(pos, L, mode="bulk"):
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
    # last block is the tail starting at (nblocks-1)*blocklen
    means[nblocks - 1, :] = tseries[(nblocks - 1) * blocklen :, :].mean(axis=0)
    mean = means.mean(axis=0)
    err = means.std(axis=0, ddof=1) / np.sqrt(nblocks)
    return mean, err
