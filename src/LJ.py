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
        p[:, 2] -= Lz * np.floor(p[:, 2] / Lz) #wrap z
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

    args:
        N (int): number of particles
        m (float): mass of particles
        T (float): desired temperature
    returns:
        array: initial velocities, with shape (N, 3)
    """
    velocities = np.random.rand(N, 3)
    #center velocities
    new_v = velocities - 0.5
    #zero the total velocity
    total_v = np.sum(new_v, axis=0)
    new_v -= total_v / N
    #get the right temperature
    current_temp = get_temperature(m, new_v)
    factor  = np.sqrt(T / current_temp)
    new_v *= factor
    return new_v


def get_temperature(mass, velocities):
    """
    calculates the instantaneous temperature
    required for: initial_velocities()
    
    args:
        mass (float): mass of particles;
        it is assumed all particles have the same mass
        velocities (array): velocities of particles,
        assumed to have shape (N, 3)
    returns:
        float: temperature according to equipartition
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

    args:
        coordinates (array): coordinates of particles,
        assumed to have shape (N, 3)
        e.g. coordinates[3,0] should give the x component
        of particle 3
        L (float): side length of cubic box,
        must be known in order to compute minimum image
    returns:
        array: table of displacements r
        such that r[i,j] is the minimum image of
        coordinates[i] - coordinates[j]
    """
    #r = np.asarray(coordinates, dtype=float)
    table = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    return minimum_image_disp(table, L, mode)


def distance_table(disp):
    return np.linalg.norm(disp, axis=-1)


def kinetic(m, v):
    """
    required for measurement

    args:
        m (float): mass of particles
        v (array): velocities of particles,
        assumed to be a 2D array of shape (N, 3)
    returns:
        float: total kinetic energy
    """
    total_vsq = np.einsum("ij,ij", v, v)
    return 0.5 * m * total_vsq


def potential(dist, rc):
    """
    LJ 12-6 with energy shift to zero at rc. All-pairs O(N^2). 
    Required for measurement.

    args:
        dist (array): distance table with shape (N, N)
        i.e. dist[i,j] is the distance
        between particle i and particle j
        in the minimum image convention
        note that the diagonal of dist can be zero
        rc (float): cutoff distance for interaction
        i.e. if dist[i,j] > rc, the pair potential between
        i and j will be 0
    returns:
        float: total potential energy
    """
    r = np.copy(dist)
    r[np.diag_indices(len(r))] = np.inf
    v = 4*np.power(r, -6)*(np.power(r, -6) - 1)
    vc = 4*np.power(rc, -6)*(np.power(rc, -6) - 1)
    v[r < rc] -= vc #shift
    v[r >= rc] = 0 #cut
    return 0.5*np.sum(v)


def force(disp, dist, rc):
    """
    Compute forces form LJ potential.
    required for: advance()

    args:
        disp (array): displacement table,
        with shape (N, N, 3)
        dist (array): distance table, with shape (N, N)
        can be calculated from displacement table,
        but since there is a separate copy available
        it is just passed in here
        rc (float): cutoff distance for interaction
        i.e. if dist[i,j] > rc, particle i will feel no force
        from particle j
    returns:
        array: forces f on all particles, with shape (N, 3)
        i.e. f[3,0] gives the force on particle i
        in the x direction
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
    Advance system according to velocity verlet

    args:
        pos (array): coordinates of particles
        val (array): velocities of particles
        mass (float): mass of particles
        dt (float): timestep by which to advance
        disp (array): displacement table
        dist (array): distance table
        rc (float): cutoff
        L (float): length of cubic box
    returns:
        array, array, array, array:
        new positions, new velocities, new displacement table,
        and new distance table
    """
    acc = force(disp, dist, rc) / mass
    v_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * v_half
    pos_new = wrap_positions(pos_new, L, mode)
    disp_new = displacement_table(pos_new, L, mode)
    dist_new = distance_table(disp_new)
    # avoid zero distances in next force call
    dist_new = np.maximum(dist_new, 1e-12)
    #repeat force calculation for new pos
    acc_new = force(disp_new, dist_new, rc) / mass
    v_new = v_half + 0.5 * dt * acc_new
    return pos_new, v_new, disp_new, dist_new


#############################
# g(r), S(k), k-vectors     #
#############################

def pair_correlation(dists, natom, nbins, dr, L):
    """ Calculate the pair correlation function g(r).

    Args:
        dists (np.array): 1d array of pair distances
        natom (int): number of atoms
        nbins (int): number of bins to histogram
        dr (float): size of bins
        L (float): scalar or (Lx, Ly, Lz)
    Return:
        array of shape (nbins,): the pair correlation g(r)
    """
    Lx, Ly, Lz = _as_box(L)
    Omega = Lx * Ly * Lz
    histogram = np.histogram(dists, bins=nbins, range=(0, nbins*dr))
    r = (histogram[1] + dr/2)[:-1] # centers of the bins
    dOmega = ((4*np.pi)/3)*((r+(dr/2))**3-(r-(dr/2))**3)
    idealhist = ((natom-1)/2)*(natom/Omega)*dOmega
    g = histogram[0] / idealhist 
    
    return g,r


def calc_rhok(kvecs, pos):
    """ 
    Calculate the fourier transform of particle density.

    Args:
        kvecs (np.array): array of k-vectors, shape (nk, ndim)
        pos (np.array): particle positions, shape (natom, ndim)
    Return:
        array of shape (nk,): fourier transformed density rho_k
    """ 
    arg = np.dot(kvecs,pos.T)
    return np.exp(-1j * arg).sum(axis=1)


def calc_sk(kvecs, pos):
    """
    Calculate the structure factor S(k).

    Args:
        kvecs (np.array): array of k-vectors, shape (nk, ndim)
        pos (np.array): particle positions, shape (natom, ndim)
    Return:
        array of shape (nk,): structure factor s(k)
    """
    rho_k = calc_rhok(kvecs, pos)
    rho_mk = calc_rhok(-kvecs, pos)
    N = pos.shape[0]
    return (rho_k * rho_mk) / N


def calc_av_sk(kvecs, pos):
    """
    Calculates the average structure factor over all k.

     Args:
        kvecs (np.array): Array of k-vectors with shape (nk, 3)
        pos (np.array): Particle positions with shape (N, 3)

    Returns:
        uniq: np.array of unique k magnitudes (|k| values)
        av: np.array of corresponding averaged structure factor S(k)
        values, averaged over all k-vectors with the same |k|.
    """
    sk = np.real(calc_sk(kvecs, pos))
    nk = np.linalg.norm(kvecs, axis=1)
    uniq, inv = np.unique(np.round(nk, 12), return_inverse=True)
    av = np.zeros(len(uniq))
    for i in range(len(uniq)):
        av[i] = np.mean(sk[inv == i])
    return uniq, av


def legal_kvecs(maxn, L):

    """ Calculate k vectors commensurate with a cubic box.

    Consider only k vectors in the all-positive octant of reciprocal space.

    Args:
        maxn : maximum value for nx, ny, nz; maxn+1 is number of k-points along each axis
        L : side length of cubic cell

    Return:
        array of shape (nk, 3): collection of k vectors
        
    """
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

def thermostat_andersen(v, m, T, prob):
    """
    Apply Andersen thermostat.

    Args:
    v : ndarray, shape (N, 3)
        Current particle velocities
    m : float
        mass
    T : float
        Target temperature
    prob : float
        Probability of collision for each particle per step (default 1%).

    Returns
    v_new : ndarray, shape (N, 3)
        Updated velocities after applying thermostat collisions.
    """
    
    N = v.shape[0]
    v_new = np.copy(v)
    
    # Standard deviation for Maxwell-Boltzmann distribution
    sigma = np.sqrt(T / m)

    # Loop over all particles
    for i in range(N):
        if np.random.rand() < prob:
            # assign new velocity from Maxwell-Boltzmann after a collision 
            v_new[i, :] = np.random.normal(loc=0.0, scale=sigma, size=3) #draw random number from distribution
    return v_new


#############################
# Distances utility (O(N^2)) #
#############################

def my_disp_in_box(drij, L, mode="bulk"):
    """ 
    Impose minimum image condition on displacement vector drij=ri-rj

    Args:
      drij (np.array): length-3 displacement vector ri-rj
      lbox (float): length of cubic cell
    Returns:
      np.array: drij under MIC
    """
    return minimum_image_disp(drij, L, mode)


def all_dists(pos, L, mode="bulk"):
    """
    get all the pairwise distances between a list of positions
    
    Args:
        pos: np array
            N x 3 array of positions
        L: float
            box length, assume cubic box
        mode: string
            bulk or slab
    
    Returns:
        dists : np array
            N x N list of pairwise distances between all particles
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

    Parameters
    ----------
    tseries : T x M array of floats
        some kind of (set of) time series data
    nblocks : int, optional
        number of blocks. The default is 5.

    Returns
    -------
    mean :  1 x M array of floats
        mean(s) of the data.
    err : 1 x M array of floats
        error(s) estimated by the error in the block means.

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
    # last block is the tail starting at (nblocks-1)*blocklen
    means[nblocks - 1, :] = tseries[(nblocks - 1) * blocklen :, :].mean(axis=0)
    mean = means.mean(axis=0)
    err = means.std(axis=0, ddof=1) / np.sqrt(nblocks)
    return mean, err

