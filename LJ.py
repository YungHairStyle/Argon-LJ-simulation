"""
LJ.py
-----
A small, fully annotated Lennard–Jones molecular dynamics toolkit
for bulk and slab geometries (argon in LJ reduced units).

Design goals
============
- Minimal dependencies (NumPy only).
- Clear, didactic structure with docstrings and inline comments.
- Ready to import into a notebook or used by `main.py` as a CLI.
- Slab geometry: PBC in x,y; open in z. Bulk: 3D PBC.
- Shifted LJ potential (U(rc)=0). Optional tail corrections for *bulk only*.

Units & Conventions
===================
All quantities are in standard LJ reduced units unless otherwise stated:
- sigma = 1, epsilon = 1, k_B = 1, particle mass m = 1 (by default).
- Time step dt is in reduced time units.
- Temperature T is in epsilon/k_B (so just "1.0" in reduced).

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def minimum_image_3d(dr: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Apply the minimum-image convention in 3D.

    Parameters
    ----------
    dr : (M, 3) ndarray
        Displacement vectors.
    L : (3,) ndarray
        Box lengths [Lx, Ly, Lz].

    Returns
    -------
    (M, 3) ndarray
        Displacements mapped into [-L/2, L/2) along each wrapped axis.
    """
    out = dr.copy()
    out -= L * np.rint(out / L)
    return out


def minimum_image_xy(dr: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """
    Apply the minimum-image convention only in x and y (slab geometry).

    Parameters
    ----------
    dr : (M, 3) ndarray
        Displacement vectors.
    Lx, Ly : float
        Box lengths in x and y.

    Returns
    -------
    (M, 3) ndarray
        x,y components wrapped; z left unchanged.
    """
    out = dr.copy()
    out[:, 0] -= Lx * np.rint(out[:, 0] / Lx)
    out[:, 1] -= Ly * np.rint(out[:, 1] / Ly)
    # z unchanged
    return out


def wrap_3d(r: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Wrap positions into the primary cell in all three dimensions.
    """
    out = r.copy()
    out -= L * np.floor(out / L)
    return out


def wrap_xy(r: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """
    Wrap positions in x and y; leave z unchanged (slab geometry).
    """
    out = r.copy()
    out[:, 0] -= Lx * np.floor(out[:, 0] / Lx)
    out[:, 1] -= Ly * np.floor(out[:, 1] / Ly)
    return out


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class Box:
    """Simple container for box lengths."""
    Lx: float
    Ly: float
    Lz: float

    @property
    def L(self) -> np.ndarray:
        return np.array([self.Lx, self.Ly, self.Lz])

    @property
    def volume(self) -> float:
        return self.Lx * self.Ly * self.Lz

    def copy(self) -> 'Box':
        return Box(self.Lx, self.Ly, self.Lz)


# -----------------------------------------------------------------------------
# Initializers
# -----------------------------------------------------------------------------

def _maxwell_boltzmann_velocities(N: int, T: float, mass: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw velocities from the Maxwell–Boltzmann distribution at temperature T.
    Center-of-mass velocity is removed.
    """
    v = rng.normal(0.0, np.sqrt(T / mass), size=(N, 3))
    v -= v.mean(axis=0, keepdims=True)
    return v


def init_fcc_bulk(n_cells: int, a: float, T: float, mass: float = 1.0,
                  rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, Box]:
    """
    Build a cubic FCC crystal with 3D PBC (bulk).

    Parameters
    ----------
    n_cells : int
        Number of FCC cells along each axis.
    a : float
        Lattice parameter (spacing) in reduced units.
    T : float
        Initial temperature.
    mass : float, default=1.0
        Particle mass.
    rng : np.random.Generator, optional
        RNG for velocities. If None, a default RNG is used.

    Returns
    -------
    r : (N, 3) ndarray
        Particle positions.
    v : (N, 3) ndarray
        Particle velocities.
    box : Box
        Simulation box.
    """
    if rng is None:
        rng = np.random.default_rng()

    # FCC basis (fractional)
    basis = np.array([[0, 0, 0],
                      [0.5, 0.5, 0],
                      [0.5, 0, 0.5],
                      [0, 0.5, 0.5]])

    pts = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                cell = a * np.array([ix, iy, iz], dtype=float)
                pts.extend(cell + a * basis)
    r = np.asarray(pts, dtype=float)
    N = r.shape[0]

    # Define box
    Lx = Ly = Lz = n_cells * a
    box = Box(Lx, Ly, Lz)

    # Velocities
    v = _maxwell_boltzmann_velocities(N, T, mass, rng)
    return r, v, box


def init_fcc_slab(n_cells_x: int, n_cells_y: int, n_layers_z: int,
                  a: float, Lz: float, T: float, mass: float = 1.0,
                  rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, Box]:
    """
    Build an FCC slab (stacked in z) inside a taller box with vacuum.
    PBC in x,y; open in z.

    The slab is centered along z.

    Parameters
    ----------
    n_cells_x, n_cells_y : int
        Number of FCC unit cells along x and y.
    n_layers_z : int
        Number of FCC unit-cell layers stacked in z.
    a : float
        FCC lattice parameter.
    Lz : float
        Total box height (includes vacuum).
    T : float
        Initial temperature.
    mass : float, default=1.0
        Particle mass.
    rng : np.random.Generator, optional
        RNG for velocities.

    Returns
    -------
    r, v, box : tuple
        Positions, velocities, and Box.
    """
    if rng is None:
        rng = np.random.default_rng()

    basis = np.array([[0, 0, 0],
                      [0.5, 0.5, 0],
                      [0.5, 0, 0.5],
                      [0, 0.5, 0.5]])

    pts = []
    for ix in range(n_cells_x):
        for iy in range(n_cells_y):
            for iz in range(n_layers_z):
                cell = a * np.array([ix, iy, iz], dtype=float)
                pts.extend(cell + a * basis)
    r = np.asarray(pts, dtype=float)
    N = r.shape[0]

    # Lateral box
    Lx = n_cells_x * a
    Ly = n_cells_y * a

    # Center the slab along z
    slab_height = n_layers_z * a
    z0 = 0.5 * (Lz - slab_height)  # lower bound of slab
    r[:, 2] = r[:, 2] - r[:, 2].min() + z0

    box = Box(Lx, Ly, Lz)

    # Velocities
    v = _maxwell_boltzmann_velocities(N, T, mass, rng)
    return r, v, box


# -----------------------------------------------------------------------------
# Neighbor list
# -----------------------------------------------------------------------------

class NeighborList:
    """
    Verlet neighbor list with a skin, supporting slab (2D PBC) or bulk (3D PBC).

    Parameters
    ----------
    rcut : float
        Force cutoff.
    skin : float, default=0.3
        Additional buffer distance for the neighbor list.
    slab_mode : bool, default=False
        If True, use 2D PBC (x,y) only; z is open.
    """

    def __init__(self, rcut: float, skin: float = 0.3, slab_mode: bool = False):
        self.rcut = float(rcut)
        self.skin = float(skin)
        self.rlist = float(rcut) + float(skin)
        self.list: Optional[list[list[int]]] = None
        self.last_pos: Optional[np.ndarray] = None
        self.slab_mode = bool(slab_mode)

    def _wrap_displacements(self, dr: np.ndarray, box: Box) -> np.ndarray:
        if self.slab_mode:
            return minimum_image_xy(dr, box.Lx, box.Ly)
        return minimum_image_3d(dr, box.L)

    def needs_rebuild(self, r: np.ndarray, box: Box) -> bool:
        """
        Rebuild when any particle moved more than half the skin distance
        since the previous build.
        """
        if self.last_pos is None:
            return True
        dr = r - self.last_pos
        if self.slab_mode:
            dr = minimum_image_xy(dr, box.Lx, box.Ly)
        else:
            dr = minimum_image_3d(dr, box.L)
        max_disp2 = np.max(np.sum(dr * dr, axis=1))
        return max_disp2 > (0.5 * (self.rlist - self.rcut))**2

    def build(self, r: np.ndarray, box: Box) -> None:
        """
        Construct the neighbor list for the current positions.
        """
        N = len(r)
        self.list = [[] for _ in range(N)]
        for i in range(N - 1):
            dr = r[i + 1:] - r[i]
            dr = self._wrap_displacements(dr, box)
            rij2 = np.einsum("ij,ij->i", dr, dr)
            mask = rij2 < self.rlist**2
            js = np.nonzero(mask)[0] + (i + 1)
            for j in js:
                self.list[i].append(int(j))
        self.last_pos = r.copy()

    def pairs_within_rcut(self, r: np.ndarray, box: Box):
        """
        Yield (i, j, rij_vec, rij2) for each pair within rcut.
        """
        assert self.list is not None, "Neighbor list not built yet."
        rc2 = self.rcut * self.rcut
        for i, js in enumerate(self.list):
            if not js:
                continue
            dr = r[js] - r[i]
            dr = self._wrap_displacements(dr, box)
            rij2 = np.einsum("ij,ij->i", dr, dr)
            mask = rij2 < rc2
            for j, vec, d2 in zip(np.array(js)[mask], dr[mask], rij2[mask]):
                yield i, int(j), vec, float(d2)


# -----------------------------------------------------------------------------
# Lennard–Jones potential and forces
# -----------------------------------------------------------------------------

def lj_shift_value(rc: float) -> float:
    """
    Energy shift so that U(rc) = 0 for the standard 12-6 LJ:
        U(r) = 4 * (r^-12 - r^-6) - U(rc)
    """
    inv2 = 1.0 / (rc * rc)
    inv6 = inv2**3
    inv12 = inv6**2
    return 4.0 * (inv12 - inv6)


def forces_energy_LJ(r: np.ndarray, box: Box, nlist: NeighborList, rc: float) -> Tuple[np.ndarray, float]:
    """
    Compute Lennard–Jones forces and potential energy with a shifted potential.
    Assumes reduced units with sigma=1 and epsilon=1.

    Returns
    -------
    F : (N,3) ndarray
        Forces.
    U : float
        Total potential energy.
    """
    N = len(r)
    F = np.zeros_like(r)
    U = 0.0
    Uc = lj_shift_value(rc)

    for i, j, rij, rij2 in nlist.pairs_within_rcut(r, box):
        inv2 = 1.0 / rij2
        inv6 = inv2**3
        inv12 = inv6**2

        # Potential (shifted)
        U_ij = 4.0 * (inv12 - inv6) - Uc
        U += U_ij

        # Force: F = 24 * (2 r^-14 - r^-8) * r_vec
        pref = 24.0 * (2.0 * inv12 - inv6) * inv2
        fij = pref * rij
        F[i] += fij
        F[j] -= fij

    return F, U


def bulk_tail_corrections(rho: float, rc: float) -> Tuple[float, float]:
    """
    Lennard–Jones long-range (tail) corrections for bulk systems.

    Parameters
    ----------
    rho : float
        Number density (N / V).
    rc : float
        Cutoff radius.

    Returns
    -------
    Utail_per_particle, Ptail : tuple of floats
        Potential energy tail per particle and pressure tail.
    """
    # Standard formulae for 12-6 LJ in 3D bulk with isotropy
    rc3 = rc**3
    rc9 = rc**9
    U_tail = (8.0 * np.pi * rho / 3.0) * ( (1.0 / (3.0 * rc9)) - (1.0 / rc3) )
    P_tail = (16.0 * np.pi * rho**2 / 3.0) * ( (2.0 / (3.0 * rc9)) - (1.0 / rc3) )
    return U_tail, P_tail


# -----------------------------------------------------------------------------
# Thermostats
# -----------------------------------------------------------------------------

def andersen_thermostat(v: np.ndarray, T: float, mass: float, dt: float,
                        nu: float = 0.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Andersen thermostat: with probability p=1-exp(-nu*dt), a particle's
    velocity is redrawn from the Maxwell–Boltzmann distribution at T.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = 1.0 - np.exp(-nu * dt)
    mask = rng.random(len(v)) < p
    if np.any(mask):
        v[mask] = np.random.default_rng().normal(0.0, np.sqrt(T / mass), size=(mask.sum(), 3))
        # keep overall COM approximately small by removing mean drift every call
    v -= v.mean(axis=0, keepdims=True)
    return v


# -----------------------------------------------------------------------------
# Integrator
# -----------------------------------------------------------------------------

def step_velocity_verlet(r: np.ndarray, v: np.ndarray, box: Box, mass: float, dt: float,
                         nlist: NeighborList, rc: float,
                         slab_mode: bool, thermostat: Optional[Callable[[np.ndarray], np.ndarray]] = None
                         ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    One step of velocity–Verlet with optional thermostat (applied at the end).

    Returns
    -------
    r, v, U, K : tuple
        Updated positions, velocities, potential and kinetic energies.
    """
    # Evaluate forces at t
    F, U = forces_energy_LJ(r, box, nlist, rc)

    # v(t+dt/2)
    v += 0.5 * dt * F / mass

    # r(t+dt)
    r += dt * v
    if slab_mode:
        r = wrap_xy(r, box.Lx, box.Ly)
    else:
        r = wrap_3d(r, box.L)

    # Rebuild neighbor list if needed
    if nlist.needs_rebuild(r, box):
        nlist.build(r, box)

    # Forces at t+dt
    F, U = forces_energy_LJ(r, box, nlist, rc)

    # v(t+dt)
    v += 0.5 * dt * F / mass

    # Thermostat (optional)
    if thermostat is not None:
        v = thermostat(v)

    # Kinetic energy
    K = 0.5 * mass * np.sum(v * v)
    return r, v, U, K


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------

def instantaneous_temperature(v: np.ndarray, mass: float) -> float:
    """
    Compute instantaneous temperature using equipartition:
        K = (3/2) N k_B T  with k_B=1 in reduced units.
    """
    N = len(v)
    K = 0.5 * mass * np.sum(v * v)
    return (2.0 / (3.0 * N)) * K


def density_profile_z(r: np.ndarray, Lz: float, nbins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram number density along z.
    Returns bin centers and number-per-unit-length ρ(z).
    """
    z = r[:, 2]
    hist, edges = np.histogram(z, bins=nbins, range=(0.0, Lz))
    dz = edges[1] - edges[0]
    rho_z = hist.astype(float) / dz  # number per unit length
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, rho_z


def rdf_inplane(r: np.ndarray, box: Box, rc: float, nbins: int = 100,
                z_window: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    In-plane radial distribution function g_||(r) using only x,y separations.
    If z_window=(zmin, zmax) is given, restrict atoms to that z range.
    """
    sel = np.ones(len(r), dtype=bool)
    if z_window is not None:
        zmin, zmax = z_window
        sel = (r[:, 2] >= zmin) & (r[:, 2] < zmax)

    pos = r[sel]
    N = len(pos)
    if N < 2:
        return np.linspace(0, rc, nbins), np.zeros(nbins)

    dists = []
    for i in range(N - 1):
        dr = pos[i + 1:, :2] - pos[i, :2]
        dr[:, 0] -= box.Lx * np.rint(dr[:, 0] / box.Lx)
        dr[:, 1] -= box.Ly * np.rint(dr[:, 1] / box.Ly)
        rij = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        dists.append(rij)
    rvals = np.concatenate(dists) if dists else np.array([])

    hist, edges = np.histogram(rvals, bins=nbins, range=(0, rc))
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]

    area = box.Lx * box.Ly
    rho2D = N / area
    shell_area = 2.0 * np.pi * r_centers * dr
    with np.errstate(divide='ignore', invalid='ignore'):
        g = hist / (rho2D * N * shell_area)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return r_centers, g


def msd_lateral(unwrap_xy_traj: np.ndarray) -> np.ndarray:
    """
    Lateral mean-squared displacement from an unwrapped x,y trajectory.
    unwrap_xy_traj: shape (T, N, 2)
    Returns msd[tau] averaged over particles and origins.
    """
    X = unwrap_xy_traj  # (T,N,2)
    T = X.shape[0]
    msd = np.zeros(T, dtype=float)
    for tau in range(T):
        disp = X[tau:] - X[:T - tau]
        msd[tau] = np.mean(np.sum(disp * disp, axis=2))
    return msd


def pressure_tensor_LJ(r: np.ndarray, v: np.ndarray, box: Box, mass: float,
                       nlist: NeighborList, rc: float) -> np.ndarray:
    """
    Irving–Kirkwood pressure tensor (instantaneous), including kinetic
    and configurational (virial) parts.
    """
    vol = box.volume

    # Kinetic contribution: P_kin = (m/V) sum_n v_n \otimes v_n
    P_kin = (mass / vol) * np.einsum("ni,nj->ij", v, v)

    # Configurational part: (1/V) sum_{i<j} r_ij ⊗ f_ij
    P_conf = np.zeros((3, 3), dtype=float)
    for i, j, rij, rij2 in nlist.pairs_within_rcut(r, box):
        inv2 = 1.0 / rij2
        inv6 = inv2**3
        inv12 = inv6**2
        pref = 24.0 * (2.0 * inv12 - inv6) * inv2
        fij = pref * rij  # force on i by j
        P_conf += np.outer(rij, fij)
    P_conf /= vol

    # Each pair counted once -> factor 1/2 for symmetric virial
    P = P_kin + 0.5 * P_conf
    return P


def surface_tension_gamma(P: np.ndarray, Lz: float) -> float:
    """
    Estimate surface tension for a slab:
        gamma = (Lz/2) * (P_zz - 0.5*(P_xx + P_yy))
    """
    return 0.5 * Lz * (P[2, 2] - 0.5 * (P[0, 0] + P[1, 1]))


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def write_gro(filename: str, r: np.ndarray, box: Box,
              title: str = "Argon", write_velocities: bool = False,
              v: Optional[np.ndarray] = None) -> None:
    """
    Write a minimal .gro file. If you need real units (nm), convert before calling.
    """
    with open(filename, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{len(r):5d}\n")
        for i, pos in enumerate(r, start=1):
            if write_velocities and v is not None:
                f.write(f"{1:5d}{'AR':>5s}{'Ar':>5s}{i:5d}"
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                        f"{v[i-1,0]:8.4f}{v[i-1,1]:8.4f}{v[i-1,2]:8.4f}\n")
            else:
                f.write(f"{1:5d}{'AR':>5s}{'Ar':>5s}{i:5d}"
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}\n")
        f.write(f"{box.Lx:10.5f}{box.Ly:10.5f}{box.Lz:10.5f}\n")