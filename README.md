# Lennard–Jones molecular dynamics of argon

Computational physics project studying bulk and slab geometries with a Lennard–Jones interaction. The implementation combines molecular dynamics with trajectory analysis and structural observables.

## Contents

- `src/LJ.py`: molecular-dynamics implementation.
- `src/main.py`: geometry, reduced-unit parameters, thermostat, sampling, and workflow switches.
- `src/analyze.py` and `src/analyze_results.py`: analysis and figures.
- `data/` and `figures/`: trajectories, thermodynamic output, and plots.
- The root-level PHYS 440 PDF records a project checkpoint; `Old/` preserves earlier code.

## Running

From the repository root, use `python src/main.py` in a Python environment with the dependencies imported by the simulation and analysis modules. Review the settings first: **the checked-in driver has simulation disabled and analysis enabled**, so its default action is to analyze existing trajectory data. Enable the simulation block to generate fresh data.

The driver includes bulk/slab geometry, an interaction cutoff, time step, equilibration interval, Andersen collision probability, and random seed. Results use reduced Lennard–Jones units. Check energy behaviour, equilibration, sampling, and finite-size effects before drawing physical conclusions.

A related class-based version is available in [Argon_LJ_sim](https://github.com/YungHairStyle/Argon_LJ_sim).

---
[Asaad Hanna — research, CV, and contact](https://yunghairstyle.github.io/)

