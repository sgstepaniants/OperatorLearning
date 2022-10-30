# Learning Partial Differential Equations in Reproducing Kernel Hilbert Spaces
This repository contains the code to replicate the results in the paper: "Learning Partial Differential Equations in Reproducing Kernel Hilbert Spaces".

The files `rkhs_functions.py` and `training_functions.py` are used to define and train RKHS estimators for the Green's functions and bias terms of fundamental solution operators that appear in many linear PDEs.

## Simulated Data (`generate_data/`)
* `karhunen_loeve.py` is used to generate all random input forcings, boundary conditions, and initial conditions through the Karhunen-Loeve expansion given an arbitrary kernel function (e.g. squared exponential, periodic squared exponential).
* `helmholtz.ipynb` generates all input forcings $f(x)$ and solutions $u(x)$ for the 1D Helmholtz equation with wavenumber $\omega$. To simulate 1D Poisson equation use $\omega = 0$.
* `schrodinger.ipynb` generates all input boundary conditions $b(x)$ and solutions $u(y_1, y_2)$ for the 2D time-independent Schrödinger equation.
* `fokker_planck.ipynb` generates all initial conditions $u_0(x)$ and solutions $u(y, t)$ for the 1D Fokker-Planck equation. Depends on the simulation toolbox of https://github.com/johnaparker/fplanck which has been modified and saved in the subfolder `fplanck/` for our simulation purposes.

## Numerical Experiments (`experiments/`)
