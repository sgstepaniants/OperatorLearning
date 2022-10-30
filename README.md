# Learning Partial Differential Equations in Reproducing Kernel Hilbert Spaces
This repository contains the code to replicate the results in the paper: "Learning Partial Differential Equations in Reproducing Kernel Hilbert Spaces".

The files `rkhs_functions.py` and `training_functions.py` are used to define and train RKHS estimators for the Green's functions and bias terms of fundamental solution operators that appear in many linear PDEs.

## Simulated Data (`generate_data/`)
* `karhunen_loeve.py` is used to generate all random input forcings, boundary conditions, and initial conditions through the Karhunen-Loeve expansion given an arbitrary kernel function (e.g. squared exponential, periodic squared exponential).
* `helmholtz.ipynb` generates all input forcings $f(x)$ and solutions $u(x)$ for the 1D Helmholtz equation with wavenumber $\omega$. To simulate 1D Poisson equation use $\omega = 0$.
* `schrodinger.ipynb` generates all input boundary conditions $b(x)$ and solutions $u(y_1, y_2)$ for the 2D time-independent Schrödinger equation.
* `fokker_planck.ipynb` generates all initial conditions $u_0(x)$ and solutions $u(y, t)$ for the 1D Fokker-Planck equation. Depends on the simulation toolbox of https://github.com/johnaparker/fplanck which has been modified and saved in the subfolder `fplanck/` for our simulation purposes.
* `heateq.ipynb` generates all forcings $f(x, s)$ and solutions $u(y, t)$ for the 1D heat equation.

## Numerical Experiments (`experiments/`)
* `poisson/poisson.ipynb` contains the code to learn the Green's function and bias term of the Poisson equation. Used to reproduce Figures 1 & 2 of the paper.
* `helmholtz/helmholtz.ipynb` learns the Green's function and bias term of the Helmholtz equation and studies the sensitivity of these estimators to samples, measurements, and noise which is used to generate Figure 3.
* `schrodinger/schrodinger.ipynb` learns the Green's function of the Schrödinger equations from boundary condition to solution which reproduces Figures 4 & 8.
* `fokker_planck/fokker_planck.ipynb` is used to learn the Green's function of the Fokker-Planck equation from initial condition to solution which reproduces Figures 5 & 9.
* `heateq/heateq.ipynb` uses time-invariance and time-causal physical constraints to learn the Green's function of the heat equation from forcing to solution. This is used to generate Figures 6 & 10 of the paper.

### RKHS Hyperparameter Sweeps (`experiments/hyperparam_sweeps`)
  * This directory contains all the code necessary to reproduce the hyperparameter sweeps for the RKHS kernel lengthscale vs. grid size on the Poisson and Helmholtz equations shown in Figure 7 of the paper.
  * These sweeps are time consuming as they iterate over several choices of RKHS kernels, kernel lengthscales, and grid sizes so these experiments were executed on the MIT Supercloud SLURM cluster.
  * The shell script `sweep_loop.sh` subdivides the list of all possible hyperparameter combinations into batches and it test each batch of hyperparameter combinations by calling `run_sweep.sh` with the sbatch command.
  * Each batch of hyperparameter combinations is then tested on the cloud and the results of the fit for each batch are then saved in a text output file of the form `sweep_results-n-N` where n is the current batch number and N is the total number of batches.
  * The outputs of all files can then be collected and saved in an npz file by running `save_sweep.py`.
  * The notebook `sweep_results.ipynb` then reads in the saved `poisson_sweep.npz` and `helmholtz_sweep.npz` files and reproduces Figure 7 of the paper for hyperparameter sweeps on the Poisson and Helmholtz equation.
