import os
import numpy as np

sizes = [4, 10, 28, 1, 5]

sweep_file = "poisson_sweep.npz"
#sweep_file = "helmholtz_sweep.npz"

if os.path.exists(sweep_file):
    sweeps = np.load(sweep_file)
    interp_res = sweeps["interp_res"]
    G_res = sweeps["G_res"]
    beta_res = sweeps["beta_res"]
else:
    interp_res = np.zeros(sizes)
    interp_res[:] = np.nan
    
    G_res = np.zeros(sizes[:-1])
    G_res[:] = np.nan

    beta_res = np.zeros(sizes[:-1])
    beta_res[:] = np.nan

def str_to_pair(line):
    inds_str = line[line.find("(")+1:line.find(")")]
    inds = tuple(np.array(inds_str.split(", ")).astype(int))
    val = float(line.split(": ")[1])
    return inds, val

all_filenames = os.listdir("./")
for filename in all_filenames:
    if filename.startswith("sweepresults-"):
        file = open(filename, 'r')
        lines = file.readlines()
        for line in lines:
            if line.startswith("Green's Function RE"):
                inds, val = str_to_pair(line)
                G_res[inds] = val
            elif line.startswith("Bias Term RE"):
                inds, val = str_to_pair(line)
                beta_res[inds] = val
            elif line.startswith("Combination"):
                inds, val = str_to_pair(line)
                interp_res[inds] = val

np.savez(sweep_file, interp_res=interp_res, G_res=G_res, beta_res=beta_res)
