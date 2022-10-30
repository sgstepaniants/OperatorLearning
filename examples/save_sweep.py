import sys
import os
import numpy as np

all_filenames = os.listdir()
filenames = []
for filename in all_filenames:
    if filename.startswith("results-"):
        filenames.append(filename)
filenames = np.array(filenames)


model_names = untrained_reps
folder = f"distances/{subset}/untrained"
if pretrained:
    model_names = pretrained_reps
    folder = f"distances/{subset}/pretrained"
if not os.path.exists(folder):
    os.makedirs(folder)
total_models = len(model_names)

dist_pairs_saved = np.zeros((total_models, total_models), dtype=bool)
if os.path.exists(f"{folder}/stats.npz"):
    dist_pairs_saved = np.load(f"{folder}/stats.npz")["dist_pairs_saved"]

dists = {}
print(folder)
existing_dist_files = os.listdir(folder)
existing_dist_names = [name[:-4] for name in existing_dist_files if name.endswith(".npy")]
for dist_name in existing_dist_names:
    dists[dist_name] = np.load(f'{folder}/{dist_name}.npy')

print(f'existing pairs {np.sum(dist_pairs_saved)} / {int(total_models * (total_models - 1) / 2)}', flush=True)

for filename in filenames:
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        if line.startswith("Computing"):
            splits = line.split(", ")
            name1 = splits[0][10:].strip()
            name2 = splits[1].strip()
            
            i1 = np.where(model_names == name1)[0][0]
            i2 = np.where(model_names == name2)[0][0]
        elif line.count(": ") == 1 and line.count(" ") == 1:
            splits = line.split(": ")
            dist_name = splits[0]
            dist = float(splits[1])
            if dist_name not in dists:
                dists[dist_name] = np.zeros((total_models, total_models))
                dists[dist_name][:] = np.nan
            dists[dist_name][i1, i2] = dist
