# Analyze reconstruction performance by computing F1 scores for different flower parameters
# This script loads predicted petal masks and calculates statistics grouped by various attributes

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# Select dataset to analyze
dataset = "testing"
# dataset = "validation"
plt.rcParams.update({"font.size": 16})

print(dataset)

dfPetal = pd.read_pickle(f"outputs/dfPetal{dataset}.pkl")

print("Mean F1 score:", np.mean(dfPetal["F1 score"]))
print("Std F1 score:", np.std(dfPetal["F1 score"]))

dataset = "validation"
print(dataset)

dfPetal = pd.read_pickle(f"outputs/dfPetal{dataset}.pkl")

print("Mean F1 score:", np.mean(dfPetal["F1 score"]))
print("Std F1 score:", np.std(dfPetal["F1 score"]))

# Compute F1 scores grouped by number of petals (validation set only)
if dataset == "validation":
    P = ["1","2","3","3-1"]
    for p in P:
        print(f"Mean F1 score for Petals={p}:", np.mean(dfPetal["F1 score"][dfPetal["Petals"]==p]))
        print(f"Std F1 score for Petals={p}:", np.std(dfPetal["F1 score"][dfPetal["Petals"]==p]))

# Compute F1 scores grouped by rotation angle (testing set only)
if dataset == "testing":
    R = ["0", "1", "2", "3", "4", "5"]
    for r in R:
        print(f"Mean F1 score for Rotation={r}:", np.mean(dfPetal["F1 score"][dfPetal["Rotation"]==r]))
        print(f"Std F1 score for Rotation={r}:", np.std(dfPetal["F1 score"][dfPetal["Rotation"]==r]))

# Compute F1 scores grouped by arthropod-flower distance
D = ["-5", "-6", "-7", "-8", "-9", "-10"]
for d in D:
    print(f"Mean F1 score for Distance={d}:", np.mean(dfPetal["F1 score"][dfPetal["Distance"]==d]))
    print(f"Std F1 score for Distance={d}:", np.std(dfPetal["F1 score"][dfPetal["Distance"]==d]))

# Compute F1 scores grouped by petal roundness parameter
C = ["0.1", "0.2", "0.4", "0.5", "0.6"]
for c in C:
    print(f"Mean F1 score for Roundness={c}:", np.mean(dfPetal["F1 score"][dfPetal["Roundness"]==c]))
    print(f"Std F1 score for Roundness={c}:", np.std(dfPetal["F1 score"][dfPetal["Roundness"]==c]))

# Compute F1 scores grouped by petal pointiness (validation set only)
if dataset == "validation":
    B = ["1","2","3","4"]
    for b in B:
        print(f"Mean F1 score for Pointness={b}:", np.mean(dfPetal["F1 score"][dfPetal["Pointness"]==b]))
        print(f"Std F1 score for Pointness={b}:", np.std(dfPetal["F1 score"][dfPetal["Pointness"]==b]))

# Compute F1 scores grouped by petal permeability
print(f"Mean F1 score for Permeability={10}:", np.mean(dfPetal["F1 score"][(dfPetal["Permeability"]=="10")]))
print(f"Std F1 score for Permeability={10}:", np.std(dfPetal["F1 score"][(dfPetal["Permeability"]=="10")]))
print(f"Mean F1 score for Permeability={20}:", np.mean(dfPetal["F1 score"][(dfPetal["Permeability"]=="20")]))
print(f"Std F1 score for Permeability={20}:", np.std(dfPetal["F1 score"][(dfPetal["Permeability"]=="20")]))

# Compute F1 scores for stretched petals (validation set only)
if dataset == "validation":
    print(f"Mean F1 score for 3 petal Stretched=NA:", np.mean(dfPetal["F1 score"][(dfPetal["Stretched"]=="") & (dfPetal["Petals"]=="3")]))
    print(f"Std F1 score for 3 petal Stretched=NA:", np.std(dfPetal["F1 score"][(dfPetal["Stretched"]=="") & (dfPetal["Petals"]=="3")]))
    print(f"Mean F1 score for 3 petal Stretched= yx2:", np.mean(dfPetal["F1 score"][(dfPetal["Stretched"]=="ystretch") & (dfPetal["Petals"]=="3")]))
    print(f"Std F1 score for 3 petal Stretched= yx2:", np.std(dfPetal["F1 score"][(dfPetal["Stretched"]=="ystretch") & (dfPetal["Petals"]=="3")]))

# Generate violin plot of F1 scores vs number of petals
dataset = "validation"
dfPetal = pd.read_pickle(f"outputs/dfPetal{dataset}.pkl")
x = []
x.append(list(dfPetal["F1 score"][dfPetal["Petals"]=="1"]))
x.append(list(dfPetal["F1 score"][dfPetal["Petals"]=="2"]))
x.append(list(dfPetal["F1 score"][dfPetal["Petals"]=="3"]))
dataset = "testing"
dfPetal = pd.read_pickle(f"outputs/dfPetal{dataset}.pkl")
x.append(list(dfPetal["F1 score"][dfPetal["Petals"]=="4"]))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

plt.violinplot(x, positions=None, vert=True, widths=0.8, showmeans=True)
ax.set_xticks([y + 1 for y in range(len(x))], labels=['1', '2', '3', '4'])
ax.set_xlabel('Numbers of petals')
ax.set_ylabel('F1 scores')
ax.set_ylim([0.5, 1])

fig.savefig(
    f"results/F1 score with petal numbers",
    transparent=True,
    bbox_inches="tight",
    dpi=300,
)
plt.close("all")

# Generate violin plot of F1 scores vs arthropod-flower distance
dataset = "validation"
dfPetal = pd.read_pickle(f"outputs/dfPetal{dataset}.pkl")
x = []
D = ["-5", "-6", "-7", "-8", "-9", "-10"]
for d in D:
    x.append(list(dfPetal["F1 score"][dfPetal["Distance"]==d]))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

plt.violinplot(x, positions=None, vert=True, widths=0.8, showmeans=True)
ax.set_xticks([y + 1 for y in range(len(x))], labels=["5", "6", "7", "8", "9", "10"])
ax.set_xlabel('Distance from flowers')
ax.set_ylabel('F1 scores')
ax.set_ylim([0.65, 1])
ax.set_title("")

fig.savefig(
    f"results/F1 score with distances from petal",
    transparent=True,
    bbox_inches="tight",
    dpi=300,
)
plt.close("all")

# Generate violin plot of F1 scores vs petal permeability
x = []
Perm = ["10", "20"]
for perm in Perm:
    x.append(list(dfPetal["F1 score"][dfPetal["Permeability"]==perm]))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

plt.violinplot(x, positions=None, vert=True, widths=0.8, showmeans=True)
ax.set_xticks([y + 1 for y in range(len(x))], labels=["10", "20"])
ax.set_xlabel('Petal permittivity')
ax.set_ylabel('F1 scores')
ax.set_ylim([0.65, 1])
ax.set_title("")

fig.savefig(
    f"results/F1 score with petal permeability",
    transparent=True,
    bbox_inches="tight",
    dpi=300,
)
plt.close("all")