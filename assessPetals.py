import os
import skimage as sm
import numpy as np
import pandas as pd

dataset = "testing"
# dataset = "validation"

folder = f"outputs/dat_{dataset}"
cwd = os.getcwd()
Fullfilenames = os.listdir(cwd + "/" + folder)
fileList = []
for file in Fullfilenames:
    if "labelled" in file:
        fileList.append(file)

def F1score(label, pred):
    return (2*np.sum(label[pred==1])/(2*np.sum(label[pred==1]) + np.sum(label[pred!=1]) + np.sum(pred[label!=1])))

def Filename(file):
    types = file.split("_")[1:-1]

    p = types[0]
    r = types[1]
    d = types[2]
    c = types[3]
    perm = types[-1]
    if "stretch" in types[-2]:
        s = types[-2]
    else:
        s=""
    if (len(types) == 6) and s == "":
        b = types[-2]
    else:
        b=""
    
    if c == "0":
        c = "0.5"
    return p, r, d, c, b, perm, s


_df = []
for file in fileList:

    label = sm.io.imread(folder + "/" + file).astype(int)//255

    pred = sm.io.imread(folder + "/" + file.replace("labelled", "pred")).astype(int)[0]
    pred[pred<128] = 0
    pred[pred>=128] = 1

    p, r, d, c, b, perm, s = Filename(file)
    
    _df.append(
        {
            "Filename": file,
            "F1 score": F1score(label, pred),
            "Petals": p,
            "Rotation": r,
            "Distance": d,
            "Roundness": c,
            "Pointness": b,
            "Permeability": perm,
            "Stretched": s,
        }
    )

dfPetal = pd.DataFrame(_df)

dfPetal.to_pickle(f"outputs/dfPetal{dataset}.pkl")