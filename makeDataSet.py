import os
from os.path import exists
import scipy as sp
import skimage as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile 
from datetime import datetime


def rotation_matrix(theta):
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R


def makeDataframe(df_e, df_p):
    _df = pd.DataFrame({"x": [], "y": [], "E_x": [], "E_y": [], "E_p": []})
    _df["x"] = df_e.iloc[:, 0]
    _df["y"] = df_e.iloc[:, 1]
    _df["E_x"] = df_e.iloc[:, 2]
    _df["E_y"] = df_e.iloc[:, 3]
    _df["E_p"] = df_p.iloc[:, 2]

    _df1 = pd.DataFrame(
        {
            "x": [float(df_e.columns[0])],
            "y": [float(df_e.columns[0])],
            "E_x": [df_e.columns[2]],
            "E_y": [df_e.columns[2]],
            "E_p": [df_p.columns[2]],
        }
    )

    df = pd.concat([_df1, _df], ignore_index=True)

    return df


def petalMask(X, Y):
    petal = np.zeros([401, 401])
    for x, y in zip(X, Y):
        petal[int(200 + 50 * x), int(200 + 50 * y)] = 255

    imgLabel = sm.measure.label(255 - petal, background=0, connectivity=1)

    # imgLabel = np.asarray(imgLabel, "uint8")
    # tifffile.imwrite(f"display/imgLabel.tif", imgLabel, imagej=True)

    label = imgLabel[200, 200]
    petal[imgLabel == label] = 255

    return petal


def petalField(df):
    X = np.linspace(-4, 4, 401)
    Y = np.linspace(-4, 4, 401)

    X, Y = np.meshgrid(X, Y)

    field = np.zeros([3, X.shape[0], X.shape[1]])

    for i in range(X.shape[1]):
        x = round(X[0, i], 2)
        field[0, i, :] = df["E_x"][df["x"] == x].iloc[::2]
        field[1, i, :] = df["E_y"][df["x"] == x].iloc[::2]
        field[2, i, :] = df["E_p"][df["x"] == x].iloc[::2]

    return field


def petalFun(p, r, s, C, b):
    if C == 0:
        C = 0.5

    if p == "3-1":
        p = 3
        ang = r * 2 * np.pi / p / 5 + np.pi / p

        if p == 1:
            C = 0
        t = np.linspace(-1, 1, 10000)
        z = np.exp(1j * np.pi * t) * (1 + C * np.cos(p * np.pi * t))
        z = z / np.max(np.abs(z))
        z2 = z[int(20000 / 3)]
        z3 = z[-1]

        z[int(20000 / 3) :].real = np.linspace(z2.real, z3.real, int(1 + 10000 / 3))
        z[int(20000 / 3) :].imag = np.linspace(z2.imag, z3.imag, int(1 + 10000 / 3))
    elif b > 0:

        t = np.linspace(-1, 1, 10000)
        B = b*(0.7/p-0.1)
        z = np.exp(1j * np.pi * t)+B*np.exp(-(p-1) * 1j * np.pi * t)
        z = z / np.max(np.abs(z))
        if p == 3:
            ang = r * 2 * np.pi / p / 5 + np.pi / p
        else:
            ang = r * 2 * np.pi / p / 5

    else:
        if p == 3:
            ang = r * 2 * np.pi / p / 5 + np.pi / p
        else:
            ang = r * 2 * np.pi / p / 5

        if p == 1:
            C = 0
        t = np.linspace(-1, 1, 10000)
        z = np.exp(1j * np.pi * t) * (1 + C * np.cos(p * np.pi * t))
        z = z / np.max(np.abs(z))

    if s == "_xstretch":
        z.real = z.real / 2
    elif s == "_ystretch":
        z.imag = z.imag / 2

    R = rotation_matrix(ang)
    r = np.transpose(np.array([z.real, z.imag]), (1, 0))
    x = np.linspace(-1, 1, 10000)
    y = np.linspace(-1, 1, 10000)
    for i in range(len(t)):
        x[i] = np.matmul(R, r[i])[0]
        y[i] = np.matmul(R, r[i])[1]

    return x, y


def check_p(p):
    if p == 6:
        p = "3-1"
    elif p == 7:
        p = "4-1"
    elif p == 8:
        p = "4-2-1"
    elif p == 9:
        p = "4-2-2"

    return p


P = [3, 4]  # petals
R = [0, 1, 2, 3, 4, 5]  # rotate petal
D = [5, 6, 7, 8, 9, 10]  # bee distance
C = [0] #roundness
B = [1,2,3,4] # pointyness
Perm = [10] # permeability
S = [""] # stretched
if False:
    i = 0
    for r in R:
        for d in D:
            for c in C:
                for perm in Perm:
                    for s in S:
                        for b in B:
                            for p in P:
                                if ((p == 1) or (p > 4)) and (c > 0):
                                    continue
                                if ((p == 1) or (p > 4)) and (s != ""):
                                    continue
                                if (s == "_ystretch") and ((c > 0.5) or (c == 0)):
                                    continue
                                if (p == 1) and (perm == 20):
                                    continue

                                # try:
                                p = check_p(p)
                                if c > 0:
                                    if s == "_ystretch":
                                        s = "_stretchy"
                                    file_name = (
                                        f"Flower_petals_{p}_rot_{r}_pc_dist_-{d}_C_{c}"
                                    )
                                    df_e = pd.read_csv(
                                        "dat_Ryan/"
                                        + file_name
                                        + f"_efield{s}_{perm}.csv"
                                    )
                                    df_p = pd.read_csv(
                                        "dat_Ryan/"
                                        + file_name
                                        + f"_potential{s}_{perm}.csv"
                                    )
                                    if s == "_stretchy":
                                        s = "_ystretch"
                                elif  b != 0:
                                    file_name = f"Flower_petals_{p}_rot_{r}_pc_dist_-{d}_B_{b}"
                                    df_e = pd.read_csv("dat_Ryan/" + file_name + f"_efield{s}_{perm}.csv")
                                    df_p = pd.read_csv("dat_Ryan/" + file_name + f"_potential{s}_{perm}.csv")
                                else:
                                    file_name = (
                                        f"Flower_petals_{p}_rot_{r}_pc_dist_-{d}"
                                    )
                                    df_e = pd.read_csv(
                                        "dat_Ryan/"
                                        + file_name
                                        + f"_efield{s}_{perm}.csv"
                                    )
                                    df_p = pd.read_csv(
                                        "dat_Ryan/"
                                        + file_name
                                        + f"_potential{s}_{perm}.csv"
                                    )

                                df = makeDataframe(df_e, df_p)

                                X, Y = petalFun(p, r, s, c, b)
                                petal = petalMask(X, Y)
                                petal = np.asarray(petal, "uint8")
                                tifffile.imwrite(
                                    f"dat/all masks new/petal_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                    petal,
                                    imagej=True,
                                )

                                field = petalField(df)

                                field[0, :, :] = (field[0, :, :] * 600000) + 2**15
                                # field[0, :, :] = (
                                #     np.sign(field[0, :, :])
                                #     * np.sqrt(np.abs(field[0, :, :]) * 200000**2)
                                #     + 2**15
                                # )
                                field[0, :, :][field[0, :, :] < 0] = 0
                                field[0, :, :][field[0, :, :] > 2**16] = 2**16 - 1

                                field[1, :, :] = (field[1, :, :] * 600000) + 2**15
                                # field[1, :, :] = (
                                #     np.sign(field[1, :, :])
                                #     * np.sqrt(np.abs(field[1, :, :]) * 200000**2)
                                #     + 2**15
                                # )
                                field[1, :, :][field[1, :, :] < 0] = 0
                                field[1, :, :][field[1, :, :] > 2**16] = 2**16 - 1

                                field[2, :, :] = (field[2, :, :] * 600000) + 2**15
                                # field[2, :, :] = (
                                #     np.sign(field[2, :, :])
                                #     * np.sqrt(np.abs(field[2, :, :]) * 200000**2)
                                #     + 2**15
                                # )
                                field[2, :, :][field[2, :, :] < 0] = 0
                                field[2, :, :][field[2, :, :] > 2**16] = 2**16 - 1

                                field = np.asarray(field, "uint16")
                                tifffile.imwrite(
                                    f"dat/all images new/field_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                    field,
                                    imagej=True,
                                )

                                # except:
                                #     continue

P = [3, 4]  # petals
R = [0, 1, 2, 3, 4, 5]  # rotate petal
D = [5, 6, 7, 8, 9, 10]  # bee distance
C = [0]
B = [1,2,3,4]
Perm = [10]
S = [""]

if True:
    # fields = np.zeros([432, 3, 401, 401])
    # i = 0
    for r in R:
        for d in D:
            for c in C:
                for perm in Perm:
                    for s in S:
                        for b in B:
                            for p in P:
                                if ((p == 1) or (p > 4)) and (c > 0):
                                    continue
                                if ((p == 1) or (p > 4)) and (s != ""):
                                    continue
                                if (s == "_ystretch") and ((c > 0.5) or (c == 0)):
                                    continue
                                if (p == 1) and (perm == 20):
                                    continue

                                try:
                                    p = check_p(p)

                                    petal = sm.io.imread(
                                        f"dat/all masks/petal_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif"
                                    ).astype(int)
                                    field = sm.io.imread(
                                        f"dat/all images/field_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif"
                                    ).astype(int)
                                    rr0, cc0 = sm.draw.disk([200, 200], 55)
                                    field[rr0, cc0] = 2**15

                                    field = np.transpose(field, (2, 0, 1))

                                    if p != 4:
                                        if r != 4:
                                            petal = np.asarray(petal, "uint8")
                                            tifffile.imwrite(
                                                f"dat/training/mask/petal_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                                petal,
                                                imagej=True,
                                            )
                                            field = np.asarray(field, "uint16")
                                            tifffile.imwrite(
                                                f"dat/training/input/field_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                                field,
                                                imagej=True,
                                            )
                                        else:
                                            petal = np.asarray(petal, "uint8")
                                            tifffile.imwrite(
                                                f"dat/validation/mask/petal_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                                petal,
                                                imagej=True,
                                            )
                                            field = np.asarray(field, "uint16")
                                            tifffile.imwrite(
                                                f"dat/validation/input/field_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                                field,
                                                imagej=True,
                                            )
                                    else:
                                        petal = np.asarray(petal, "uint8")
                                        tifffile.imwrite(
                                            f"dat/testing/mask/petal_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                            petal,
                                            imagej=True,
                                        )
                                        field = np.asarray(field, "uint16")
                                        tifffile.imwrite(
                                            f"dat/testing/input/field_{p}_{r}_-{d}_{c}{s}_{b}_{perm}.tif",
                                            field,
                                            imagej=True,
                                        )

                                except:
                                    continue
