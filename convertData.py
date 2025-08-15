import os
import scipy as sp
import skimage as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile


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


def petalFun(p, r, s, C):
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


d = 10
p = 3
r = 1
s = ""  # "_xstretch"
c = 0
b=0
perm = 10

if c > 0:
    if s == "_ystretch":
        s = "_stretchy"
    file_name = f"Flower_petals_{p}_rot_{r}_pc_dist_-{d}_C_{c}"
    df_e = pd.read_csv("dat_Ryan/" + file_name + f"_efield{s}_{perm}.csv")
    df_p = pd.read_csv("dat_Ryan/" + file_name + f"_potential{s}_{perm}.csv")
    if s == "_stretchy":
        s = "_ystretch"
elif  b != 0:
    file_name = f"Flower_petals_{p}_rot_{r}_pc_dist_-{d}_B_{b}"
    df_e = pd.read_csv("dat_Ryan/" + file_name + f"_efield{s}_{perm}.csv")
    df_p = pd.read_csv("dat_Ryan/" + file_name + f"_potential{s}_{perm}.csv")

else:
    file_name = f"Flower_petals_{p}_rot_{r}_pc_dist_-{d}"
    df_e = pd.read_csv("dat_Ryan/" + file_name + f"_efield{s}_{perm}.csv")
    df_p = pd.read_csv("dat_Ryan/" + file_name + f"_potential{s}_{perm}.csv")

X, Y = petalFun(p, r, s, c)
petal = petalMask(X, Y)
petal = np.asarray(petal, "uint8")
tifffile.imwrite(f"display/petal_{p}_{r}_-{d}_{c}{s}_{perm}.tif", petal, imagej=True)

df = makeDataframe(df_e, df_p)

X = np.linspace(-4, 4, 401)
Y = np.linspace(-4, 4, 401)

X, Y = np.meshgrid(X, Y)

E_x = np.zeros([X.shape[0], X.shape[1]])
E_y = np.zeros([X.shape[0], X.shape[1]])
E_p = np.zeros([X.shape[0], X.shape[1]])

for i in range(X.shape[1]):
    x = round(X[0, i], 2)
    E_x[i] = df["E_x"][df["x"] == x].iloc[::2]
    E_y[i] = df["E_y"][df["x"] == x].iloc[::2]
    E_p[i] = df["E_p"][df["x"] == x].iloc[::2]

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
x, y = np.mgrid[-4:4.02:0.02, -4:4.02:0.02]
_c = c

c = ax[0].pcolor(
    x,
    y,
    E_x,
    cmap="RdBu_r",
    vmax=0.15,
    vmin=-0.15,
    shading="auto",
)
fig.colorbar(c, ax=ax[0])

c = ax[1].pcolor(
    x,
    y,
    E_y,
    cmap="RdBu_r",
    vmax=0.15,
    vmin=-0.15,
    shading="auto",
)
fig.colorbar(c, ax=ax[1])

c = ax[2].pcolor(
    x,
    y,
    E_p,
    cmap="RdBu_r",
    vmax=0.15,
    vmin=-0.15,
    shading="auto",
)
fig.colorbar(c, ax=ax[2])

fig.savefig(
    f"display/electric field petals={p} rot={r} dist=-{d} stretch={s} C={int(10*_c)}",
    dpi=300,
    transparent=True,
)
plt.close("all")
