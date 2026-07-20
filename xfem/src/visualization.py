"""Visualization for the FEM results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_stress_field(x_test, stress, tip1, tip2, crack_center,
                      save="stress_field.png"):
    n = int(np.sqrt(len(x_test)))
    X, Y = x_test[:,0].reshape(n,n), x_test[:,1].reshape(n,n)
    crack_x = [tip2[0], tip1[0]]; crack_y = [crack_center[1]]*2

    fields = [(stress["sigma_xx"], r"$\sigma_{xx}$", "RdBu_r"),
              (stress["sigma_yy"], r"$\sigma_{yy}$", "RdBu_r"),
              (stress["sigma_xy"], r"$\sigma_{xy}$", "RdBu_r"),
              (stress["von_mises"], "Von Mises",      "jet")]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (raw, title, cmap) in zip(axes, fields):
        vmax = np.percentile(np.abs(raw), 99)
        data = raw.reshape(n, n)
        im = ax.contourf(X, Y,
                         np.clip(data, 0, vmax) if cmap=="jet" else data,
                         levels=50, cmap=cmap,
                         **({} if cmap=="jet" else {"vmin":-vmax,"vmax":vmax}))
        ax.plot(crack_x, crack_y, "k-", lw=3)
        ax.plot([tip1[0],tip2[0]], [tip1[1],tip2[1]], "ko", ms=7,
                markerfacecolor="white", markeredgewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(True, alpha=0.3, ls="--")

    plt.suptitle("FEM — Stress Fields", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"Saved: {save}"); plt.show()


def plot_displacement_field(x_test, u_pred, tip1, tip2, crack_center,
                             save="displacement.png"):
    n = int(np.sqrt(len(x_test)))
    X, Y = x_test[:,0].reshape(n,n), x_test[:,1].reshape(n,n)
    crack_x = [tip2[0], tip1[0]]; crack_y = [crack_center[1]]*2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (data, title) in zip(axes, [(u_pred[:,0], r"$u_x$"), (u_pred[:,1], r"$u_y$")]):
        im = ax.contourf(X, Y, data.reshape(n,n), levels=50, cmap="RdBu_r")
        ax.plot(crack_x, crack_y, "k-", lw=3)
        ax.plot([tip1[0],tip2[0]], [tip1[1],tip2[1]], "ko", ms=7,
                markerfacecolor="white", markeredgewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(True, alpha=0.3, ls="--")

    plt.suptitle("FEM — Displacement Fields", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"Saved: {save}"); plt.show()


def plot_ki_path_independence(ki_j, ki_analytical, save="ki_path_independence.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for tip, c in [("tip1","steelblue"), ("tip2","tomato")]:
        r = ki_j[tip]
        ax.plot(r["radii"], r["K_I_values"], "o-", color=c, lw=2, ms=7,
                label=f"{tip}  mean={r['K_I_mean']:.4f}")
    ax.axhline(ki_analytical, color="k", ls="--", lw=2,
               label=f"Analytical={ki_analytical:.4f}")
    ax.set_xlabel("Contour radius ρ"); ax.set_ylabel("K_I")
    ax.set_title("FEM — K_I Path Independence (J-Integral)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save, dpi=120, bbox_inches="tight")
    print(f"Saved: {save}"); plt.show()


def plot_ki_cod(ki_cod, ki_analytical, save="ki_cod.png"):
    r  = np.array(ki_cod["r_values"])
    ki = np.array(ki_cod["K_I_values"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.sqrt(r), ki, "b.-", lw=1.5, ms=4, label="K_I from COD")
    ax.axhline(ki_analytical,               color="k",      ls="--", lw=2,
               label=f"Analytical={ki_analytical:.4f}")
    ax.axhline(ki_cod["K_I_extrapolated"], color="tomato", ls="-.", lw=2,
               label=f"Extrapolated={ki_cod['K_I_extrapolated']:.4f}")
    ax.set_xlabel("√r"); ax.set_ylabel("K_I")
    ax.set_title("FEM — K_I via COD Extrapolation")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save, dpi=120, bbox_inches="tight")
    print(f"Saved: {save}"); plt.show()
