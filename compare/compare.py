#!/usr/bin/env python3
"""
Cross-comparison: FEM (ground truth) vs X-PINN.

Loads saved .npz / .json from both solvers and produces:
  - Side-by-side stress field comparison (2×4 panel)
  - Side-by-side displacement comparison (2×2 panel)
  - K_I bar chart + path-independence curves
  - Printed summary table

Usage
-----
  python compare/compare.py
  python compare/compare.py --xfem xfem/results/data --xpinn xpinn/results/data

Run xfem/run.py and xpinn/run.py first to generate the data.
"""

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(HERE, "results", "figures")
DATA_DIR = os.path.join(HERE, "results", "data")
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ── load helpers ──────────────────────────────────────────────────────────────
def load_stress(data_dir):
    d = np.load(f"{data_dir}/stress_field.npz")
    return d["x_test"], {k: d[k] for k in ["sigma_xx","sigma_yy","sigma_xy","von_mises"]}


def load_displacement(data_dir):
    d = np.load(f"{data_dir}/displacement.npz")
    return d["x_test"], np.column_stack([d["u_x"], d["u_y"]])


def load_ki(data_dir):
    with open(f"{data_dir}/ki_results.json") as f:
        return json.load(f)


# ── stress comparison (2-row × 4-col) ────────────────────────────────────────
def plot_stress_comparison(x_test, stress_fem, stress_pinn,
                            tip1, tip2, crack_cy,
                            save="stress_comparison.png"):
    n = int(np.sqrt(len(x_test)))
    X, Y = x_test[:,0].reshape(n,n), x_test[:,1].reshape(n,n)
    cx = [tip2[0], tip1[0]]; cy = [crack_cy, crack_cy]

    keys   = ["sigma_xx","sigma_yy","sigma_xy","von_mises"]
    titles = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$", "Von Mises"]
    cmaps  = ["RdBu_r","RdBu_r","RdBu_r","jet"]
    rows   = [("FEM  (ground truth)", stress_fem),
              ("X-PINN",              stress_pinn)]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ri, (label, stress) in enumerate(rows):
        axes[ri, 0].set_ylabel(label, fontsize=12, fontweight="bold", labelpad=12)
        for ci, (key, title, cmap) in enumerate(zip(keys, titles, cmaps)):
            ax = axes[ri, ci]
            raw  = stress[key]
            vmax = np.percentile(np.abs(raw), 99)
            data = raw.reshape(n, n)
            im = ax.contourf(X, Y,
                             np.clip(data, 0, vmax) if cmap=="jet" else data,
                             levels=50, cmap=cmap,
                             **({} if cmap=="jet" else {"vmin":-vmax,"vmax":vmax}))
            ax.plot(cx, cy, "k-", lw=3)
            ax.plot([tip1[0],tip2[0]], [tip1[1],tip2[1]], "ko", ms=6,
                    markerfacecolor="white", markeredgewidth=2)
            if ri == 0: ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("x"); ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.grid(True, alpha=0.25, ls="--")

    plt.suptitle("Stress Field Comparison: FEM vs X-PINN",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save}", dpi=130, bbox_inches="tight")
    print(f"Saved: {FIG_DIR}/{save}"); plt.show()


# ── displacement comparison (2-row × 2-col) ───────────────────────────────────
def plot_displacement_comparison(x_test, u_fem, u_pinn,
                                  tip1, tip2, crack_cy,
                                  save="displacement_comparison.png"):
    n = int(np.sqrt(len(x_test)))
    X, Y = x_test[:,0].reshape(n,n), x_test[:,1].reshape(n,n)
    cx = [tip2[0], tip1[0]]; cy = [crack_cy, crack_cy]

    comps = [(0, r"$u_x$"), (1, r"$u_y$")]
    rows  = [("FEM", u_fem), ("X-PINN", u_pinn)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ri, (label, u) in enumerate(rows):
        axes[ri, 0].set_ylabel(label, fontsize=12, fontweight="bold", labelpad=12)
        for ci, (idx, title) in enumerate(comps):
            ax = axes[ri, ci]
            im = ax.contourf(X, Y, u[:,idx].reshape(n,n), levels=50, cmap="RdBu_r")
            ax.plot(cx, cy, "k-", lw=3)
            ax.plot([tip1[0],tip2[0]], [tip1[1],tip2[1]], "ko", ms=6,
                    markerfacecolor="white", markeredgewidth=2)
            if ri == 0: ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("x"); ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.grid(True, alpha=0.25, ls="--")

    plt.suptitle("Displacement Comparison: FEM vs X-PINN",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save}", dpi=130, bbox_inches="tight")
    print(f"Saved: {FIG_DIR}/{save}"); plt.show()


# ── K_I summary plot ──────────────────────────────────────────────────────────
def plot_ki_comparison(ki_fem, ki_pinn, save="ki_comparison.png"):
    ki_ana = ki_fem["analytical"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    labels = ["Analytical", "FEM tip1", "FEM tip2", "PINN tip1", "PINN tip2"]
    vals   = [ki_ana,
              ki_fem["j_integral"]["tip1"]["K_I_mean"],
              ki_fem["j_integral"]["tip2"]["K_I_mean"],
              ki_pinn["j_integral"]["tip1"]["K_I_mean"],
              ki_pinn["j_integral"]["tip2"]["K_I_mean"]]
    errs   = [0,
              ki_fem["j_integral"]["tip1"]["K_I_std"],
              ki_fem["j_integral"]["tip2"]["K_I_std"],
              ki_pinn["j_integral"]["tip1"]["K_I_std"],
              ki_pinn["j_integral"]["tip2"]["K_I_std"]]
    colors = ["k", "steelblue", "cornflowerblue", "tomato", "salmon"]

    bars = ax1.bar(labels, vals, color=colors, yerr=errs, capsize=5,
                   edgecolor="black", linewidth=0.8)
    ax1.axhline(ki_ana, color="k", ls="--", lw=1.5, alpha=0.4)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("K_I"); ax1.set_title("K_I Comparison")
    ax1.tick_params(axis="x", labelsize=9); ax1.grid(axis="y", alpha=0.3)

    # Path-independence curves
    r_f = ki_fem["j_integral"]["tip1"]["radii"]
    ax2.plot(r_f, ki_fem["j_integral"]["tip1"]["K_I_values"],
             "s-", color="steelblue", lw=2, ms=7, label="FEM tip1")
    ax2.plot(r_f, ki_fem["j_integral"]["tip2"]["K_I_values"],
             "s--", color="cornflowerblue", lw=2, ms=7, label="FEM tip2")
    r_p = ki_pinn["j_integral"]["tip1"]["radii"]
    ax2.plot(r_p, ki_pinn["j_integral"]["tip1"]["K_I_values"],
             "o-", color="tomato", lw=2, ms=7, label="PINN tip1")
    ax2.plot(r_p, ki_pinn["j_integral"]["tip2"]["K_I_values"],
             "o--", color="salmon", lw=2, ms=7, label="PINN tip2")
    ax2.axhline(ki_ana, color="k", ls="--", lw=2,
                label=f"Analytical={ki_ana:.4f}")
    ax2.set_xlabel("Contour radius ρ"); ax2.set_ylabel("K_I")
    ax2.set_title("K_I Path Independence"); ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("K_I: FEM vs X-PINN", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save}", dpi=130, bbox_inches="tight")
    print(f"Saved: {FIG_DIR}/{save}"); plt.show()


# ── difference maps ───────────────────────────────────────────────────────────
def plot_difference_maps(x_test, stress_fem, stress_pinn,
                          tip1, tip2, crack_cy,
                          save="difference_maps.png"):
    """Absolute difference |FEM - PINN| for each stress component."""
    n = int(np.sqrt(len(x_test)))
    X, Y = x_test[:,0].reshape(n,n), x_test[:,1].reshape(n,n)
    cx = [tip2[0], tip1[0]]; cy_line = [crack_cy, crack_cy]

    keys   = ["sigma_xx","sigma_yy","sigma_xy","von_mises"]
    titles = [r"|Δ$\sigma_{xx}$|", r"|Δ$\sigma_{yy}$|",
              r"|Δ$\sigma_{xy}$|", r"|ΔVon Mises|"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, key, title in zip(axes, keys, titles):
        diff = np.abs(stress_fem[key] - stress_pinn[key])
        vmax = np.percentile(diff, 99)
        im   = ax.contourf(X, Y, diff.reshape(n,n), levels=50,
                            cmap="Reds", vmin=0, vmax=vmax)
        ax.plot(cx, cy_line, "k-", lw=3)
        ax.plot([tip1[0],tip2[0]], [tip1[1],tip2[1]], "ko", ms=6,
                markerfacecolor="white", markeredgewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(True, alpha=0.25, ls="--")

    plt.suptitle("|FEM − X-PINN| Stress Difference", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save}", dpi=130, bbox_inches="tight")
    print(f"Saved: {FIG_DIR}/{save}"); plt.show()


# ── summary table ─────────────────────────────────────────────────────────────
def print_summary(ki_fem, ki_pinn):
    ki_ana = ki_fem["analytical"]
    def err(v): return abs(v-ki_ana)/ki_ana*100

    print("\n" + "="*70)
    print("COMPARISON SUMMARY — Mode I Forward Problem")
    print("="*70)
    print(f"{'Method':<30} {'K_I tip1':<14} {'K_I tip2':<14} {'Err tip1 %':<12}")
    print("-"*70)
    print(f"{'Analytical (Westergaard)':<30} {ki_ana:<14.5f} {'—':<14} {'0.00':<12}")

    f1 = ki_fem["j_integral"]["tip1"]["K_I_mean"]
    f2 = ki_fem["j_integral"]["tip2"]["K_I_mean"]
    print(f"{'FEM  (Q4, 80×80 mesh)':<30} {f1:<14.5f} {f2:<14.5f} {err(f1):<12.2f}")

    p1 = ki_pinn["j_integral"]["tip1"]["K_I_mean"]
    p2 = ki_pinn["j_integral"]["tip2"]["K_I_mean"]
    print(f"{'X-PINN':<30} {p1:<14.5f} {p2:<14.5f} {err(p1):<12.2f}")
    print("="*70)

    # Save summary JSON
    summary = {
        "analytical": ki_ana,
        "fem":  {"tip1": f1, "tip2": f2, "err_pct": err(f1)},
        "pinn": {"tip1": p1, "tip2": p2, "err_pct": err(p1)},
    }
    import json
    with open(f"{DATA_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {DATA_DIR}/summary.json")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xfem",  default=os.path.join(HERE, "..", "xfem",  "results", "data"))
    p.add_argument("--xpinn", default=os.path.join(HERE, "..", "xpinn", "results", "data"))
    args = p.parse_args()

    xfem_dir  = os.path.normpath(args.xfem)
    xpinn_dir = os.path.normpath(args.xpinn)

    print(f"Loading FEM  data from:  {xfem_dir}")
    print(f"Loading PINN data from:  {xpinn_dir}")

    x_test,  stress_fem  = load_stress(xfem_dir)
    _,       stress_pinn = load_stress(xpinn_dir)
    _,       u_fem       = load_displacement(xfem_dir)
    _,       u_pinn      = load_displacement(xpinn_dir)
    ki_fem               = load_ki(xfem_dir)
    ki_pinn              = load_ki(xpinn_dir)

    # Geometry (crack at y=0.5, tips at x=0.35 and x=0.65)
    tip1 = np.array([0.65, 0.5]); tip2 = np.array([0.35, 0.5]); crack_cy = 0.5

    print_summary(ki_fem, ki_pinn)

    print("\nGenerating comparison figures...")
    plot_stress_comparison(x_test, stress_fem, stress_pinn, tip1, tip2, crack_cy)
    plot_displacement_comparison(x_test, u_fem, u_pinn, tip1, tip2, crack_cy)
    plot_ki_comparison(ki_fem, ki_pinn)
    plot_difference_maps(x_test, stress_fem, stress_pinn, tip1, tip2, crack_cy)

    print("\nDone. Comparison figures in compare/results/figures/")


if __name__ == "__main__":
    main()
