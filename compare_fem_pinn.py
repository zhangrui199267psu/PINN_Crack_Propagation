#!/usr/bin/env python3
"""
Forward problem comparison: Classical FEM (slit mesh) vs X-PINN.

Both solve the SAME forward problem:
  Domain  : [0,1] × [0,1] plate
  Crack   : horizontal, centre (0.5,0.5), half-length 0.15
  BCs     : u_y = ±0.1 on top/bottom, u_x = 0 on left, traction-free crack faces
  Goal    : displacement field u(x,y) → stress σ(x,y) → K_I

FEM  — structured Q4 mesh (slit crack), scipy direct solver.
X-PINN — 3-network decomposition: u = u_C + H·u_D + ψ·u_S, trained with DeepXDE.

Usage
-----
  python compare_fem_pinn.py                        # FEM only (fast, no training)
  python compare_fem_pinn.py --pinn                 # FEM + train PINN then compare
  python compare_fem_pinn.py --pinn --config configs/mode1_quick.yaml
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fem_solver import FEMCrackSolver
from xpinn_fracture import (
    MaterialProperties, TrainingConfig, CrackGeometry, XFEMCrackSolver,
    westergaard_ki_analytical, compute_ki_j_integral, compute_stress_components,
)
from xpinn_fracture.postprocess import compute_ki_displacement_extrapolation

# ── shared problem setup ──────────────────────────────────────────────────────
CRACK_CENTER     = (0.5, 0.5)
CRACK_HALF_LEN   = 0.15
E, NU, TRACTION  = 1.0, 0.3, 0.1
FORMULATION      = "plane_stress"
J_RADII          = [0.03, 0.05, 0.07]
N_GRID           = 80      # grid resolution for plots
FEM_NX = FEM_NY  = 80     # FEM mesh


# ── build shared objects ──────────────────────────────────────────────────────
def make_shared():
    geom = CrackGeometry(
        center=np.array(CRACK_CENTER),
        half_length=CRACK_HALF_LEN,
        angle=0.0,
        domain_size=(1.0, 1.0),
        enrichment_radius=0.12,
        enrichment_inner=0.05,
    )
    mat = MaterialProperties(E=E, nu=NU, traction=TRACTION, formulation=FORMULATION)
    return geom, mat


# ── test grid ─────────────────────────────────────────────────────────────────
def make_grid(n=N_GRID):
    xs, ys = np.linspace(0, 1, n), np.linspace(0, 1, n)
    return np.column_stack([v.ravel() for v in np.meshgrid(xs, ys)])


# ── FEM run ───────────────────────────────────────────────────────────────────
def run_fem(geom, mat):
    print("\n" + "="*55)
    print("CLASSICAL FEM (Q4 slit mesh)")
    print("="*55)
    fem = FEMCrackSolver(
        Nx=FEM_NX, Ny=FEM_NY,
        E=mat.E, nu=mat.nu, traction=mat.traction,
        crack_center=CRACK_CENTER,
        crack_half_length=CRACK_HALF_LEN,
        formulation=FORMULATION,
    )
    fem.solve()

    x_test = make_grid()
    stress  = fem.stress_at(x_test)
    ki_j    = compute_ki_j_integral(fem.predict, geom, mat, radii=J_RADII)
    ki_disp = compute_ki_displacement_extrapolation(fem.predict, geom.tip1, geom, mat)

    print(f"  K_I J-integral (tip1): {ki_j['tip1']['K_I_mean']:.5f} "
          f"± {ki_j['tip1']['K_I_std']:.5f}")
    print(f"  K_I J-integral (tip2): {ki_j['tip2']['K_I_mean']:.5f} "
          f"± {ki_j['tip2']['K_I_std']:.5f}")
    print(f"  K_I COD extrap (tip1): {ki_disp['K_I_extrapolated']:.5f}")
    return fem, x_test, stress, ki_j, ki_disp


# ── PINN run ──────────────────────────────────────────────────────────────────
def run_pinn(geom, mat, config_path=None):
    print("\n" + "="*55)
    print("X-PINN SOLVER")
    print("="*55)

    if config_path:
        import yaml
        with open(config_path) as f: c = yaml.safe_load(f)
        t = c["training"]
        cfg = TrainingConfig(
            num_domain=t["num_domain"], num_boundary=t["num_boundary"],
            num_crack_surface=t.get("num_crack_surface", 100),
            adam_iterations=t["adam_iterations"], adam_lr=t["adam_lr"],
            use_lbfgs=t.get("use_lbfgs", False),
            lbfgs_iterations=t.get("lbfgs_iterations", 1000),
            adaptive_sampling=t.get("adaptive_sampling", True),
            resample_every=t.get("resample_every", 500),
            loss_weight_pde=t.get("loss_weight_pde", 1.),
            loss_weight_bc=t.get("loss_weight_bc", 10.),
            loss_weight_crack=t.get("loss_weight_crack", 10.),
        )
    else:
        cfg = TrainingConfig(
            num_domain=2000, num_boundary=100, num_crack_surface=100,
            adam_iterations=3000, adam_lr=1e-3, use_lbfgs=False,
            adaptive_sampling=True, resample_every=500,
        )

    solver = XFEMCrackSolver(geom, mat, cfg)
    solver.setup()
    solver.train()

    x_test = make_grid()
    stress  = compute_stress_components(x_test, solver.net, mat)
    ki_j    = compute_ki_j_integral(solver.predict, geom, mat, radii=J_RADII)
    ki_disp = compute_ki_displacement_extrapolation(solver.predict, geom.tip1, geom, mat)

    print(f"  K_I J-integral (tip1): {ki_j['tip1']['K_I_mean']:.5f} "
          f"± {ki_j['tip1']['K_I_std']:.5f}")
    print(f"  K_I J-integral (tip2): {ki_j['tip2']['K_I_mean']:.5f} "
          f"± {ki_j['tip2']['K_I_std']:.5f}")
    print(f"  K_I COD extrap (tip1): {ki_disp['K_I_extrapolated']:.5f}")
    return solver, x_test, stress, ki_j, ki_disp


# ── comparison plots ──────────────────────────────────────────────────────────
def plot_stress_comparison(x_test, stress_fem, stress_pinn, crack,
                           save="comparison_stress.png"):
    """2-row, 4-column panel: row0=FEM, row1=PINN."""
    n = int(np.sqrt(len(x_test)))
    X = x_test[:, 0].reshape(n, n)
    Y = x_test[:, 1].reshape(n, n)
    crack_x = [crack.tip2[0], crack.tip1[0]]
    crack_y = [crack.center[1], crack.center[1]]

    keys    = ["sigma_xx", "sigma_yy", "sigma_xy", "von_mises"]
    titles  = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$", r"Von Mises"]
    cmaps   = ["RdBu_r", "RdBu_r", "RdBu_r", "jet"]
    labels  = ["FEM (slit mesh)", "X-PINN"]
    stresses = [stress_fem, stress_pinn]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for row, (label, stress) in enumerate(zip(labels, stresses)):
        axes[row, 0].set_ylabel(label, fontsize=13, fontweight="bold", rotation=90, labelpad=10)
        for col, (key, title, cmap) in enumerate(zip(keys, titles, cmaps)):
            ax = axes[row, col]
            raw  = stress[key]
            vmax = np.percentile(np.abs(raw), 99)
            data = raw.reshape(n, n)
            if cmap == "jet":
                im = ax.contourf(X, Y, np.clip(data, 0, vmax), levels=50, cmap=cmap)
            else:
                im = ax.contourf(X, Y, data, levels=50, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.plot(crack_x, crack_y, "k-", lw=3)
            ax.plot([crack.tip1[0], crack.tip2[0]],
                    [crack.tip1[1], crack.tip2[1]], "ko", ms=6,
                    markerfacecolor="white", markeredgewidth=2)
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("x"); ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.grid(True, alpha=0.25, ls="--")

    plt.suptitle("Stress Field Comparison: FEM vs X-PINN", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight")
    print(f"Saved: {save}")
    plt.show()


def plot_ki_comparison(ki_fem, ki_pinn, ki_analytical, save="comparison_ki.png"):
    """Bar chart + path-independence curves for both methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- bar chart ---
    methods = ["Analytical\n(Westergaard)", "FEM tip1", "FEM tip2", "X-PINN tip1", "X-PINN tip2"]
    values  = [
        ki_analytical,
        ki_fem["tip1"]["K_I_mean"],  ki_fem["tip2"]["K_I_mean"],
        ki_pinn["tip1"]["K_I_mean"] if ki_pinn else 0,
        ki_pinn["tip2"]["K_I_mean"] if ki_pinn else 0,
    ]
    errors = [0,
              ki_fem["tip1"]["K_I_std"],  ki_fem["tip2"]["K_I_std"],
              ki_pinn["tip1"]["K_I_std"]  if ki_pinn else 0,
              ki_pinn["tip2"]["K_I_std"]  if ki_pinn else 0]
    colors = ["k", "steelblue", "cornflowerblue", "tomato", "salmon"]
    bars   = ax1.bar(methods, values, color=colors, yerr=errors,
                     capsize=5, edgecolor="black", linewidth=0.8)
    ax1.axhline(ki_analytical, color="k", ls="--", lw=1.5, alpha=0.5)
    ax1.set_ylabel("K_I"); ax1.set_title("K_I Comparison")
    ax1.tick_params(axis="x", labelsize=9)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # --- path-independence curves ---
    r_fem  = ki_fem["tip1"]["radii"]
    ax2.plot(r_fem, ki_fem["tip1"]["K_I_values"],  "s-",  color="steelblue",  lw=2, ms=7, label="FEM tip1")
    ax2.plot(r_fem, ki_fem["tip2"]["K_I_values"],  "s--", color="cornflowerblue", lw=2, ms=7, label="FEM tip2")
    if ki_pinn:
        r_pinn = ki_pinn["tip1"]["radii"]
        ax2.plot(r_pinn, ki_pinn["tip1"]["K_I_values"], "o-",  color="tomato",  lw=2, ms=7, label="PINN tip1")
        ax2.plot(r_pinn, ki_pinn["tip2"]["K_I_values"], "o--", color="salmon",  lw=2, ms=7, label="PINN tip2")
    ax2.axhline(ki_analytical, color="k", ls="--", lw=2, label=f"Analytical={ki_analytical:.4f}")
    ax2.set_xlabel("Contour radius ρ")
    ax2.set_ylabel("K_I")
    ax2.set_title("K_I Path Independence (J-Integral)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.suptitle("Stress Intensity Factor: FEM vs X-PINN", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight")
    print(f"Saved: {save}")
    plt.show()


def print_summary(ki_analytical, ki_fem, ki_pinn=None):
    print("\n" + "="*65)
    print("SUMMARY — Forward Problem: Mode I Central Crack")
    print("="*65)
    print(f"{'Method':<30} {'K_I (tip1)':<15} {'K_I (tip2)':<15} {'Err %':<10}")
    print("-"*65)

    def err(v): return abs(v - ki_analytical) / ki_analytical * 100

    print(f"{'Analytical (Westergaard)':<30} {ki_analytical:<15.5f} {'—':<15} {'0.00':<10}")
    fem_k1 = ki_fem["tip1"]["K_I_mean"]; fem_k2 = ki_fem["tip2"]["K_I_mean"]
    print(f"{'FEM (Q4 slit, 80×80)':<30} {fem_k1:<15.5f} {fem_k2:<15.5f} {err(fem_k1):<10.2f}")
    if ki_pinn:
        p1 = ki_pinn["tip1"]["K_I_mean"]; p2 = ki_pinn["tip2"]["K_I_mean"]
        print(f"{'X-PINN':<30} {p1:<15.5f} {p2:<15.5f} {err(p1):<10.2f}")
    print("="*65)
    print("\nNotes:")
    print("  FEM  — J-integral K_I limited by mesh resolution near tip;")
    print("          quarter-point elements or mesh refinement improves accuracy.")
    print("  PINN — XFEM enrichment (√r tip functions) captures singularity")
    print("          analytically, giving better K_I without mesh refinement.")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn",   action="store_true", help="Also train and compare X-PINN")
    parser.add_argument("--config", default=None,        help="Path to YAML config for PINN")
    args = parser.parse_args()

    geom, mat = make_shared()
    ki_analytical = westergaard_ki_analytical(mat, geom)
    print(f"\nAnalytical K_I (Westergaard + Feddersen): {ki_analytical:.5f}")

    # Always run FEM (fast: ~1–5 s)
    fem, x_test, stress_fem, ki_fem, ki_disp_fem = run_fem(geom, mat)

    stress_pinn = ki_pinn = None
    if args.pinn:
        solver, x_test, stress_pinn, ki_pinn, _ = run_pinn(geom, mat, args.config)

    print_summary(ki_analytical, ki_fem, ki_pinn)

    # Plots
    if stress_pinn is not None:
        plot_stress_comparison(x_test, stress_fem, stress_pinn, geom)
        plot_ki_comparison(ki_fem, ki_pinn, ki_analytical)
    else:
        # FEM-only plots
        from xpinn_fracture.visualization import plot_stress_field, plot_ki_path_independence
        plot_stress_field(x_test, stress_fem, geom, save_path="fem_stress_field.png")
        plot_ki_path_independence(ki_fem, ki_analytical, save_path="fem_ki_path.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
