#!/usr/bin/env python3
"""
X-PINN forward problem runner.

Usage
-----
  cd xpinn
  python run.py                               # configs/quick.yaml
  python run.py --config configs/medium.yaml
  python run.py --config configs/production.yaml

Outputs
-------
  results/figures/stress_field.png
  results/figures/displacement.png
  results/figures/ki_path_independence.png
  results/figures/ki_cod.png
  results/figures/loss_history.png
  results/data/stress_field.npz
  results/data/displacement.npz
  results/data/ki_results.json
  results/data/loss_history.npy
"""

import argparse, os, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    MaterialProperties, TrainingConfig, CrackGeometry, XFEMCrackSolver,
    westergaard_ki_analytical, validate_against_williams,
    compute_ki_j_integral, compute_ki_displacement_extrapolation,
    compute_stress_components,
    plot_stress_field, plot_loss_history,
    plot_ki_path_independence, plot_displacement_extrapolation,
)

FIG_DIR  = os.path.join(os.path.dirname(__file__), "results", "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "data")
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

_DEFAULT = os.path.join(os.path.dirname(__file__), "configs", "quick.yaml")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_objects(cfg):
    g, m, t = cfg["geometry"], cfg["material"], cfg["training"]
    geom = CrackGeometry(
        center           = np.array(g["center"]),
        half_length      = g["half_length"],
        angle            = g.get("angle", 0.0),
        domain_size      = tuple(g["domain_size"]),
        enrichment_radius= g["enrichment_radius"],
        enrichment_inner = g["enrichment_inner"],
    )
    mat = MaterialProperties(E=m["E"], nu=m["nu"],
                             traction=m["traction"], formulation=m["formulation"])
    cfg_ = TrainingConfig(
        num_domain       = t["num_domain"],
        num_boundary     = t["num_boundary"],
        num_crack_surface= t.get("num_crack_surface", 200),
        num_test         = t.get("num_test", 200),
        adam_iterations  = t["adam_iterations"],
        adam_lr          = t["adam_lr"],
        use_lbfgs        = t.get("use_lbfgs", False),
        lbfgs_iterations = t.get("lbfgs_iterations", 1000),
        adaptive_sampling= t.get("adaptive_sampling", True),
        resample_every   = t.get("resample_every", 500),
        loss_weight_pde  = t.get("loss_weight_pde", 1.0),
        loss_weight_bc   = t.get("loss_weight_bc", 10.0),
        loss_weight_crack= t.get("loss_weight_crack", 10.0),
    )
    return geom, mat, cfg_


def save_results(x_test, stress, u_pred, ki_j, ki_cod, ki_analytical, loss_history):
    import json

    np.savez(f"{DATA_DIR}/stress_field.npz",
             x_test    = x_test,
             sigma_xx  = stress["sigma_xx"],
             sigma_yy  = stress["sigma_yy"],
             sigma_xy  = stress["sigma_xy"],
             von_mises = stress["von_mises"])

    np.savez(f"{DATA_DIR}/displacement.npz",
             x_test = x_test,
             u_x    = u_pred[:, 0],
             u_y    = u_pred[:, 1])

    if loss_history is not None:
        raw = np.asarray(loss_history.loss_train)
        total = raw.sum(1) if raw.ndim > 1 else raw
        np.save(f"{DATA_DIR}/loss_history.npy", total)

    ki_out = {
        "analytical": ki_analytical,
        "j_integral": ki_j,
        "cod":        ki_cod,
    }
    with open(f"{DATA_DIR}/ki_results.json", "w") as f:
        json.dump(ki_out, f, indent=2)

    print(f"Data saved → {DATA_DIR}/")


def main(config_path=None):
    if config_path is None:
        config_path = _DEFAULT
    cfg_raw = load_config(config_path)
    geom, mat, cfg = build_objects(cfg_raw)

    ki_analytical = westergaard_ki_analytical(mat, geom)
    print(f"\nAnalytical K_I (Westergaard + Feddersen): {ki_analytical:.6f}\n")

    # ── train ─────────────────────────────────────────────────────────────────
    solver = XFEMCrackSolver(geom, mat, cfg)
    solver.setup()
    solver.train()

    # ── evaluation grid ───────────────────────────────────────────────────────
    n  = 100
    xs, ys = np.linspace(0, 1, n), np.linspace(0, 1, n)
    x_test = np.column_stack([v.ravel() for v in np.meshgrid(xs, ys)])
    u_pred = solver.predict(x_test)

    # ── stress + K_I ─────────────────────────────────────────────────────────
    print("Computing stress fields...")
    stress = compute_stress_components(x_test, solver.net, mat)

    print("Computing K_I via J-integral...")
    ki_j = compute_ki_j_integral(solver.predict, geom, mat, radii=[0.03, 0.05, 0.07])
    print(f"  tip1: {ki_j['tip1']['K_I_mean']:.6f} ± {ki_j['tip1']['K_I_std']:.6f}")
    print(f"  tip2: {ki_j['tip2']['K_I_mean']:.6f} ± {ki_j['tip2']['K_I_std']:.6f}")

    print("Computing K_I via COD...")
    ki_cod = compute_ki_displacement_extrapolation(solver.predict, geom.tip1, geom, mat)
    print(f"  COD extrapolated: {ki_cod['K_I_extrapolated']:.6f}")

    ki_pinn = (ki_j["tip1"]["K_I_mean"] + ki_j["tip2"]["K_I_mean"]) / 2

    # ── validation ───────────────────────────────────────────────────────────
    val = validate_against_williams(solver.predict, geom, mat, ki_analytical, ki_pinn)

    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)
    print(f"  K_I analytical:          {ki_analytical:.6f}")
    print(f"  K_I J-integral (tip1):   {ki_j['tip1']['K_I_mean']:.6f}")
    print(f"  K_I J-integral (tip2):   {ki_j['tip2']['K_I_mean']:.6f}")
    print(f"  K_I COD extrap:          {ki_cod['K_I_extrapolated']:.6f}")
    print(f"  K_I relative error:      {val['K_I_relative_error_pct']:.2f}%")
    print(f"  Path-independence tip1:  {ki_j['tip1']['path_independence_pct']:.2f}%")
    if "tip1" in val:
        print(f"  L2 vs Williams (tip1):   {val['tip1']['l2_error_vs_williams']:.4f}")
    print("="*55)

    # ── save data ─────────────────────────────────────────────────────────────
    save_results(x_test, stress, u_pred, ki_j, ki_cod, ki_analytical, solver._loss_history)

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_stress_field(x_test, stress, geom, save_path=f"{FIG_DIR}/stress_field.png")
    plot_loss_history(solver._loss_history,  save_path=f"{FIG_DIR}/loss_history.png")
    plot_ki_path_independence(ki_j, ki_analytical, save_path=f"{FIG_DIR}/ki_path_independence.png")
    plot_displacement_extrapolation(ki_cod, ki_analytical, save_path=f"{FIG_DIR}/ki_cod.png")

    # displacement field
    import matplotlib.pyplot as plt
    n_g = int(np.sqrt(len(x_test)))
    X, Y = x_test[:,0].reshape(n_g,n_g), x_test[:,1].reshape(n_g,n_g)
    cx, cy = geom.center
    t1, t2 = geom.tip1, geom.tip2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (data, title) in zip(axes, [(u_pred[:,0], r"$u_x$"), (u_pred[:,1], r"$u_y$")]):
        im = ax.contourf(X, Y, data.reshape(n_g,n_g), levels=50, cmap="RdBu_r")
        ax.plot([t2[0],t1[0]], [cy,cy], "k-", lw=3)
        ax.plot([t1[0],t2[0]], [t1[1],t2[1]], "ko", ms=7, markerfacecolor="white", markeredgewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("X-PINN — Displacement Fields", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/displacement.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIG_DIR}/displacement.png"); plt.show()

    print("\nDone. Results in xpinn/results/")
    return solver


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args = p.parse_args()
    main(args.config)
