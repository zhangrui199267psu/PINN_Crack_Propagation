#!/usr/bin/env python3
"""
XFEM (classical Q4 FEM) forward problem runner.

Usage
-----
  cd xfem
  python run.py                          # uses configs/problem.yaml
  python run.py --config configs/problem.yaml

Outputs
-------
  results/figures/stress_field.png
  results/figures/displacement.png
  results/figures/ki_path_independence.png
  results/figures/ki_cod.png
  results/data/stress_field.npz
  results/data/displacement.npz
  results/data/ki_results.json
"""

import argparse, os, sys
import numpy as np
import yaml

# allow running as  python xfem/run.py  from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.solver       import FEMCrackSolver
from src.postprocess  import (westergaard_ki, compute_ki_j_integral,
                               compute_ki_cod, save_results)
from src.visualization import (plot_stress_field, plot_displacement_field,
                                plot_ki_path_independence, plot_ki_cod)

FIG_DIR  = os.path.join(os.path.dirname(__file__), "results", "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "data")
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "problem.yaml")
    cfg = load_config(config_path)

    g   = cfg["geometry"]
    m   = cfg["material"]
    f   = cfg["fem"]
    pp  = cfg["postprocess"]

    crack_center = tuple(g["crack_center"])
    crack_half   = g["crack_half_length"]
    domain_size  = tuple(g["domain_size"])
    E, nu, T     = m["E"], m["nu"], m["traction"]
    form         = m["formulation"]

    # ── analytical reference ──────────────────────────────────────────────────
    ki_analytical = westergaard_ki(E, nu, T, crack_half, domain_size, form)
    print(f"\nAnalytical K_I (Westergaard + Feddersen): {ki_analytical:.6f}\n")

    # ── FEM solve ─────────────────────────────────────────────────────────────
    fem = FEMCrackSolver(
        Nx=f["Nx"], Ny=f["Ny"],
        E=E, nu=nu, traction=T,
        crack_center=crack_center,
        crack_half_length=crack_half,
        formulation=form,
    )
    fem.solve()

    # ── evaluation grid ───────────────────────────────────────────────────────
    n = f["n_grid"]
    xs, ys = np.linspace(0, domain_size[0], n), np.linspace(0, domain_size[1], n)
    x_test = np.column_stack([v.ravel() for v in np.meshgrid(xs, ys)])

    u_pred = fem.predict(x_test)
    stress = fem.stress_at(x_test)

    # ── K_I extraction ────────────────────────────────────────────────────────
    print("Computing K_I via J-integral...")
    ki_j = compute_ki_j_integral(
        fem.predict, fem.tip1, fem.tip2, E, nu,
        radii=pp["j_integral_radii"], formulation=form,
    )
    print(f"  tip1: {ki_j['tip1']['K_I_mean']:.6f} ± {ki_j['tip1']['K_I_std']:.6f}")
    print(f"  tip2: {ki_j['tip2']['K_I_mean']:.6f} ± {ki_j['tip2']['K_I_std']:.6f}")

    print("Computing K_I via COD extrapolation...")
    ki_cod = compute_ki_cod(
        fem.predict, fem.tip1, E, nu,
        formulation=form, n=pp["cod_n_pts"],
    )
    print(f"  COD extrapolated: {ki_cod['K_I_extrapolated']:.6f}")

    k1 = ki_j["tip1"]["K_I_mean"]
    print(f"\nK_I relative error (J-integral tip1): {abs(k1-ki_analytical)/ki_analytical*100:.2f}%")

    # ── save data ─────────────────────────────────────────────────────────────
    save_results(DATA_DIR, x_test, stress, u_pred, ki_j, ki_cod, ki_analytical)

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_stress_field(x_test, stress, fem.tip1, fem.tip2, crack_center,
                      save=f"{FIG_DIR}/stress_field.png")
    plot_displacement_field(x_test, u_pred, fem.tip1, fem.tip2, crack_center,
                            save=f"{FIG_DIR}/displacement.png")
    plot_ki_path_independence(ki_j, ki_analytical,
                              save=f"{FIG_DIR}/ki_path_independence.png")
    plot_ki_cod(ki_cod, ki_analytical,
                save=f"{FIG_DIR}/ki_cod.png")
    print("\nDone. Results in xfem/results/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args = p.parse_args()
    main(args.config)
