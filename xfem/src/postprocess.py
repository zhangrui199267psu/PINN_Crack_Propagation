"""
Post-processing for the FEM solver: K_I extraction and data saving.
Pure numpy — no torch or deepxde dependency.
"""

import json
import numpy as np

_FD   = 1e-3   # central-difference step as fraction of contour radius
_COD  = 1e-5   # crack-face offset (matches slit mesh offset)
_EPS  = 1e-14  # guard against division by zero


# ── material helpers ───────────────────────────────────────────────────────────
def _elastic_constants(E, nu, formulation="plane_stress"):
    mu  = E / (2*(1+nu))
    lam = E*nu/(1-nu**2) if formulation == "plane_stress" else E*nu/((1+nu)*(1-2*nu))
    kappa = (3-nu)/(1+nu) if formulation == "plane_stress" else 3-4*nu
    return lam, mu, kappa


# ── analytical reference ───────────────────────────────────────────────────────
def westergaard_ki(E, nu, traction, half_length, domain_size, formulation="plane_stress"):
    """Westergaard K_I with Feddersen finite-width correction."""
    W, H = domain_size
    sigma_inf = E * (2*traction / H)
    F = np.sqrt(1.0 / np.cos(np.pi * half_length / W))
    return float(sigma_inf * np.sqrt(np.pi * half_length) * F)


# ── J-integral K_I ────────────────────────────────────────────────────────────
def _central_diff(predict_fn, pts, h):
    u_px = predict_fn(pts + [h, 0]); u_mx = predict_fn(pts - [h, 0])
    u_py = predict_fn(pts + [0, h]); u_my = predict_fn(pts - [0, h])
    return ((u_px[:,0]-u_mx[:,0])/(2*h), (u_py[:,0]-u_my[:,0])/(2*h),
            (u_px[:,1]-u_mx[:,1])/(2*h), (u_py[:,1]-u_my[:,1])/(2*h))


def j_integral(predict_fn, tip, E, nu, rho=0.04, n=500, formulation="plane_stress"):
    lam, mu, _ = _elastic_constants(E, nu, formulation)
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.c_[tip[0]+rho*np.cos(phi), tip[1]+rho*np.sin(phi)]
    h   = rho * _FD
    dux_dx, dux_dy, duy_dx, duy_dy = _central_diff(predict_fn, pts, h)
    eps_xy = 0.5*(dux_dy+duy_dx)
    tr = dux_dx+duy_dy
    sxx = lam*tr+2*mu*dux_dx; syy = lam*tr+2*mu*duy_dy; sxy = 2*mu*eps_xy
    W   = 0.5*(sxx*dux_dx + syy*duy_dy + 2*sxy*eps_xy)
    nx, ny = np.cos(phi), np.sin(phi)
    integ = W*nx - ((sxx*nx+sxy*ny)*dux_dx + (sxy*nx+syy*ny)*duy_dx)
    return float(np.trapezoid(integ, dx=2*np.pi/n) * rho)


def compute_ki_j_integral(predict_fn, tip1, tip2, E, nu,
                           radii=(0.03, 0.05, 0.07), formulation="plane_stress"):
    results = {}
    for name, tip in [("tip1", tip1), ("tip2", tip2)]:
        ki_vals = []
        for rho in radii:
            J = j_integral(predict_fn, tip, E, nu, rho=rho, formulation=formulation)
            ki = np.sqrt(abs(J)*E)*np.sign(J) if J != 0 else 0.0
            ki_vals.append(float(ki))
        results[name] = {
            "radii":               list(radii),
            "K_I_values":          ki_vals,
            "K_I_mean":            float(np.mean(ki_vals)),
            "K_I_std":             float(np.std(ki_vals)),
            "path_independence_pct": float(np.std(ki_vals)/(abs(np.mean(ki_vals))+_EPS)*100),
        }
    return results


# ── COD K_I ───────────────────────────────────────────────────────────────────
def compute_ki_cod(predict_fn, tip, E, nu, formulation="plane_stress", n=50):
    lam, mu, kappa = _elastic_constants(E, nu, formulation)
    r   = np.linspace(0.005, 0.06, n)
    pts_up  = np.c_[tip[0]-r, np.full(n, tip[1]+_COD)]
    pts_lo  = np.c_[tip[0]-r, np.full(n, tip[1]-_COD)]
    cod     = predict_fn(pts_up)[:,1] - predict_fn(pts_lo)[:,1]
    ki      = (mu*(kappa+1)/4) * np.sqrt(2*np.pi/r) * cod
    coeffs  = np.polyfit(np.sqrt(r), ki, 1)
    return {
        "r_values":         r.tolist(),
        "K_I_values":       ki.tolist(),
        "K_I_extrapolated": float(coeffs[1]),
        "K_I_near_tip":     float(ki[0]),
    }


# ── save results ───────────────────────────────────────────────────────────────
def save_results(data_dir, x_test, stress, u_pred, ki_j, ki_cod, ki_analytical):
    """
    Save all numerical results to data_dir.

    Files written
    -------------
    stress_field.npz   : x_test + sigma_xx/yy/xy + von_mises
    displacement.npz   : x_test + u_x + u_y
    ki_results.json    : K_I from J-integral, COD, and analytical reference
    """
    import os; os.makedirs(data_dir, exist_ok=True)

    np.savez(f"{data_dir}/stress_field.npz",
             x_test     = x_test,
             sigma_xx   = stress["sigma_xx"],
             sigma_yy   = stress["sigma_yy"],
             sigma_xy   = stress["sigma_xy"],
             von_mises  = stress["von_mises"])

    np.savez(f"{data_dir}/displacement.npz",
             x_test = x_test,
             u_x    = u_pred[:, 0],
             u_y    = u_pred[:, 1])

    ki_out = {
        "analytical":  ki_analytical,
        "j_integral":  ki_j,
        "cod":         ki_cod,
    }
    with open(f"{data_dir}/ki_results.json", "w") as f:
        json.dump(ki_out, f, indent=2)

    print(f"Data saved → {data_dir}/")
