#!/usr/bin/env python3
"""
X-PINN for 2D Mode I fracture mechanics — single-file version.

Usage:
    python pinn_crack.py                             # quick run (built-in defaults)
    python pinn_crack.py --config configs/mode1_quick.yaml
"""
import os, argparse
import numpy as np
import torch
import torch.nn as nn
import deepxde as dde
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

# ── global precision ──────────────────────────────────────────────────────────
os.environ.setdefault("DDE_BACKEND", "pytorch")
dde.config.set_default_float("float64")
torch.set_default_dtype(torch.float64)

try:
    from deepxde.icbc import PointSetOperatorBC
except ImportError:
    raise ImportError("Upgrade deepxde: pip install 'deepxde>=1.10.0'")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Material:
    E: float = 1.0
    nu: float = 0.3
    traction: float = 0.1
    formulation: str = "plane_stress"

    @property
    def mu(self):  return self.E / (2 * (1 + self.nu))

    @property
    def lam(self):
        if self.formulation == "plane_stress":
            return self.E * self.nu / (1 - self.nu**2)
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def kappa(self):
        return (3 - self.nu) / (1 + self.nu) if self.formulation == "plane_stress" else 3 - 4 * self.nu


@dataclass
class Config:
    num_domain: int = 2000
    num_boundary: int = 100
    num_crack_surface: int = 100
    num_test: int = 200
    adam_iterations: int = 3000
    adam_lr: float = 1e-3
    use_lbfgs: bool = False
    lbfgs_iterations: int = 1000
    adaptive_sampling: bool = True
    resample_every: int = 500
    loss_weight_pde: float = 1.0
    loss_weight_bc: float = 10.0
    loss_weight_crack: float = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Crack:
    center: np.ndarray
    half_length: float
    angle: float = 0.0
    domain_size: tuple = (1.0, 1.0)
    enrichment_radius: float = 0.12
    enrichment_inner: float = 0.05

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        dx = self.half_length * np.cos(self.angle)
        dy = self.half_length * np.sin(self.angle)
        self.tip1 = self.center + np.array([dx, dy])
        self.tip2 = self.center - np.array([dx, dy])

    def polar_from_tip(self, x, tip):
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        dx, dy = x[:, 0:1] - tip[0], x[:, 1:2] - tip[1]
        xl =  dx * cos_a + dy * sin_a
        yl = -dx * sin_a + dy * cos_a
        r = np.sqrt(xl**2 + yl**2 + 1e-14)
        return r, np.arctan2(yl, xl)

    def crack_surface_points(self, n=100):
        xs = np.linspace(self.tip2[0], self.tip1[0], n)
        y0 = self.center[1]
        upper = np.column_stack([xs, np.full(n, y0 + 1e-5)])
        lower = np.column_stack([xs, np.full(n, y0 - 1e-5)])
        return np.vstack([upper, lower])


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK  u = u_C + H·u_D + ψ·u_S
# ══════════════════════════════════════════════════════════════════════════════
def _fnn(sizes, act="tanh"):
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "swish": nn.SiLU}
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(acts.get(act, nn.Tanh)())
    return nn.Sequential(*layers)


class XPINNNet(nn.Module):
    def __init__(self, crack: Crack):
        super().__init__()
        self._cos_a = float(np.cos(crack.angle))
        self._sin_a = float(np.sin(crack.angle))
        self._cx, self._cy = float(crack.center[0]), float(crack.center[1])
        self._tip1 = crack.tip1.tolist()
        self._tip2 = crack.tip2.tolist()
        self._r_s, self._r_b = crack.enrichment_inner, crack.enrichment_radius

        self.net_C = _fnn([2, 40, 40, 40, 40, 40, 40, 2])   # smooth background
        self.net_D = _fnn([3, 20, 20, 20, 20, 2])            # discontinuous jump
        self.net_S = _fnn([10, 20, 20, 20, 20, 2])           # singular near-tip
        self.regularizer = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _tip_feats(self, x, tip):
        dx, dy = x[:, 0:1] - tip[0], x[:, 1:2] - tip[1]
        xl =  dx * self._cos_a + dy * self._sin_a
        yl = -dx * self._sin_a + dy * self._cos_a
        r  = torch.sqrt(xl**2 + yl**2 + 1e-14)
        th = torch.atan2(yl, xl)
        sr = torch.sqrt(r)
        return torch.cat([sr*torch.sin(th/2), sr*torch.cos(th/2),
                          sr*torch.sin(th/2)*torch.sin(th),
                          sr*torch.cos(th/2)*torch.sin(th)], dim=1)

    def forward(self, x):
        # Heaviside: sign of perpendicular distance from crack line
        dx, dy = x[:, 0:1] - self._cx, x[:, 1:2] - self._cy
        y_loc = -dx * self._sin_a + dy * self._cos_a
        H = torch.sign(y_loc)
        H = torch.where(H == 0, torch.ones_like(H), H)

        # Partition-of-unity C2 smoothstep
        d1 = torch.sqrt((x[:, 0:1]-self._tip1[0])**2 + (x[:, 1:2]-self._tip1[1])**2)
        d2 = torch.sqrt((x[:, 0:1]-self._tip2[0])**2 + (x[:, 1:2]-self._tip2[1])**2)
        t  = torch.clamp((torch.minimum(d1, d2) - self._r_s) / (self._r_b - self._r_s), 0., 1.)
        psi = 1. - 3.*t**2 + 2.*t**3

        Phi = torch.cat([self._tip_feats(x, self._tip1), self._tip_feats(x, self._tip2)], dim=1)

        return self.net_C(x) + H * self.net_D(torch.cat([x, H], dim=1)) \
                             + psi * self.net_S(torch.cat([x, Phi], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS  (2D linear elasticity, equilibrium + BCs)
# ══════════════════════════════════════════════════════════════════════════════
def _stresses_dde(x, y, mat: Material):
    u_x, u_y = y[:, 0:1], y[:, 1:2]
    lam, mu = mat.lam, mat.mu
    u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)
    u_xy = dde.grad.jacobian(u_x, x, i=0, j=1)
    u_yx = dde.grad.jacobian(u_y, x, i=0, j=0)
    u_yy = dde.grad.jacobian(u_y, x, i=0, j=1)
    tr = u_xx + u_yy
    return lam*tr + 2*mu*u_xx, lam*tr + 2*mu*u_yy, mu*(u_xy + u_yx)


def pde_residual(mat: Material):
    def _pde(x, y):
        s_xx, s_yy, s_xy = _stresses_dde(x, y, mat)
        eq_x = dde.grad.jacobian(s_xx, x, i=0, j=0) + dde.grad.jacobian(s_xy, x, i=0, j=1)
        eq_y = dde.grad.jacobian(s_xy, x, i=0, j=0) + dde.grad.jacobian(s_yy, x, i=0, j=1)
        return [eq_x, eq_y]
    return _pde


def build_bcs(geom_dde, crack: Crack, mat: Material, n_crack=100):
    H_dom = crack.domain_size[1]
    t = mat.traction
    on_bottom = lambda x, b: b and np.isclose(x[1], 0.0)
    on_top    = lambda x, b: b and np.isclose(x[1], H_dom)
    on_left   = lambda x, b: b and np.isclose(x[0], 0.0)

    bc_bot  = dde.icbc.DirichletBC(geom_dde, lambda x: np.full((len(x),1), -t), on_bottom, component=1)
    bc_top  = dde.icbc.DirichletBC(geom_dde, lambda x: np.full((len(x),1),  t), on_top,    component=1)
    bc_left = dde.icbc.DirichletBC(geom_dde, lambda x: np.zeros((len(x),1)),    on_left,   component=0)

    cpts = crack.crack_surface_points(n_crack // 2)
    zeros = np.zeros((len(cpts), 1))

    def syy(x, y, X): return _stresses_dde(x, y, mat)[1]
    def sxy(x, y, X): return _stresses_dde(x, y, mat)[2]

    bc_syy = PointSetOperatorBC(cpts, zeros, syy)
    bc_sxy = PointSetOperatorBC(cpts, zeros, sxy)
    return [bc_bot, bc_top, bc_left, bc_syy, bc_sxy], cpts


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER
# ══════════════════════════════════════════════════════════════════════════════
class Solver:
    def __init__(self, crack: Crack, mat: Material, cfg: Config):
        self.crack, self.mat, self.cfg = crack, mat, cfg
        self.net: Optional[XPINNNet] = None
        self.model = None
        self.loss_history = None

    def setup(self):
        geom_dde = dde.geometry.Rectangle([0., 0.], list(self.crack.domain_size))
        bcs, cpts = build_bcs(geom_dde, self.crack, self.mat, self.cfg.num_crack_surface)
        self._n_terms = 7  # 2 PDE + 3 BC + 2 crack
        self.data = dde.data.PDE(
            geom_dde, pde_residual(self.mat), bcs,
            num_domain=self.cfg.num_domain,
            num_boundary=self.cfg.num_boundary,
            num_test=self.cfg.num_test,
            anchors=cpts,
        )
        self.net = XPINNNet(self.crack)
        self.model = dde.Model(self.data, self.net)
        self._geom_dde = geom_dde

    def _weights(self):
        c = self.cfg
        return [c.loss_weight_pde]*2 + [c.loss_weight_bc]*3 + [c.loss_weight_crack]*2

    def _pde_residual_mag(self, X):
        x_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            y_t = self.net(x_t)
            lam, mu = self.mat.lam, self.mat.mu
            def jac(u, j): return torch.autograd.grad(u.sum(), x_t, create_graph=True)[0][:, j:j+1]
            u_x, u_y = y_t[:, 0:1], y_t[:, 1:2]
            u_xx, u_xy = jac(u_x, 0), jac(u_x, 1)
            u_yx, u_yy = jac(u_y, 0), jac(u_y, 1)
            tr = u_xx + u_yy
            s_xx = lam*tr + 2*mu*u_xx; s_yy = lam*tr + 2*mu*u_yy; s_xy = mu*(u_xy+u_yx)
            eq_x = jac(s_xx, 0) + jac(s_xy, 1)
            eq_y = jac(s_xy, 0) + jac(s_yy, 1)
            return (eq_x**2 + eq_y**2).detach().numpy().flatten()

    def _rad_resample(self, n):
        X = self._geom_dde.random_points(10000)
        w = np.abs(self._pde_residual_mag(X)) + 1e-14
        w = w / w.mean() + self.cfg.loss_weight_pde
        w /= w.sum()
        return X[np.random.choice(len(X), size=min(n, len(X)), replace=False, p=w)]

    def train(self):
        cfg, weights = self.cfg, self._weights()
        all_loss = []

        print("--- Adam ---")
        self.model.compile("adam", lr=cfg.adam_lr, loss_weights=weights)
        t0 = perf_counter()
        if cfg.adaptive_sampling and cfg.resample_every > 0:
            for epoch in range(0, cfg.adam_iterations, cfg.resample_every):
                n_iters = min(cfg.resample_every, cfg.adam_iterations - epoch)
                lh, _ = self.model.train(iterations=n_iters, display_every=200)
                raw = np.asarray(lh.loss_train)
                all_loss.extend((raw.sum(1) if raw.ndim > 1 else raw).tolist())
                if epoch + n_iters < cfg.adam_iterations:
                    X_new = self._rad_resample(cfg.num_domain)
                    self.data.train_x_all = self.data.train_x = X_new
        else:
            lh, _ = self.model.train(iterations=cfg.adam_iterations, display_every=500)
            raw = np.asarray(lh.loss_train)
            all_loss = (raw.sum(1) if raw.ndim > 1 else raw).tolist()
        print(f"Adam done in {perf_counter()-t0:.1f}s")

        if cfg.use_lbfgs:
            print("--- L-BFGS ---")
            self.model.compile("L-BFGS", loss_weights=weights)
            lh2, _ = self.model.train(iterations=cfg.lbfgs_iterations)
            raw2 = np.asarray(lh2.loss_train)
            all_loss.extend((raw2.sum(1) if raw2.ndim > 1 else raw2).tolist())
            lh2.loss_train = all_loss
            self.loss_history = lh2
        else:
            lh.loss_train = all_loss
            self.loss_history = lh

    def predict(self, x): return np.asarray(self.model.predict(x), dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def stress_components(x_test, net, mat: Material):
    x_t = torch.tensor(x_test, dtype=torch.float64, requires_grad=True)
    with torch.enable_grad():
        y_t = net(x_t)
        u_x, u_y = y_t[:, 0:1], y_t[:, 1:2]
        gx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=False, retain_graph=True)[0]
        gy = torch.autograd.grad(u_y, x_t, torch.ones_like(u_y), create_graph=False)[0]
        lam, mu = mat.lam, mat.mu
        tr = gx[:, 0:1] + gy[:, 1:2]
        s_xx = lam*tr + 2*mu*gx[:, 0:1]
        s_yy = lam*tr + 2*mu*gy[:, 1:2]
        s_xy = mu*(gx[:, 1:2] + gy[:, 0:1])
        vm = torch.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2 + 1e-14)
    return {k: v.detach().numpy().flatten()
            for k, v in zip(["sigma_xx","sigma_yy","sigma_xy","von_mises"], [s_xx,s_yy,s_xy,vm])}


def j_integral(predict_fn, tip, mat: Material, rho=0.04, n=500):
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.column_stack([tip[0] + rho*np.cos(phi), tip[1] + rho*np.sin(phi)])
    h = rho * 1e-3
    u_px = predict_fn(pts + [h, 0]); u_mx = predict_fn(pts - [h, 0])
    u_py = predict_fn(pts + [0, h]); u_my = predict_fn(pts - [0, h])
    du_x_dx = (u_px[:,0]-u_mx[:,0])/(2*h); du_x_dy = (u_py[:,0]-u_my[:,0])/(2*h)
    du_y_dx = (u_px[:,1]-u_mx[:,1])/(2*h); du_y_dy = (u_py[:,1]-u_my[:,1])/(2*h)
    lam, mu = mat.lam, mat.mu
    eps_xy = 0.5*(du_x_dy + du_y_dx)
    tr = du_x_dx + du_y_dy
    s_xx = lam*tr + 2*mu*du_x_dx; s_yy = lam*tr + 2*mu*du_y_dy; s_xy = 2*mu*eps_xy
    W = 0.5*(s_xx*du_x_dx + s_yy*du_y_dy + 2*s_xy*eps_xy)
    nx, ny = np.cos(phi), np.sin(phi)
    integrand = W*nx - ((s_xx*nx + s_xy*ny)*du_x_dx + (s_xy*nx + s_yy*ny)*du_y_dx)
    return float(np.trapezoid(integrand, dx=2*np.pi/n) * rho)


def ki_j_integral(predict_fn, crack: Crack, mat: Material, radii=(0.03, 0.05, 0.07)):
    results = {}
    for name, tip in [("tip1", crack.tip1), ("tip2", crack.tip2)]:
        ki_vals = [np.sqrt(abs(j := j_integral(predict_fn, tip, mat, rho=r)) * mat.E) * np.sign(j)
                   for r in radii]
        results[name] = dict(radii=list(radii), K_I_values=ki_vals,
                             K_I_mean=float(np.mean(ki_vals)), K_I_std=float(np.std(ki_vals)),
                             path_independence_pct=float(np.std(ki_vals)/(abs(np.mean(ki_vals))+1e-14)*100))
    return results


def ki_displacement_extrap(predict_fn, tip, crack: Crack, mat: Material, n=50):
    r = np.linspace(0.005, 0.06, n)
    upper = np.column_stack([tip[0]-r, np.full(n, tip[1]+1e-5)])
    lower = np.column_stack([tip[0]-r, np.full(n, tip[1]-1e-5)])
    cod   = predict_fn(upper)[:,1] - predict_fn(lower)[:,1]
    ki    = (mat.mu*(mat.kappa+1)/4) * np.sqrt(2*np.pi/r) * cod
    coeffs = np.polyfit(np.sqrt(r), ki, 1)
    return dict(r_values=r.tolist(), K_I_values=ki.tolist(),
                K_I_extrapolated=float(coeffs[1]), K_I_near_tip=float(ki[0]))


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def westergaard_ki(mat: Material, crack: Crack):
    a, (W, H) = crack.half_length, crack.domain_size
    sigma = mat.E * (2*mat.traction / H)
    F = np.sqrt(1.0 / np.cos(np.pi * a / W))
    return float(sigma * np.sqrt(np.pi * a) * F)


def validate(predict_fn, crack: Crack, mat: Material, ki_analytical, ki_pinn, n=500, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    for name, tip in [("tip1", crack.tip1), ("tip2", crack.tip2)]:
        r = rng.uniform(0.01, crack.enrichment_inner, n)
        th = rng.uniform(-np.pi+1e-6, np.pi-1e-6, n)
        cos_a, sin_a = np.cos(crack.angle), np.sin(crack.angle)
        xl, yl = r*np.cos(th), r*np.sin(th)
        xg = tip[0] + xl*cos_a - yl*sin_a
        yg = tip[1] + xl*sin_a + yl*cos_a
        pts = np.column_stack([xg, yg])
        mask = ((pts[:,0]>=0)&(pts[:,0]<=crack.domain_size[0])&
                (pts[:,1]>=0)&(pts[:,1]<=crack.domain_size[1]))
        pts = pts[mask]
        if len(pts) == 0: continue
        u_pinn = predict_fn(pts)
        # Williams leading-term reference
        r_w, th_w = crack.polar_from_tip(pts, tip)
        fac = ki_analytical / (2*mat.mu) * np.sqrt(r_w / (2*np.pi))
        u_ref = np.concatenate([
            fac * np.cos(th_w/2) * (mat.kappa - 1 + 2*np.sin(th_w/2)**2),
            fac * np.sin(th_w/2) * (mat.kappa + 1 - 2*np.cos(th_w/2)**2),
        ], axis=1)
        l2 = np.linalg.norm(u_pinn - u_ref) / (np.linalg.norm(u_ref) + 1e-14)
        results[name] = {"l2_vs_williams": float(l2), "n_pts": len(pts)}
    results.update(K_I_analytical=ki_analytical, K_I_pinn=ki_pinn,
                   K_I_err_pct=float(abs(ki_pinn-ki_analytical)/(abs(ki_analytical)+1e-14)*100))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_stress(x_test, stress, crack: Crack, path="stress_field.png"):
    n = int(np.sqrt(len(x_test)))
    X, Y = x_test[:,0].reshape(n,n), x_test[:,1].reshape(n,n)
    fields = [(stress["sigma_xx"],r"$\sigma_{xx}$","RdBu_r"),
              (stress["sigma_yy"],r"$\sigma_{yy}$","RdBu_r"),
              (stress["sigma_xy"],r"$\sigma_{xy}$","RdBu_r"),
              (stress["von_mises"],r"Von Mises","jet")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (raw, title, cmap) in zip(axes, fields):
        vmax = np.percentile(np.abs(raw), 99)
        data = raw.reshape(n, n)
        im = ax.contourf(X, Y, np.clip(data, 0, vmax) if cmap=="jet" else data,
                         levels=50, cmap=cmap,
                         **({} if cmap=="jet" else {"vmin":-vmax,"vmax":vmax}))
        ax.plot([crack.tip2[0], crack.tip1[0]], [crack.center[1]]*2, "k-", lw=3)
        ax.plot([crack.tip1[0], crack.tip2[0]], [crack.tip1[1], crack.tip2[1]], "ko", ms=7,
                markerfacecolor="white", markeredgewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}"); plt.show()


def plot_loss(lh, path="loss_history.png"):
    if lh is None: return
    raw = np.asarray(lh.loss_train)
    total = raw.sum(1) if raw.ndim > 1 else raw
    win = max(1, len(total)//50)
    mean_e = np.convolve(total, np.ones(win)/win, mode="same")
    plt.figure(figsize=(10, 5))
    plt.semilogy(total, alpha=0.3, color="lightblue", label="Raw loss")
    plt.semilogy(mean_e, "b-", lw=2, label="Rolling mean")
    plt.xlabel("Epoch"); plt.ylabel("Total Loss")
    plt.title("Training Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}"); plt.show()


def plot_ki_paths(ki_results, ki_analytical, path="ki_path_independence.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for tip, c in zip(["tip1","tip2"], ["steelblue","tomato"]):
        r = ki_results[tip]
        ax.plot(r["radii"], r["K_I_values"], "o-", color=c, lw=2, ms=7,
                label=f"{tip}  mean={r['K_I_mean']:.4f}")
    ax.axhline(ki_analytical, color="k", ls="--", lw=2, label=f"Analytical={ki_analytical:.4f}")
    ax.set_xlabel("Contour radius"); ax.set_ylabel("K_I")
    ax.set_title("K_I Path-Independence"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}"); plt.show()


def plot_extrap(extrap, ki_analytical, path="ki_extrapolation.png"):
    r, ki = np.array(extrap["r_values"]), np.array(extrap["K_I_values"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.sqrt(r), ki, "b.-", lw=1.5, ms=4, label="K_I from COD")
    ax.axhline(ki_analytical,            color="k",      ls="--", lw=2, label=f"Analytical={ki_analytical:.4f}")
    ax.axhline(extrap["K_I_extrapolated"],color="tomato", ls="-.", lw=2, label=f"Extrapolated={extrap['K_I_extrapolated']:.4f}")
    ax.set_xlabel("sqrt(r)"); ax.set_ylabel("K_I")
    ax.set_title("K_I via COD Extrapolation"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}"); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run(crack: Crack = None, mat: Material = None, cfg: Config = None):
    if crack is None:
        crack = Crack(center=np.array([0.5, 0.5]), half_length=0.15,
                      domain_size=(1.0, 1.0), enrichment_radius=0.12, enrichment_inner=0.05)
    if mat is None:
        mat = Material(E=1.0, nu=0.3, traction=0.1)
    if cfg is None:
        cfg = Config()

    ki_ana = westergaard_ki(mat, crack)
    print(f"\nAnalytical K_I: {ki_ana:.6f}\n")

    solver = Solver(crack, mat, cfg)
    solver.setup()
    solver.train()

    n = 100
    xs, ys = np.linspace(0, crack.domain_size[0], n), np.linspace(0, crack.domain_size[1], n)
    x_test = np.column_stack([v.ravel() for v in np.meshgrid(xs, ys)])

    stress = stress_components(x_test, solver.net, mat)

    ki_j    = ki_j_integral(solver.predict, crack, mat)
    ki_disp = ki_displacement_extrap(solver.predict, crack.tip1, crack, mat)
    ki_pinn = (ki_j["tip1"]["K_I_mean"] + ki_j["tip2"]["K_I_mean"]) / 2

    val = validate(solver.predict, crack, mat, ki_ana, ki_pinn)

    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)
    print(f"  K_I analytical:           {ki_ana:.6f}")
    print(f"  K_I J-integral (tip1):    {ki_j['tip1']['K_I_mean']:.6f}")
    print(f"  K_I J-integral (tip2):    {ki_j['tip2']['K_I_mean']:.6f}")
    print(f"  K_I COD extrap:           {ki_disp['K_I_extrapolated']:.6f}")
    print(f"  K_I relative error:       {val['K_I_err_pct']:.2f}%")
    print(f"  Path-independence tip1:   {ki_j['tip1']['path_independence_pct']:.2f}%")
    if "tip1" in val:
        print(f"  L2 vs Williams (tip1):    {val['tip1']['l2_vs_williams']:.4f}")
    print("="*55)

    plot_stress(x_test, stress, crack)
    plot_loss(solver.loss_history)
    plot_ki_paths(ki_j, ki_ana)
    plot_extrap(ki_disp, ki_ana)
    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f: c = yaml.safe_load(f)
        g, m, t = c["geometry"], c["material"], c["training"]
        crack = Crack(center=np.array(g["center"]), half_length=g["half_length"],
                      angle=g.get("angle", 0.), domain_size=tuple(g["domain_size"]),
                      enrichment_radius=g["enrichment_radius"], enrichment_inner=g["enrichment_inner"])
        mat = Material(E=m["E"], nu=m["nu"], traction=m["traction"], formulation=m["formulation"])
        cfg = Config(num_domain=t["num_domain"], num_boundary=t["num_boundary"],
                     num_crack_surface=t.get("num_crack_surface", 100),
                     adam_iterations=t["adam_iterations"], adam_lr=t["adam_lr"],
                     use_lbfgs=t.get("use_lbfgs", False), lbfgs_iterations=t.get("lbfgs_iterations", 1000),
                     adaptive_sampling=t.get("adaptive_sampling", True),
                     resample_every=t.get("resample_every", 500),
                     loss_weight_pde=t.get("loss_weight_pde", 1.),
                     loss_weight_bc=t.get("loss_weight_bc", 10.),
                     loss_weight_crack=t.get("loss_weight_crack", 10.))
        run(crack, mat, cfg)
    else:
        run()
