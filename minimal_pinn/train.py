"""
Personal minimal PINN crack example (2 files only):
- model.py: neural network
- train.py: physics loss, BC loss, simple training loop

Run:
    python minimal_pinn/train.py
"""

import torch
from model import CrackPINN


def pde_residual(model, x, E=1.0, nu=0.3):
    """Linear elasticity residual: div(sigma)=0 in 2D."""
    x.requires_grad_(True)
    u = model(x)
    ux, uy = u[:, 0:1], u[:, 1:2]

    lam = E * nu / (1 - nu**2)  # plane stress
    mu = E / (2 * (1 + nu))

    def grad(v, j):
        return torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, j:j+1]

    ux_x, ux_y = grad(ux, 0), grad(ux, 1)
    uy_x, uy_y = grad(uy, 0), grad(uy, 1)

    tr = ux_x + uy_y
    sxx = lam * tr + 2 * mu * ux_x
    syy = lam * tr + 2 * mu * uy_y
    sxy = mu * (ux_y + uy_x)

    rx = grad(sxx, 0) + grad(sxy, 1)
    ry = grad(sxy, 0) + grad(syy, 1)
    return (rx.pow(2) + ry.pow(2)).mean()


def bc_loss(model, n=200, load=0.1):
    """Simple Mode-I-like BCs on unit square."""
    x = torch.rand(n, 2)

    top = x.clone(); top[:, 1] = 1.0
    bot = x.clone(); bot[:, 1] = 0.0
    left = x.clone(); left[:, 0] = 0.0

    u_top = model(top)
    u_bot = model(bot)
    u_left = model(left)

    l_top = (u_top[:, 1:2] - load).pow(2).mean()
    l_bot = (u_bot[:, 1:2] + load).pow(2).mean()
    l_left = (u_left[:, 0:1]).pow(2).mean()
    return l_top + l_bot + l_left


def crack_jump_loss(model, n=200, y0=0.5, half_len=0.15, eps=1e-3):
    """Encourage displacement discontinuity across crack line segment."""
    xline = torch.rand(n, 1) * (2 * half_len) + (0.5 - half_len)
    up = torch.cat([xline, torch.full_like(xline, y0 + eps)], dim=1)
    dn = torch.cat([xline, torch.full_like(xline, y0 - eps)], dim=1)

    jump = model(up)[:, 1:2] - model(dn)[:, 1:2]
    target_jump = 0.02
    return (jump - target_jump).pow(2).mean()


def train(steps=3000, lr=1e-3):
    model = CrackPINN(width=32, depth=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(steps):
        x_domain = torch.rand(1024, 2)
        loss_pde = pde_residual(model, x_domain)
        loss_bc = bc_loss(model)
        loss_crack = crack_jump_loss(model)

        loss = loss_pde + 10.0 * loss_bc + 5.0 * loss_crack
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 500 == 0:
            print(f"step={i:4d} total={loss.item():.4e} pde={loss_pde.item():.4e} bc={loss_bc.item():.4e} crack={loss_crack.item():.4e}")

    return model


if __name__ == "__main__":
    train()
