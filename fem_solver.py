#!/usr/bin/env python3
"""
Classical 2D Q4 FEM for Mode I crack — reference comparison for X-PINN.

Crack is modeled as a **slit mesh**: nodes on the crack line are split
into upper/lower sets so crack faces are traction-free by construction
(no Neumann condition needed).

Provides:
  fem.solve()          → solves K u = f
  fem.predict(pts)     → (N,2) displacements (same interface as PINN solver)
  fem.stress_at(pts)   → dict of stress components

This lets you pass fem.predict directly to the existing postprocess functions
(compute_ki_j_integral, etc.) without any changes.
"""

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    _SCIPY = True
except ImportError:
    _SCIPY = False


class FEMCrackSolver:
    """
    Structured Q4 FEM with slit mesh for a central horizontal crack.

    Same BCs as the X-PINN:
      - u_y = -traction on bottom,  u_y = +traction on top
      - u_x = 0 on left edge
      - Crack faces: traction-free (automatic via split nodes)

    Parameters
    ----------
    Nx, Ny           : mesh divisions (recommend >= 60 for decent K_I)
    crack_center     : (cx, cy) — must lie on a mesh node row
    crack_half_length: a
    """

    def __init__(self, Nx=80, Ny=80, E=1.0, nu=0.3, traction=0.1,
                 crack_center=(0.5, 0.5), crack_half_length=0.15,
                 formulation="plane_stress"):
        self.Nx, self.Ny = Nx, Ny
        self.E, self.nu, self.T = E, nu, traction
        self.cx, self.cy = crack_center
        self.a  = crack_half_length
        self.hx, self.hy = 1.0 / Nx, 1.0 / Ny

        # Crack row and column bounds (snapped to mesh)
        self.j_ck = round(self.cy / self.hy)
        self.i_s  = round((self.cx - self.a) / self.hx)
        self.i_e  = round((self.cx + self.a) / self.hx)

        # Tip coordinates
        self.tip1 = np.array([self.cx + self.a, self.cy])   # right
        self.tip2 = np.array([self.cx - self.a, self.cy])   # left

        # Plane-stress material matrix  C (3×3)
        if formulation == "plane_stress":
            c = E / (1.0 - nu**2)
            self.Cmat = c * np.array([[1, nu, 0],
                                       [nu, 1, 0],
                                       [0,  0, (1-nu)/2]])
        else:
            lam = E * nu / ((1+nu)*(1-2*nu))
            mu  = E / (2*(1+nu))
            self.Cmat = np.array([[lam+2*mu, lam, 0],
                                   [lam, lam+2*mu, 0],
                                   [0,   0,        mu]])

        self.nodes = self.elems = self.u = None

    # ── mesh ──────────────────────────────────────────────────────────────────
    def _build_mesh(self):
        Nx, Ny = self.Nx, self.Ny
        nb = (Nx+1) * (Ny+1)                 # base nodes
        ns = self.i_e - self.i_s + 1         # extra split nodes (upper crack surface)

        nodes = np.zeros((nb + ns, 2))
        for j in range(Ny+1):
            for i in range(Nx+1):
                nodes[j*(Nx+1)+i] = [i*self.hx, j*self.hy]
        for k in range(ns):                   # split nodes share position with originals
            nodes[nb+k] = nodes[self.j_ck*(Nx+1) + self.i_s + k]
        self.nodes = nodes
        self.n_nodes = nb + ns

        def nid(i, j, up=False):
            """Node index; up=True returns split (upper surface) node on crack line."""
            if up and j == self.j_ck and self.i_s <= i <= self.i_e:
                return nb + (i - self.i_s)
            return j*(Nx+1) + i

        elems = []
        for j in range(Ny):
            for i in range(Nx):
                # Elements immediately ABOVE the crack use split nodes on their bottom edge
                above = (j == self.j_ck and self.i_s <= i and i+1 <= self.i_e)
                elems.append([nid(i,   j,   above),
                               nid(i+1, j,   above),
                               nid(i+1, j+1, False),
                               nid(i,   j+1, False)])
        self.elems = np.array(elems, dtype=int)

    # ── element stiffness (precomputed once — uniform mesh) ───────────────────
    def _ke_ref(self):
        """8×8 Ke for an hx×hy rectangle (same for every element on uniform mesh)."""
        xe = np.array([0., self.hx, self.hx, 0.])
        ye = np.array([0., 0.,      self.hy, self.hy])
        gp = np.array([-1., 1.]) / np.sqrt(3.)
        Ke = np.zeros((8, 8))
        for xi in gp:
            for eta in gp:
                dNx = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)]) / 4.
                dNe = np.array([-(1-xi),  -(1+xi), (1+xi),  (1-xi)]) / 4.
                J    = np.array([[dNx@xe, dNx@ye], [dNe@xe, dNe@ye]])
                Ji   = np.linalg.inv(J);  detJ = np.linalg.det(J)
                dNdx = Ji[0,0]*dNx + Ji[0,1]*dNe
                dNdy = Ji[1,0]*dNx + Ji[1,1]*dNe
                B = np.zeros((3, 8))
                B[0, 0::2] = dNdx;  B[1, 1::2] = dNdy
                B[2, 0::2] = dNdy;  B[2, 1::2] = dNdx
                Ke += B.T @ self.Cmat @ B * detJ
        return Ke

    # ── global stiffness assembly ──────────────────────────────────────────────
    def _assemble(self):
        Ke = self._ke_ref()
        ne, nd = len(self.elems), 2*self.n_nodes

        # DOF indices per element: [2n0, 2n0+1, 2n1, 2n1+1, ...]  shape (ne, 8)
        ed = np.array([[2*n, 2*n+1] for e in self.elems for n in e]).reshape(ne, 8)

        # COO entries (Ke same for all elements on uniform mesh)
        rows = np.repeat(ed, 8, axis=1).flatten()   # row dof for each Ke[i,j]
        cols = np.tile(ed,   (1, 8)).flatten()       # col dof for each Ke[i,j]
        vals = np.tile(Ke.flatten(), ne)             # Ke[i,j] repeated ne times

        if _SCIPY:
            return coo_matrix((vals, (rows, cols)), shape=(nd, nd)).tocsr()
        K = np.zeros((nd, nd));  np.add.at(K, (rows, cols), vals);  return K

    # ── Dirichlet BCs via penalty method ──────────────────────────────────────
    def _apply_bcs(self, K, f):
        P   = 1e30
        tol = 0.4 * min(self.hx, self.hy)
        x_n, y_n = self.nodes[:, 0], self.nodes[:, 1]

        diag_add = np.zeros(2*self.n_nodes)
        rhs_add  = np.zeros(2*self.n_nodes)

        def pin(dofs, vals):
            diag_add[dofs] += P
            rhs_add[dofs]  += P * np.asarray(vals)

        bot = np.where(y_n < tol)[0];         pin(2*bot+1, -self.T)
        top = np.where(y_n > 1.0 - tol)[0];  pin(2*top+1,  self.T)
        lft = np.where(x_n < tol)[0];        pin(2*lft,    0.0)

        if _SCIPY:
            from scipy.sparse import diags
            K = K + diags(diag_add)
        else:
            np.fill_diagonal(K, K.diagonal() + diag_add)

        f += rhs_add
        return K, f

    # ── solve ─────────────────────────────────────────────────────────────────
    def solve(self):
        """Build mesh, assemble K, apply BCs, solve Ku=f. Returns displacement vector."""
        self._build_mesh()
        K = self._assemble()
        f = np.zeros(2*self.n_nodes)
        K, f = self._apply_bcs(K, f)

        self.u = spsolve(K, f) if _SCIPY else np.linalg.solve(K, f)
        print(f"FEM solved: {self.n_nodes} nodes, {len(self.elems)} elements, "
              f"{2*self.n_nodes} DOFs")
        return self.u

    # ── helpers for interpolation ──────────────────────────────────────────────
    def _locate(self, pts):
        """Return (element_id, xi, eta) for each query point — O(1) for uniform mesh."""
        pts = np.asarray(pts, dtype=float)
        i   = np.clip((pts[:, 0] / self.hx).astype(int), 0, self.Nx-1)
        j   = np.clip((pts[:, 1] / self.hy).astype(int), 0, self.Ny-1)
        xi  = np.clip(2*(pts[:, 0] - i*self.hx) / self.hx - 1., -1., 1.)
        eta = np.clip(2*(pts[:, 1] - j*self.hy) / self.hy - 1., -1., 1.)
        return j*self.Nx + i, xi, eta

    # ── predict  (matches PINN solver interface) ───────────────────────────────
    def predict(self, pts):
        """
        Interpolate displacements. pts: (N,2) array → (N,2) [u_x, u_y].
        Compatible with postprocess.compute_ki_j_integral() and friends.
        """
        eid, xi, eta = self._locate(pts)
        e = self.elems[eid]                                            # (N,4)
        N = np.c_[(1-xi)*(1-eta), (1+xi)*(1-eta),
                  (1+xi)*(1+eta), (1-xi)*(1+eta)] / 4.                # (N,4)
        ux = (N * self.u[2*e  ]).sum(1)
        uy = (N * self.u[2*e+1]).sum(1)
        return np.column_stack([ux, uy])

    # ── stress fields ──────────────────────────────────────────────────────────
    def stress_at(self, pts):
        """
        Compute stress components at pts (N,2).
        Returns dict: sigma_xx, sigma_yy, sigma_xy, von_mises — each (N,) array.
        Same key names as postprocess.compute_stress_components().
        """
        eid, xi, eta = self._locate(pts)
        e   = self.elems[eid]
        xe, ye = self.nodes[e, 0], self.nodes[e, 1]                   # (N,4)

        dNx_ = np.c_[-(1-eta), (1-eta),  (1+eta), -(1+eta)] / 4.
        dNe_ = np.c_[-(1-xi),  -(1+xi),  (1+xi),  (1-xi) ] / 4.

        J00 = (dNx_*xe).sum(1);  J01 = (dNx_*ye).sum(1)
        J10 = (dNe_*xe).sum(1);  J11 = (dNe_*ye).sum(1)
        det = J00*J11 - J01*J10

        dNdx = ( J11/det)[:,None]*dNx_ + (-J01/det)[:,None]*dNe_
        dNdy = (-J10/det)[:,None]*dNx_ + ( J00/det)[:,None]*dNe_

        ux_e, uy_e = self.u[2*e], self.u[2*e+1]                       # (N,4)
        eps = np.c_[(dNdx*ux_e).sum(1),                               # ε_xx
                    (dNdy*uy_e).sum(1),                               # ε_yy
                    (dNdy*ux_e + dNdx*uy_e).sum(1)]                   # γ_xy

        sig = eps @ self.Cmat                                          # (N,3)
        sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
        vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2 + 1e-14)
        return {"sigma_xx": sxx, "sigma_yy": syy, "sigma_xy": sxy, "von_mises": vm}
