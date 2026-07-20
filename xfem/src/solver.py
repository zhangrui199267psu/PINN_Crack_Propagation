"""
Classical 2D Q4 FEM with slit mesh for Mode I central crack.

Crack is modeled as a slit: nodes on the crack line are duplicated into
upper/lower sets so crack faces are traction-free by construction.

Public interface
----------------
  fem = FEMCrackSolver(...)
  fem.solve()            → solves K u = f
  fem.predict(pts)       → (N,2) [u_x, u_y]  (same API as PINN solver)
  fem.stress_at(pts)     → dict with sigma_xx, sigma_yy, sigma_xy, von_mises
"""

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    _SCIPY = True
except ImportError:
    _SCIPY = False


class FEMCrackSolver:
    def __init__(self, Nx=80, Ny=80, E=1.0, nu=0.3, traction=0.1,
                 crack_center=(0.5, 0.5), crack_half_length=0.15,
                 formulation="plane_stress"):
        self.Nx, self.Ny   = Nx, Ny
        self.E, self.nu, self.T = E, nu, traction
        self.cx, self.cy   = crack_center
        self.a             = crack_half_length
        self.hx, self.hy   = 1.0/Nx, 1.0/Ny

        self.j_ck = round(self.cy / self.hy)
        self.i_s  = round((self.cx - self.a) / self.hx)
        self.i_e  = round((self.cx + self.a) / self.hx)
        self.tip1 = np.array([self.cx + self.a, self.cy])
        self.tip2 = np.array([self.cx - self.a, self.cy])

        if formulation == "plane_stress":
            c = E / (1.0 - nu**2)
            self.Cmat = c * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        else:
            lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
            self.Cmat = np.array([[lam+2*mu, lam, 0], [lam, lam+2*mu, 0], [0, 0, mu]])

        self.nodes = self.elems = self.u = None

    # ── mesh ──────────────────────────────────────────────────────────────────
    def _build_mesh(self):
        Nx, Ny = self.Nx, self.Ny
        nb = (Nx+1)*(Ny+1)
        ns = self.i_e - self.i_s + 1

        nodes = np.zeros((nb+ns, 2))
        for j in range(Ny+1):
            for i in range(Nx+1):
                nodes[j*(Nx+1)+i] = [i*self.hx, j*self.hy]
        for k in range(ns):
            nodes[nb+k] = nodes[self.j_ck*(Nx+1) + self.i_s + k]
        self.nodes = nodes; self.n_nodes = nb+ns

        def nid(i, j, up=False):
            if up and j == self.j_ck and self.i_s <= i <= self.i_e:
                return nb + (i - self.i_s)
            return j*(Nx+1)+i

        elems = []
        for j in range(Ny):
            for i in range(Nx):
                above = (j == self.j_ck and self.i_s <= i and i+1 <= self.i_e)
                elems.append([nid(i,   j,   above),
                               nid(i+1, j,   above),
                               nid(i+1, j+1, False),
                               nid(i,   j+1, False)])
        self.elems = np.array(elems, dtype=int)

    # ── element stiffness (precomputed once for uniform mesh) ─────────────────
    def _ke_ref(self):
        xe = np.array([0., self.hx, self.hx, 0.])
        ye = np.array([0., 0.,      self.hy, self.hy])
        gp = np.array([-1., 1.]) / np.sqrt(3.)
        Ke = np.zeros((8, 8))
        for xi in gp:
            for eta in gp:
                dNx = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)]) / 4.
                dNe = np.array([-(1-xi),  -(1+xi), (1+xi),  (1-xi) ]) / 4.
                J    = np.array([[dNx@xe, dNx@ye], [dNe@xe, dNe@ye]])
                Ji   = np.linalg.inv(J); detJ = np.linalg.det(J)
                dNdx = Ji[0,0]*dNx + Ji[0,1]*dNe
                dNdy = Ji[1,0]*dNx + Ji[1,1]*dNe
                B = np.zeros((3, 8))
                B[0, 0::2] = dNdx; B[1, 1::2] = dNdy
                B[2, 0::2] = dNdy; B[2, 1::2] = dNdx
                Ke += B.T @ self.Cmat @ B * detJ
        return Ke

    # ── global assembly ────────────────────────────────────────────────────────
    def _assemble(self):
        Ke = self._ke_ref()
        ne, nd = len(self.elems), 2*self.n_nodes
        ed   = np.array([[2*n, 2*n+1] for e in self.elems for n in e]).reshape(ne, 8)
        rows = np.repeat(ed, 8, axis=1).flatten()
        cols = np.tile(ed, (1, 8)).flatten()
        vals = np.tile(Ke.flatten(), ne)
        if _SCIPY:
            return coo_matrix((vals, (rows, cols)), shape=(nd, nd)).tocsr()
        K = np.zeros((nd, nd)); np.add.at(K, (rows, cols), vals); return K

    # ── penalty BCs ───────────────────────────────────────────────────────────
    def _apply_bcs(self, K, f):
        P = 1e30; tol = 0.4*min(self.hx, self.hy)
        xn, yn = self.nodes[:, 0], self.nodes[:, 1]
        diag = np.zeros(2*self.n_nodes); rhs = np.zeros(2*self.n_nodes)

        bot = np.where(yn < tol)[0];        diag[2*bot+1]+=P; rhs[2*bot+1]+=P*(-self.T)
        top = np.where(yn > 1.0-tol)[0];   diag[2*top+1]+=P; rhs[2*top+1]+=P*(self.T)
        lft = np.where(xn < tol)[0];       diag[2*lft]  +=P

        if _SCIPY:
            from scipy.sparse import diags
            K = K + diags(diag)
        else:
            np.fill_diagonal(K, K.diagonal() + diag)
        f += rhs; return K, f

    # ── solve ──────────────────────────────────────────────────────────────────
    def solve(self):
        self._build_mesh()
        K = self._assemble()
        f = np.zeros(2*self.n_nodes)
        K, f = self._apply_bcs(K, f)
        self.u = spsolve(K, f) if _SCIPY else np.linalg.solve(K, f)
        print(f"FEM solved: {self.n_nodes} nodes | {len(self.elems)} elements | {2*self.n_nodes} DOFs")
        return self.u

    # ── interpolation helpers ──────────────────────────────────────────────────
    def _locate(self, pts):
        pts = np.asarray(pts, float)
        i   = np.clip((pts[:, 0]/self.hx).astype(int), 0, self.Nx-1)
        j   = np.clip((pts[:, 1]/self.hy).astype(int), 0, self.Ny-1)
        xi  = np.clip(2*(pts[:,0]-i*self.hx)/self.hx-1., -1., 1.)
        eta = np.clip(2*(pts[:,1]-j*self.hy)/self.hy-1., -1., 1.)
        return j*self.Nx+i, xi, eta

    # ── predict (matches PINN interface) ──────────────────────────────────────
    def predict(self, pts):
        eid, xi, eta = self._locate(pts)
        e = self.elems[eid]
        N = np.c_[(1-xi)*(1-eta), (1+xi)*(1-eta),
                  (1+xi)*(1+eta), (1-xi)*(1+eta)] / 4.
        return np.c_[(N*self.u[2*e]).sum(1), (N*self.u[2*e+1]).sum(1)]

    # ── stress fields ──────────────────────────────────────────────────────────
    def stress_at(self, pts):
        eid, xi, eta = self._locate(pts)
        e   = self.elems[eid]
        xe, ye = self.nodes[e, 0], self.nodes[e, 1]
        dNx_ = np.c_[-(1-eta), (1-eta),  (1+eta), -(1+eta)] / 4.
        dNe_ = np.c_[-(1-xi),  -(1+xi),  (1+xi),  (1-xi) ] / 4.
        J00=(dNx_*xe).sum(1); J01=(dNx_*ye).sum(1)
        J10=(dNe_*xe).sum(1); J11=(dNe_*ye).sum(1)
        det = J00*J11 - J01*J10
        dNdx = ( J11/det)[:,None]*dNx_ + (-J01/det)[:,None]*dNe_
        dNdy = (-J10/det)[:,None]*dNx_ + ( J00/det)[:,None]*dNe_
        ux_e, uy_e = self.u[2*e], self.u[2*e+1]
        eps = np.c_[(dNdx*ux_e).sum(1),
                    (dNdy*uy_e).sum(1),
                    (dNdy*ux_e+dNdx*uy_e).sum(1)]
        sig = eps @ self.Cmat
        sxx, syy, sxy = sig[:,0], sig[:,1], sig[:,2]
        vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2 + 1e-14)
        return {"sigma_xx": sxx, "sigma_yy": syy, "sigma_xy": sxy, "von_mises": vm}
