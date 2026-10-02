r"""gen_layer_table.py: the generator of the soft layer table (Stage 1c table lane, 2026-10-02).

Writes jolt_soft_layer_table.gen.h in the format of TABLE_FORMAT.v2.md (the desk's TABLE_FORMAT.md plus the K / G axis
of the lead's decision 4(b)). SCons never runs this file; the header is committed beside it.

The FE is the physics critic's validated plane-strain solver (critique\physics_desk.py: four-node quadrilaterals on a
tensor mesh, selective reduced integration for the nearly incompressible rubber, the base bonded to a rigid backing, a
rigid indenter, the contact set found by an active set), extended as SOFT-CONTACT.brief.md 3.1 asks:
  - to axisymmetry: the same element with the hoop strain u_r / r and the ring weight 2 pi r (a sphere, a disc);
  - to Coulomb friction on the contact face, loaded monotonically from first touch in small steps: a node that touches
    sticks where it touched, slips at mu p against its slip, sticks again if its slip reverses; the bonded column is
    the same path with no slip limit;
  - with the layer condensed onto the top nodes that can touch (a Schur complement of the same stiffness, factorised
    once per mesh by a banded Cholesky in depth-fastest order), so each contact iteration is a small dense solve.
The plane-strain element is the critic's q4_rect, copied verbatim below; validate_fe.py checks the copy against the
critic's function and reproduces the critic's runs (Hertz on a 30 mm layer, the P6 face run) and Gent and Lindley.

Non-dimensional: h = 1, E = 1, the material fixed by K / G. Lengths are over h, plane-strain loads P' / (E h), sphere
loads F / (E h^2).

Usage (idle priority, at most --workers processes):
  python gen_layer_table.py screen   the K / G screen (bonded face, K/G 600 against KG_REF, every node)
  python gen_layer_table.py run      every CYL and SPH solve (checkpointed, resumable), then T3 and T4
  python gen_layer_table.py header   assemble the checkpoints into jolt_soft_layer_table.gen.h
  python gen_layer_table.py status   what is done
"""
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.linalg as sla
from scipy.sparse import coo_matrix

GEN_VERSION = "2.0"
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
HEADER = os.path.join(HERE, "jolt_soft_layer_table.gen.h")

# ------------------------------------------------------------------------------------------------ the table's axes
RH = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 512.0]
# squash/h: the brief's seven log steps 1e-4 to 0.1, plus one more (0.316, TABLE_FORMAT.v2.md 3): the brief's own U6 rods
# R 1 and R 2 into the 2 mm pad at 0.5 N/mm squash 0.146 and 0.115 depths, a pen held in a fingertip pad
SH = [1e-4 * 10 ** (k / 2.0) for k in range(8)]
FACE_MU = [0.0, 0.3, 0.6, 0.9, -1.0]            # -1: bonded (no slip)
FACE_MODE = ["frictionless", "coulomb", "coulomb", "coulomb", "noslip"]
BONDED_MU = 1.2
KG = [600.0, 1200.0, 2500.0, 5000.0]
KG_REF = 2500.0
I_KG_REF = KG.index(KG_REF)
# T3 and T4 are read linearly in ln S and ln(c/h) (TABLE_FORMAT.md 4.7); at the desk's eight and seven nodes that missed
# U7's 2 % at off-grid points by up to 4.1 % and 2.9 % (u7_t34.out.txt, first run), so the nodes are denser (data only,
# the reader takes T3_NUM_S and T4_NUM_C from the header): S steps of sqrt(2), c/h steps of 0.05 below 0.5
T3_S = [0.25 * 2 ** (k / 2.0) for k in range(15)]
T4_C = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
T_SQUASH = 1e-3                                  # T3 and T4 are homogeneous of degree 1: one squash
INTERP_ORDER = 3
SCREEN_FLAG = 0.01                               # TABLE_FORMAT.v2.md 3: carries the K/G axis above 1 %
NEIGHBOUR_LIMIT = 0.02                           # and every neighbour is checked to 2 %

# ------------------------------------------------------------------------------------------------ mesh rules
N_FINE = 200          # uniform steps over the fine region [0, X_f]
GROW_X = 1.15         # geometric growth beyond it, to X_f + FAR
FAR = 12.0            # lateral extent beyond the fine region, in depths
GROW_Y = 1.15         # geometric growth of the depth steps from the top
MIN_LAYERS = 24
N_PATH = 200          # load steps from first touch, delta_k = s (k / N_PATH)^2
FILL_LO, FILL_HI = 0.40, 0.88   # the contact half-width must sit in this share of X_f, else the mesh is resized

SIMPSON_W = (math.pi / 48.0) * np.array([1, 4, 2, 4, 2, 4, 2, 4, 1], dtype=float)
THETA = np.array([math.pi * k / 16.0 for k in range(9)])
STATION = np.sin(THETA)


def idle():
    """Idle priority class (Windows), so the table never competes with the app or a build."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000040)


def peak_mb():
    if sys.platform != "win32":
        return float("nan")
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                    ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = k32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2 ** 20


# ------------------------------------------------------------------------------------------------ material
def material_kg(kg, E=1.0):
    """G, K and nu for Young's modulus E and K / G = kg."""
    nu = (3.0 * kg - 2.0) / (2.0 * (3.0 * kg + 1.0))
    G = E / (2.0 * (1.0 + nu))
    return G, kg * G, nu


def elastic_from_E_K(E, K):
    G = 3 * K * E / (9 * K - E)
    return G, (3 * K - 2 * G) / (2 * (3 * K + G))


# ------------------------------------------------------------------------------------------------ elements
def q4_rect(dx, dy, Ddev, Dvol):
    # the physics critic's element (critique\physics_desk.py q4_rect), copied verbatim; validate_fe.py checks it
    g = 1 / math.sqrt(3)
    xi_n = np.array([-1, 1, 1, -1.0]); et_n = np.array([-1, -1, 1, 1.0])

    def B_at(xi, et):
        dNdxi = 0.25 * xi_n * (1 + et * et_n)
        dNdet = 0.25 * et_n * (1 + xi * xi_n)
        dNdx = dNdxi * 2 / dx; dNdy = dNdet * 2 / dy
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx; B[1, 1::2] = dNdy
        B[2, 0::2] = dNdy; B[2, 1::2] = dNdx
        return B
    detJ = dx * dy / 4
    K = np.zeros((8, 8))
    for xi in (-g, g):
        for et in (-g, g):
            B = B_at(xi, et)
            K += B.T @ Ddev @ B * detJ
    B = B_at(0, 0)
    K += B.T @ Dvol @ B * detJ * 4
    return K


G_PT = 1.0 / math.sqrt(3.0)
XI_N = np.array([-1.0, 1.0, 1.0, -1.0])
ET_N = np.array([-1.0, -1.0, 1.0, 1.0])


def _plane_matrices(G, K):
    Ddev = np.array([[4 * G / 3, -2 * G / 3, 0], [-2 * G / 3, 4 * G / 3, 0], [0, 0, G]])
    Dvol = K * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0.0]])
    return Ddev, Dvol


def _axi_matrices(G, K):
    Ddev = G * np.array([[4 / 3, -2 / 3, -2 / 3, 0], [-2 / 3, 4 / 3, -2 / 3, 0], [-2 / 3, -2 / 3, 4 / 3, 0], [0, 0, 0, 1.0]])
    m = np.array([1.0, 1.0, 1.0, 0.0])
    return Ddev, K * np.outer(m, m)


def q4_axi_batch(r0, dr, dz, Ddev, Dvol):
    """Axisymmetric four-node elements, the critic's integration rule (deviatoric part at 2 x 2 Gauss points,
    volumetric part at the centre), strains (e_rr, e_zz, e_tt, g_rz), nodes counter-clockwise from (r0, z0), dofs
    (u_r, u_z) per node, ring weight 2 pi r included. Returns (n, 8, 8)."""
    r0 = np.asarray(r0, float); dr = np.asarray(dr, float); dz = np.asarray(dz, float)
    n = r0.size
    Ke = np.zeros((n, 8, 8))
    detJ = dr * dz / 4.0

    def B_at(xi, et):
        N = 0.25 * (1 + xi * XI_N) * (1 + et * ET_N)
        dNdxi = 0.25 * XI_N * (1 + et * ET_N)
        dNdet = 0.25 * ET_N * (1 + xi * XI_N)
        dNdr = dNdxi[None, :] * (2.0 / dr)[:, None]
        dNdz = dNdet[None, :] * (2.0 / dz)[:, None]
        r = r0 + 0.5 * dr * (1 + xi)
        B = np.zeros((n, 4, 8))
        B[:, 0, 0::2] = dNdr
        B[:, 1, 1::2] = dNdz
        B[:, 2, 0::2] = N[None, :] / r[:, None]
        B[:, 3, 0::2] = dNdz
        B[:, 3, 1::2] = dNdr
        return B, r
    for xi in (-G_PT, G_PT):
        for et in (-G_PT, G_PT):
            B, r = B_at(xi, et)
            w = detJ * 2 * math.pi * r
            Ke += np.einsum("nki,kl,nlj->nij", B, Ddev, B) * w[:, None, None]
    B, r = B_at(0.0, 0.0)
    w = 4 * detJ * 2 * math.pi * r
    Ke += np.einsum("nki,kl,nlj->nij", B, Dvol, B) * w[:, None, None]
    return Ke


# ------------------------------------------------------------------------------------------------ mesh axes
def graded_axis(fine_end, dx_fine, total_end, growth=GROW_X, start=0.0):
    """Nodes from `start`: uniform at about dx_fine up to fine_end, then geometric (ratio at most `growth`) to
    total_end exactly."""
    nf = max(1, int(round((fine_end - start) / dx_fine)))
    fine = np.linspace(start, fine_end, nf + 1)
    dx = (fine_end - start) / nf
    span = total_end - fine_end
    if span <= 1e-12 * max(1.0, abs(total_end)):
        return fine
    n = 1
    while dx * growth * (growth ** n - 1) / (growth - 1) < span:
        n += 1
    lo, hi = 1.0 + 1e-9, growth
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = dx * mid * (mid ** n - 1) / (mid - 1)
        if s > span:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    steps = dx * r ** np.arange(1, n + 1)
    steps *= span / steps.sum()
    return np.concatenate([fine, fine_end + np.cumsum(steps)])


def depth_axis(h, top_step, growth=GROW_Y, min_layers=MIN_LAYERS):
    """Nodes 0 (base) .. h (top): the top step (at most h / min_layers), growing geometrically downward (ratio at most
    `growth`), at least min_layers layers."""
    top_step = min(top_step, h / min_layers)
    n = 1
    while top_step * (growth ** n - 1) / (growth - 1) < h:
        n += 1
    n = max(n, min_layers)
    if top_step * n >= h * (1 - 1e-12):
        return np.linspace(0.0, h, n + 1)
    lo, hi = 1.0 + 1e-12, growth
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = top_step * (mid ** n - 1) / (mid - 1)
        if s > h:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    steps = top_step * r ** np.arange(n)
    steps *= h / steps.sum()
    ys = h - np.concatenate([[0.0], np.cumsum(steps)])
    ys[-1] = 0.0
    return ys[::-1]


def graded_toward(x0, x1, dx_at_x1, dx_at_x0, growth=1.1):
    """Nodes on [x0, x1], step dx_at_x1 at x1 growing toward x0 up to dx_at_x0."""
    steps = []
    s, acc, L = dx_at_x1, 0.0, x1 - x0
    while acc + s < L:
        steps.append(s)
        acc += s
        s = min(s * growth, dx_at_x0)
    steps.append(L - acc)
    if len(steps) > 1 and steps[-1] < 0.3 * steps[-2]:
        steps[-2] += steps[-1]
        steps.pop()
    pts = x1 - np.concatenate([[0.0], np.cumsum(steps)])
    pts[-1] = x0
    return pts[::-1]


def two_sided(x0, x1, dx_end, dx_mid, growth=1.1):
    """Nodes on [x0, x1] refined toward BOTH ends, at most dx_mid in the middle."""
    m = 0.5 * (x0 + x1)
    right = graded_toward(m, x1, dx_end, dx_mid, growth)      # fine at x1
    left = (x0 + x1) - right[::-1]                             # its mirror, fine at x0
    return np.unique(np.concatenate([left, right]).round(14))


def extend_axis(x_start, dx0, length, growth=1.15):
    """Nodes beyond x_start (not including it): the first step dx0 * growth, growing geometrically (ratio at most
    `growth`) to cover `length` exactly."""
    n = 1
    while dx0 * growth * (growth ** n - 1) / (growth - 1) < length:
        n += 1
    lo, hi = 1.0 + 1e-9, growth
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dx0 * mid * (mid ** n - 1) / (mid - 1) > length:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    steps = dx0 * r ** np.arange(1, n + 1)
    steps *= length / steps.sum()
    return x_start + np.cumsum(steps)


# ------------------------------------------------------------------------------------------------ the layer
class LayerFE:
    """A layer of depth h = ys[-1] on a rigid base (bottom nodes fixed), its top loaded by a rigid indenter at the
    candidate nodes, condensed onto those nodes' dofs.
    kind 'ps' (plane strain, per unit length) or 'axi' (axisymmetric, xs radii from 0).
    sym: the half model x >= 0 with u_x = 0 at x = 0 (always for 'axi').
    cand: a boolean mask over the top nodes (default: all)."""

    def __init__(self, kind, xs, ys, E, K=None, nu=None, sym=False, cand=None, block=64):
        self.kind = kind
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)
        if K is None:
            G = E / (2 * (1 + nu)); K = E / (3 * (1 - 2 * nu))
        else:
            G, nu = elastic_from_E_K(E, K)
        self.E, self.K, self.G, self.nu = E, K, G, nu
        self.h = self.ys[-1] - self.ys[0]
        self.sym = sym or kind == "axi"
        nx, ny = len(self.xs), len(self.ys)
        self.nx, self.ny = nx, ny
        N = 2 * nx * ny
        self.N = N
        I, J = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
        I, J = I.ravel(), J.ravel()
        dx = np.diff(self.xs)[I]
        dy = np.diff(self.ys)[J]
        if kind == "ps":
            Ddev, Dvol = _plane_matrices(G, K)
            Ke = np.empty((I.size, 8, 8))
            cache = {}
            for e in range(I.size):
                key = (dx[e], dy[e])
                if key not in cache:
                    cache[key] = q4_rect(dx[e], dy[e], Ddev, Dvol)
                Ke[e] = cache[key]
        else:
            Ddev, Dvol = _axi_matrices(G, K)
            Ke = q4_axi_batch(self.xs[I], dx, dy, Ddev, Dvol)
        n0 = J * nx + I
        nodes = np.stack([n0, n0 + 1, n0 + 1 + nx, n0 + nx], axis=1)
        dofs = np.empty((I.size, 8), dtype=np.int64)
        dofs[:, 0::2] = 2 * nodes
        dofs[:, 1::2] = 2 * nodes + 1
        rows = np.repeat(dofs, 8, axis=1).ravel()
        cols = np.tile(dofs, (1, 8)).ravel()
        Kg = coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()
        del Ke, rows, cols
        fixed = np.zeros(N, dtype=bool)
        bottom = np.arange(nx)
        fixed[2 * bottom] = True
        fixed[2 * bottom + 1] = True
        if self.sym:
            if abs(self.xs[0]) > 1e-12 * max(1.0, abs(self.xs[-1])):
                raise ValueError("a symmetric model starts at x = 0")
            fixed[2 * (np.arange(ny) * nx)] = True
        self.top = (ny - 1) * nx + np.arange(nx)
        self.cand = np.ones(nx, dtype=bool) if cand is None else np.asarray(cand, bool)
        cn = self.top[self.cand]
        self.cx = self.xs[self.cand]
        cdofs, jx, jy = [], [], []
        for n in cn:
            if not fixed[2 * n]:
                jx.append(len(cdofs)); cdofs.append(2 * n)
            else:
                jx.append(-1)
            jy.append(len(cdofs)); cdofs.append(2 * n + 1)
        self.jx, self.jy = np.array(jx), np.array(jy)
        cdofs = np.array(cdofs)
        isc = np.zeros(N, dtype=bool)
        isc[cdofs] = True
        idofs = np.where(~fixed & ~isc)[0]
        # depth-fastest order: column of nodes (i) slowest, then height (j), then the dof
        node = idofs // 2
        perm = np.lexsort((idofs % 2, node // nx, node % nx))
        idofs = idofs[perm]
        Kii = Kg[idofs][:, idofs].tocoo()
        low = Kii.row >= Kii.col
        bw = int((Kii.row[low] - Kii.col[low]).max())
        ni = len(idofs)
        ab = np.zeros((bw + 1, ni))
        np.add.at(ab, (Kii.row[low] - Kii.col[low], Kii.col[low]), Kii.data[low])
        del Kii
        cb = sla.cholesky_banded(ab, lower=True, check_finite=False)
        del ab
        Kic = Kg[idofs][:, cdofs].tocsc()
        Kcc = Kg[cdofs][:, cdofs].toarray()
        S = Kcc.copy()
        for c0 in range(0, len(cdofs), block):
            c1 = min(c0 + block, len(cdofs))
            B = Kic[:, c0:c1].toarray()
            X = sla.cho_solve_banded((cb, True), B, check_finite=False)
            S[:, c0:c1] -= Kic.T @ X
            del X, B
        self.S = 0.5 * (S + S.T)
        self.ndof_c = len(cdofs)
        self.dofs_total = int((~fixed).sum())
        self.bandwidth = bw
        del cb, Kic, Kcc, Kg
        x = self.xs
        trib = np.zeros(nx)
        d = np.diff(x)
        if kind == "ps":
            trib[:-1] += d / 2
            trib[1:] += d / 2
        else:
            trib[:-1] += 2 * math.pi * d / 6 * (2 * x[:-1] + x[1:])
            trib[1:] += 2 * math.pi * d / 6 * (x[:-1] + 2 * x[1:])
        self.trib = trib[self.cand]

    # -------------------------------------------------------------------------------------------- one contact state
    def _solve_state(self, t, active, mode, stick, sigma, anchor, mu):
        S, jx, jy = self.S, self.jx, self.jy
        n = self.ndof_c
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        for k in range(len(jy)):
            iy, ix = jy[k], jx[k]
            if active[k]:
                A[iy, iy] = 1.0
                rhs[iy] = t[k]
                if ix >= 0:
                    if mode == "frictionless":
                        A[ix] = S[ix]
                    elif mode in ("noslip", "glued") or stick[k]:
                        A[ix, ix] = 1.0
                        rhs[ix] = anchor[k]
                    else:
                        A[ix] = S[ix] - mu * sigma[k] * S[iy]
            else:
                A[iy] = S[iy]
                if ix >= 0:
                    A[ix] = S[ix]
        u = sla.solve(A, rhs, check_finite=False)
        return u, S @ u

    def contact(self, t, mode="frictionless", mu=0.0, state=None, max_iter=400, delta=None):
        """The contact state at one load level. t: per candidate node, the indenter surface's height (the layer's top
        node may not rise above it: u_y <= t). state carries the previous level's sets and anchors (monotonic
        loading); None starts from no contact. Modes: 'frictionless'; 'coulomb' (incremental: a node anchors where it
        touched, a slipping node's anchor follows it); 'noslip' (incremental, never slips); 'glued' (every touching
        node held at its undeformed position, the critics' 'bonded'); 'coulomb_total' (the critic's P6 form: anchors
        at the undeformed position, one level). delta: this level's squash, so a node that touches during the step is
        anchored where it touched (its free motion over the last two levels extrapolated to the moment its gap closes;
        without it the anchor is its position at the previous level, an error of one step's slide per node)."""
        nc = len(self.jy)
        if state is None:
            state = {"active": np.zeros(nc, bool), "stick": np.zeros(nc, bool), "sigma": np.zeros(nc),
                     "anchor": np.zeros(nc), "ux": np.zeros(nc)}
        active = state["active"].copy()
        stick = state["stick"].copy()
        sigma = state["sigma"].copy()
        anchor = state["anchor"].copy()
        ux_prev = state["ux"].copy()
        incremental = mode in ("coulomb", "noslip")
        pred_anchor = ux_prev.copy()
        if incremental and delta is not None and state.get("delta") is not None and state.get("delta_m2") is not None:
            d1, d0 = state["delta"], state["delta_m2"]
            if d1 > d0:
                r = (delta - d1) / (d1 - d0)
                ux1, uy1 = state["ux"], state["uy"]
                dux = (ux1 - state["ux_m2"]) * r
                duy = (uy1 - state["uy_m2"]) * r
                g1 = state["t"] - uy1
                gk = t - (uy1 + duy)
                with np.errstate(invalid="ignore", divide="ignore"):
                    alpha = np.where((gk < 0) & (g1 > 0) & np.isfinite(g1) & np.isfinite(gk), g1 / (g1 - gk), 1.0)
                pred_anchor = ux1 + np.clip(alpha, 0.0, 1.0) * dux
        seen = set()
        single = False
        converged = False
        flips = np.zeros(nc, dtype=int)   # a node that touched in this level and let go again
        tf = np.abs(t[np.isfinite(t)])
        scale_u = max(tf.max() if tf.size else 0.0, 1e-300)
        it = 0
        for it in range(max_iter):
            u, f = self._solve_state(t, active, mode, stick, sigma, anchor, mu)
            uy = u[self.jy]
            ux = np.where(self.jx >= 0, u[np.maximum(self.jx, 0)], 0.0)
            fy = f[self.jy]
            fx = np.where(self.jx >= 0, f[np.maximum(self.jx, 0)], 0.0)
            p = -fy
            scale_f = max(np.abs(p).max(), 1e-300)
            viol = []
            for k in range(nc):
                if active[k]:
                    if p[k] < -1e-10 * scale_f:
                        viol.append((-p[k] / scale_f, k, "open"))
                    elif mode in ("coulomb", "coulomb_total") and self.jx[k] >= 0:
                        lim = mu * max(p[k], 0.0)
                        if stick[k]:
                            if abs(fx[k]) > lim * (1 + 1e-9) + 1e-12 * scale_f:
                                viol.append(((abs(fx[k]) - lim) / scale_f, k, "slip"))
                        else:
                            s = ux[k] - anchor[k]
                            if s * sigma[k] < -1e-10 * scale_u:
                                viol.append((abs(s) / scale_u, k, "stick"))
                else:
                    if uy[k] > t[k] + 1e-10 * scale_u:
                        viol.append(((uy[k] - t[k]) / scale_u, k, "close"))
            if not viol:
                converged = True
                break
            sig = (active.tobytes(), stick.tobytes(), sigma.tobytes())
            if sig in seen:
                single = True
            seen.add(sig)
            if single:
                viol = [max(viol)]
            for _, k, kind in viol:
                if kind == "open":
                    active[k] = False
                    stick[k] = False
                    flips[k] += 1
                elif kind == "close":
                    active[k] = True
                    stick[k] = mode != "frictionless"
                    sigma[k] = 0.0
                    if mode in ("glued", "coulomb_total"):
                        anchor[k] = 0.0
                    elif incremental:
                        # a node that closed, pulled in tension at its predicted anchor and let go, penetrating by a hair
                        # when open, touched where it now is: anchor it there (the predicted anchor's error, not physics,
                        # made it cycle; at its present position it presses with a small positive force)
                        anchor[k] = ux[k] if flips[k] >= 1 else pred_anchor[k]
                elif kind == "slip":
                    stick[k] = False
                    sigma[k] = -math.copysign(1.0, fx[k])
                elif kind == "stick":
                    stick[k] = True
        if incremental:
            slipping = active & ~stick
            anchor = np.where(slipping, ux, anchor)
        return {"active": active, "stick": stick, "sigma": sigma, "anchor": anchor, "ux": ux, "uy": uy,
                "u": u, "f": f, "p": np.where(active, p, 0.0), "fx": np.where(active, fx, 0.0),
                "converged": converged, "iters": it + 1, "t": t, "delta": delta,
                "ux_m2": state.get("ux"), "uy_m2": state.get("uy"), "delta_m2": state.get("delta")}

    # -------------------------------------------------------------------------------------------- figures
    def figures(self, st, prof=None):
        """Load, centre, contact half-width (or radius), m1, spin radius and the station profile from a contact
        state. The half-width 'b': beyond the last touching node, the gap of the two first open nodes extrapolated
        as gap^(2/3) linear in x (the smooth edge of a curved contact), clamped between the last touching node and
        the first open one; 'b_span' the critic's node span plus half a step; 'b_p' the pressure-side estimate
        (p^2 linear)."""
        p = st["p"]
        x = self.cx
        act = st["active"]
        out = {"converged": bool(st["converged"]), "iters": int(st["iters"])}
        F = p.sum() * (2.0 if (self.kind == "ps" and self.sym) else 1.0)
        out["F"] = float(F)
        if F <= 0 or not act.any():
            out["empty"] = True
            return out
        out["empty"] = False
        out["c"] = float((p * x).sum() / p.sum())
        if self.kind == "ps":
            out["m1"] = float((p * np.abs(x)).sum() / p.sum()) if self.sym else float((p * np.abs(x - out["c"])).sum() / p.sum())
        else:
            rs = float((p * x).sum() / p.sum())
            out["r_s"] = rs
            out["m1"] = 2.0 / math.pi * rs
        gap = st["t"] - st["uy"]
        idx = np.where(act)[0]
        edges = []
        for side in ((+1,) if self.sym else (+1, -1)):
            k = idx.max() if side > 0 else idx.min()
            k1, k2 = k + side, k + 2 * side
            if 0 <= k1 < len(x) and 0 <= k2 < len(x):
                g1, g2 = max(gap[k1], 0.0), max(gap[k2], 0.0)
                d = (g2 ** (2 / 3) - g1 ** (2 / 3)) if (np.isfinite(g1) and np.isfinite(g2)) else -1.0
                if d > 0:
                    xe = x[k1] - side * g1 ** (2 / 3) * abs(x[k2] - x[k1]) / d
                    lo, hi = sorted((x[k], x[k1]))
                    xe = min(max(xe, lo), hi)
                else:
                    xe = 0.5 * (x[k] + x[k1])
                span_b = x[k] + side * 0.5 * abs(x[k1] - x[k])
                pk = p[k] / self.trib[k]
                km = k - side
                if 0 <= km < len(x) and act[km]:
                    pm = p[km] / self.trib[km]
                    dd = pm * pm - pk * pk
                    xp = x[k] + side * pk * pk * abs(x[k] - x[km]) / dd if dd > 0 else x[k1]
                    lo, hi = sorted((x[k], x[k1]))
                    xp = min(max(xp, lo), hi)
                else:
                    xp = x[k1]
                edges.append((xe, span_b, xp))
            else:
                edges.append((x[k], x[k], x[k]))
        out["touch_to_end"] = bool(idx.max() >= len(x) - 3)
        if self.sym:
            out["b"], out["b_span"], out["b_p"] = map(float, edges[0])
        else:
            out["edge_r"], out["edge_l"] = float(edges[0][0]), float(edges[1][0])
            out["b"] = 0.5 * (out["edge_r"] - out["edge_l"])
            out["b_span"] = 0.5 * (edges[0][1] - edges[1][1])
        out["stick_share"] = float((st["stick"] & act).sum() / max(act.sum(), 1))
        out["n_active"] = int(act.sum())
        if prof is not None and self.sym:
            out["profile"] = [float(v) for v in station_profile(x, p / self.trib, out["b"], prof)]
        return out


def station_profile(x, pl, b, kind):
    """TABLE_FORMAT.md 3: the lumped nodal pressure at x = b sin(theta_k), linear between nodes, normalised so the
    Simpson rule in theta gives exactly the load ('cyl': sum w p cos = 1; 'sph': 2 sum w p sin cos = 1)."""
    xs = b * np.sin(THETA)
    pv = np.interp(xs, x, pl, right=0.0)
    pv = np.maximum(pv, 0.0)
    if kind == "cyl":
        norm = float(np.sum(SIMPSON_W * pv * np.cos(THETA)))
    else:
        norm = float(2 * np.sum(SIMPSON_W * pv * np.sin(THETA) * np.cos(THETA)))
    return pv / norm


# ------------------------------------------------------------------------------------------------ indenters, paths
def circle_gap(x, R, delta):
    """The rigid circle's (or sphere's) surface height over the undeformed top, its lowest point delta below:
    t = (R - sqrt(R^2 - x^2)) - delta (exact circle; beyond x = R no surface below it: +inf)."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.inf)
    m = np.abs(x) < R
    out[m] = (R - np.sqrt(R * R - x[m] ** 2)) - delta
    return out


def parabola_gap(x, R, delta):
    return np.asarray(x, float) ** 2 / (2 * R) - delta


def flat_gap(x, c, delta):
    """A flat rigid face of half-width c (np.inf for an unbounded face)."""
    x = np.asarray(x, float)
    return np.where(np.abs(x) <= c + 1e-12, -delta, np.inf)


def load_path(fe, gapfun, delta, mode, mu=0.0, n_steps=N_PATH, power=2.0, keep=None):
    """Monotonic loading from first touch to delta in n_steps (delta_k = delta (k/n)^power) for the path-dependent
    modes ('coulomb', 'noslip'); one solve for 'frictionless', 'glued' and 'coulomb_total'. keep: delta values at which
    to return the state (the last is delta). Returns [(delta, state)]."""
    if mode in ("frictionless", "glued", "coulomb_total"):
        return [(d, fe.contact(gapfun(d), mode=mode, mu=mu)) for d in (keep or [delta])]
    sched = [delta * (k / n_steps) ** power for k in range(1, n_steps + 1)]
    marks = sorted(keep or [delta])
    sched = sorted(set(sched) | set(marks))
    st = None
    out = []
    nonconv = 0
    for d in sched:
        st = fe.contact(gapfun(d), mode=mode, mu=mu, state=st, delta=d)
        nonconv += 0 if st["converged"] else 1
        if any(abs(d - m) <= 1e-15 * max(1.0, m) for m in marks):
            st["nonconv_levels"] = nonconv
            out.append((d, st))
    return out


# ------------------------------------------------------------------------------------------------ one table node
def node_mesh(kind, Xf, n_fine=N_FINE, far=FAR):
    """The half (or axisymmetric) mesh for a contact inside [0, Xf]: uniform fine steps there, graded beyond."""
    dx = Xf / n_fine
    xs = graded_axis(Xf, dx, Xf + far)
    ys = depth_axis(1.0, dx)
    return xs, ys


def solve_curved(kind, rh, sh, face, kg, Xf=None, n_fine=N_FINE, n_path=N_PATH, max_tries=4, verbose=False):
    """One table solve: a rigid circle (kind 'cyl', plane strain) or sphere ('sph') of radius rh pressed sh into the
    unit layer with K / G = kg and the face condition of column `face` (or ('mu', value) for any Coulomb mu, U7's
    fresh FE). Returns the figures dict (non-dimensional) with the mesh it used."""
    G, K, nu = material_kg(kg)
    if isinstance(face, tuple):
        mode, mu = ("coulomb", face[1]) if face[1] > 0 else ("frictionless", 0.0)
    else:
        mode, mu = FACE_MODE[face], max(FACE_MU[face], 0.0)
    bg = math.sqrt(max(2 * rh * sh - sh * sh, 1e-300))
    if Xf is None:
        Xf = 1.6 * bg
    t0 = time.time()
    tries = []
    for attempt in range(max_tries):
        xs, ys = node_mesh(kind, Xf, n_fine)
        cand = xs <= Xf * (1 + 1e-12)
        fe = LayerFE("ps" if kind == "cyl" else "axi", xs, ys, 1.0, K=K, sym=True, cand=cand)
        res = load_path(fe, lambda d: circle_gap(fe.cx, rh, d), sh, mode, mu=mu, n_steps=n_path)
        d, st = res[-1]
        fig = fe.figures(st, prof=kind)
        fill = fig.get("b", 0.0) / Xf
        tries.append({"Xf": Xf, "fill": fill, "nodes": [len(xs), len(ys)], "ndof_c": fe.ndof_c, "dofs": fe.dofs_total})
        if verbose:
            print(f"   try {attempt}: Xf {Xf:.4g} fill {fill:.3f} F {fig.get('F', 0):.5e} conv {st['converged']}", flush=True)
        if fig.get("empty") or not fig.get("touch_to_end", False) and FILL_LO <= fill <= FILL_HI:
            break
        target = fig.get("b", bg) / 0.65 if not fig.get("empty") and fill > 0.05 else Xf * 0.5
        if fig.get("touch_to_end"):
            target = max(target, Xf * 1.6)
        Xf = target
    fig["mesh"] = tries
    fig["seconds"] = time.time() - t0
    fig["nonconv_levels"] = int(st.get("nonconv_levels", 0))
    fig["mode"] = mode
    fig["mu"] = mu
    fig["kg"] = kg
    fig["nu"] = nu
    ok = (not fig.get("empty")) and FILL_LO * 0.9 <= fig["b"] / tries[-1]["Xf"] <= FILL_HI * 1.04 \
        and fig["converged"] and not fig.get("touch_to_end")
    fig["ok"] = bool(ok)
    return fig


def solve_t3(S, face, kg, n_mid=None):
    """T3: a strip of half-width S (in depths) in full contact with a flat rigid face, free side faces, plane strain,
    squash T_SQUASH, one level (proportional loading). Returns F (per unit length, the whole strip) and the stick
    share. The mesh is refined toward the free edge and toward the top and the base."""
    G, K, nu = material_kg(kg)
    mode = "frictionless" if FACE_MU[face] == 0.0 else ("glued" if FACE_MU[face] < 0 else "coulomb_total")
    mu = max(FACE_MU[face], 0.0)
    dx_end = 1.0 / 60.0
    xs = graded_toward(0.0, S, dx_end, min(0.25, max(S / 40.0, dx_end)), 1.08)
    ys = two_sided(0.0, 1.0, 1.0 / 60.0, 1.0 / 12.0, 1.12)
    fe = LayerFE("ps", xs, ys, 1.0, K=K, sym=True)
    st = fe.contact(flat_gap(fe.cx, 1e9, T_SQUASH), mode=mode, mu=mu)
    fig = fe.figures(st)
    return {"F": fig["F"], "stick_share": fig["stick_share"], "converged": fig["converged"],
            "nodes": [len(xs), len(ys)], "open_nodes": int((~st["active"]).sum())}


def solve_t4(c, face, kg):
    """T4: a flat rigid punch of half-width c (in depths) on a wide layer, plane strain, squash T_SQUASH, one level.
    Returns F per unit length (the whole punch). The mesh is refined toward the punch's edge from both sides."""
    G, K, nu = material_kg(kg)
    mode = "frictionless" if FACE_MU[face] == 0.0 else ("glued" if FACE_MU[face] < 0 else "coulomb_total")
    mu = max(FACE_MU[face], 0.0)
    dx_e = 1.0 / 80.0
    inner = graded_toward(0.0, c, dx_e, min(0.1, max(c / 20.0, dx_e)), 1.08)
    outer = c + (c - graded_toward(c - 3.0, c, dx_e, 0.25, 1.1)[::-1])[1:]
    xs = np.unique(np.concatenate([inner, outer, extend_axis(c + 3.0, 0.25, FAR)]).round(14))
    ys = two_sided(0.0, 1.0, 1.0 / 80.0, 1.0 / 12.0, 1.12)
    cand = xs <= c + 1e-12
    fe = LayerFE("ps", xs, ys, 1.0, K=K, sym=True, cand=cand)
    st = fe.contact(flat_gap(fe.cx, c, T_SQUASH), mode=mode, mu=mu)
    fig = fe.figures(st)
    return {"F": fig["F"], "stick_share": fig["stick_share"], "converged": fig["converged"],
            "nodes": [len(xs), len(ys)], "open_nodes": int((~st["active"]).sum())}


def split_law_punch(c, kg, n=4001):
    """T4's F_law: the BONDED split law (design 2.2) on the engine's extended domain in plane strain, in its fine-grid
    limit: the punch |x| <= c in contact, the layer free to |x| = c + h with q = 0 there (brief 3.1), complementarity
    (an active node with p < 0 lets go). Non-dimensional (E 1, h 1). Returns F per unit length (the whole punch)."""
    G, K, nu = material_kg(kg)
    E, h = 1.0, 1.0
    kW = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * h)
    kI = 12.0 * G / h ** 3
    lam2 = kI / (kW - E / h)
    L = c + h
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    active = np.abs(x) <= c + 1e-12
    from scipy.linalg import solve_banded
    for _ in range(50):
        # unknown q at interior nodes; active: w = d (p = E/h d + q); inactive: w = -(h/E) q, so the q equation gains
        # kI h / E on the diagonal there
        m = n - 2
        a = active[1:-1]
        main = lam2 + 2 / dx ** 2 + np.where(a, 0.0, kI * h / E)
        ab = np.zeros((3, m))
        ab[0, 1:] = -1 / dx ** 2
        ab[1] = main
        ab[2, :-1] = -1 / dx ** 2
        rhs = np.where(a, kI * T_SQUASH, 0.0)
        q = np.zeros(n)
        q[1:-1] = solve_banded((1, 1), ab, rhs)
        p = np.where(active, E / h * T_SQUASH + q, 0.0)
        new = active & (p >= 0)
        if np.array_equal(new, active):
            break
        active = new
    return float(np.trapezoid(p, x))


# ------------------------------------------------------------------------------------------------ checkpoints
def ck_name(kind, i, j, f, k):
    return os.path.join(CKPT, f"{kind}_rh{i}_sh{j}_f{f}_kg{k}.json")


def ck_write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def ck_read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def unit_task(args):
    """A worker's unit: one (kind, R/h, squash/h, K/G) node, the listed faces, each checkpointed as it finishes."""
    kind, i, j, k, faces, tag = args
    idle()
    done = []
    for f in faces:
        path = ck_name(kind, i, j, f, k) if tag == "grid" else os.path.join(CKPT, "screen", f"{kind}_rh{i}_sh{j}_f{f}_kg{k}.json")
        if ck_read(path) is not None:
            done.append(path)
            continue
        t0 = time.time()
        try:
            fig = solve_curved(kind, RH[i], SH[j], f, KG[k])
            fig["error"] = None
        except Exception as ex:   # recorded, the node becomes STATUS 1
            fig = {"ok": False, "error": repr(ex)}
        fig.update({"kind": kind, "i_rh": i, "i_sh": j, "face": f, "i_kg": k, "rh": RH[i], "sh": SH[j],
                    "wall": time.time() - t0, "peak_mb": peak_mb(), "gen": GEN_VERSION})
        ck_write(path, fig)
        done.append(path)
    return done


def run_pool(tasks, workers, label):
    import multiprocessing as mp
    t0 = time.time()
    n = len(tasks)
    print(f"{label}: {n} units on {workers} workers", flush=True)
    with mp.Pool(processes=workers, initializer=idle) as pool:
        for c, _ in enumerate(pool.imap_unordered(unit_task, tasks), 1):
            if c % 10 == 0 or c == n:
                print(f"   {label}: {c}/{n} units, {time.time() - t0:7.0f} s", flush=True)
    return time.time() - t0


# ------------------------------------------------------------------------------------------------ the K / G screen
def screen(workers):
    """TABLE_FORMAT.v2.md 3: every (kind, R/h, squash/h) with the bonded face at K/G 600 and KG_REF."""
    tasks = []
    for kind in ("cyl", "sph"):
        for i in range(len(RH)):
            for j in range(len(SH)):
                for k in (0, I_KG_REF):
                    tasks.append((kind, i, j, k, [4], "screen"))
    return run_pool(tasks, workers, "screen")


def screen_result():
    """The screen's figures: per kind a (NUM_RH, NUM_SH) array of F(600) / F(KG_REF) - 1 (bonded face)."""
    out = {}
    for kind in ("cyl", "sph"):
        a = np.full((len(RH), len(SH)), np.nan)
        for i in range(len(RH)):
            for j in range(len(SH)):
                r0 = ck_read(os.path.join(CKPT, "screen", f"{kind}_rh{i}_sh{j}_f4_kg0.json"))
                r1 = ck_read(os.path.join(CKPT, "screen", f"{kind}_rh{i}_sh{j}_f4_kg{I_KG_REF}.json"))
                if r0 and r1 and r0.get("F") and r1.get("F"):
                    a[i, j] = r0["F"] / r1["F"] - 1.0
        out[kind] = a
    return out


def kg_axis_flags(scr, extra=None):
    """1 where the screen moved the bonded force by more than SCREEN_FLAG; `extra` adds nodes the neighbour check
    found above NEIGHBOUR_LIMIT."""
    flags = {}
    for kind in ("cyl", "sph"):
        a = scr[kind]
        fl = (np.abs(np.nan_to_num(a, nan=1.0)) > SCREEN_FLAG).astype(int)
        if extra and kind in extra:
            for i, j in extra[kind]:
                fl[i, j] = 1
        flags[kind] = fl
    return flags


# ------------------------------------------------------------------------------------------------ the full run
def grid_tasks(flags):
    tasks = []
    for kind in ("cyl", "sph"):
        for i in range(len(RH)):
            for j in range(len(SH)):
                ks = range(len(KG)) if flags[kind][i, j] else [I_KG_REF]
                for k in ks:
                    faces = [f for f in range(len(FACE_MU)) if ck_read(ck_name(kind, i, j, f, k)) is None]
                    if faces:
                        tasks.append((kind, i, j, k, faces, "grid"))
    # widest first (the slowest units start early)
    tasks.sort(key=lambda t: -(RH[t[1]] * SH[t[2]]))
    return tasks


def t34_task(args):
    which, idx, f, k = args
    idle()
    if which == "t3":
        p = os.path.join(CKPT, "t3", f"t3_s{idx}_f{f}_kg{k}.json")
        if ck_read(p) is None:
            t0 = time.time()
            r = solve_t3(T3_S[idx], f, KG[k])
            r["wall"] = time.time() - t0
            r["peak_mb"] = peak_mb()
            ck_write(p, r)
    else:
        p = os.path.join(CKPT, "t4", f"t4_c{idx}_f{f}_kg{k}.json")
        if ck_read(p) is None:
            t0 = time.time()
            r = solve_t4(T4_C[idx], f, KG[k])
            r["F_law"] = split_law_punch(T4_C[idx], KG[k])
            r["wall"] = time.time() - t0
            r["peak_mb"] = peak_mb()
            ck_write(p, r)
    return p


def run_t34(workers):
    import multiprocessing as mp
    tasks = [("t3", s, f, k) for k in range(len(KG)) for f in range(len(FACE_MU)) for s in range(len(T3_S))]
    tasks += [("t4", c, f, k) for k in range(len(KG)) for f in range(len(FACE_MU)) for c in range(len(T4_C))]
    tasks.sort(key=lambda t: -(T3_S[t[1]] if t[0] == "t3" else T4_C[t[1]]))
    t0 = time.time()
    print(f"t3/t4: {len(tasks)} solves on {workers} workers", flush=True)
    with mp.Pool(processes=workers, initializer=idle) as pool:
        for c, _ in enumerate(pool.imap_unordered(t34_task, tasks), 1):
            if c % 20 == 0 or c == len(tasks):
                print(f"   t3/t4: {c}/{len(tasks)}, {time.time() - t0:6.0f} s", flush=True)
    return time.time() - t0


# ------------------------------------------------------------------------------------------------ the header
def assemble(flags):
    """The v2 dict from the checkpoints. A node whose solve failed (or did not converge, or ran out of its mesh) is
    STATUS 1 with placeholder values (the reader never weights it)."""
    nk, nf, nr, ns = len(KG), len(FACE_MU), len(RH), len(SH)
    d = {"VERSION": 2, "INTERP_ORDER": INTERP_ORDER, "POISSON": material_kg(KG_REF)[2], "BONDED_MU": BONDED_MU,
         "KG_REF": KG_REF, "NUM_KG": nk, "KG": np.array(KG), "NUM_RH": nr, "RH": np.array(RH), "NUM_SH": ns,
         "SH": np.array(SH), "NUM_FACE": nf, "FACE_MU": np.array(FACE_MU), "NUM_ST": 9, "STATION": STATION,
         "T3_NUM_S": len(T3_S), "T3_S": np.array(T3_S), "T4_NUM_C": len(T4_C), "T4_C": np.array(T4_C)}
    notes = []
    for kind, pre, names in (("cyl", "CYL_", [("FORCE", "F"), ("HALFWIDTH", "b"), ("M1", "m1")]),
                             ("sph", "SPH_", [("FORCE", "F"), ("RADIUS", "b"), ("M1", "m1"), ("SPIN", "r_s")])):
        arrs = {nm: np.ones((nk, nf, nr, ns)) for nm, _ in names}
        prof = np.zeros((nk, nf, nr, ns, 9))
        status = np.zeros((nk, nf, nr, ns), dtype=int)
        for f in range(nf):
            for i in range(nr):
                for j in range(ns):
                    for k in range(nk):
                        kk = k if flags[kind][i, j] else I_KG_REF
                        r = ck_read(ck_name(kind, i, j, f, kk))
                        good = r is not None and r.get("ok") and not r.get("error")
                        if good:
                            for nm, key in names:
                                arrs[nm][k, f, i, j] = r[key]
                            prof[k, f, i, j] = r["profile"]
                        else:
                            status[k, f, i, j] = 1
                            if k == kk:
                                why = "missing" if r is None else (r.get("error") or f"ok false (fill {r.get('b', 0) / r['mesh'][-1]['Xf'] if r.get('mesh') else 0:.2f}, conv {r.get('converged')})")
                                notes.append(f"{pre}STATUS 1 at face {f}, R/h {RH[i]}, squash/h {SH[j]:.3g}, K/G {KG[kk]}: {why}")
        for nm, _ in names:
            d[pre + nm] = arrs[nm]
        d[pre + "PROFILE"] = prof
        d[pre + "STATUS"] = status
        d[pre + "KG_AXIS"] = flags[kind].astype(int)
    t3 = np.ones((nk, nf, len(T3_S)))
    t4 = np.ones((nk, nf, len(T4_C)))
    for k in range(nk):
        for s_i in range(len(T3_S)):
            rb = ck_read(os.path.join(CKPT, "t3", f"t3_s{s_i}_f4_kg{k}.json"))
            for f in range(nf):
                r = ck_read(os.path.join(CKPT, "t3", f"t3_s{s_i}_f{f}_kg{k}.json"))
                t3[k, f, s_i] = r["F"] / rb["F"]
        for c_i in range(len(T4_C)):
            for f in range(nf):
                r = ck_read(os.path.join(CKPT, "t4", f"t4_c{c_i}_f{f}_kg{k}.json"))
                t4[k, f, c_i] = r["F"] / r["F_law"]
    d["T3_RATIO"] = t3
    d["T4_RATIO"] = t4
    return d, notes


def _c_float(x):
    x = float(np.float32(x))
    t = "%.9g" % x
    if "e" not in t and "." not in t and "inf" not in t and "nan" not in t:
        t += ".0"
    return t + "f"


def _c_init(arr, is_int, indent=1):
    if arr.ndim == 1:
        items = [str(int(v)) if is_int else _c_float(v) for v in arr]
        lines, line = [], ""
        for it in items:
            if len(line) + len(it) + 2 > 110:
                lines.append(line.rstrip())
                line = ""
            line += it + ", "
        lines.append(line.rstrip().rstrip(","))
        if len(lines) == 1:
            return "{ " + lines[0] + " }"
        pad = "\t" * indent
        return "{\n" + ",\n".join(pad + l.rstrip(",") for l in lines) + "\n" + "\t" * (indent - 1) + "}"
    pad = "\t" * indent
    return "{\n" + ",\n".join(pad + _c_init(sub, is_int, indent + 1) for sub in arr) + "\n" + "\t" * (indent - 1) + "}"


ORDER_V2 = ["VERSION", "INTERP_ORDER", "POISSON", "BONDED_MU", "KG_REF",
            "NUM_KG", "KG", "NUM_RH", "RH", "NUM_SH", "SH", "NUM_FACE", "FACE_MU", "NUM_ST", "STATION",
            "CYL_KG_AXIS", "CYL_FORCE", "CYL_HALFWIDTH", "CYL_M1", "CYL_PROFILE", "CYL_STATUS",
            "SPH_KG_AXIS", "SPH_FORCE", "SPH_RADIUS", "SPH_M1", "SPH_SPIN", "SPH_PROFILE", "SPH_STATUS",
            "T3_NUM_S", "T3_S", "T3_RATIO", "T4_NUM_C", "T4_C", "T4_RATIO"]
INTS = {"VERSION", "INTERP_ORDER", "NUM_RH", "NUM_SH", "NUM_FACE", "NUM_ST", "T3_NUM_S", "T4_NUM_C", "CYL_STATUS",
        "SPH_STATUS", "NUM_KG", "CYL_KG_AXIS", "SPH_KG_AXIS"}
_A4 = ["NUM_KG", "NUM_FACE", "NUM_RH", "NUM_SH"]
DIMS_V2 = {"KG": ["NUM_KG"], "RH": ["NUM_RH"], "SH": ["NUM_SH"], "FACE_MU": ["NUM_FACE"], "STATION": ["NUM_ST"],
           "CYL_KG_AXIS": ["NUM_RH", "NUM_SH"], "SPH_KG_AXIS": ["NUM_RH", "NUM_SH"],
           "CYL_FORCE": _A4, "CYL_HALFWIDTH": _A4, "CYL_M1": _A4, "CYL_PROFILE": _A4 + ["NUM_ST"], "CYL_STATUS": _A4,
           "SPH_FORCE": _A4, "SPH_RADIUS": _A4, "SPH_M1": _A4, "SPH_SPIN": _A4, "SPH_PROFILE": _A4 + ["NUM_ST"],
           "SPH_STATUS": _A4, "T3_S": ["T3_NUM_S"], "T3_RATIO": ["NUM_KG", "NUM_FACE", "T3_NUM_S"],
           "T4_C": ["T4_NUM_C"], "T4_RATIO": ["NUM_KG", "NUM_FACE", "T4_NUM_C"]}


def write_header(d, path, provenance):
    """TABLE_FORMAT.v2.md 5 (the same text soft_ref.write_table_header writes; validate_header.py checks they agree)."""
    out = ["// clang-format off", "// GENERATED by " + provenance[0] + "; do not edit by hand."]
    out += ["// " + p for p in provenance[1:]]
    out += ["", "#pragma once", "", "namespace jolt_soft_layer_table {", ""]
    for name in ORDER_V2:
        v = d[name]
        is_int = name in INTS
        typ = "int" if is_int else "float"
        if name in DIMS_V2:
            arr = np.asarray(v)
            want = tuple(int(d[dn]) for dn in DIMS_V2[name])
            if arr.shape != want:
                raise ValueError(f"{name}: shape {arr.shape}, the format says {want}")
            dims = "".join(f"[{dn}]" for dn in DIMS_V2[name])
            out.append(f"constexpr {typ} {name}{dims} = {_c_init(arr, is_int)};")
        else:
            out.append(f"constexpr {typ} {name} = {int(v) if is_int else _c_float(v)};")
    out += ["", "} // namespace jolt_soft_layer_table", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))


def status():
    n = {"grid": 0, "screen": 0, "t3": 0, "t4": 0}
    for root, _, files in os.walk(CKPT):
        for fn in files:
            if fn.endswith(".json"):
                key = os.path.basename(root) if os.path.basename(root) in n else "grid"
                n[key] += 1
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "run", "t34", "header", "status"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--flags", default=os.path.join(HERE, "kg_flags.json"))
    a = ap.parse_args()
    idle()
    if a.cmd == "status":
        print(status())
    elif a.cmd == "screen":
        screen(a.workers)
    elif a.cmd == "t34":
        print(f"t3/t4 wall {run_t34(a.workers):.0f} s")
    elif a.cmd == "run":
        with open(a.flags, encoding="utf-8") as fh:
            fl = json.load(fh)
        flags = {k: np.array(fl[k]) for k in ("cyl", "sph")}
        tasks = grid_tasks(flags)
        wall = run_pool(tasks, a.workers, "grid") if tasks else 0.0
        print(f"grid wall {wall:.0f} s")
    elif a.cmd == "header":
        with open(a.flags, encoding="utf-8") as fh:
            fl = json.load(fh)
        flags = {k: np.array(fl[k]) for k in ("cyl", "sph")}
        d, notes = assemble(flags)
        prov = json.load(open(os.path.join(HERE, "provenance.json"), encoding="utf-8")) if os.path.exists(
            os.path.join(HERE, "provenance.json")) else ["gen_layer_table.py"]
        write_header(d, HEADER, prov + notes)
        print("wrote", HEADER, os.path.getsize(HEADER), "bytes;", len(notes), "STATUS notes")


if __name__ == "__main__":
    main()
