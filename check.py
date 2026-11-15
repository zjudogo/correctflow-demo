"""Step 4, check and filter: exact feasibility check (sec:select, eq:P).

Independent float64 numpy re-implementation of the route, sampled every 0.02 m.  A route is
feasible when curvature, 1 mm endpoint error and rectangle-body collision (against the
eroded obstacles and the map border) all pass.  demo.py keeps the shortest feasible one.
"""
# Paper (TeX labels of main-en.tex) -> code:
#   eq:repr, eq:D-psi..eq:D-y  dense route in float64 ..... route()
#   eq:P-coll, eq:D-coll  rectangle body vs eroded cells .. body_hits()
#   eq:P, sec:select     feasibility ...................... check()
import math

import numpy as np

from config import CFG, HALF, K, KM


def route(z, start, goal, step=None):
    """Densify a 65-parameter route every `step` m (plus every arc end); float64."""
    # Same as poses(): eq:repr, then exact integration eq:D-psi..eq:D-y in float64
    step = step or CFG['check_step']
    z = np.asarray(z, np.float64)
    length = float(np.hypot(goal[0] - start[0], goal[1] - start[1])) * (1 + float(np.logaddexp(0, z[0])))
    k = KM * np.clip(z[1:], -1, 1)
    seg = np.full(K, length / K)
    ends = np.r_[0, np.cumsum(seg)]
    q = np.empty((K + 1, 3))
    q[0] = start
    for i in range(K):
        a = k[i] * seg[i]
        d = seg[i] * np.sinc(a / (2 * np.pi))
        q[i + 1] = q[i] + [d * np.cos(q[i, 2] + a / 2), d * np.sin(q[i, 2] + a / 2), a]
    s = np.unique(np.r_[np.arange(0, ends[-1], step), ends])
    ix = np.clip(np.searchsorted(ends, s, side='right') - 1, 0, K - 1)
    dl = s - ends[ix]
    a = k[ix] * dl
    d = dl * np.sinc(a / (2 * np.pi))
    out = np.c_[q[ix, 0] + d * np.cos(q[ix, 2] + a / 2), q[ix, 1] + d * np.sin(q[ix, 2] + a / 2), q[ix, 2] + a]
    return out, s, length


def body_hits(eroded, sdf, poses_):
    """Exact rotated-rectangle vs occupied-1cm-cell test; True per colliding pose."""
    # Body constraint B(q(s)) in F, eq:P-coll / eq:D-coll: rectangle body against eroded cells and border
    res, side = CFG['fine_res'], CFG['fine_cells']
    hl, hw, off = CFG['body_length'] / 2, CFG['body_width'] / 2, CFG['body_offset']
    co, si = np.cos(poses_[:, 2]), np.sin(poses_[:, 2])
    cx, cy = poses_[:, 0] + off * co, poses_[:, 1] + off * si
    bx, by = np.abs(co) * hl + np.abs(si) * hw, np.abs(si) * hl + np.abs(co) * hw
    hit = (np.abs(cx) + bx >= HALF) | (np.abs(cy) + by >= HALF)
    col = np.floor((cx + HALF) / res).astype(int)
    row = np.floor((HALF - cy) / res).astype(int)
    inside = (row >= 0) & (row < side) & (col >= 0) & (col < side)
    far = np.zeros(len(poses_), bool)
    far[inside] = sdf[row[inside], col[inside]] > math.hypot(hl, hw) + 2 * res
    for i in np.where(~hit & ~far)[0]:
        c0 = max(0, int(math.floor((cx[i] - bx[i] + HALF) / res)) - 1)
        c1 = min(side - 1, int(math.floor((cx[i] + bx[i] + HALF) / res)) + 1)
        r0 = max(0, int(math.floor((HALF - cy[i] - by[i]) / res)) - 1)
        r1 = min(side - 1, int(math.floor((HALF - cy[i] + by[i]) / res)) + 1)
        rr, cc = np.nonzero(eroded[r0:r1 + 1, c0:c1 + 1])
        if not len(rr):
            continue
        dx = -HALF + (cc + c0 + .5) * res - cx[i]
        dy = HALF - (rr + r0 + .5) * res - cy[i]
        proj = res * (abs(co[i]) + abs(si[i])) / 2
        ok = ((np.abs(dx) <= bx[i] + res / 2) & (np.abs(dy) <= by[i] + res / 2)
              & (np.abs(dx * co[i] + dy * si[i]) <= hl + proj) & (np.abs(-dx * si[i] + dy * co[i]) <= hw + proj))
        hit[i] = bool(ok.any())
    return hit


def check(z, start, goal, fine):
    """Feasibility of one route: curvature, 1 mm endpoint, body collision every 2 cm."""
    # Feasible path of eq:P, sec:select: curvature (eq:P-curv), endpoint (eq:P-goal), body (eq:P-coll)
    q, s, length = route(z, start, goal)
    ds = np.diff(s)
    dyaw = np.diff(np.unwrap(q[:, 2]))
    v = ds > 1e-10
    kappa = float(np.abs(dyaw[v] / ds[v]).max(initial=0))
    chord = np.diff(q[:, :2], axis=0)
    tangent = np.arctan2(chord[:, 1], chord[:, 0])
    terr = np.abs(np.angle(np.exp(1j * (tangent - (q[:-1, 2] + dyaw / 2)))))
    terr = float(np.rad2deg(terr[np.linalg.norm(chord, axis=1) > 1e-10]).max(initial=0))
    violation = kappa > KM * (1 + CFG['kappa_tol']) or terr > 1.
    endpoint = float(np.hypot(q[-1, 0] - goal[0], q[-1, 1] - goal[1]))
    collision = bool(body_hits(fine[0], fine[1], q).any())
    return dict(success=not collision and not violation and endpoint <= CFG['endpoint_tol'],
                collision=collision, violation=violation, endpoint_m=endpoint, length_m=length, max_kappa=kappa)

