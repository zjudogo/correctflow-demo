"""Shared by every step: route geometry and eroded maps (constants are in config.py).

Route: 65 parameters theta = [a, c_1..c_64] -> total length S = d (1 + softplus(a)), with d the
start-goal distance, and 64 equal arcs of curvature kappa_max clip(c_k, -1, 1) (eq:repr).
Map: NIRRT random_2d rectangles and circles (0.05 m/px), merged and ERODED by 0.30 m (sec:setup).
World frame in metres, origin at the map centre, y up, extent [-5.6, 5.6]^2.
Pose = rear-axle [x, y, yaw]; body 0.58 x 0.31 m, centre 0.17145 m ahead of the rear axle.
"""
# Paper (TeX labels of main-en.tex) -> code:
#   sec:setup            merge and erode by 0.30 m ............ fine_maps(), net_map()
#   eq:kine-s, eq:D      arc-length model, discrete form ...... poses()
#   eq:repr              theta -> (S, kappa_k) ................ poses(), box_clip()
#   eq:x1hat             projection Pi ........................ project()
#   eq:D-goal, eq:Le     endpoint p_K, Jacobian J_e ........... endpoints(), endpoint_jacobian()
#   eq:C, eq:sdf, eq:eij circles o_ij, sdf, collision cost .... circle_centers(), clearance(), collision_cost()
#   sec:fm               network input image .................. net_map(), heat_images()
import math

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F

from config import CFG, HALF, K, KM


# ----------------------------------------------------------------------------- maps
def fine_maps(env):
    """
    Eroded 1 cm occupancy grid and its signed distance field, for guidance, correction and check.

    env: NIRRT map, rectangles [x, y, w, h] and circles [x, y, r] in 0.05 m pixels, y down.
    Returns (eroded, sdf), both fine_cells x fine_cells (row 0 = top of the map):
      eroded  bool, cells still occupied after shrinking all obstacles by CFG['erosion'];
      sdf     float32 metres, sdf_obs of eq:C and eq:sdf (obstacles only, no map border),
              positive in free space; a lower bound of the true distance (see below).
    """
    # Rasterise conservatively: a cell is occupied if it touches any obstacle.
    res, side, src = CFG['fine_res'], CFG['fine_cells'], CFG['src_res']
    occ = np.zeros((side, side), bool)
    for x, y, w, h in env['rectangle_obstacles']:
        c0, c1 = max(0, math.floor(x * src / res)), min(side,
                                                        math.ceil((x + w) * src / res))
        r0, r1 = max(0, math.floor(y * src / res)), min(side,
                                                        math.ceil((y + h) * src / res))
        if r1 > r0 and c1 > c0:
            occ[r0:r1, c0:c1] = True
    for x, y, rad in env['circle_obstacles']:
        cx, cy, rr = x * src / res, y * src / res, rad * src / res
        c0, c1 = max(0, math.floor(cx - rr - 1)
                     ), min(side, math.ceil(cx + rr + 1))
        r0, r1 = max(0, math.floor(cy - rr - 1)
                     ), min(side, math.ceil(cy + rr + 1))
        rows, cols = np.ogrid[r0:r1, c0:c1]
        dx = np.maximum(np.maximum(cols - cx, cx - (cols + 1)), 0)
        dy = np.maximum(np.maximum(rows - cy, cy - (rows + 1)), 0)
        # dx, dy: gap from the circle centre to the nearest point of each cell (0 if inside its span)
        occ[r0:r1, c0:c1] |= dx * dx + dy * dy <= rr * rr
    half_diag = math.sqrt(2) * res / 2
    # Pad with free cells so obstacles touching the map edge shrink from that side as well.
    pad = math.ceil(CFG['erosion'] / res) + 2
    inside = distance_transform_edt(np.pad(occ, pad))[pad:-pad, pad:-pad] * res
    # Keep a cell if it is at least `erosion` from free space (half-diagonal allowance: cell, not centre).
    eroded = occ & (inside + half_diag >= CFG['erosion'])
    sdf = distance_transform_edt(~eroded) * res - \
        distance_transform_edt(eroded) * res - half_diag
    # centre-to-centre minus half diagonal, rounded down: never too large
    sdf = np.floor(sdf / res) * res
    return eroded, sdf.astype(np.float32)


def net_map(env):
    """Eroded occupancy image fed to the network (channel 0 of heat_images).

    net_cells x net_cells float32 at 0.05 m/px (row 0 = top), 1 = occupied.  A coarse version of
    fine_maps used only as network input; all checks use fine_maps.
    """
    n = CFG['net_cells']
    rr, cc = np.ogrid[:n, :n]
    occ = np.zeros((n, n), bool)
    for x, y, w, h in env['rectangle_obstacles']:
        occ[int(y):int(y + h), int(x):int(x + w)] = True
    for x, y, r in env['circle_obstacles']:
        occ[(cc - x) ** 2 + (rr - y) ** 2 <= r * r] = True
    return (distance_transform_edt(occ) * CFG['net_res'] > CFG['erosion']).astype(np.float32)


def heat_images(maps, starts, goals):
    """Three-channel network input: eroded map, start heat map, goal heat map.

    maps (B, n, n) from net_map; starts (B, 3) and goals (B, 2) in metres.
    Returns (B, 3, n, n); channels 1-2 are Gaussians of width CFG['heat_sigma'] at the start and
    goal positions, evaluated at pixel centres.
    """
    n = CFG['net_cells']
    axis = (torch.arange(n, device=maps.device,
            dtype=maps.dtype) + .5 - n // 2) * CFG['net_res']
    yy, xx = torch.meshgrid(-axis, axis, indexing='ij')
    s2 = 2 * CFG['heat_sigma'] * CFG['heat_sigma']
    h1 = torch.exp(-((xx - starts[:, 0, None, None]) **
                   2 + (yy - starts[:, 1, None, None]) ** 2) / s2)
    h2 = torch.exp(-((xx - goals[:, 0, None, None]) **
                   2 + (yy - goals[:, 1, None, None]) ** 2) / s2)
    return torch.cat([maps[:, None], h1[:, None], h2[:, None]], dim=1)


def clearance(centers, sdf, wall):
    """Signed distance (m) at points: bilinear lookup in sdf, optionally including the map border.

    centers (B, ..., 2) world points; sdf (B, H, W) from fine_maps.  Points outside the map read 0.
    wall=True gives sdf of eq:sdf (border included); wall=False gives sdf_obs of eq:C.
    """
    grid = torch.stack(
        [centers[..., 0] / HALF, -centers[..., 1] / HALF], dim=-1)
    d = F.grid_sample(sdf[:, None], grid.reshape(len(sdf), -1, 1, 2), mode='bilinear',
                      padding_mode='zeros', align_corners=False).reshape(centers.shape[:-1])
    if wall:
        # eq:sdf: sdf(o) = min(sdf_obs(o), 5.6 - max(abs(o_x), abs(o_y)))
        d = torch.minimum(d, HALF - centers.abs().amax(dim=-1))
    return d


# ----------------------------------------------------------------------------- route geometry
def box_clip(c):
    """clip(c_k, -1, 1) of eq:repr; the gradient is 1 where -1 < c_k < 1 and 0 elsewhere."""
    inside = (c.abs() < 1.).to(c.dtype)
    return c.clamp(-1., 1.).detach() + (c - c.detach()) * inside


def project(z):
    """Projection Pi of eq:x1hat, alg:infer, alg:correct, alg:lm: clip the curvature parameters, keep a.

    z (B, 65) = [a, c_1..c_64] in physical (not normalised) units.
    """
    return torch.cat([z[:, :1], box_clip(z[:, 1:])], dim=1)


def poses(z, start, goal, sub=1):
    """Poses along the route by exact constant-curvature integration (eq:repr, eq:D-psi..eq:D-y).

    z (B, 65) route parameters; start (B, 3) pose [x, y, yaw]; goal (B, 2) position.
    sub: poses per segment, at fractions 1/sub .. 1 of it (the start pose itself is not included).
    Returns (q, endpoint, length): q (B, 64*sub, 3), endpoint (B, 2) = p_K, length (B,) = S.
    Differentiable in z.
    """
    c = box_clip(z[:, 1:])
    # eq:repr: S = d (1 + softplus(a)), kappa_k = kappa_max clip(c_k, -1, 1)
    length = torch.linalg.vector_norm(
        goal - start[:, :2], dim=1) * (1. + F.softplus(z[:, 0]))
    seg = length[:, None] / K  # Delta s = S / K
    turn = KM * c * seg  # kappa_k Delta s
    # psi_{k-1} for every segment, eq:D-psi, starting from q_0 = q_s (eq:D-start)
    heading = start[:, 2:3] + torch.cumsum(F.pad(turn[:, :-1], (1, 0)), dim=1)
    # torch.sinc(u) = sin(pi u)/(pi u), so this is Delta s sinc(kappa_k Delta s / 2) of eq:D-x, eq:D-y
    travel = seg * torch.sinc(turn / (2 * math.pi))
    shift = torch.stack([travel * torch.cos(heading + turn / 2),
                        travel * torch.sin(heading + turn / 2)], dim=-1)
    base = start[:, None, :2] + \
        torch.cumsum(F.pad(shift[:, :-1], (0, 0, 1, 0)), dim=1)
    # States inside segment k: Delta s replaced by beta in [0, Delta s] in eq:D-psi..eq:D-y
    local = seg[:, :, None] * \
        torch.arange(1, sub + 1, device=z.device, dtype=z.dtype) / sub
    small = (KM * c)[:, :, None] * local
    yaw = heading[:, :, None] + small
    xy = base[:, :, None, :] + (local * torch.sinc(small / (2 * math.pi)))[..., None] * torch.stack(
        [torch.cos(heading[:, :, None] + small / 2), torch.sin(heading[:, :, None] + small / 2)], dim=-1)
    q = torch.cat([xy, yaw[..., None]], dim=-1).reshape(len(z), K * sub, 3)
    return q, q[:, -1, :2], length


def midpoint_poses(z, starts, goals):
    """Pose at the middle of each of the 64 segments, (B, 64, 3); position features of the network.

    Plain clamp instead of box_clip: the network calls it without gradients.
    """
    zz = torch.cat([z[:, :1], z[:, 1:].clamp(-1., 1.)], dim=1)
    q, _, _ = poses(zz, starts, goals, 2)
    return q.reshape(len(z), K, 2, 3)[:, :, 0]


def endpoints(z, start, goal):
    """Route endpoint p_K(theta) = (x_K, y_K) of eq:D-goal, eq:Le, eq:re-lin; (B, 2)."""
    return poses(z, start, goal, 1)[1]


def endpoint_jacobian(z, start, goal):
    """Endpoint and its Jacobian J_e = d p_K / d theta (sec:correction).

    z must require grad.  Returns (endpoint (B, 2), J_e (B, 2, 65)).  Multiplied by sigma it is
    J_1 of eq:nullproj.  Summing over the batch is exact: each endpoint depends on its own z only.
    """
    end = endpoints(z, start, goal)
    dx = torch.autograd.grad(end[:, 0].sum(), z, retain_graph=True)[0]
    dy = torch.autograd.grad(end[:, 1].sum(), z)[0]
    return end.detach(), torch.stack([dx, dy], dim=1)


def circle_centers(z, start, goal):
    """Centres o_ij(theta) of the three body circles (eq:C, eq:eij); (B, N_p=128, 3, 2).

    Poses at segment midpoints and ends; circles on the body axis at CFG['circle_offsets'] ahead of
    the rear axle.  Heading from the chord to the previous point (an approximation used only for
    the costs; the exact check in check.py uses the true heading).
    """
    q, _, _ = poses(z, start, goal, 2)
    xy = q[..., :2]
    delta = torch.diff(xy, dim=1, prepend=xy[:, :1])
    yaw = torch.atan2(delta[..., 1], delta[..., 0] + 1e-8)
    yaw = torch.cat([start[:, 2, None], yaw[:, 1:]], dim=1)
    off = torch.as_tensor(CFG['circle_offsets'],
                          device=z.device, dtype=z.dtype)
    direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return xy[:, :, None] + off[None, None, :, None] * direction[:, :, None]


def collision_cost(z, start, goal, sdf, margin, wall):
    """Mean squared circle penetration per route, (B,).

    margin=0, wall=False: C_c of eq:C, mean of max(0, rho - sdf_obs)^2 over the 3 N_p circles.
    margin=m, wall=True: mean of the squared residuals r_c of eq:eij, i.e. squared norm of
    r_c / (3 N_p); used for the accept test of alg:correct.
    """
    d = clearance(circle_centers(z, start, goal), sdf, wall)
    return F.relu(CFG['circle_radius'] + margin - d).square().mean((1, 2))
