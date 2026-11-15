"""Step 3, post-processing: endpoint and collision correction (sec:correction).

lm_endpoint: Levenberg-Marquardt on the 2-D endpoint residual (alg:lm).
refine: constrained Gauss-Newton on the circle residuals, with the linearised endpoint
residual as an equality constraint, solved through the KKT system (alg:correct).
"""
# Paper (TeX labels of main-en.tex) -> code; each equation is quoted in LaTeX where it is used.
#   eq:re-lin, eq:lm-e, eq:lm-e-sol, alg:lm  endpoint LM .. lm_endpoint()
#   eq:sdf, eq:eij, eq:e-lin  collision residual, J_c ..... _single_residual()
#   eq:eq-con, eq:lm-c, eq:kkt, eq:nu, eq:dthc, alg:correct  refine()
# Notation (sec:correction):
# \[
# r_e(\theta)=p_K(\theta)-g, \qquad
# J_e=\partial p_K/\partial\theta\in\mathbb R^{2\times(K+1)}
# \]
import torch

from common import circle_centers, collision_cost, endpoint_jacobian, endpoints, project
from config import CFG, HALF


def lm_endpoint(z, start, goal, steps=None, require_tol=True):
    """Levenberg-Marquardt endpoint correction, alg:lm (LM-Endpoint).

    At most CFG['gn_iters'] = 20 iterations; stops once every endpoint error is at most
    epsilon = 1 mm (require_tol=True).  steps=1, require_tol=False gives the single
    endpoint step used inside refine().
    """
    # eq:re-lin
    # \[
    # r_e(\theta+\Delta\theta_e)
    # \approx
    # r_e+J_e\Delta\theta_e.
    # \]
    # eq:lm-e
    # \[
    # \min_{\Delta\theta_e}
    # \ \lVert r_e+J_e\Delta\theta_e\rVert^2
    # +\lambda_e\lVert\Delta\theta_e\rVert^2,
    # \]
    steps = steps or CFG['gn_iters']
    tol = CFG['endpoint_tol']
    z = project(z.detach())
    damping = torch.full((len(z),), CFG['gn_damping'], device=z.device, dtype=z.dtype)  # lambda_e = lambda_0
    eye = torch.eye(2, device=z.device, dtype=z.dtype)[None]
    for _ in range(steps):
        before = torch.linalg.vector_norm(endpoints(z, start, goal) - goal, dim=1)  # norm of r_e(theta)
        if require_tol and bool(torch.all(before <= tol)):
            break
        active = before > tol if require_tol else torch.ones_like(before, dtype=torch.bool)
        work = z.detach().requires_grad_(True)
        end, jac = endpoint_jacobian(work, start, goal)  # p_K(theta), J_e
        # eq:lm-e-sol (the code uses the second form: only a 2 x 2 solve)
        # \[
        # \begin{aligned}
        # \Delta\theta_e
        # &=
        # -\bigl(J_e^\top J_e+\lambda_e I\bigr)^{-1}J_e^\top r_e\\
        # &=
        # -J_e^\top
        # \bigl(J_eJ_e^\top+\lambda_e I\bigr)^{-1}r_e.
        # \end{aligned}
        # \]
        dual = torch.linalg.solve(jac @ jac.transpose(1, 2) + damping[:, None, None] * eye, (end - goal)[..., None])
        cand = project(work.detach() - (jac.transpose(1, 2) @ dual).squeeze(-1))  # theta' = Pi(theta + Delta theta_e)
        after = torch.linalg.vector_norm(endpoints(cand, start, goal) - goal, dim=1)
        # alg:lm: accept if the residual decreases (lambda_e / 2), else reject (lambda_e x 10)
        accept = active & torch.isfinite(after) & (after < before)
        z = torch.where(accept[:, None], cand, z)
        damping = torch.where(accept, damping * CFG['gn_down'], damping * CFG['gn_up'])
    return z


def _single_residual(z, start, goal, sdf):
    """Collision residual r_c (3 N_p = 384 circles) and its Jacobian J_c (384 x 65) for one route.

    sdf includes the map border (eq:sdf); its value and gradient come from bilinear
    interpolation of the grid.  By the chain rule, J_c = -(grad sdf)^T d o_ij / d theta on
    circles with a positive residual, and 0 elsewhere.
    """
    def centers(v):
        return circle_centers(v[None], start[None], goal[None])[0].reshape(384, 2)
    c = centers(z)
    cj = torch.func.jacfwd(centers)(z)  # d o_ij / d theta, 384 x 2 x 65
    h, w = sdf.shape
    px = (c[:, 0] / HALF + 1) * w / 2 - .5
    py = (-c[:, 1] / HALF + 1) * h / 2 - .5
    x0, y0 = torch.floor(px).long(), torch.floor(py).long()
    fx, fy = px - x0, py - y0

    def at(x, y):
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        return sdf[y.clamp(0, h - 1), x.clamp(0, w - 1)] * ok.to(sdf.dtype)
    v00, v10, v01, v11 = at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1)
    d = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11  # sdf_obs(o_ij)
    g = torch.stack([((1 - fy) * (v10 - v00) + fy * (v11 - v01)) * w / (2 * HALF),
                     -((1 - fx) * (v01 - v00) + fx * (v11 - v10)) * h / (2 * HALF)], dim=-1)
    # eq:sdf (value and gradient of whichever term is smaller)
    # \[
    # \mathrm{sdf}(o)
    # =
    # \min\bigl(
    # \mathrm{sdf}_{\mathrm{obs}}(o),
    # \ 5.6-\max(|o_x|,|o_y|)
    # \bigr),
    # \]
    axis = c.abs().argmax(dim=-1)
    wall = HALF - c.abs().amax(dim=-1)
    wg = torch.zeros_like(g).scatter_(1, axis[:, None], -torch.sign(c.gather(1, axis[:, None])))
    g = torch.where((wall < d)[:, None], wg, g)
    d = torch.minimum(d, wall)
    # eq:eij
    # \[
    # r_{c,ij}(\theta)
    # =
    # \max\bigl(
    # 0,\,
    # \rho+m-\mathrm{sdf}(o_{ij}(\theta))
    # \bigr),
    # \]
    radius = CFG['circle_radius'] + CFG['refine_margin']  # rho + m
    res = torch.relu(radius - d)
    # eq:e-lin, with J_c = \partial r_c/\partial\theta
    # \[
    # r_c(\theta+\Delta\theta_c)
    # \approx
    # r_c+J_c\Delta\theta_c.
    # \]
    jac = -torch.einsum('ni,nij->nj', g, cj) * ((radius - d) > 0)[:, None]
    return res, jac


def refine(z, start, goal, sdf):
    """Collision correction, alg:correct (Correct), after its first LM-Endpoint call (done in plan()).

    The input z is theta-bar of alg:correct: it is returned if the final endpoint correction
    fails.  Each correction step solves eq:lm-c through the KKT system eq:kkt.
    """
    # eq:eq-con
    # \[
    # r_e+J_e\Delta\theta_c=0.
    # \]
    # eq:lm-c
    # \[
    # \min_{\Delta\theta_c}
    # \ \lVert r_c+J_c\Delta\theta_c\rVert^2
    # +\lambda_c\lVert\Delta\theta_c\rVert^2
    # \quad
    # \text{s.t.}
    # \quad
    # r_e+J_e\Delta\theta_c=0.
    # \]
    margin, tol = CFG['refine_margin'], CFG['endpoint_tol']
    z = project(z.detach())
    original = z.clone()  # theta-bar
    with torch.no_grad():
        cost = collision_cost(z, start, goal, sdf, margin, True)  # squared norm of r_c / (3 N_p)
    err = torch.linalg.vector_norm(endpoints(z, start, goal) - goal, dim=1)
    active = (cost > 0) & (err <= tol)  # alg:correct: still colliding and endpoint reached
    failures = torch.zeros(len(z), dtype=torch.int32, device=z.device)
    damping = torch.full((len(z),), .01, dtype=z.dtype, device=z.device)  # lambda_c = lambda_0
    eye = torch.eye(65, dtype=z.dtype, device=z.device)[None]
    for _ in range(CFG['refine_steps']):  # alg:correct: at most 3 correction steps
        ids = torch.where(active)[0]
        if not len(ids):
            break
        with torch.no_grad():
            e, jc = torch.vmap(_single_residual)(z[ids], start[ids], goal[ids], sdf[ids])  # r_c, J_c
        work = z[ids].detach().requires_grad_(True)
        end, je = endpoint_jacobian(work, start[ids], goal[ids])  # p_K(theta), J_e
        r = end - goal[ids]  # r_e
        selected, sel_cost = z[ids].clone(), cost[ids].clone()
        accepted = torch.zeros(len(ids), dtype=torch.bool, device=z.device)
        for _ in range(3):  # alg:correct: up to 3 re-solves with larger lambda_c
            rem = torch.where(~accepted)[0]
            if not len(rem):
                break
            n = len(rem)
            # eq:kkt (solved directly, 67 x 67; [:65] keeps Delta theta_c and drops nu)
            # \[
            # \begin{bmatrix}
            # J_c^\top J_c+\lambda_c I & J_e^\top\\
            # J_e & 0
            # \end{bmatrix}
            # \begin{bmatrix}
            # \Delta\theta_c\\
            # \nu
            # \end{bmatrix}
            # =
            # \begin{bmatrix}
            # -J_c^\top r_c\\
            # -r_e
            # \end{bmatrix}.
            # \]
            # eq:nu, eq:dthc: the equivalent elimination, with H=J_c^\top J_c+\lambda_c I, b=J_c^\top r_c
            # (not used here; same Delta theta_c)
            # \[
            # \begin{aligned}
            # \nu
            # &=
            # \bigl(J_eH^{-1}J_e^\top\bigr)^{-1}
            # \bigl(r_e-J_eH^{-1}b\bigr),\\
            # \Delta\theta_c
            # &=
            # -H^{-1}\bigl(b+J_e^\top\nu\bigr).
            # \end{aligned}
            # \]
            kkt = torch.zeros(n, 67, 67, dtype=z.dtype, device=z.device)
            kkt[:, :65, :65] = jc[rem].transpose(1, 2) @ jc[rem] + damping[ids[rem], None, None] * eye
            kkt[:, :65, 65:] = je[rem].transpose(1, 2)
            kkt[:, 65:, :65] = je[rem]
            rhs = torch.cat([-(jc[rem].transpose(1, 2) @ e[rem, :, None]).squeeze(-1), -r[rem]], dim=1)
            delta = torch.linalg.solve(kkt, rhs[..., None]).squeeze(-1)[:, :65]
            # theta' = Pi(theta + Delta theta_c), then 1 endpoint-correction step eq:lm-e-sol
            trial = lm_endpoint(project(z[ids[rem]] + delta), start[ids[rem]], goal[ids[rem]],
                                CFG['refine_repull'], require_tol=False)
            with torch.no_grad():
                nc = collision_cost(trial, start[ids[rem]], goal[ids[rem]], sdf[ids[rem]], margin, True)
                ne = torch.linalg.vector_norm(endpoints(trial, start[ids[rem]], goal[ids[rem]]) - goal[ids[rem]], dim=1)
                good = (nc < cost[ids[rem]]) & (ne <= tol)  # alg:correct: r_c decreases and r_e within epsilon
            # accept: lambda_c / 2; reject: lambda_c x 10 and solve again
            selected[rem[good]] = trial[good]
            sel_cost[rem[good]] = nc[good]
            accepted[rem[good]] = True
            damping[ids[rem[good]]] *= .5
            damping[ids[rem[~good]]] *= 10.
        z[ids] = selected
        cost[ids] = sel_cost
        failures[ids] = torch.where(accepted, 0, failures[ids] + 1)
        active[ids[failures[ids] >= CFG['refine_max_failed']]] = False
        active[ids[cost[ids] <= 0]] = False  # alg:correct: stop when collision-free
    final = lm_endpoint(z, start, goal)  # alg:correct: final endpoint correction
    err = torch.linalg.vector_norm(endpoints(final, start, goal) - goal, dim=1)
    return torch.where((err <= tol)[:, None], final, original)  # theta' if endpoint within epsilon, else theta-bar
