"""Step 2, network output: flow-matching network and guided sampling (sec:fm, sec:guidance).

FlowModel predicts the velocity v_phi over the 65 normalised route parameters.  sample() runs
5 Euler steps from Gaussian noise; the last 2 add endpoint guidance and collision guidance
projected onto the null space of the endpoint Jacobian.  Weights: weights/flow_seed{0,1,2}.pt.
"""
# Paper (TeX labels of main-en.tex) -> code:
#   sec:fm, eq:token     network, parameter tokens ........ FlowModel
#   eq:norm              label normalisation .............. FlowModel.mean / .std, sample()
#   eq:ode, eq:euler     Euler steps ...................... sample()
#   eq:x1hat             clean estimate, projection Pi .... sample()
#   eq:Le, eq:Fe         endpoint guidance ................ sample()
#   eq:gradC, eq:nullproj, eq:Fc  collision guidance ...... sample(), null_project()
#   eq:guided            guided update .................... sample()
import math

import torch
from torch import nn
from torch.nn import functional as F

from common import collision_cost, endpoint_jacobian, endpoints, heat_images, midpoint_poses, project
from config import CFG, HALF, K


def sincos_2d(x, y, channels):
    quarter = channels // 4
    freq = torch.exp(-math.log(10000.) * torch.arange(quarter, device=x.device, dtype=x.dtype) / max(quarter - 1, 1))
    x, y = x[..., None] * freq, y[..., None] * freq
    return torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, tokens, map_tokens):
        q = self.self_norm(tokens)
        tokens = tokens + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.cross_norm(tokens)
        tokens = tokens + self.cross_attn(q, map_tokens, map_tokens, need_weights=False)[0]
        return tokens + self.ff(self.ff_norm(tokens))


class FlowModel(nn.Module):
    """Velocity field over the 65 route tokens, conditioned on map, start and goal."""

    def __init__(self, mean, std):
        super().__init__()
        dim = CFG['dim']
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.value = nn.Linear(1, dim)
        self.index = nn.Embedding(K + 1, dim)
        self.condition = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cnn = nn.Sequential(nn.Conv2d(3, 24, 5, 2, 2), nn.SiLU(), nn.Conv2d(24, 48, 5, 2, 2), nn.SiLU(),
                                 nn.Conv2d(48, 96, 3, 2, 1), nn.SiLU(), nn.Conv2d(96, dim, 3, 2, 1), nn.SiLU())
        self.map_projection = nn.Linear(dim, dim)
        self.pose_projection = nn.Linear(dim + 2, dim)
        self.local_projection = nn.Linear(3 * dim, dim)
        self.blocks = nn.ModuleList([Block(dim, CFG['heads']) for _ in range(CFG['blocks'])])
        self.head = nn.Linear(dim, 1)

    @classmethod
    def load(cls, path, device='cpu'):
        ck = torch.load(path, map_location=device, weights_only=True)
        net = cls(ck['mean'], ck['std']).to(device).eval()
        net.load_state_dict(ck['model'])
        return net

    def forward(self, x, t, images, starts, goals):
        b, dim = x.shape[0], CFG['dim']
        # eq:token: h_theta[k] = h_value + h_index(k) + h_z + h_tau, k = 0..K
        cond = torch.stack([starts[:, 0] / HALF, starts[:, 1] / HALF, torch.sin(starts[:, 2]),
                            torch.cos(starts[:, 2]), goals[:, 0] / HALF, goals[:, 1] / HALF], dim=-1)
        tokens = self.value(x[..., None]) + self.index(torch.arange(K + 1, device=x.device))[None]
        tokens = tokens + self.condition(cond)[:, None] + self.time(t[:, None])[:, None]
        # Position injection (sec:fm): F_m is 14 x 14 x 128; map tokens h_m get 2D encoding
        feature_map = self.cnn(images)
        n = CFG['feature_cells']
        yy, xx = torch.meshgrid(torch.arange(n, device=x.device, dtype=x.dtype),
                                torch.arange(n, device=x.device, dtype=x.dtype), indexing='ij')
        map_tokens = self.map_projection(feature_map.flatten(2).transpose(1, 2)) + sincos_2d(xx, yy, dim).reshape(1, n * n, dim)
        # Midpoint state of each segment from eq:D-psi..eq:D-y, after undoing eq:norm
        with torch.no_grad():
            q = midpoint_poses(x * self.std[None] + self.mean[None], starts, goals)
            xy, yaw = q[..., :2], q[..., 2]
            encoded = torch.cat([sincos_2d(xy[..., 0] / HALF, xy[..., 1] / HALF, dim),
                                 torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]], dim=-1)
        tokens = torch.cat([tokens[:, :1], tokens[:, 1:] + self.pose_projection(encoded)], dim=1)
        # Features at the midpoint and 0.3 m before/after it along the heading
        direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        sample = torch.stack([xy - .3 * direction, xy, xy + .3 * direction], dim=2)
        grid = torch.stack([sample[..., 0] / HALF, -sample[..., 1] / HALF], dim=-1).reshape(b, K * 3, 1, 2)
        local = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        local = local.squeeze(-1).transpose(1, 2).reshape(b, K, 3 * dim)
        tokens = torch.cat([tokens[:, :1], tokens[:, 1:] + self.local_projection(local)], dim=1)
        for block in self.blocks:
            tokens = block(tokens, map_tokens)
        return self.head(tokens).squeeze(-1)  # v_phi(theta~, tau, z) in eq:ode, eq:euler, eq:x1hat


def null_project(jac, grad):
    # eq:nullproj: P_1 grad = (I - J_1^+ J_1) grad, with J_1^+ = J_1^T (J_1 J_1^T + 1e-6 I)^-1
    eye = torch.eye(2, device=grad.device, dtype=grad.dtype)[None]
    dual = torch.linalg.solve(jac @ jac.transpose(1, 2) + CFG['nullspace_damping'] * eye, jac @ grad[..., None])
    return grad - (jac.transpose(1, 2) @ dual).squeeze(-1)


def sample(model, starts, goals, images, sdfs, n=16, generator=None):
    """Five Euler flow steps; the last two add endpoint and null-space collision guidance."""
    # alg:infer, sampling loop. x is theta~ (normalised, eq:norm); theta~_0 ~ N(0, I) as in eq:ode
    steps, guided = CFG['flow_steps'], CFG['guided_steps']
    x = torch.randn(len(starts) * n, 65, generator=generator, dtype=torch.float32,
                    device=generator.device if generator is not None else 'cpu').to(starts.device)
    st, go = starts.repeat_interleave(n, 0), goals.repeat_interleave(n, 0)
    img = heat_images(images, starts, goals).repeat_interleave(n, 0)
    sdf = sdfs.repeat_interleave(n, 0)
    mean, std = model.mean[None], model.std[None]
    for step in range(steps):
        t = torch.full((len(x),), step / steps, device=x.device)  # tau_n = n / N
        if step >= steps - guided:
            work = x.detach().requires_grad_(True)
            v = model(work, t, img, st, go)
            clean = work + (1 - step / steps) * v  # theta^_1, eq:x1hat
            phys = project(clean * std + mean)  # theta^ = Pi(sigma . theta^_1 + mu), eq:x1hat
            e_cost = (endpoints(phys, st, go) - go).square().sum()  # C_e, eq:Le
            c_cost = collision_cost(phys, st, go, sdf, 0., False).sum()  # C_c, eq:C
            g_end = torch.autograd.grad(CFG['guide_endpoint'] * e_cost, work, retain_graph=True)[0]  # w_e F_e, eq:Fe
            g_col = torch.autograd.grad(CFG['guide_collision'] * c_cost, clean, retain_graph=True)[0]  # w_c grad_{theta^_1} C_c
            det = phys.detach().requires_grad_(True)
            _, jac = endpoint_jacobian(det, st, go)
            # J_1 = J_e sigma (chain rule through eq:x1hat), project with P_1 (eq:nullproj), then
            # pull back through d theta^_1 / d theta~_n: w_c F_c of eq:Fc (cf. eq:gradC)
            g_col = torch.autograd.grad(clean, work, grad_outputs=null_project(jac * std[:, None, :], g_col))[0]
            x = (work + (v - g_end - g_col) / steps).detach()  # guided update, eq:guided
        else:
            with torch.no_grad():
                x = x + model(x, t, img, st, go) / steps  # unguided Euler step, eq:euler
        with torch.no_grad():
            # alg:infer: to physical units, clip curvatures (Pi), renormalise (eq:norm)
            x = (project(x * std + mean) - mean) / std
    return project(x * std + mean)  # theta = Pi(sigma . theta~_N + mu)

