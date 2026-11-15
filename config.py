"""Initialisation: every constant of the demo and the global torch settings, in one place.

Units are metres and radians.  Values follow sec:setup of the paper (TeX labels of main-en.tex).
demo.py calls init() once before loading the network.
"""
from pathlib import Path

import torch

CFG = dict(
    # K, kappa_max (eq:kine-s, eq:repr); erosion 0.30 m (sec:setup); circle_radius = rho (eq:C, eq:eij);
    # refine_margin = m (eq:eij); guide_endpoint = w_e, guide_collision = w_c (eq:guided);
    # gn_damping = lambda_0 (alg:correct, alg:lm); flow_steps = N (eq:euler); endpoint_tol = epsilon (eq:P-goal)
    seed=20260926, k_segments=64, kappa_max=1.7495, half=5.6,
    # Pixels and images.  src_res: NIRRT obstacle pixels; net_cells x net_cells network image at
    # net_res; fine_cells x fine_cells grid at fine_res for sdf and exact check; feature_cells x
    # feature_cells CNN feature map (F_m, sec:fm); heat_sigma: start/goal heat-map width.
    net_res=0.05, fine_res=0.01, src_res=0.05, fine_cells=1120, erosion=0.30,
    net_cells=224, feature_cells=14, heat_sigma=0.12,
    body_length=0.58, body_width=0.31, body_offset=0.17145,
    circle_offsets=(-0.0219, 0.1715, 0.3648), circle_radius=0.183, refine_margin=0.05,
    dim=128, heads=4, blocks=4,
    flow_steps=5, guided_steps=2, guide_endpoint=0.01524, guide_collision=67.51,
    gn_iters=20, gn_damping=0.01, gn_down=0.5, gn_up=10.0, nullspace_damping=1e-6,
    refine_steps=3, refine_max_failed=3, refine_repull=1,
    endpoint_tol=1e-3, check_step=0.02, kappa_tol=1e-3,
    # Figure: size (inches), dpi, candidate drawing step (m), body outline every n route points.
    fig_size=6, fig_dpi=150, plot_step=0.05, body_every=25,
)
K = CFG['k_segments']
KM = CFG['kappa_max']
HALF = CFG['half']
OUT_DIR = Path(__file__).resolve().parent / 'out'  # figures; fixed, not a command-line option


def init():
    """Global torch settings: strict float32 (TF32 off), as in all paper experiments; create OUT_DIR."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
