"""Minimal demo: plan the bundled example queries and draw the result.

Pipeline, one file per step:
    (1) demo.py        read examples/*.json (one map and one query per file), run the steps below, print a summary
    (2) model.py       network output: guided flow sampling of 16 candidates
    (3) postprocess.py endpoint LM and collision correction
    (4) check.py       exact feasibility check; demo.py keeps the shortest feasible route
    (5) plot.py        figure out/<query>.png
    config.py          initialisation: all constants (pixels, images, figure) and torch settings
    common.py          route geometry and eroded maps shared by all steps

    python demo.py                      # all examples, GPU, seed-0 weights
    python demo.py --device cpu --weights weights/flow_seed1.pt
    python demo.py --queries examples/test_hw_2000_250.json   # one query
    python demo.py --list data/test_hw_2000_nf.csv --index 250 777      # rows of a query list
    python demo.py --list data/test_hw_2000_nf.csv --no-plot            # the whole test set

For each query: 16 candidates -> keep those that pass the exact check -> report the
shortest one.  Figures go to out/<query>.png.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from check import check
from common import fine_maps, net_map
from config import CFG, OUT_DIR, init
from model import FlowModel, sample
from plot import draw
from postprocess import lm_endpoint, refine

HERE = Path(__file__).resolve().parent


def plan(model, env, start, goal, n=16, seed=0, maps=None):
    """Plan one query. Returns (all candidate parameters [n,65], fine maps)."""
    device = model.mean.device
    fine = maps or fine_maps(env)
    st = torch.tensor([start], dtype=torch.float32, device=device)
    go = torch.tensor([goal], dtype=torch.float32, device=device)
    image = torch.from_numpy(net_map(env))[None].to(device)
    sdf = torch.from_numpy(fine[1])[None].to(device)
    gen = torch.Generator(device=device).manual_seed(CFG['seed'] + seed)
    # alg:infer for n = 16 candidates in parallel: sampling, then Correct (alg:correct)
    raw = sample(model, st, go, image, sdf, n, gen)
    ss, gg = st.repeat_interleave(n, 0), go.repeat_interleave(n, 0)
    # alg:correct, first step: initial endpoint correction (alg:lm)
    z = lm_endpoint(raw, ss, gg)
    z = refine(z, ss, gg, sdf.repeat_interleave(n, 0))
    return z.detach().cpu().numpy(), fine


def sync(device):
    """Wait for queued GPU work, so that wall-clock timing covers it."""
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def load(paths):
    """Read query files; a directory stands for all its *.json files, in natural order.

    Each file holds one map and one query: {"map": {...}, "query": {"name", "start", "goal"}}.
    """
    files = []
    for path in map(Path, paths):
        files += sorted(path.glob('*.json'), key=lambda f: (len(f.stem), f.stem)) if path.is_dir() else [path]
    return [json.loads(f.read_text()) for f in files]


def load_list(list_file, maps_file, index):
    """Read rows of a query list (CSV); column map_id is the position of the map in maps_file.

    `index` are 0-based data rows of the CSV; empty means all rows.
    """
    envs = json.loads(Path(maps_file).read_text())
    with open(list_file, newline='') as f:
        rows = list(csv.DictReader(f))
    stem = Path(list_file).stem
    return [dict(map=envs[int(rows[i]['map_id'])],
                 query=dict(name=f'{stem}[{i}]',
                            start=[float(rows[i][k]) for k in ('start_x_m', 'start_y_m', 'start_yaw_rad')],
                            goal=[float(rows[i][k]) for k in ('goal_x_m', 'goal_y_m')]))
            for i in (index or range(len(rows)))]


def warm_up(model, data, args):
    """Untimed pass over the first queries (at most 6) before timing.

    CUDA init and the first vmap / cuSOLVER call in the collision correction cost about 1 s
    once; the paper timing also warms up first.  Several queries are used so that every code
    path of the timed pass (including collision correction) has most likely run once.
    """
    t = time.perf_counter()
    data = data[:6]
    for item in data:
        env, qy = item['map'], item['query']
        plan(model, env, qy['start'], qy['goal'],
             args.samples, args.seed, fine_maps(env))
    sync(args.device)
    print(
        f"warm-up: {len(data)} queries, {time.perf_counter() - t:.2f}s, not timed", flush=True)


def main():
    p = argparse.ArgumentParser()
    # default weights for demo
    p.add_argument('--weights', default=str(HERE / 'weights/flow_seed0.pt'))
    # query files or directories of them; default: all bundled examples
    p.add_argument('--queries', nargs='+', default=[str(HERE / 'examples')])
    # alternative input: a query list (CSV) whose map_id column points into --maps
    p.add_argument('--list', default=None)
    p.add_argument('--maps', default=str(HERE / 'data/envs.json'))
    # rows of --list to plan (0-based); default: all rows
    p.add_argument('--index', type=int, nargs='*', default=[])
    # skip the figures, e.g. when planning a whole list
    p.add_argument('--no-plot', action='store_true')
    # number of candidates to sample per query
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)  # random seed for sampling
    # device to run the model on (e.g., 'cuda' or 'cpu')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    init()

    data = load_list(args.list, args.maps, args.index) if args.list else load(args.queries)
    model = FlowModel.load(args.weights, args.device)
    warm_up(model, data, args)

    solved, times = 0, []
    for item in data:
        env, qy = item['map'], item['query']
        start, goal = np.array(qy['start']), np.array(qy['goal'])
        # fine: 1120x1120 (sdf + check)
        fine = fine_maps(env)
        sync(args.device)
        t = time.perf_counter()
        # fine: 1120x1120 (sdf)
        cands, _ = plan(model, env, qy['start'],
                        qy['goal'], args.samples, args.seed, fine)
        sync(args.device)
        plan_s = time.perf_counter() - t
        times.append(plan_s)
        # fine: 1120x1120 (check)
        results = [check(z, start, goal, fine) for z in cands]
        good = [i for i, r in enumerate(results) if r['success']]
        # sec:select: shortest feasible candidate (objective S, eq:P-obj); none -> report failure
        best = min(
            good, key=lambda i: results[i]['length_m']) if good else None
        solved += best is not None
        line = f"{qy['name']:>18}  feasible {len(good):2d}/{len(cands)}  plan {plan_s:5.2f}s"
        if best is not None:
            r = results[best]
            line += f"  length {r['length_m']:.2f} m  endpoint {r['endpoint_m'] * 1e3:.3f} mm  max kappa {r['max_kappa']:.4f}"
        else:
            line += '  no feasible route'
        print(line, flush=True)
        if not args.no_plot:
            draw(OUT_DIR / f"{qy['name'].replace('[', '_').replace(']', '')}.png",
                 env, fine, start, goal, cands, results, best)
    print(
        f'solved {solved}/{len(data)} queries, plan time mean {np.mean(times):.2f}s max {np.max(times):.2f}s')


if __name__ == '__main__':
    main()
