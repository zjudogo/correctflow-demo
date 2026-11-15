"""Step 5, figure: map, all candidates and the selected route with its body outline."""
import numpy as np

from check import route
from config import CFG, HALF


def draw(path, env, fine, start, goal, cands, results, best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(CFG['fig_size'], CFG['fig_size']))
    c0, res = CFG['net_cells'] // 2, CFG['src_res']  # NIRRT pixel (c0, c0) is the map centre
    for x, y, w, h in env['rectangle_obstacles']:
        ax.add_patch(Rectangle(((x - c0) * res, (c0 - y - h) * res), w * res, h * res, color='0.85', lw=0))
    for x, y, r in env['circle_obstacles']:
        ax.add_patch(Circle(((x - c0) * res, (c0 - y) * res), r * res, color='0.85', lw=0))
    ax.imshow(np.ma.masked_where(~fine[0], fine[0]), cmap='Greys', vmin=0, vmax=1.2,
              extent=[-HALF, HALF, -HALF, HALF], origin='upper', interpolation='nearest', zorder=2)
    for z, res in zip(cands, results):
        q, _, _ = route(z, start, goal, CFG['plot_step'])
        ax.plot(q[:, 0], q[:, 1], color='tab:green' if res['success'] else 'tab:red', lw=.6, alpha=.6)
    if best is not None:
        q, _, _ = route(cands[best], start, goal, CFG['check_step'])
        ax.plot(q[:, 0], q[:, 1], color='tab:blue', lw=2)
        hl, hw, off = CFG['body_length'] / 2, CFG['body_width'] / 2, CFG['body_offset']
        for x, y, a in q[::CFG['body_every']]:
            c, s = np.cos(a), np.sin(a)
            pts = [(off + u * hl, v * hw) for u, v in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            ax.add_patch(Polygon([(x + c * u - s * v, y + s * u + c * v) for u, v in pts],
                                 fill=False, ec='tab:blue', lw=.6))
    ax.plot(*start[:2], 'ko')
    ax.arrow(start[0], start[1], .4 * np.cos(start[2]), .4 * np.sin(start[2]), width=.03, color='k')
    ax.plot(*goal, 'k*', ms=12)
    ax.set(xlim=(-HALF, HALF), ylim=(-HALF, HALF), aspect='equal', xlabel='x [m]', ylabel='y [m]')
    ax.set_title('light: original obstacles, dark: eroded by 0.30 m\n'
                 'green/red: feasible/infeasible candidates, blue: selected', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=CFG['fig_dpi'])
    plt.close(fig)
