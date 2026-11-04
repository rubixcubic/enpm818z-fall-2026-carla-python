"""L9 hands-on: a local planner on one Town03 road, and planners for a parking garage.

  python3 l9_handson.py --part scene    # the road in the map and in the Frenet frame
  python3 l9_handson.py --part sample   # candidate paths, their costs, the cheapest
  python3 l9_handson.py --part drive    # re-planning every 0.5 s while passing the parked car
  python3 l9_handson.py --part rrt      # an RRT growing in the garage
  python3 l9_handson.py --part search   # grid search and hybrid A* results (read only)
  python3 l9_handson.py --part sample --extra   # with your edge_cost from l9_exercise.py
Add --snapshot out.png to save a still, --html out.html for a page (needs mpld3).

numpy and matplotlib only: no CARLA server, no GPU. The road comes from
l9_scene.npz (export_scene.py read it from CARLA's Town03 map). The grid search
and hybrid A* results in l9_results/ were computed in advance; this folder has no
search code (GP4 Task 2 asks you to write A* yourself).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Slider

import l9common as c

HERE = Path(__file__).resolve().parent
COL = {"av": "#1E2939", "car": "#2D6CA2", "best": "#1E9E74", "kept": "#7A828C",
       "coll": "#D9694A", "sharp": "#7B5EA7", "road": "#F2F2F2", "edge": "#7A828C"}


def draw_road(ax, ref, s0, s1, frenet=False):
    """Right edge, lane line and left edge of the two lanes, in the map or in (s, d)."""
    W = ref["lane_width"]
    s = np.arange(s0, s1 + 1e-9, 0.5)
    for d, ls in ((-W / 2, "-"), (W / 2, "--"), (1.5 * W, "-")):
        if frenet:
            ax.plot([s0, s1], [d, d], color=COL["edge"], ls=ls, lw=1.2)
        else:
            ax.plot(*c.from_frenet(ref, s, d), color=COL["edge"], ls=ls, lw=1.2)


def car_box(ax, ref, s, d, color, frenet=False):
    """A 4.5 m by 1.9 m box centered at (s, d), drawn in the map or in (s, d)."""
    L, Wd = c.AV["length"], c.AV["width"]
    if frenet:
        ax.add_patch(Rectangle((s - L / 2, d - Wd / 2), L, Wd, fc=color, ec=color))
        return
    x, y = c.from_frenet(ref, s, d)
    _, _, h, _ = c.ref_at(ref, s)
    loc = np.array([[-L / 2, -Wd / 2], [L / 2, -Wd / 2], [L / 2, Wd / 2], [-L / 2, Wd / 2]])
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    ax.fill(*(loc @ R.T + [x, y]).T, color=color)


def av_box(ax, ref, s, d, frenet=False, color=COL["av"]):
    """The AV's box: its base link is 0.9 m in from the rear bumper, so the box's center
    is 1.35 m ahead of (s, d)."""
    car_box(ax, ref, s + c.AV["length"] / 2 - c.AV["rear_to_base"], d, color, frenet)


# ----------------------------------------------------------------- part 1: scene
def part_scene(fig, ref, car):
    """A point P in the map and in the Frenet frame. Move it with the s and d sliders:
    in the map it follows the curve, in (s, d) it moves along straight lines."""
    ax_m = fig.add_axes([0.06, 0.40, 0.88, 0.52])
    ax_f = fig.add_axes([0.06, 0.13, 0.88, 0.20])
    s_sl = Slider(fig.add_axes([0.15, 0.05, 0.3, 0.03]), "s (m)", 0.0, float(ref["s"][-1]), valinit=140.0)
    d_sl = Slider(fig.add_axes([0.60, 0.05, 0.3, 0.03]), "d (m)", -1.75, 5.25, valinit=3.0)
    s0, s1 = 0.0, float(ref["s"][-1])
    draw_road(ax_m, ref, s0, s1)
    ax_m.plot(ref["x"], ref["y"], ":", color=COL["av"], lw=1, label="reference: the right lane's center")
    av_box(ax_m, ref, c.SCENE["av_s"], 0.0)
    car_box(ax_m, ref, c.SCENE["car_s"], c.SCENE["car_d"], COL["car"])
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("x (m)")
    ax_m.set_ylabel("y (m)")
    draw_road(ax_f, ref, s0, s1, frenet=True)
    av_box(ax_f, ref, c.SCENE["av_s"], 0.0, frenet=True)
    car_box(ax_f, ref, c.SCENE["car_s"], c.SCENE["car_d"], COL["car"], frenet=True)
    ax_f.set_xlabel("s (m)")
    ax_f.set_ylabel("d (m)")
    ax_f.set_ylim(-3, 6.5)
    P_m, = ax_m.plot([], [], "o", color=COL["coll"], ms=8, label="P")
    F_m, = ax_m.plot([], [], "o", mfc="white", mec=COL["av"], ms=7, label="foot point")
    seg, = ax_m.plot([], [], "-", color=COL["coll"], lw=2)
    P_f, = ax_f.plot([], [], "o", color=COL["coll"], ms=8)
    ax_m.legend(loc="upper left", fontsize=9)
    title = ax_m.set_title("")

    def update(_=None):
        s, d = s_sl.val, d_sl.val
        x, y = c.from_frenet(ref, s, d)
        fx, fy = c.from_frenet(ref, s, 0.0)
        sb, db, _, _ = c.to_frenet(ref, float(x), float(y))
        P_m.set_data([x], [y])
        F_m.set_data([fx], [fy])
        seg.set_data([fx, x], [fy, y])
        P_f.set_data([s], [d])
        title.set_text(f"P = ({float(x):.2f}, {float(y):.2f}) m in the map;  back to Frenet: "
                       f"s = {sb:.2f} m, d = {db:.2f} m")
        fig.canvas.draw_idle()
    s_sl.on_changed(update)
    d_sl.on_changed(update)
    update()
    fig._keep = (s_sl, d_sl)


# ----------------------------------------------------------------- part 2: sample
def draw_candidates(ax_f, ax_m, ref, plan_out, s0):
    best = plan_out[0][0]
    for p, sc in plan_out[::-1]:
        if p is best:
            st = dict(color=COL["best"], lw=3, zorder=5)
        elif p.get("stop"):
            st = dict(color=COL["av"], lw=1.6, ls=":", zorder=4)
        elif sc["dropped"] == "collision":
            st = dict(color=COL["coll"], lw=0.9, zorder=2)
        elif sc["dropped"] == "too sharp":
            st = dict(color=COL["sharp"], lw=1.0, ls="--", zorder=2)
        else:
            st = dict(color=COL["kept"], lw=1.2, zorder=3)
        ax_f.plot(p["s"], p["d"], **st)
        ax_m.plot(p["x"], p["y"], **st)


def table_text(plan_out, n=6):
    extra = "extra" in plan_out[0][1]["terms"]
    lines = ["rank  d_end  length  comfort clear  lane  progress" + ("  extra" if extra else "") + "  total"]
    for k, (p, sc) in enumerate(plan_out[:n]):
        t = sc["terms"]
        name = "stop" if p.get("stop") else f"{p['d_end']:5.1f}  {p['maneuver']:4.0f} m"
        lines.append(f"{k + 1:>3}   {name:>13}  {t['comfort']:6.2f} {t['clearance']:5.2f} {t['lane']:5.2f}"
                     f"  {t['progress']:7.2f}" + (f"  {t['extra']:5.2f}" if extra else "")
                     + f"  {sc['total']:6.2f}")
    dropped = sum(sc["dropped"] is not None for _, sc in plan_out)
    lines.append(f"{len(plan_out)} candidates, {dropped} dropped (collision or too sharp)")
    return "\n".join(lines)


def part_sample(fig, ref, car, extra=None):
    """Every candidate from the AV at s = 60 m, scored. The sliders change the weights
    and where the parked car is; the cheapest path is green."""
    ax_f = fig.add_axes([0.06, 0.58, 0.60, 0.36])
    ax_m = fig.add_axes([0.06, 0.25, 0.60, 0.26])
    ax_t = fig.add_axes([0.68, 0.25, 0.31, 0.69])
    ax_t.axis("off")
    names = ("comfort", "clearance", "lane", "progress")
    sls = {}
    for k, n in enumerate(names):
        sls[n] = Slider(fig.add_axes([0.14, 0.17 - 0.035 * k, 0.3, 0.025]), f"w {n}", 0.0, 10.0,
                        valinit=c.WEIGHTS[n])
    car_sl = Slider(fig.add_axes([0.60, 0.17, 0.3, 0.025]), "car s (m)", 80.0, 140.0, valinit=c.SCENE["car_s"])
    s0 = c.SCENE["av_s"]

    def update(_=None):
        for ax in (ax_f, ax_m):
            ax.clear()
        w = dict(c.WEIGHTS)
        for n in names:
            w[n] = sls[n].val
        scene = dict(c.SCENE, car_s=car_sl.val)
        cr = c.parked_car(ref, scene)
        out = c.plan(ref, s0, (0, 0, 0), [cr], weights=w, scene=scene, extra=extra)
        draw_road(ax_f, ref, s0 - 2, s0 + 72, frenet=True)
        draw_road(ax_m, ref, s0 - 2, s0 + 82)
        car_box(ax_f, ref, scene["car_s"], scene["car_d"], COL["car"], frenet=True)
        car_box(ax_m, ref, scene["car_s"], scene["car_d"], COL["car"])
        av_box(ax_m, ref, s0, 0.0)
        draw_candidates(ax_f, ax_m, ref, out, s0)
        ax_f.set_xlabel("s (m)")
        ax_f.set_ylabel("d (m)")
        ax_f.set_title("candidates in the Frenet frame (d axis stretched)")
        ax_m.set_aspect("equal")
        ax_m.set_xlabel("x (m)")
        ax_t.clear()
        ax_t.axis("off")
        ax_t.text(0, 1, table_text(out, n=9), family="monospace", fontsize=7 if extra else 8, va="top")
        fig.canvas.draw_idle()
    for s in list(sls.values()) + [car_sl]:
        s.on_changed(update)
    update()
    fig._keep = (sls, car_sl)


# ----------------------------------------------------------------- part 3: drive
def part_drive(fig, ref, car):
    """The planner runs every 0.5 s; the AV drives the first 0.5 s of the cheapest path
    and plans again. The slider steps through the plans."""
    states, plans = c.drive(ref, [car], duration=14.0)
    ax_m = fig.add_axes([0.06, 0.45, 0.9, 0.48])
    ax_d = fig.add_axes([0.06, 0.16, 0.9, 0.22])
    sl = Slider(fig.add_axes([0.2, 0.05, 0.6, 0.03]), "plan", 0, len(plans) - 1, valinit=0, valstep=1)
    S = np.array(states)

    def update(_=None):
        i = int(sl.val)
        ax_m.clear()
        ax_d.clear()
        draw_road(ax_m, ref, 55, 230)
        car_box(ax_m, ref, c.SCENE["car_s"], c.SCENE["car_d"], COL["car"])
        best = plans[i]["best"]
        ax_m.plot(best["x"], best["y"], color=COL["best"], lw=3)
        xs, ys = c.from_frenet(ref, S[: i + 1, 0], S[: i + 1, 1])
        ax_m.plot(xs, ys, ":", color=COL["av"])
        s0, d0 = states[i]
        av_box(ax_m, ref, s0, d0)
        ax_m.set_aspect("equal")
        ax_m.set_title(f"t = {0.5 * i:.1f} s: the AV at s = {s0:.1f} m, d = {d0:.2f} m; plan ends at "
                       f"d = {best['d_end']:.1f} m over {best['maneuver']:.0f} m")
        draw_road(ax_d, ref, 55, 230, frenet=True)
        car_box(ax_d, ref, c.SCENE["car_s"], c.SCENE["car_d"], COL["car"], frenet=True)
        ax_d.plot(S[:, 0], S[:, 1], "o-", color=COL["kept"], ms=3)
        ax_d.plot(best["s"], best["d"], color=COL["best"], lw=2.5)
        ax_d.plot([s0], [d0], "o", color=COL["av"])
        ax_d.set_xlabel("s (m)")
        ax_d.set_ylabel("d (m)")
        fig.canvas.draw_idle()
    sl.on_changed(update)
    update()
    fig._keep = sl


# ----------------------------------------------------------------- garage
def draw_garage(ax):
    Wg, Hg = c.GARAGE["size"]
    ax.add_patch(Rectangle((0, 0), Wg, Hg, fc="white", ec=COL["av"], lw=1.5))
    top, bottom = c.garage_spots()
    for x0, y0, x1, y1 in top + bottom:
        ax.plot([x0, x0, x1, x1], [y0, y1, y1, y0], color=COL["edge"], lw=0.6)
    for x0, y0, x1, y1 in c.garage_obstacles():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=COL["car"], alpha=0.7))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, Wg + 0.5)
    ax.set_ylim(-0.5, Hg + 0.5)


def part_rrt(fig, ref, car, seed=4):
    """An RRT from the AV's start to the spot's entrance. The slider shows the tree after
    each number of added nodes; the path appears when the tree reaches the goal."""
    goal = np.array([c.garage_goal()[0], 9.0])
    res = c.rrt(c.GARAGE["start"], goal, c.garage_obstacles(), goal_bias=0.05, seed=seed)
    nodes, par = res["nodes"], res["parents"]
    ax = fig.add_axes([0.06, 0.15, 0.9, 0.78])
    sl = Slider(fig.add_axes([0.2, 0.05, 0.6, 0.03]), "nodes", 1, len(nodes), valinit=len(nodes), valstep=1)

    def update(_=None):
        n = int(sl.val)
        ax.clear()
        draw_garage(ax)
        for i in range(1, n):
            p = nodes[par[i]]
            ax.plot([p[0], nodes[i][0]], [p[1], nodes[i][1]], color=COL["kept"], lw=1)
        ax.plot(nodes[:n, 0], nodes[:n, 1], "o", color=COL["kept"], ms=3)
        ax.plot(*goal, "*", color=COL["coll"], ms=14)
        if n == len(nodes) and res["path"] is not None:
            ax.plot(res["path"][:, 0], res["path"][:, 1], color=COL["best"], lw=3)
        ax.set_title(f"seed {seed}: {n} of {len(nodes)} nodes; goal reached at iteration {res['iterations']}")
        fig.canvas.draw_idle()
    sl.on_changed(update)
    update()
    fig._keep = sl


def load_search(path=HERE / "l9_results" / "garage_search.json"):
    if not Path(path).exists():
        raise FileNotFoundError(f"missing results file: {path}")
    d = json.loads(Path(path).read_text())
    for k in ("toy", "grid", "hybrid"):
        if k not in d:
            raise KeyError(f"{path} has no '{k}'")
    return d


def part_search(fig, ref, car):
    """Results only: the toy grid's step numbers, the garage grid path, and hybrid A*'s
    path with the poses it expanded (forward solid, reverse dashed)."""
    d = load_search()
    T = d["toy"]
    steps = np.array(T["steps"])
    free = np.array(T["free"], bool)
    ax0 = fig.add_axes([0.03, 0.55, 0.28, 0.4])
    ax1 = fig.add_axes([0.36, 0.55, 0.6, 0.4])
    ax2 = fig.add_axes([0.36, 0.06, 0.6, 0.4])
    nr, nc = free.shape
    for i in range(nr):
        for j in range(nc):
            ax0.add_patch(Rectangle((j, nr - 1 - i), 1, 1, fc="white" if free[i, j] else COL["car"], ec=COL["edge"]))
            if steps[i, j] >= 0:
                ax0.text(j + 0.5, nr - 0.5 - i, str(steps[i, j]), ha="center", va="center")
    ax0.set_xlim(0, nc)
    ax0.set_ylim(0, nr)
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.set_title("toy grid: steps from the start")
    draw_garage(ax1)
    P = np.array(d["grid"]["path_xy"])
    ax1.plot(P[:, 0], P[:, 1], color=COL["best"], lw=2.5)
    ax1.set_title(f"grid search: {d['grid']['n_expanded']} cells expanded, {d['grid']['length']:.1f} m")
    draw_garage(ax2)
    H = d["hybrid"]
    E = np.array(H["expanded"])
    Q = np.array(H["poses"])
    D = np.array(H["dirs"])
    ax2.plot(E[:, 0], E[:, 1], ".", color="#C9CDD2", ms=1)
    k = int(np.argmax(D < 0)) - 1
    ax2.plot(Q[: k + 1, 0], Q[: k + 1, 1], color=COL["best"], lw=2.5)
    ax2.plot(Q[k:, 0], Q[k:, 1], "--", color=COL["best"], lw=2.5)
    ax2.set_title(f"hybrid A*: {H['n_expanded']} poses expanded, {H['length']:.1f} m, "
                  f"{H['switches']} switch to reverse")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["scene", "sample", "drive", "rrt", "search"], required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--html")
    ap.add_argument("--seed", type=int, default=4, help="rrt: random seed (default 4, as on the slides)")
    ap.add_argument("--extra", action="store_true",
                    help="sample: add edge_cost from l9_exercise.py as a fifth cost term")
    a = ap.parse_args(argv)
    ref = c.load_scene()
    car = c.parked_car(ref)
    fig = plt.figure(figsize=(13, 8))
    if a.part == "rrt":
        part_rrt(fig, ref, car, seed=a.seed)
    elif a.part == "sample" and a.extra:
        import l9_exercise
        part_sample(fig, ref, car, extra=l9_exercise.edge_cost)
    else:
        {"scene": part_scene, "sample": part_sample, "drive": part_drive, "search": part_search}[a.part](fig, ref, car)
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=110, facecolor="white")
    elif a.html:
        try:
            import mpld3
        except ImportError:
            sys.exit("--html needs mpld3 (pip install mpld3); or use --snapshot")
        Path(a.html).write_text(mpld3.fig_to_html(fig))
    else:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
