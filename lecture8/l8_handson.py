"""L8 hands-on: where will the others be? Four parts on one Town01 junction.

  python3 l8_handson.py --part scene      # the junction, the AV, car A and car B over time
  python3 l8_handson.py --part physics    # CV, CTRV and CTRA for car A, from any "now"
  python3 l8_handson.py --part modes      # map-aware prediction: two futures, their probabilities
  python3 l8_handson.py --part interact   # car B behind A (IDM), and A's gap to the AV

Drag the "now" slider to predict from another moment. Add --now T to open at T seconds,
--snapshot out.png to save a still instead of opening a window, or --html out.html to
save an animation over "now". Needs numpy and matplotlib only (no CARLA, no ROS).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

import l8sim as m
import vehicle_icons as vi

HERE = Path(__file__).resolve().parent
CV, CTRV, CTRA, MAP, TRUTH = "#2D6CA2", "#7B4FA0", "#9A6A00", "#B0478A", "#1E2939"
COLORS = {"AV": "#1E2939", "A": "#2E9E4A", "B": "#747474"}


# ----------------------------------------------------------------- drawing
def draw_road(ax, sc):
    for e in sc["edges"]:
        poly = np.r_[e[:, :2], e[::-1, 2:]]
        ax.fill(poly[:, 0], poly[:, 1], fc="#E4E6E9", ec="#E4E6E9", lw=1.0, zorder=0)
    for e, junc in zip(sc["edges"], sc["edge_in_junction"]):
        if not junc:
            ax.plot(e[:, 0], e[:, 1], color="white", lw=1.2, ls=(0, (4, 4)), zorder=1)
            ax.plot(e[:, 2], e[:, 3], color="#9AA1A9", lw=0.8, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")


class Car:
    """A vehicle icon (icons/: the AV, car A, car B), sharp at any zoom, with its name above."""
    def __init__(self, ax, name):
        self.icon = vi.Vehicle(ax, name, color=COLORS[name], zorder=8)
        self.label = ax.text(0, 0, name, fontsize=10, ha="center", va="bottom", zorder=9)

    def set(self, xy, heading):
        self.icon.set(xy, heading)
        self.label.set_position((xy[0], xy[1] + 2.2))


def state_at(o, name, now):
    k = int(np.argmin(abs(o["t"] - now)))
    return o[name]["xy"][k], o[name]["heading"][k], k


def fit_at(o, name, now):
    k = int(np.argmin(abs(o["t"] - now)))
    return m.fit_state(o["t"][:k + 1], o[name]["track"][:k + 1], now=o["t"][k]), k


def future(o, name, k, n):
    return o[name]["xy"][k + 1:k + 1 + n]


# ----------------------------------------------------------------- the parts
def part_scene(fig, sc, o, now):
    ax = fig.add_subplot(111)
    draw_road(ax, sc)
    ax.set_xlim(98, 214)
    ax.set_ylim(-26, 12)
    cp = m.conflict_point(sc)["xy"]
    ax.plot(*cp, "x", color="#C0392B", ms=10, mew=2.5, zorder=7)
    ax.annotate("crossing", cp, xytext=(10, -14), textcoords="offset points", color="#C0392B")
    cars = {n: Car(ax, n) for n in ("AV", "A", "B")}
    trails = {n: ax.plot([], [], ".", color=COLORS[n], ms=3, zorder=6)[0] for n in ("A", "B")}
    title = ax.set_title("")

    def update(now):
        for n, car in cars.items():
            xy, hd, k = state_at(o, n, now)
            car.set(xy, hd)
            if n in trails:
                past = o["t"] <= now + 1e-9
                trails[n].set_data(o[n]["track"][past, 0], o[n]["track"][past, 1])
        title.set_text(f"t = {now:+.1f} s   (dots: the tracked positions the AV sees)")
    return update


def part_physics(fig, sc, o, now):
    ax = fig.add_subplot(121)
    draw_road(ax, sc)
    ax.set_xlim(132, 190)
    ax.set_ylim(-30, 10)
    ax2 = fig.add_subplot(122)
    h = m.horizons()
    lines = {n: ax.plot([], [], color=c, lw=2, ls=ls, label=n)[0]
             for n, c, ls in (("truth", TRUTH, "-"), ("CV", CV, "-"), ("CTRV", CTRV, "--"), ("CTRA", CTRA, "-."))}
    errs = {n: ax2.plot([], [], color=c, lw=2, ls=ls, label=n)[0]
            for n, c, ls in (("CV", CV, "-"), ("CTRV", CTRV, "--"), ("CTRA", CTRA, "-."))}
    car = Car(ax, "A")
    ax.legend(loc="lower right", fontsize=9)
    ax2.set_xlabel("horizon (s)")
    ax2.set_ylabel("error (m)")
    ax2.set_xlim(0, m.HORIZON)
    ax2.axhline(m.MISS, color="0.5", lw=0.8, ls=":")
    ax2.legend(loc="upper left", fontsize=9)
    title = ax.set_title("")

    def update(now):
        st, k = fit_at(o, "A", now)
        tr = future(o, "A", k, len(h))
        hh = h[:len(tr)]
        car.set(o["A"]["xy"][k], o["A"]["heading"][k])
        lines["truth"].set_data(tr[:, 0], tr[:, 1])
        preds = {"CV": m.predict_cv(st, hh), "CTRV": m.predict_ctrv(st, hh), "CTRA": m.predict_ctra(st, hh)}
        top = 1.0
        for n, p in preds.items():
            lines[n].set_data(p[:, 0], p[:, 1])
            e = m.errors_by_horizon(p, tr)
            errs[n].set_data(hh, e)
            top = max(top, e.max())
        ax2.set_ylim(0, top * 1.1)
        ax2.set_title("FDE at 5 s: " + ", ".join(f"{n} {m.fde(p, tr):.1f} m" for n, p in preds.items()), fontsize=10)
        title.set_text(f"car A from t = {now:+.1f} s: speed {st['speed']:.2f} m/s,\n"
                       f"yaw rate {st['omega']:+.3f} rad/s, speeding up by {st['a_along']:+.2f} m/s²")
        title.set_fontsize(10)
    return update


def part_modes(fig, sc, o, now):
    ax = fig.add_subplot(121)
    draw_road(ax, sc)
    ax.set_xlim(132, 190)
    ax.set_ylim(-30, 10)
    ax2 = fig.add_subplot(122)
    h = m.horizons()
    ts = np.round(np.arange(-1.5, 3.01, 0.1), 2)
    ps = []
    for tn in ts:
        st, _ = fit_at(o, "A", tn)
        ps.append(sum(md["prob"] for md in m.predict_modes(sc, st) if md["name"] == "west_to_south"))
    ax2.plot(ts, ps, color=MAP, lw=2)
    mark = ax2.axvline(now, color="0.4", ls=":")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("P(left turn)")
    truth, = ax.plot([], [], color=TRUTH, lw=2, label="truth", zorder=7)
    mode_lines = [ax.plot([], [], color=MAP, lw=2, ls=ls)[0] for ls in ("-", "--", ":")]
    car = Car(ax, "A")
    title = ax.set_title("")
    ax.legend(loc="lower right", fontsize=9)

    def update(now):
        st, k = fit_at(o, "A", now)
        tr = future(o, "A", k, len(h))
        hh = h[:len(tr)]
        car.set(o["A"]["xy"][k], o["A"]["heading"][k])
        truth.set_data(tr[:, 0], tr[:, 1])
        modes = m.predict_modes(sc, st, hh)
        text = []
        for i, line in enumerate(mode_lines):
            if i < len(modes):
                p = modes[i]["traj"]
                line.set_data(p[:, 0], p[:, 1])
                line.set_linewidth(1 + 4 * modes[i]["prob"])
                text.append(f"{m.MOVES[modes[i]['name']]}: p = {modes[i]['prob']:.2f}, FDE {m.fde(p, tr):.1f} m")
            else:
                line.set_data([], [])
        mark.set_xdata([now, now])
        title.set_text(f"t = {now:+.1f} s\n" + "\n".join(text))
        title.set_fontsize(9)
    return update


def part_interact(fig, sc, o, now, o2=None):
    from matplotlib.widgets import CheckButtons
    runs = {"run #1": o, "run #2": o2 if o2 is not None else m.our_scene(sc, av_closer=24.0)}
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    h = m.horizons()
    lav = sc["lanes"]["east_to_east"]
    la = sc["lanes"]["west_to_south"]
    state = {"run": "run #1"}
    check_ax = fig.add_axes([0.01, 0.02, 0.16, 0.08])
    check = CheckButtons(check_ax, ["AV 24 m closer"], [False])
    fig._l8_check = check

    def update(now):
        oo = runs[state["run"]]
        ax.clear()
        ax2.clear()
        stA, k = fit_at(oo, "A", now)
        stB, _ = fit_at(oo, "B", now)
        trA, trB = future(oo, "A", k, len(h)), future(oo, "B", k, len(h))
        hh = h[:len(trA)]
        av_s = lav.project(oo["AV"]["xy"][k])[0]
        plain = m.predict_modes(sc, stA, hh)
        gap = m.predict_modes_with_av(sc, stA, av_s, m.V_CRUISE, hh)
        s0 = la.project(stA["p0"])[0]
        along = lambda xy: np.array([la.project(p)[0] for p in xy]) - s0
        ax.plot(hh, along(trA), color=TRUTH, lw=2, label="truth")
        for md, ls, lab in ((plain[0], "--", "most likely mode, AV ignored"),
                            (gap[0], "-", "most likely mode, with the gap rule")):
            ax.plot(hh, along(md["traj"]), color=MAP, lw=2, ls=ls, label=lab)
        g = next((md.get("gap") for md in gap if "gap" in md), None)
        ax.set_title(f"{state['run']}, car A from t = {now:+.1f} s" +
                     (f": gap to the AV {g:.1f} s (A turns if it is at least {m.GAP} s)" if g is not None else ""),
                     fontsize=10)
        ax.set_xlabel("horizon (s)")
        ax.set_ylabel("car A: meters driven")
        ax.legend(fontsize=9, loc="upper left")
        pB, vB = m.predict_follower(sc, "west_to_west", stB, gap[0]["traj"], hh)
        ax2.plot(hh, oo["B"]["v"][k + 1:k + 1 + len(hh)], color=TRUTH, lw=2, label="truth")
        ax2.plot(hh, np.full_like(hh, stB["speed"]), color=CV, lw=2, label="CV")
        ax2.plot(hh, vB, color=MAP, lw=2, ls="--", label="IDM behind A's likely future")
        ax2.set_title(f"car B: FDE CV {m.fde(m.predict_cv(stB, hh), trB):.1f} m, IDM {m.fde(pB, trB):.1f} m",
                      fontsize=10)
        ax2.set_xlabel("horizon (s)")
        ax2.set_ylabel("car B's speed (m/s)")
        ax2.legend(fontsize=9, loc="lower left")
        fig.canvas.draw_idle()

    def toggle(_label):
        state["run"] = "run #2" if check.get_status()[0] else "run #1"
        update(fig._l8_now())
    check.on_clicked(toggle)
    return update


PARTS = {"scene": part_scene, "physics": part_physics, "modes": part_modes, "interact": part_interact}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=PARTS, required=True)
    ap.add_argument("--now", type=float, default=0.0, help="the moment to predict from (s), -0.5 to 3")
    ap.add_argument("--snapshot", help="save a still to this PNG instead of opening a window")
    ap.add_argument("--html", help="save an animation over 'now' to this HTML file")
    a = ap.parse_args(argv)
    if a.snapshot or a.html:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    sc = m.load_scene()
    o = m.our_scene(sc)
    fig = plt.figure(figsize=(13, 6.5))
    update = PARTS[a.part](fig, sc, o, a.now)
    fig.subplots_adjust(bottom=0.2, top=0.86, wspace=0.25)
    s_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    lo = -0.5 if a.part != "scene" else -2.0
    slider = Slider(s_ax, "now (s)", lo, 3.0, valinit=float(np.clip(a.now, lo, 3.0)), valstep=0.1)
    fig._l8_now = lambda: slider.val
    slider.on_changed(update)
    update(slider.val)
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=110)
        print("wrote", a.snapshot)
    elif a.html:
        from matplotlib.animation import FuncAnimation
        frames = np.round(np.arange(lo, 3.01, 0.2), 2)
        anim = FuncAnimation(fig, lambda f: update(f), frames=frames, interval=200)
        Path(a.html).write_text(anim.to_jshtml())
        print("wrote", a.html)
    else:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
