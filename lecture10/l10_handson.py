"""L10 hands-on: from a path to steering, throttle and brake, on one Town03 turn.

  python3 l10_handson.py --part bicycle   # the bicycle model: one steering angle, one circle
  python3 l10_handson.py --part profile   # the speed profile, and two ways to slow down
  python3 l10_handson.py --part pid       # speed control: P, PI, PID, windup (precomputed)
  python3 l10_handson.py --part track     # steering: Pure Pursuit and Stanley (precomputed)
Add --snapshot out.png to save a still, --html out.html for a page (needs mpld3).

The pid and track parts only display runs computed in advance (l10_results/); this
folder has no controller code. You write the controllers in GP4.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

import l10common as c

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, RED, GRAY = "#1f6fb4", "#c9682f", "#2e7d32", "#b03a2e", "0.55"


# ------------------------------------------------------------------ part: bicycle
def part_bicycle(fig, curve, delta0=0.30, v0=3.46):
    """Hold one steering angle: the rear axle drives a circle of radius L / tan(delta).
    Left: 3 s of driving from the origin, and the circle. Right: the radius against the
    steering angle, with the two arcs of our turn marked."""
    ax = fig.add_axes([0.06, 0.22, 0.45, 0.70])
    ax2 = fig.add_axes([0.60, 0.22, 0.36, 0.70])
    s_d = Slider(fig.add_axes([0.15, 0.09, 0.30, 0.03]), "steering (rad)", -c.MAX_STEER,
                 c.MAX_STEER, valinit=delta0)
    s_v = Slider(fig.add_axes([0.15, 0.04, 0.30, 0.03]), "speed (m/s)", 1.0, 10.0, valinit=v0)

    d = np.linspace(0.02, c.MAX_STEER, 200)
    ax2.plot(d, c.turning_radius(d), color=BLUE)
    for k, lab in ((0.1000, "first arc"), (0.1673, "second arc")):
        dk = np.arctan(c.WHEELBASE * k)
        ax2.plot([dk], [1 / k], "o", color=ORANGE)
        ax2.annotate(f"{lab}: R = {1 / k:.2f} m, delta = {dk:.3f} rad", (dk, 1 / k),
                     xytext=(0.30, 30 if k < 0.15 else 24), textcoords="data", fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=GRAY))
    ax2.set_ylim(0, 40); ax2.set_xlabel("steering angle delta (rad)")
    ax2.set_ylabel("turning radius R (m)"); ax2.set_title("R = L / tan(delta), L = 2.875 m")
    ax2.grid(alpha=0.3)
    mark, = ax2.plot([], [], "s", color=GREEN, ms=8)

    def draw(_=None):
        dl, v = s_d.val, s_v.val
        ax.clear()
        xs, ys, ths = c.drive_constant(0.0, 0.0, 0.0, v, dl, 3.0)
        ax.plot(xs, ys, color=GREEN, lw=2, label="rear axle, 3 s")
        if abs(dl) > 1e-3:
            R = c.turning_radius(dl)
            cy = np.sign(dl) * R
            a = np.linspace(0, 2 * np.pi, 300)
            ax.plot(R * np.cos(a), cy + R * np.sin(a), ":", color=GRAY, label=f"circle, R = {R:.2f} m")
            mark.set_data([abs(dl)], [R])
        c.draw_av(ax, 0.0, 0.0, 0.0)
        x1, y1, th1, _ = c.bicycle_step(0.0, 0.0, 0.0, v, dl, 0.0, 0.05)
        ax.set_title(f"one step of 0.05 s: x += {x1:.4f} m, y += {y1:.4f} m, "
                     f"theta += {th1:.4f} rad", fontsize=10)
        ax.set_aspect("equal"); ax.set_xlim(-12, 28); ax.set_ylim(-22, 22)
        ax.set_xlabel("x (m), forward at the start"); ax.set_ylabel("y (m), left at the start")
        ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
        fig.canvas.draw_idle()

    s_d.on_changed(draw); s_v.on_changed(draw)
    draw()
    fig._keep = (s_d, s_v)


# ------------------------------------------------------------------ part: profile
def part_profile(fig, curve):
    """Top: the turn's speed profile and the time to drive it.
    Bottom: slowing from 8 m/s to the turn's speed, at a constant 2 m/s^2 and along a
    quintic that never passes 2 m/s^2: the acceleration, then the jerk."""
    s, k = curve["s"], curve["kappa"]
    v = c.speed_profile(s, k)
    t = c.time_along(s, v)
    ax = fig.add_axes([0.07, 0.58, 0.88, 0.34])
    ax.plot(s, np.minimum(8.0, np.sqrt(2.0 / np.maximum(np.abs(k), 1e-9))), ":", color=GRAY,
            label="limit: 8 m/s and sqrt(2.0 / |kappa|)")
    ax.plot(s, v, color=BLUE, lw=2, label="speed profile (forward and backward passes)")
    ax.axvspan(*curve["s_turn"], color=ORANGE, alpha=0.12, label="the junction lane")
    ax.set_ylim(0, 9); ax.set_xlabel("distance along the path s (m)"); ax.set_ylabel("speed (m/s)")
    ax.set_title(f"the profile drives the {s[-1]:.0f} m in {t[-1]:.1f} s; slowest "
                 f"{v.min():.2f} m/s on the second arc")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)

    sd = c.slow_down(8.0, float(v.min()), 2.0)
    q, cs = sd["quintic"], sd["constant"]
    a1 = fig.add_axes([0.07, 0.08, 0.40, 0.36])
    a2 = fig.add_axes([0.56, 0.08, 0.40, 0.36])
    a1.step(cs["t"], cs["a"], where="post", color=RED, label=f"constant: {cs['T']:.2f} s, {cs['D']:.1f} m")
    a1.plot(q["t"], q["a"], color=GREEN, label=f"quintic: {q['T']:.2f} s, {q['D']:.1f} m")
    a1.set_xlabel("time from the start of braking (s)"); a1.set_ylabel("acceleration (m/s^2)")
    a1.legend(fontsize=9); a1.grid(alpha=0.3); a1.set_title("acceleration")
    a2.plot(cs["t"], cs["jerk"], color=RED, label="constant (spikes at the two jumps)")
    a2.plot(q["t"], q["jerk"], color=GREEN, label=f"quintic: at most {q['peak_jerk']:.2f} m/s^3")
    a2.set_xlabel("time from the start of braking (s)"); a2.set_ylabel("jerk (m/s^3)")
    a2.legend(fontsize=9); a2.grid(alpha=0.3); a2.set_title("jerk, sampled at 20 Hz")


# ------------------------------------------------------------------ part: pid
def part_pid(fig, res):
    """Speed tests: the AV at rest, the target steps to 8 m/s. Pick the gains and the
    clamp; the runs were computed in advance on our speed model (not CARLA's)."""
    g = res["pid_grid"]
    gains, t = g["gains"], g["t"]
    kps, kis, kds = (sorted(set(gains[:, i])) for i in range(3))
    state = {"kp": 0.5, "ki": 0.3, "kd": 0.0, "clamp": False}
    ax_v = fig.add_axes([0.27, 0.55, 0.68, 0.38])
    ax_u = fig.add_axes([0.27, 0.08, 0.68, 0.36])
    r_kp = RadioButtons(fig.add_axes([0.02, 0.70, 0.12, 0.22]), [f"Kp {k:g}" for k in kps], active=kps.index(0.5))
    r_ki = RadioButtons(fig.add_axes([0.02, 0.42, 0.12, 0.25]), [f"Ki {k:g}" for k in kis], active=kis.index(0.3))
    r_kd = RadioButtons(fig.add_axes([0.02, 0.20, 0.12, 0.20]), [f"Kd {k:g}" for k in kds], active=0)
    cb = CheckButtons(fig.add_axes([0.02, 0.06, 0.18, 0.10]), ["clamp the integral at 1 m"], [False])

    def row():
        cl = 1.0 if state["clamp"] else np.nan
        m = (gains[:, 0] == state["kp"]) & (gains[:, 1] == state["ki"]) & (gains[:, 2] == state["kd"])
        m &= np.isclose(gains[:, 3], cl) if state["clamp"] else np.isnan(gains[:, 3])
        return int(np.flatnonzero(m)[0])

    def draw(_=None):
        i = row()
        v, u = g["v"][i], g["u"][i]
        n = np.isfinite(v)
        ax_v.clear(); ax_u.clear()
        ax_v.axhline(8.0, color=GRAY, ls="--", label="target 8 m/s")
        ax_v.plot(t[n], v[n], color=BLUE, lw=2, label="speed")
        ax_v.set_ylim(0, 14); ax_v.set_ylabel("speed (m/s)"); ax_v.grid(alpha=0.3)
        ax_v.set_title(f"peak {np.nanmax(v):.2f} m/s, after 30 s {v[n][-1]:.2f} m/s")
        ax_v.legend(loc="lower right", fontsize=9)
        ax_u.axhline(0.75, color=GRAY, ls=":", label="throttle limit 0.75")
        ax_u.plot(t[n], g["p"][i][n], color=ORANGE, label="P term")
        ax_u.plot(t[n], g["i"][i][n], color=GREEN, label="I term")
        ax_u.plot(t[n], g["d"][i][n], color=RED, label="D term")
        ax_u.plot(t[n], u[n], color="k", lw=1.5, label="command (clipped to -1 to 1)")
        ax_u.set_ylim(-2.5, 4.5); ax_u.set_xlabel("time (s)"); ax_u.set_ylabel("command")
        ax_u.legend(loc="upper right", fontsize=8, ncol=2); ax_u.grid(alpha=0.3)
        fig.canvas.draw_idle()

    def pick(key):
        def f(label):
            state[key] = float(label.split()[1]); draw()
        return f

    r_kp.on_clicked(pick("kp")); r_ki.on_clicked(pick("ki")); r_kd.on_clicked(pick("kd"))

    def on_cb(_):
        state["clamp"] = not state["clamp"]; draw()

    cb.on_clicked(on_cb)
    draw()
    fig._keep = (r_kp, r_ki, r_kd, cb)


# ------------------------------------------------------------------ part: track
NAMES = {"pp": "Pure Pursuit, lookahead {:g} m", "ppv": "Pure Pursuit, lookahead {:g} s x speed (at least 3 m)",
         "st": "Stanley, gain {:g}"}


def _label_text(lab):
    kind, val = lab.split("_")
    return NAMES[kind].format(float(val.lstrip("Ldkv")))


def part_track(fig, curve, res, choice="pp_Ld5"):
    """The turn, driven by a controller from l10_results/track_grid.npz. Every run
    starts 1 m left of the lane center at 8 m/s, with the same speed PID following
    the speed profile. Left: from above. Right: the rear axle's distance from the lane
    center and the steering angle, against the distance along the path."""
    g = res["track_grid"]
    labels = [str(x) for x in g["labels"]]
    ax = fig.add_axes([0.27, 0.08, 0.32, 0.84])
    ae = fig.add_axes([0.71, 0.56, 0.27, 0.36])
    ad = fig.add_axes([0.71, 0.08, 0.27, 0.36])
    radio = RadioButtons(fig.add_axes([0.01, 0.08, 0.21, 0.84]), [_label_text(l) for l in labels],
                         active=labels.index(choice))
    for txt in radio.labels:
        txt.set_fontsize(8)
    hw = 1.75                                                   # half the 3.5 m lane

    def draw(label):
        i = [_label_text(l) for l in labels].index(label)
        n = np.isfinite(g["t"][i])
        x, y, th, s = g["x"][i][n], g["y"][i][n], g["th"][i][n], g["s"][i][n]
        e, dl = g["e_rear"][i][n], g["delta"][i][n]
        ax.clear(); ae.clear(); ad.clear()
        nx, ny = -np.sin(curve["heading"]), np.cos(curve["heading"])
        for side in (-1, 1):
            ax.plot(curve["x"] + side * hw * nx, curve["y"] + side * hw * ny, color="k", lw=1)
        ax.plot(curve["x"], curve["y"], "--", color=GRAY, lw=1, label="lane center (the path)")
        ax.plot(x, y, color=BLUE, lw=1.5, label="rear axle")
        k = int(np.argmin(np.abs(s - 84.0)))
        c.draw_av(ax, x[k], y[k], th[k])
        ax.set_aspect("equal"); ax.set_xlim(55, 100); ax.set_ylim(-52, 8)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="lower left", fontsize=8)
        ax.set_title(label, fontsize=10)
        ae.axhline(0, color=GRAY, lw=0.8)
        ae.axvspan(*curve["s_turn"], color=ORANGE, alpha=0.12)
        ae.plot(s, e, color=BLUE)
        ae.set_ylabel("off center (m), + left"); ae.set_ylim(-3, 3); ae.grid(alpha=0.3)
        ad.axvspan(*curve["s_turn"], color=ORANGE, alpha=0.12)
        ad.plot(s, dl, color=GREEN)
        ad.set_ylabel("steering angle (rad)"); ad.set_xlabel("distance along the path s (m)")
        ad.set_ylim(-0.75, 0.75); ad.grid(alpha=0.3)
        fig.canvas.draw_idle()

    radio.on_clicked(draw)
    draw(_label_text(choice))
    fig._keep = radio


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["bicycle", "profile", "pid", "track"], required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--html")
    ap.add_argument("--run", default="pp_Ld5", help="track: the run to open (e.g. pp_Ld10, st_k1)")
    a = ap.parse_args(argv)
    curve = c.load_curve()
    fig = plt.figure(figsize=(13, 7.5))
    if a.part == "bicycle":
        part_bicycle(fig, curve)
    elif a.part == "profile":
        part_profile(fig, curve)
    elif a.part == "pid":
        part_pid(fig, c.load_results())
    else:
        part_track(fig, curve, c.load_results(), choice=a.run)
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=120, facecolor="white")
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
