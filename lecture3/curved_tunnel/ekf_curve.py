# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""An extended Kalman filter for a car in a curved tunnel, with a live window.

The tunnel bends 120 degrees to the left. Turning puts a cosine in the motion,
so no matrix F can describe it. The camera recognizes each emergency exit sign
on the wall and matches it against the HD map, which knows where every sign is.
It measures the sign's RANGE (how far) and BEARING (at what angle to the car's
nose): a square root and an arctangent, so no matrix H describes it either.

The EKF uses the real functions f and h for the mean, and their tangents, the
Jacobians F_k and H_k, for everything that touches the covariance:

    predict:  x^- = f(x, u)              P^- = F_k P F_k^T + Q_k
    update:   nu = z - h(x^-)            S = H_k P^- H_k^T + R
              K = P^- H_k^T S^-1         x = x^- + K nu     P = (I - K H_k) P^-

State x = [px, py, theta] (east, north in m; heading in rad, 0 = east).
Control u = [v, omega]: wheel speed (m/s) and gyro turn rate (rad/s).
Measurement z = [range (m), bearing (rad)] to one matched exit sign.

Usage
    python3 ekf_curve.py --make-csv               # write curve_drive.csv
    python3 ekf_curve.py curve_drive.csv          # run the filter, live window
    python3 ekf_curve.py curve_drive.csv --html ekf_curve.html   # web page instead
    python3 ekf_curve.py curve_drive.csv --snapshot frame.png    # one still image
    python3 ekf_curve.py --check-jacobian         # compare F_k, H_k with numerical ones

Needs numpy and matplotlib only. ukf_curve.py reuses this file's model and data.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.1                      # s between wheel/gyro readings, one predict step each
SPEED = 10.0                  # m/s
RADIUS = 60.0                 # m, the bend
BEND = np.radians(120.0)      # how far the tunnel turns
LEG_IN, LEG_OUT = 20.0, 70.0  # m of straight tunnel before and after the bend
SIGN_SPACING = 25.0           # m between exit signs, measured along the tunnel
SIGN_OFFSET = 4.0             # m: signs hang on the left wall, 4 m from the lane center
MATCH_RANGE = 30.0            # m: the camera matches a sign once it is this close
SIGMA_V = 0.2                 # m/s, wheel speed noise
SIGMA_W = 0.02                # rad/s (about 1.1 deg/s), gyro noise in the data
SIGMA_W_FILTER = np.radians(2.0)  # rad/s: what the filter assumes. Bigger than the
                              # real noise on purpose: Q must also cover the gyro
                              # bias, which the filter does not model.
GYRO_BIAS = 0.003             # rad/s, a small bias the filter does not know about
SIGMA_RANGE = 1.0             # m, camera range to a matched sign (same 1 m as the KF)
SIGMA_BEARING = np.radians(2.0)   # rad, camera bearing to a matched sign
HEADING0_GUESS = np.radians(4.0)  # the heading handed over at the entrance is 4 deg off


# ---------------------------------------------------------------------------
# 1. The tunnel, the HD map, and the dataset
# ---------------------------------------------------------------------------
def tunnel_length():
    return LEG_IN + BEND * RADIUS + LEG_OUT


def true_turn_rate(t):
    """The tunnel's turn rate (rad/s) at time t: v/R in the bend, 0 on the straights."""
    t = np.asarray(t, float)
    t_in, t_out = LEG_IN / SPEED, (LEG_IN + BEND * RADIUS) / SPEED
    return np.where((t >= t_in) & (t < t_out), SPEED / RADIUS, 0.0)


def centerline(s):
    """Lane-center position and direction at distance s (m) along the tunnel.

    The lane is laid out by driving the slides' own motion rule f at a constant
    speed and the tunnel's turn rate, so the filter's model matches the road
    exactly and only the sensors' noise and the gyro bias are left to estimate.
    """
    s = np.atleast_1d(np.asarray(s, float))
    m = int(np.ceil((tunnel_length() + 60) / SPEED / DT)) + 1
    pts = np.zeros((m, 3))
    for k in range(1, m):
        pts[k] = f(pts[k - 1], (SPEED, float(true_turn_rate((k - 1) * DT))))
    idx = np.clip(np.round(s / SPEED / DT).astype(int), 0, m - 1)
    pts[:, 2] = np.unwrap(pts[:, 2])
    return pts[idx, 0], pts[idx, 1], pts[idx, 2]


def hd_map():
    """The HD map's exit signs: (N, 2) positions, on the left wall."""
    s = np.arange(SIGN_SPACING, tunnel_length() + 30, SIGN_SPACING)
    x, y, th = centerline(s)
    return np.column_stack([x - SIGN_OFFSET * np.sin(th), y + SIGN_OFFSET * np.cos(th)])


def h(x, sign):
    """Measurement model: the range and bearing the camera should report.

    Works on one state (3,) or many (N, 3): ukf_curve.py pushes sigma points
    through it. Bearing is measured from the car's nose, positive to the left.
    """
    x = np.asarray(x, float)
    dx, dy = sign[0] - x[..., 0], sign[1] - x[..., 1]
    return np.stack([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[..., 2])], axis=-1)


def make_csv(path, seed=818):
    rng = np.random.default_rng(seed)
    n = int(round((tunnel_length() - 5) / SPEED / DT)) + 1
    t = np.arange(n) * DT
    x, y, th = centerline(SPEED * t)
    omega = true_turn_rate(t)
    wheel_v = SPEED + rng.normal(0, SIGMA_V, n)
    gyro_w = omega + GYRO_BIAS + rng.normal(0, SIGMA_W, n)

    # a sign is matched once, on the first frame it is within MATCH_RANGE ahead
    signs = hd_map()
    matched = set()
    sid = np.full(n, -1)
    rng_m, brg_m = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(n):
        truth = np.array([x[k], y[k], th[k]])
        for j, sg in enumerate(signs):
            if j in matched:
                continue
            r, b = h(truth, sg)
            if r <= MATCH_RANGE and abs(b) < np.radians(60):
                matched.add(j)
                sid[k] = j
                rng_m[k] = r + rng.normal(0, SIGMA_RANGE)
                brg_m[k] = b + rng.normal(0, SIGMA_BEARING)
                break
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "true_x", "true_y", "true_theta", "wheel_v", "gyro_w",
                    "sign_id", "sign_range", "sign_bearing"])
        for k in range(n):
            has = sid[k] >= 0
            w.writerow([f"{t[k]:.1f}", f"{x[k]:.4f}", f"{y[k]:.4f}",
                        f"{wrap(th[k]):.5f}", f"{wheel_v[k]:.4f}", f"{gyro_w[k]:.5f}",
                        sid[k] if has else "",
                        f"{rng_m[k]:.3f}" if has else "",
                        f"{brg_m[k]:.5f}" if has else ""])
    print(f"wrote {path}: {n} rows, {int((sid >= 0).sum())} sign matches")


def load_csv(path):
    raw = np.genfromtxt(path, delimiter=",", names=True)
    return {k: raw[k] for k in raw.dtype.names}


def wrap(a):
    """Wrap an angle, or an array of them, into (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------------------
# 2. The motion model and the two Jacobians
# ---------------------------------------------------------------------------
def f(x, u):
    """Motion model: drive v*dt along the current heading, then turn by omega*dt.

    Works on one state (3,) or many (N, 3) at once.
    """
    v, w = u
    x = np.asarray(x, float)
    out = x.copy()
    out[..., 0] = x[..., 0] + v * DT * np.cos(x[..., 2])
    out[..., 1] = x[..., 1] + v * DT * np.sin(x[..., 2])
    out[..., 2] = wrap(x[..., 2] + w * DT)
    return out


def jacobian_f(x, u, wrong_sign=False):
    """F_k = df/dx at the current estimate: the table of slopes.

    Row 1, column 3: nudge the heading and x moves by -v dt sin(theta) times the
    nudge. wrong_sign flips that one entry, the classic hand-derivation bug.
    """
    v, _ = u
    s, c = np.sin(x[2]), np.cos(x[2])
    F = np.array([[1.0, 0.0, -v * DT * s],
                  [0.0, 1.0,  v * DT * c],
                  [0.0, 0.0,  1.0]])
    if wrong_sign:
        F[0, 2] = -F[0, 2]
    return F


def jacobian_h(x, sign):
    """H_k = dh/dx at the predicted state, for one sign.

    Range row:   moving the car toward the sign shortens the range.
    Bearing row: moving the car sideways swings the sign across the image, and
                 turning the car's nose left by one radian moves the sign one
                 radian to the right: the -1 in the last column.
    """
    dx, dy = sign[0] - x[0], sign[1] - x[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / r2, -dx / r2, -1.0]])


def process_noise(x, u, sigma_v, sigma_w):
    """Q_k: the wheel and gyro noise pushed through G = df/du, rebuilt every step."""
    s, c = np.sin(x[2]), np.cos(x[2])
    G = np.array([[DT * c, 0.0],
                  [DT * s, 0.0],
                  [0.0,    DT]])
    return G @ np.diag([sigma_v**2, sigma_w**2]) @ G.T


def initial_belief(d, sigma_theta0=np.radians(5.0)):
    """Where the filter starts: the belief handed over at the tunnel entrance."""
    x = np.array([d["true_x"][0] + 0.5, d["true_y"][0] - 0.3,
                  d["true_theta"][0] + HEADING0_GUESS])
    P = np.diag([1.0**2, 1.0**2, sigma_theta0**2])
    return x, P


def match_rows(d, dark_until=0.0):
    """Rows with a sign match, ignoring any in an unlit stretch before dark_until (s)."""
    return (d["sign_id"] >= 0) & (d["t"] >= dark_until)


# ---------------------------------------------------------------------------
# 3. The filter
# ---------------------------------------------------------------------------
def run_ekf(d, sigma_w=SIGMA_W_FILTER, sigma_range=SIGMA_RANGE, wrong_sign=False,
            sigma_theta0=np.radians(5.0), dark_until=0.0):
    """Run the EKF over the whole drive. Returns every step's x and P."""
    signs = hd_map()
    x, P = initial_belief(d, sigma_theta0)
    R = np.diag([sigma_range**2, SIGMA_BEARING**2])
    matches = match_rows(d, dark_until)
    n = len(d["t"])
    xs, Ps = np.zeros((n, 3)), np.zeros((n, 3, 3))
    for k in range(n):
        if k > 0:
            # PREDICT. The control from row k-1 acts during [t[k-1], t[k]).
            u = (d["wheel_v"][k - 1], d["gyro_w"][k - 1])
            # F_k and Q_k are evaluated at the estimate BEFORE moving it:
            # that is where the tangent is drawn.
            F = jacobian_f(x, u, wrong_sign)
            Q = process_noise(x, u, SIGMA_V, sigma_w)
            x = f(x, u)                       # the mean goes through the real f
            P = F @ P @ F.T + Q               # the covariance goes through the tangent
        if matches[k]:
            # UPDATE with one matched sign. The HD map says where the sign is.
            sign = signs[int(d["sign_id"][k])]
            z = np.array([d["sign_range"][k], d["sign_bearing"][k]])
            Hk = jacobian_h(x, sign)          # the tangent of h, at the prediction
            nu = z - h(x, sign)               # the surprise goes through the real h
            nu[1] = wrap(nu[1])               # angles wrap: place one
            S = Hk @ P @ Hk.T + R
            K = P @ Hk.T @ np.linalg.inv(S)   # 3x2
            x = x + K @ nu
            x[2] = wrap(x[2])                 # angles wrap: place two
            P = (np.eye(3) - K @ Hk) @ P
        xs[k], Ps[k] = x, P
    return xs, Ps


def check_jacobian(eps=1e-6):
    """Nudge each input, see how each output moves, compare with the formulas."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        x = rng.normal(0, 5, 3); x[2] = rng.uniform(-np.pi, np.pi)
        u = (rng.uniform(0, 20), rng.uniform(-1, 1))
        sign = rng.normal(0, 20, 2)
        nf, nh = np.zeros((3, 3)), np.zeros((2, 3))
        for j in range(3):
            dx = np.zeros(3); dx[j] = eps
            df = f(x + dx, u) - f(x - dx, u); df[2] = wrap(df[2])
            dh = h(x + dx, sign) - h(x - dx, sign); dh[1] = wrap(dh[1])
            nf[:, j], nh[:, j] = df / (2 * eps), dh / (2 * eps)
        worst = max(worst, np.abs(nf - jacobian_f(x, u)).max(),
                    np.abs(nh - jacobian_h(x, sign)).max())
    verdict = "OK" if worst < 1e-6 else "WRONG, fix the Jacobians"
    print(f"Jacobian check (F_k and H_k): largest difference {worst:.2e}  ->  {verdict}")


def score(d, xs, Ps):
    """Position RMSE (m), heading RMSE (deg), and how often the true heading
    error falls inside the +/- 1 sigma the filter reports (honest: about 68%)."""
    pos = np.hypot(xs[:, 0] - d["true_x"], xs[:, 1] - d["true_y"])
    he = wrap(xs[:, 2] - d["true_theta"])
    return (np.sqrt(np.mean(pos**2)), np.degrees(np.sqrt(np.mean(he**2))),
            np.mean(np.abs(he) <= np.sqrt(Ps[:, 2, 2])))


# ---------------------------------------------------------------------------
# 4. The live window
# ---------------------------------------------------------------------------
def ellipse_xy(P2, n_sigma=1.0):
    w, V = np.linalg.eigh(P2)
    a = np.linspace(0, 2 * np.pi, 60)
    circle = np.stack([np.cos(a), np.sin(a)])
    return (V @ np.diag(n_sigma * np.sqrt(np.maximum(w, 0))) @ circle).T


def draw_tunnel(ax):
    """The tunnel from above: road, walls, lane line, and the HD map's signs."""
    s = np.linspace(-15, tunnel_length() + 15, 900)
    x, y, th = centerline(s)
    nx, ny = -np.sin(th), np.cos(th)
    for off in (SIGN_OFFSET + 0.6, -3.0):
        ax.plot(x + off * nx, y + off * ny, color="#1E2939", lw=3, zorder=1)
    ax.fill(np.r_[x + (SIGN_OFFSET + 0.6) * nx, (x - 3.0 * nx)[::-1]],
            np.r_[y + (SIGN_OFFSET + 0.6) * ny, (y - 3.0 * ny)[::-1]],
            color="#E9ECEF", lw=0, zorder=0)
    ax.plot(x + 1.75 * nx, y + 1.75 * ny, color="white", lw=2, ls=(0, (8, 6)), zorder=1)
    sg = hd_map()
    ax.scatter(sg[:, 0], sg[:, 1], marker="s", s=60, color="#1E9E74", zorder=3,
               label="exit sign (HD map)")
    return sg



def add_help_panel(fig, sections, rect=(0.745, 0.10, 0.245, 0.84), width=56, fs=8.2):
    """A text panel beside the plots: what each panel shows, and what to try.

    sections: [(heading, [(subheading, [bullet, ...]), ...]), ...]
    Headings are bold, subheadings semibold, bullets indented under them.
    """
    import textwrap
    from matplotlib.patches import FancyBboxPatch
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.015",
                                transform=ax.transAxes, fc="#F6F7F9", ec="#C6CBD1", lw=1))
    line = fs * 1.28 / (fig.get_figheight() * 72 * rect[3])   # one text line, axes units
    y = 0.978
    for heading, groups in sections:
        ax.text(0.035, y, heading, transform=ax.transAxes, va="top", fontsize=fs + 1.2,
                fontweight="bold", color="#1E2939")
        y -= line * 1.5
        for sub, bullets in groups:
            ax.text(0.05, y, sub, transform=ax.transAxes, va="top", fontsize=fs + 0.2,
                    fontweight="semibold", color="#2D6CA2")
            y -= line * 1.2
            for b in bullets:
                lines = textwrap.wrap(b, width, initial_indent="\u2022 ",
                                      subsequent_indent="   ", break_on_hyphens=False)
                ax.text(0.075, y, "\n".join(lines), transform=ax.transAxes, va="top",
                        fontsize=fs, color="#1E2939", linespacing=1.28)
                y -= line * len(lines)
            y -= line * 0.45
        y -= line * 0.5
    return ax

def build_figure(d, sigma_w, sigma_range, interactive=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider

    t = d["t"]
    n = len(t)
    mrows = np.where(d["sign_id"] >= 0)[0]
    state = {"k": 0, "playing": True, "sw": sigma_w, "sr": sigma_range, "wrong": False}
    res = {}

    fig = plt.figure(figsize=(17.5, 8.4))
    if interactive:
        fig.canvas.manager.set_window_title("EKF in a curved tunnel")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], hspace=0.42, wspace=0.16,
                          left=0.03, right=0.725, top=0.90, bottom=0.15)

    ax_top = fig.add_subplot(gs[:, 0])
    ax_top.set_title("The curved tunnel", loc="left", fontsize=12,
                     fontweight="bold")
    signs = draw_tunnel(ax_top)
    trail, = ax_top.plot([], [], color="#2D6CA2", lw=1, alpha=0.6)
    true_dot, = ax_top.plot([], [], "s", color="#1E2939", ms=10, label="true car")
    est_dot, = ax_top.plot([], [], "o", color="#2D6CA2", ms=7, label="estimate")
    ell, = ax_top.plot([], [], color="#2D6CA2", lw=2, label="1-sigma ellipse (P)")
    arrow, = ax_top.plot([], [], color="#2D6CA2", lw=2.5)
    fan, = ax_top.plot([], [], color="#2D6CA2", lw=1, ls="--")
    ray, = ax_top.plot([], [], color="#D9694A", lw=2, label="camera: range and bearing (z)")
    ax_top.set_aspect("equal"); ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.legend(loc="lower left", fontsize=9, frameon=True)
    info = ax_top.text(0.02, 0.97, "", transform=ax_top.transAxes, va="top",
                       fontsize=10, family="monospace",
                       bbox=dict(fc="white", ec="#C6CBD1", alpha=0.9))

    ax_sig = fig.add_subplot(gs[0, 1])
    ax_sig.set_title("sigma of the heading (deg)", loc="left", fontsize=11)
    for m in mrows:
        ax_sig.axvline(t[m], color="#D9694A", lw=0.8, alpha=0.35)
    line_sig, = ax_sig.plot([], [], color="#2D6CA2", lw=2)
    now_sig = ax_sig.axvline(0, color="#1E2939", lw=1)
    ax_sig.set_xlim(t[0], t[-1]); ax_sig.set_ylabel("deg")

    ax_err = fig.add_subplot(gs[1, 1])
    ax_err.set_title("true heading error, and the +/- sigma the filter reports",
                     loc="left", fontsize=11)
    band = ax_err.fill(np.r_[t, t[::-1]], np.zeros(2 * n), color="#2D6CA2", alpha=0.18,
                       lw=0)[0]
    line_err, = ax_err.plot([], [], color="#D9694A", lw=2)
    ax_err.axhline(0, color="#7A828C", lw=1)
    now_err = ax_err.axvline(0, color="#1E2939", lw=1)
    ax_err.set_xlim(t[0], t[-1]); ax_err.set_xlabel("time (s)"); ax_err.set_ylabel("deg")
    t_in, t_out = LEG_IN / SPEED, (LEG_IN + BEND * RADIUS) / SPEED
    for a in (ax_sig, ax_err):
        a.axvspan(t_in, t_out, color="#FFF4D6", zorder=0)
    ax_sig.text((t_in + t_out) / 2, 0.96, "in the bend", ha="center", va="top",
                transform=ax_sig.get_xaxis_transform(), fontsize=9, color="#7A828C")

    stats = fig.text(0.38, 0.955, "", ha="center", fontsize=12, fontweight="bold")
    add_help_panel(fig, [
        ("What you see", [
            ("Left: the curved tunnel", [
                "dark square: the true car",
                "blue dot and arrow: the estimate and its heading",
                "dashed fan: +/- 1 sigma of heading",
                "blue ellipse: the position uncertainty",
                "orange line: the camera's range and bearing to the sign "
                "it just matched",
            ]),
            ("Top right: sigma of the heading", [
                "climbs while only the gyro drives the prediction",
                "drops at each match: the bearing tells the filter which "
                "way the nose points",
                "yellow shading: the car is in the bend",
            ]),
            ("Bottom right: the honesty check", [
                "The orange line is the true heading error: the estimated "
                "heading minus the real one. Positive: the estimate points too "
                "far left.",
                "The blue band is the filter's own claim about that error: "
                "plus or minus one sigma.",
                "A bell curve holds 68% of its values within one sigma, so an "
                "honest filter keeps the line inside the band about 68% of the time.",
                "Much less than 68%: the band is too thin, overconfident. Much "
                "more: the band is too wide, underconfident.",
            ]),
        ]),
        ("Try this", [
            ("Raise Q (gyro noise)", [
                "sigma climbs faster between signs; the band widens",
                "at 8 deg/s: 92% inside, underconfident",
            ]),
            ("Lower Q", [
                "the band is too thin to cover the gyro's bias",
                "at 0.1 deg/s: 23% inside, overconfident",
            ]),
            ("Raise R (range noise)", [
                "the filter trusts each match less; corrections shrink",
            ]),
            ("Lower R", [
                "at 0.1 m it chases every match; the position error grows",
            ]),
            ("Tick 'wrong sign in the Jacobian'", [
                "the filter still runs, with no error message",
                "the heading error triples and leaves the band",
                "that is why you run --check-jacobian first",
            ]),
        ]),
    ])

    def rerun():
        res["xs"], res["Ps"] = run_ekf(d, state["sw"], state["sr"], state["wrong"])
        res["sig"] = np.degrees(np.sqrt(res["Ps"][:, 2, 2]))
        res["err"] = np.degrees(wrap(res["xs"][:, 2] - d["true_theta"]))
        line_sig.set_data(t, res["sig"])
        line_err.set_data(t, res["err"])
        band.set_xy(np.column_stack([np.r_[t, t[::-1]],
                                     np.r_[res["sig"], -res["sig"][::-1]]]))
        pos, head, inside = score(d, res["xs"], res["Ps"])
        tag = "   WRONG JACOBIAN SIGN" if state["wrong"] else ""
        stats.set_text(f"position error (RMSE) {pos:.2f} m    heading error {head:.1f} deg"
                       f"    heading inside 1-sigma: {100 * inside:.0f}%  "
                       f"(honest: about 68%){tag}")
        stats.set_color("#D9694A" if state["wrong"] else "#1E2939")
        ax_sig.set_ylim(0, max(6.0, 1.1 * res["sig"].max()))
        lim = max(6.0, 1.15 * np.abs(res["err"]).max())
        ax_err.set_ylim(-lim, lim)

    rerun()

    def draw(k):
        k = int(k)
        xs, Ps = res["xs"], res["Ps"]
        cx, cy = d["true_x"][k], d["true_y"][k]
        ax_top.set_xlim(cx - 26, cx + 26); ax_top.set_ylim(cy - 34, cy + 34)
        true_dot.set_data([cx], [cy])
        est_dot.set_data([xs[k, 0]], [xs[k, 1]])
        trail.set_data(xs[:k + 1, 0], xs[:k + 1, 1])
        e = ellipse_xy(Ps[k, :2, :2]) + xs[k, :2]
        ell.set_data(e[:, 0], e[:, 1])
        L, th, sg = 7.0, xs[k, 2], np.sqrt(Ps[k, 2, 2])
        arrow.set_data([xs[k, 0], xs[k, 0] + L * np.cos(th)],
                       [xs[k, 1], xs[k, 1] + L * np.sin(th)])
        fan.set_data([xs[k, 0] + L * np.cos(th - sg), xs[k, 0], xs[k, 0] + L * np.cos(th + sg)],
                     [xs[k, 1] + L * np.sin(th - sg), xs[k, 1], xs[k, 1] + L * np.sin(th + sg)])
        recent = [m for m in mrows if 0 <= k - m < 8]
        if recent:
            m = recent[-1]
            sgn = signs[int(d["sign_id"][m])]
            ray.set_data([d["true_x"][m], sgn[0]], [d["true_y"][m], sgn[1]])
        else:
            ray.set_data([], [])
        pos_sig = np.sqrt(np.linalg.eigvalsh(Ps[k, :2, :2]).max())
        info.set_text(f"t {t[k]:5.1f} s\nheading sigma {res['sig'][k]:4.1f} deg\n"
                      f"position sigma {pos_sig:4.2f} m")
        now_sig.set_xdata([t[k], t[k]]); now_err.set_xdata([t[k], t[k]])
        return true_dot, est_dot, ell, arrow, fan, ray, info, now_sig, now_err, trail

    if not interactive:
        return fig, draw, n

    ax_play = fig.add_axes([0.04, 0.035, 0.06, 0.045])
    ax_rst = fig.add_axes([0.11, 0.035, 0.06, 0.045])
    ax_sw = fig.add_axes([0.33, 0.06, 0.17, 0.025])
    ax_sr = fig.add_axes([0.33, 0.02, 0.17, 0.025])
    ax_chk = fig.add_axes([0.66, 0.02, 0.22, 0.065])
    b_play = Button(ax_play, "Pause")
    b_rst = Button(ax_rst, "Restart")
    s_sw = Slider(ax_sw, "Q: gyro noise (deg/s)", 0.05, 10.0, valinit=np.degrees(sigma_w))
    s_sr = Slider(ax_sr, "R: sign range noise (m)", 0.05, 5.0, valinit=sigma_range)
    chk = CheckButtons(ax_chk, ["wrong sign in the Jacobian"], [False])

    def on_play(_):
        state["playing"] = not state["playing"]
        b_play.label.set_text("Pause" if state["playing"] else "Play")
    def on_rst(_):
        state["k"] = 0
    def on_change(_):
        state["sw"], state["sr"] = np.radians(s_sw.val), s_sr.val
        state["wrong"] = chk.get_status()[0]
        rerun(); draw(state["k"]); fig.canvas.draw_idle()
    b_play.on_clicked(on_play); b_rst.on_clicked(on_rst)
    s_sw.on_changed(on_change); s_sr.on_changed(on_change); chk.on_clicked(on_change)
    fig._keep = (b_play, b_rst, s_sw, s_sr, chk)

    def step(_):
        if state["playing"]:
            state["k"] = (state["k"] + 1) % n
        return draw(state["k"])
    return fig, step, n


def animate_or_save(a, build, title, frame_stride=3):
    """Shared by ekf_curve.py and ukf_curve.py: --snapshot, --html, or the live window."""
    import matplotlib
    if a.html or a.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if a.snapshot:
        fig, step, n = build(True)
        for _ in range(int(round(a.frame / DT))):
            step(None)
        fig.savefig(a.snapshot, dpi=150)
        print("wrote", a.snapshot)
        return
    if a.html:
        fig, draw, n = build(False)
        fig.set_dpi(70)
        write_html(fig, draw, range(0, n, frame_stride), a.html, title, fps=10)
        return
    fig, step, n = build(True)
    anim = FuncAnimation(fig, step, interval=int(DT * 1000), blit=False,
                         cache_frame_data=False)
    fig._anim = anim
    plt.show()



def write_html(fig, draw, frames, path, title, fps):
    """Render the animation to a self-contained web page, with progress.

    Rendering takes about 15 to 30 seconds and prints nothing on its own, which
    looks like a hang. So it counts the frames as it goes, and it writes the
    file only once everything is rendered: stopping it early leaves no broken
    half-page behind.
    """
    import os
    import matplotlib
    from matplotlib.animation import FuncAnimation
    frames = list(frames)
    done = [0]

    def draw_with_progress(k):
        done[0] += 1
        print(f"\r  rendering frame {min(done[0], len(frames))} of {len(frames)}",
              end="", flush=True)
        return draw(k)

    print(f"rendering {len(frames)} frames into {path} "
          f"(about {max(10, len(frames) // 6)} s, a web page of ~20 MB) ...")
    anim = FuncAnimation(fig, draw_with_progress, frames=frames, interval=200, blit=False)
    matplotlib.rcParams["animation.embed_limit"] = 200
    page = anim.to_jshtml(fps=fps, default_mode="loop")
    print()
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>{title}</title></head>"
                 "<body style='font-family:sans-serif'>")
        fh.write(page)
        fh.write("</body></html>")
    os.replace(tmp, path)
    print("wrote", path, "- open it in a web browser")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", default=os.path.join(HERE, "curve_drive.csv"))
    ap.add_argument("--make-csv", action="store_true", help="write the dataset and exit")
    ap.add_argument("--check-jacobian", action="store_true",
                    help="compare the Jacobians with numerical derivatives and exit")
    ap.add_argument("--gyro-noise", type=float, default=np.degrees(SIGMA_W_FILTER),
                    help="Q: gyro noise the filter assumes (deg/s)")
    ap.add_argument("--range-noise", type=float, default=SIGMA_RANGE,
                    help="R: sign range noise the filter assumes (m)")
    ap.add_argument("--wrong-sign", action="store_true", help="flip one Jacobian sign")
    ap.add_argument("--html", help="write the animation to this .html file instead")
    ap.add_argument("--snapshot", help="write one still frame to this .png file")
    ap.add_argument("--frame", type=float, default=9.0, help="time (s) for --snapshot")
    a = ap.parse_args()

    if a.check_jacobian:
        check_jacobian()
        return
    if a.make_csv:
        make_csv(a.csv)
        return
    if not os.path.exists(a.csv):
        sys.exit(f"{a.csv} not found. Run: python3 ekf_curve.py --make-csv")
    d = load_csv(a.csv)

    sw = np.radians(a.gyro_noise)
    xs, Ps = run_ekf(d, sw, a.range_noise, a.wrong_sign)
    pos, head, inside = score(d, xs, Ps)
    print(f"gyro noise {a.gyro_noise:.2f} deg/s, range noise {a.range_noise:.2f} m"
          f"{', WRONG Jacobian sign' if a.wrong_sign else ''}: position RMSE {pos:.2f} m, "
          f"heading RMSE {head:.1f} deg, heading inside 1-sigma {100 * inside:.0f}%")
    animate_or_save(a, lambda live: build_figure(d, sw, a.range_noise, live),
                    "EKF in a curved tunnel")


if __name__ == "__main__":
    main()
