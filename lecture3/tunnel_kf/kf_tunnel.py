# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""A Kalman filter for a car in a tunnel, with a live window.

The car has lost GNSS in a tunnel. Its IMU measures acceleration (the control
input u), and every 25 m its camera recognizes an emergency exit sign and
matches it against the HD map, which gives a position (the measurement z).
The filter combines the two, exactly as in the L3 slides:

    predict:  x^- = F x + B u          P^- = F P F^T + Q
    update:   K = P^- H^T (H P^- H^T + R)^-1
              x = x^- + K (z - H x^-)  P = (I - K H) P^-

State x = [px, py, vx, vy]  (east, north position in m; east, north speed in m/s).
The sign match has sigma 1 m along the tunnel (the slider) and 0.2 m across it
(SIGN_SIGMA_Y): the wall is close, so the sideways distance is measured well.

Usage
    python3 kf_tunnel.py --make-csv               # write tunnel_drive.csv
    python3 kf_tunnel.py tunnel_drive.csv         # run the filter, live window
    python3 kf_tunnel.py tunnel_drive.csv --html tunnel_kf.html   # web page instead
    python3 kf_tunnel.py tunnel_drive.csv --snapshot frame.png    # one still image
    python3 kf_tunnel.py tunnel_drive.csv --no-control  # ignore the IMU: brake goes into w

Needs numpy and matplotlib only.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.1                 # s between IMU readings, one predict step each
SIGN_SPACING = 25.0      # m between emergency exit signs in the HD map
SIGN_Y = 3.4             # signs hang on the north wall
LANE_Y = -1.75           # the car drives in the south lane
SIGN_SIGMA_Y = 0.2       # m: a sign on the wall fixes the distance to the wall well
SIGMA_A_Y = 0.3          # m/s^2: Q's sideways part, the IMU's sideways noise
BRAKE = (8.0, 12.0, -1.0)    # s, s, m/s^2: brakes for a slower car
SPEED_UP = (20.0, 24.0, 0.8) # s, s, m/s^2: speeds up again


# ---------------------------------------------------------------------------
# 1. The dataset
# ---------------------------------------------------------------------------
def make_csv(path, seed=702):
    """Simulate a 45 s drive through a straight tunnel and write it as a CSV."""
    rng = np.random.default_rng(seed)
    n = int(45.0 / DT) + 1
    t = np.arange(n) * DT

    # what the car really does: cruise, brake for a slower car, speed up again
    ax = np.zeros(n)
    for t0, t1, a in (BRAKE, SPEED_UP):
        ax[(t >= t0) & (t < t1)] = a
    ax += 0.3 * np.sin(2 * np.pi * t / 15.0) * (t > 28)
    # a small sideways wander inside the lane. The sideways speed starts at
    # -A so that it swings evenly around zero; starting it at zero would make it
    # never negative, and the car would drift steadily into the next lane.
    A_Y, T_Y = 0.05, 9.0                          # m/s^2, s
    ay = A_Y * np.sin(2 * np.pi * t / T_Y)
    vy0 = -A_Y * T_Y / (2 * np.pi)                # about -0.07 m/s: +/- 0.1 m wander

    x = np.zeros((n, 4))
    x[0] = [0.0, LANE_Y, 12.0, vy0]
    for k in range(1, n):
        px, py, vx, vy = x[k - 1]
        x[k] = [px + vx * DT + 0.5 * ax[k - 1] * DT**2,
                py + vy * DT + 0.5 * ay[k - 1] * DT**2,
                vx + ax[k - 1] * DT,
                vy + ay[k - 1] * DT]

    # the IMU reads that acceleration, with noise and a small constant bias the
    # filter does not know about. Q has to cover it, so dead reckoning drifts.
    imu_ax = ax + 0.05 + rng.normal(0, 0.3, n)
    imu_ay = ay + rng.normal(0, 0.3, n)

    # a sign match: once per sign, when the sign comes into camera range (15 m)
    sign_x = np.full(n, np.nan)
    sign_y = np.full(n, np.nan)
    sign_id = np.full(n, -1)
    next_sign = 1
    for k in range(n):
        s = next_sign * SIGN_SPACING
        if x[k, 0] >= s - 15.0:
            sign_x[k] = x[k, 0] + rng.normal(0, 1.0)
            sign_y[k] = x[k, 1] + rng.normal(0, SIGN_SIGMA_Y)
            sign_id[k] = next_sign
            next_sign += 1

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "true_x", "true_y", "true_vx", "true_vy",
                    "imu_ax", "imu_ay", "sign_x", "sign_y", "sign_id"])
        for k in range(n):
            has = not np.isnan(sign_x[k])
            w.writerow([f"{t[k]:.1f}", *(f"{v:.4f}" for v in x[k]),
                        f"{imu_ax[k]:.4f}", f"{imu_ay[k]:.4f}",
                        f"{sign_x[k]:.3f}" if has else "",
                        f"{sign_y[k]:.3f}" if has else "",
                        sign_id[k] if has else ""])
    print(f"wrote {path}: {n} rows, {np.sum(~np.isnan(sign_x))} sign matches")


def load_csv(path):
    raw = np.genfromtxt(path, delimiter=",", names=True)
    return {k: raw[k] for k in raw.dtype.names}


# ---------------------------------------------------------------------------
# 2. The Kalman filter: everything the slides built, in about 30 lines
# ---------------------------------------------------------------------------
def run_kf(d, sigma_a=0.5, sigma_sign=1.0, use_control=True, with_prior=False):
    """Run the filter over the whole drive. Returns every step's x, P and K.

    with_prior=True also returns the prediction x^-, P^- at every step, before
    any update: the slides' "after predict" ellipse comes from there.
    """
    F = np.array([[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], float)          # motion rule: constant speed
    B = np.array([[0.5 * DT**2, 0],
                  [0, 0.5 * DT**2],
                  [DT, 0],
                  [0, DT]])                       # acceleration -> state change
    # how wrong the rule usually is: the acceleration the IMU gets wrong,
    # pushed through B. Along the tunnel sigma_a (the slider) must also cover
    # the IMU's bias; across it only the IMU's noise, 0.3 m/s^2, because a car
    # in its lane has no sideways bias to cover. So predict stretches the
    # ellipse most along the tunnel.
    Q = B @ np.diag([sigma_a**2, SIGMA_A_Y**2]) @ B.T
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], float)           # the sign match gives position
    # how noisy the sign match is: sigma_sign along the tunnel, SIGN_SIGMA_Y
    # across it (the wall is close, so the sideways distance is measured well)
    R = np.diag([sigma_sign**2, SIGN_SIGMA_Y**2])

    # start: the last GNSS fix at the tunnel entrance
    x = np.array([d["true_x"][0] + 0.5, d["true_y"][0] - 0.3, 12.0, 0.0])
    P = np.diag([1.0**2, 1.0**2, 0.5**2, 0.5**2])

    # History buffers, one entry per CSV row, so the window can replay the run:
    #   xs[k]  (4,)    the estimate after everything known at time t[k]
    #   Ps[k]  (4, 4)  its covariance
    #   Ks[k]          the along-tunnel gain K[0, 0]; NaN on rows with no update
    n = len(d["t"])
    xs, Ps, Ks = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.full(n, np.nan)
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
    for k in range(n):
        if k > 0:
            # ---- PREDICT: carry the belief from t[k-1] to t[k] -----------------
            # The IMU sample from row k-1 is the acceleration applied during the
            # interval [t[k-1], t[k]), so it drives this step: a zero-order hold.
            # Using row k instead would let the filter see the future by 0.1 s.
            u = np.array([d["imu_ax"][k - 1], d["imu_ay"][k - 1]])   # (2,)
            # use_control=False is the filter from before B u existed: it
            # ignores the IMU, so the brake is no longer predicted and falls
            # back into w. Q must then be large enough to cover a 1 m/s^2 brake.
            if not use_control:
                u = np.zeros(2)

            # Mean: constant-velocity motion (F) plus the known push (B u).
            # F couples position to velocity: p += v*dt. B u adds 0.5*a*dt^2 to
            # position and a*dt to velocity.
            x = F @ x + B @ u                                          # (4,)

            # Covariance: F P F^T moves the old uncertainty through the motion
            # (it also creates the position-velocity correlation in the
            # off-diagonal blocks), and + Q adds what the model cannot know:
            # IMU noise and bias. With no update, trace(P) only grows.
            P = F @ P @ F.T + Q                                        # (4, 4)

        xp[k], Pp[k] = x, P     # the prediction, before any update
        # Most rows have no sign match: sign_x is NaN there, so we skip the
        # update and the prediction stands as the estimate.
        if not np.isnan(d["sign_x"][k]):
            # ---- UPDATE: correct the prediction with the sign match ----------
            z = np.array([d["sign_x"][k], d["sign_y"][k]])            # (2,)

            # Innovation (the surprise): measured minus expected position.
            # H picks the two position rows out of the state.
            nu = z - H @ x                                             # (2,)

            # Innovation covariance: how big nu should be if everything is
            # honest. HPH^T is our own position uncertainty, R the sensor's.
            # Comparing nu with S (nu^T S^-1 nu, the NIS) is the consistency
            # test from the "Is the Covariance Honest?" section.
            S = H @ P @ H.T + R                                        # (2, 2)

            # Kalman gain: how much of the surprise to act on, per state
            # variable. K is 4x2, so a position surprise also corrects velocity,
            # through the position-velocity correlation that predict built in P.
            # (np.linalg.solve(S.T, (P @ H.T).T).T avoids forming S^-1 and is
            # the more robust choice for larger or badly scaled problems; for a
            # well-conditioned 2x2 the explicit inverse is fine and easier to read.)
            K = P @ H.T @ np.linalg.inv(S)                             # (4, 2)

            # Mean: move toward the measurement by the fraction K of the surprise.
            x = x + K @ nu

            # Covariance: the short form (I - KH) P. It is exact only for the
            # optimal K and can lose symmetry through rounding over many steps.
            # The Joseph form (I-KH) P (I-KH)^T + K R K^T stays symmetric and
            # positive definite for any K; use it if you ever tune K by hand.
            P = (np.eye(4) - K @ H) @ P

            Ks[k] = K[0, 0]     # along-tunnel position gain, for the readout
        xs[k], Ps[k] = x, P
    if with_prior:
        return xs, Ps, Ks, xp, Pp
    return xs, Ps, Ks


# ---------------------------------------------------------------------------
# 3. The live window
# ---------------------------------------------------------------------------
def ellipse_xy(P2, n_sigma=1.0):
    """Points on the n-sigma ellipse of a 2x2 covariance, centered at 0."""
    w, V = np.linalg.eigh(P2)
    a = np.linspace(0, 2 * np.pi, 60)
    circle = np.stack([np.cos(a), np.sin(a)])
    return (V @ np.diag(n_sigma * np.sqrt(np.maximum(w, 0))) @ circle).T



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

def build_figure(d, sigma_a, sigma_sign, interactive=True, use_control=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider

    t = d["t"]
    n = len(t)
    matches = np.where(~np.isnan(d["sign_x"]))[0]
    state = {"k": 0, "playing": True}
    res = {}

    def rerun():
        res["xs"], res["Ps"], res["Ks"] = run_kf(d, state["sa"], state["ss"], state["uc"])
        res["sig"] = np.sqrt(res["Ps"][:, 0, 0])
        res["err"] = res["xs"][:, 0] - d["true_x"]
        line_sig.set_data(t, res["sig"])
        line_err.set_data(t, res["err"])
        band.set_xy(np.column_stack([np.r_[t, t[::-1]],
                                     np.r_[res["sig"], -res["sig"][::-1]]]))
        rmse = np.sqrt(np.mean(res["err"]**2))
        inside = np.mean(np.abs(res["err"]) <= res["sig"])
        tag = "" if state["uc"] else "    IMU IGNORED: the brake is in w"
        stats.set_text(f"error (RMSE) {rmse:.2f} m    true error inside 1-sigma: "
                       f"{100 * inside:.0f}%  (honest: about 68%){tag}")
        stats.set_color("#1E2939" if state["uc"] else "#D9694A")
        ax_sig.set_ylim(0, max(2.2, 1.1 * res["sig"].max()))
        lim = max(3.0, 1.2 * np.abs(res["err"]).max())
        ax_err.set_ylim(-lim, lim)

    state["sa"], state["ss"], state["uc"] = sigma_a, sigma_sign, use_control
    fig = plt.figure(figsize=(17.5, 8))
    fig.canvas.manager.set_window_title("Kalman filter in a tunnel") if interactive else None
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 0.22], hspace=0.55,
                          left=0.05, right=0.725, top=0.93, bottom=0.05)

    # --- top: the tunnel from above, following the car
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_title("GNSS lost: the tunnel", loc="left", fontsize=13,
                     fontweight="bold")
    ax_top.axhspan(-3.5, 3.5, color="#E9ECEF", zorder=0)
    ax_top.axhline(3.5, color="#1E2939", lw=3); ax_top.axhline(-3.5, color="#1E2939", lw=3)
    ax_top.axhline(0, color="white", lw=2, ls=(0, (8, 6)))
    signs = np.arange(1, 40) * SIGN_SPACING
    ax_top.scatter(signs, np.full_like(signs, SIGN_Y), marker="s", s=80, color="#1E9E74",
                   zorder=3, label="exit sign (HD map)")
    true_dot, = ax_top.plot([], [], "s", color="#1E2939", ms=11, label="true car")
    est_dot, = ax_top.plot([], [], "o", color="#2D6CA2", ms=8, label="estimate")
    ell, = ax_top.plot([], [], color="#2D6CA2", lw=2, label="1-sigma ellipse (P)")
    match_dot, = ax_top.plot([], [], "X", color="#D9694A", ms=13, label="sign match (z)")
    ax_top.set_ylim(-6, 6); ax_top.set_aspect("equal")
    ax_top.set_xlabel("along the tunnel (m)"); ax_top.set_yticks([])
    ax_top.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=5, frameon=False,
                  fontsize=10, handletextpad=0.4, columnspacing=1.2)
    # a zoom on the car, so the ellipse's shape is visible: it grows along the
    # tunnel between signs and snaps long and thin at each sign match
    ax_zoom = ax_top.inset_axes([0.795, 0.04, 0.2, 0.92])
    ax_zoom.set_facecolor("#E9ECEF")
    ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
    for sp in ax_zoom.spines.values():
        sp.set_color("#1E2939"); sp.set_linewidth(1.5)
    ax_zoom.text(0.5, 0.97, "zoom: grid = 1 m", transform=ax_zoom.transAxes, ha="center",
                 va="top", fontsize=9, color="#1E2939",
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    true_z, = ax_zoom.plot([], [], "s", color="#1E2939", ms=9)
    ell_z, = ax_zoom.plot([], [], color="#2D6CA2", lw=2.2)
    est_z, = ax_zoom.plot([], [], "o", color="#2D6CA2", ms=6)
    match_z, = ax_zoom.plot([], [], "X", color="#D9694A", ms=11)
    grid_z = [ax_zoom.axvline(0, color="white", lw=0.8) for _ in range(9)] + \
             [ax_zoom.axhline(0, color="white", lw=0.8) for _ in range(9)]
    info = ax_top.text(0.01, 0.95, "", transform=ax_top.transAxes, va="top",
                       fontsize=11, family="monospace",
                       bbox=dict(fc="white", ec="#C6CBD1", alpha=0.9))

    # --- bottom left: sigma over time (the sawtooth)
    ax_sig = fig.add_subplot(gs[1, 0])
    ax_sig.set_title("sigma of the position along the tunnel", loc="left", fontsize=12)
    for m in matches:
        ax_sig.axvline(t[m], color="#D9694A", lw=0.8, alpha=0.4)
    line_sig, = ax_sig.plot([], [], color="#2D6CA2", lw=2)
    now_sig = ax_sig.axvline(0, color="#1E2939", lw=1)
    ax_sig.set_xlim(t[0], t[-1]); ax_sig.set_xlabel("time (s)"); ax_sig.set_ylabel("m")

    # --- bottom right: true error against the band the filter reports
    ax_err = fig.add_subplot(gs[1, 1])
    ax_err.set_title("true error along the tunnel, and the +/- sigma the filter reports",
                     loc="left", fontsize=12)
    band = ax_err.fill(np.r_[t, t[::-1]], np.zeros(2 * n), color="#2D6CA2", alpha=0.18,
                       lw=0)[0]           # a Polygon: update it with set_xy
    line_err, = ax_err.plot([], [], color="#D9694A", lw=2)
    ax_err.axhline(0, color="#7A828C", lw=1)
    now_err = ax_err.axvline(0, color="#1E2939", lw=1)
    ax_err.set_xlim(t[0], t[-1]); ax_err.set_xlabel("time (s)"); ax_err.set_ylabel("m")
    # when the car brakes and speeds up: the IMU sees it, and B u predicts it
    for a_ in (ax_sig, ax_err):
        a_.axvspan(BRAKE[0], BRAKE[1], color="#FBE3DC", zorder=0)
        a_.axvspan(SPEED_UP[0], SPEED_UP[1], color="#E3F1EC", zorder=0)
    for t0, t1, label in ((BRAKE[0], BRAKE[1], "brake"), (SPEED_UP[0], SPEED_UP[1], "speed up")):
        ax_sig.text((t0 + t1) / 2, 0.97, label, transform=ax_sig.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="#7A828C")

    stats = fig.text(0.39, 0.965, "", ha="center", fontsize=12, fontweight="bold")
    add_help_panel(fig, [
        ("What you see", [
            ("Top: the tunnel", [
                "dark square: the true car",
                "blue dot: the estimate",
                "blue ellipse: its 1-sigma uncertainty, P",
                "green squares: exit signs in the HD map",
                "orange X: a sign match (the measurement z)",
                "zoom, right: the ellipse's shape, long along the tunnel "
                "and thin across it",
            ]),
            ("Bottom left: sigma along the tunnel", [
                "climbs while only the IMU drives the prediction",
                "drops at every sign match (thin orange lines)",
                "red shading: braking; green: speeding up",
            ]),
            ("Bottom right: the honesty check", [
                "orange line: the true error",
                "blue band: the +/- sigma the filter reports",
                "honest: the line stays inside about 68% of the time",
            ]),
        ]),
        ("Try this", [
            ("Raise Q (sigma_a)", [
                "the ellipse and the band grow faster between signs",
                "each match pulls the estimate harder",
                "at 3: 84% inside, underconfident",
            ]),
            ("Lower Q", [
                "the band gets thin and the error walks out of it",
                "at 0.01: 25% inside, overconfident (the dangerous case)",
            ]),
            ("Raise R (sigma_sign)", [
                "smaller jumps at each match",
                "the estimate drifts with the IMU's bias between signs",
            ]),
            ("Lower R", [
                "the estimate snaps to every match, noise and all",
            ]),
            ("Untick 'use the IMU'", [
                "the brake is no longer predicted",
                "8 to 12 s: the estimate runs ahead of the car and out of the band",
            ]),
        ]),
    ])
    rerun()

    def draw(k):
        k = int(k)
        xs, Ps = res["xs"], res["Ps"]
        cx = d["true_x"][k]
        ax_top.set_xlim(cx - 30, cx + 30)
        true_dot.set_data([cx], [d["true_y"][k]])
        est_dot.set_data([xs[k, 0]], [xs[k, 1]])
        e = ellipse_xy(Ps[k, :2, :2]) + xs[k, :2]
        ell.set_data(e[:, 0], e[:, 1])
        recent = [m for m in matches if 0 <= k - m < 8]
        if recent:
            m = recent[-1]
            match_dot.set_data([d["sign_x"][m]], [d["sign_y"][m]])
        else:
            match_dot.set_data([], [])
        # the zoom follows the estimate, on a fixed 1 m grid
        ex, ey = xs[k, 0], xs[k, 1]
        ax_zoom.set_xlim(ex - 3.2, ex + 3.2); ax_zoom.set_ylim(ey - 3.6, ey + 3.6)
        gx0, gy0 = np.floor(ex) - 4, np.floor(ey) - 4
        for i, g in enumerate(grid_z[:9]):
            g.set_xdata([gx0 + i, gx0 + i])
        for i, g in enumerate(grid_z[9:]):
            g.set_ydata([gy0 + i, gy0 + i])
        true_z.set_data([cx], [d["true_y"][k]])
        ell_z.set_data(e[:, 0], e[:, 1]); est_z.set_data([ex], [ey])
        match_z.set_data(*match_dot.get_data())
        last_k = [m for m in matches if m <= k]
        kline = f"K {res['Ks'][last_k[-1]]:.2f}" if last_k else "K  -  "
        doing = ("BRAKING" if BRAKE[0] <= t[k] < BRAKE[1] else
                 "speeding up" if SPEED_UP[0] <= t[k] < SPEED_UP[1] else "cruising")
        info.set_text(f"t {t[k]:5.1f} s   sigma {res['sig'][k]:.2f} m   {kline}   "
                      f"speed {xs[k, 2]:.1f} m/s   u = IMU a_x {d['imu_ax'][k]:+.2f} m/s^2"
                      f"   {doing}")
        now_sig.set_xdata([t[k], t[k]]); now_err.set_xdata([t[k], t[k]])
        return (true_dot, est_dot, ell, match_dot, info, now_sig, now_err,
                true_z, ell_z, est_z, match_z)

    if not interactive:
        return fig, draw, n

    # --- controls
    ax_play = fig.add_axes([0.06, 0.012, 0.07, 0.04])
    ax_rst = fig.add_axes([0.14, 0.012, 0.07, 0.04])
    ax_sa = fig.add_axes([0.36, 0.042, 0.22, 0.022])
    ax_ss = fig.add_axes([0.36, 0.010, 0.22, 0.022])
    ax_uc = fig.add_axes([0.72, 0.008, 0.20, 0.055])
    b_play = Button(ax_play, "Pause")
    b_rst = Button(ax_rst, "Restart")
    s_sa = Slider(ax_sa, "Q: sigma_a (m/s^2)", 0.01, 3.0, valinit=sigma_a)
    s_ss = Slider(ax_ss, "R: sigma_sign (m)", 0.1, 6.0, valinit=sigma_sign)
    chk = CheckButtons(ax_uc, ["use the IMU (B u)"], [use_control])

    def on_play(_):
        state["playing"] = not state["playing"]
        b_play.label.set_text("Pause" if state["playing"] else "Play")
    def on_rst(_):
        state["k"] = 0
    def on_slide(_):
        state["sa"], state["ss"] = s_sa.val, s_ss.val
        state["uc"] = chk.get_status()[0]
        rerun(); draw(state["k"]); fig.canvas.draw_idle()
    b_play.on_clicked(on_play); b_rst.on_clicked(on_rst)
    s_sa.on_changed(on_slide); s_ss.on_changed(on_slide); chk.on_clicked(on_slide)
    fig._keep = (b_play, b_rst, s_sa, s_ss, chk)     # keep widgets alive

    def step(_):
        if state["playing"]:
            state["k"] = (state["k"] + 1) % n
        return draw(state["k"])
    return fig, step, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", default=os.path.join(HERE, "tunnel_drive.csv"))
    ap.add_argument("--make-csv", action="store_true", help="write the dataset and exit")
    ap.add_argument("--sigma-a", type=float, default=0.5, help="Q: sigma of the unknown acceleration")
    ap.add_argument("--sigma-sign", type=float, default=1.0, help="R: sigma of a sign match")
    ap.add_argument("--no-control", action="store_true",
                    help="ignore the IMU (B u = 0): the brake falls back into w")
    ap.add_argument("--html", help="write the animation to this .html file instead")
    ap.add_argument("--snapshot", help="write one still frame to this .png file")
    ap.add_argument("--frame", type=float, default=11.0, help="time (s) for --snapshot")
    a = ap.parse_args()

    if a.make_csv:
        make_csv(a.csv)
        return
    if not os.path.exists(a.csv):
        sys.exit(f"{a.csv} not found. Run: python3 kf_tunnel.py --make-csv")
    d = load_csv(a.csv)

    uc = not a.no_control
    xs, Ps, _ = run_kf(d, a.sigma_a, a.sigma_sign, uc)
    err = xs[:, 0] - d["true_x"]
    rmse = np.sqrt(np.mean(err**2))
    inside = np.mean(np.abs(err) <= np.sqrt(Ps[:, 0, 0]))
    print(f"{'IMU ignored (no B u), ' if not uc else ''}"
          f"sigma_a {a.sigma_a:.2f}, sigma_sign {a.sigma_sign:.2f}: "
          f"RMSE along the tunnel {rmse:.2f} m, final sigma {np.sqrt(Ps[-1, 0, 0]):.2f} m, "
          f"inside 1-sigma {100 * inside:.0f}%")

    import matplotlib
    if a.html or a.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if a.snapshot:
        # the full window, controls included, stopped at time --frame
        fig, step, n = build_figure(d, a.sigma_a, a.sigma_sign, interactive=True, use_control=uc)
        for _ in range(int(round(a.frame / DT))):
            step(None)
        fig.savefig(a.snapshot, dpi=150)
        print("wrote", a.snapshot)
        return
    if a.html:
        fig, draw, n = build_figure(d, a.sigma_a, a.sigma_sign, interactive=False, use_control=uc)
        fig.set_dpi(70)                                   # keeps the page small
        anim = FuncAnimation(fig, draw, frames=range(0, n, 4), interval=200, blit=False)
        matplotlib.rcParams["animation.embed_limit"] = 200
        with open(a.html, "w") as f:
            f.write("<html><head><meta charset='utf-8'><title>Kalman filter in a "
                    "tunnel</title></head><body style='font-family:sans-serif'>")
            f.write(anim.to_jshtml(fps=8, default_mode="loop"))
            f.write("</body></html>")
        print("wrote", a.html)
        return

    fig, step, n = build_figure(d, a.sigma_a, a.sigma_sign, interactive=True, use_control=uc)
    anim = FuncAnimation(fig, step, interval=50, blit=False, cache_frame_data=False)
    fig._anim = anim
    plt.show()


if __name__ == "__main__":
    main()
