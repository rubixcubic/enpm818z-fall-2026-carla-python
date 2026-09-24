# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""An unscented Kalman filter next to the EKF, in the same curved tunnel.

Same car, same data, same f and h as ekf_curve.py. The difference is how each
filter pushes its uncertainty through those curves:

    EKF: one tangent (the Jacobian) at the estimate, for the whole ellipse.
    UKF: 2n + 1 = 7 sigma points, each pushed through the REAL f or h, then a
         new mean and covariance rebuilt from where they land. No Jacobians.

To see who is right we also compute the exact belief by brute force: thousands
of possible cars, each driven with its own draw of the wheel and gyro noise and
weighted by how well it explains each sign match (the particle filter of the
next section, used here as a referee). A filter is right when its mean and
ellipse match that cloud.

Two sliders set how hard the job is:
    heading sigma at the entrance   how well the heading is known going in
    unlit stretch                   the camera matches no sign for this long, so
                                    the car drives the bend on wheels and gyro only

Usage
    python3 ukf_curve.py                          # live window (uses curve_drive.csv)
    python3 ukf_curve.py --html ukf_curve.html    # web page instead
    python3 ukf_curve.py --snapshot frame.png --frame 5

Needs numpy and matplotlib only.
"""
import argparse
import os
import sys

import numpy as np

import ekf_curve as ek
from ekf_curve import DT, f, h, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
N_STATE = 3
KAPPA = 0.0      # the usual choice n + kappa = 3. With n = 3 that makes kappa 0,
                 # so the center point gets weight 0 and the six others 1/6 each.
N_CLOUD = 4000   # possible cars in the brute-force reference


# ---------------------------------------------------------------------------
# 1. The UKF
# ---------------------------------------------------------------------------
def sigma_points(x, P):
    """Step 1, place: the mean, plus one step out and one step back along each
    axis of the ellipse. One step is sqrt(n + kappa) standard deviations; the
    axes are the columns of the Cholesky factor of P."""
    L = np.linalg.cholesky((N_STATE + KAPPA) * P)
    pts = [x] + [x + L[:, i] for i in range(N_STATE)] + [x - L[:, i] for i in range(N_STATE)]
    pts = np.array(pts)
    pts[:, 2] = wrap(pts[:, 2])
    W = np.full(2 * N_STATE + 1, 1.0 / (2 * (N_STATE + KAPPA)))
    W[0] = KAPPA / (N_STATE + KAPPA)
    return pts, W


def mean_and_cov(Y, W, angle_col):
    """Step 3, rebuild: weighted average and weighted spread. Angles cannot be
    averaged directly (179 and -179 deg average to 0, not 180), so each angle is
    taken relative to the first point, wrapped, averaged, and added back."""
    ref = Y[0].copy()
    D = Y - ref
    if angle_col is not None:
        D[:, angle_col] = wrap(D[:, angle_col])
    mean = ref + W @ D
    if angle_col is not None:
        mean[angle_col] = wrap(mean[angle_col])
    R = Y - mean
    if angle_col is not None:
        R[:, angle_col] = wrap(R[:, angle_col])
    return mean, (W[:, None] * R).T @ R, R


def run_ukf(d, sigma_w=ek.SIGMA_W_FILTER, sigma_range=ek.SIGMA_RANGE,
            sigma_theta0=np.radians(5.0), dark_until=0.0):
    signs = ek.hd_map()
    x, P = ek.initial_belief(d, sigma_theta0)
    Rm = np.diag([sigma_range**2, ek.SIGMA_BEARING**2])
    matches = ek.match_rows(d, dark_until)
    n = len(d["t"])
    xs, Ps, Xs = np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 7, 3))
    for k in range(n):
        if k > 0:
            u = (d["wheel_v"][k - 1], d["gyro_w"][k - 1])
            X, W = sigma_points(x, P)                 # 1. place
            Y = f(X, u)                               # 2. push, through the real f
            x, P, _ = mean_and_cov(Y, W, 2)           # 3. rebuild
            P = P + ek.process_noise(x, u, ek.SIGMA_V, sigma_w)   # plus Q, as always
        if matches[k]:
            # the same recipe, through h: expected reading and its spread S
            sign = signs[int(d["sign_id"][k])]
            X, W = sigma_points(x, P)
            Z = h(X, sign)
            zhat, S, Rz = mean_and_cov(Z, W, 1)
            S = S + Rm
            Dx = X - x
            Dx[:, 2] = wrap(Dx[:, 2])
            Pxz = (W[:, None] * Dx).T @ Rz            # how state and reading move together
            K = Pxz @ np.linalg.inv(S)                # the same gain, no Jacobian
            nu = np.array([d["sign_range"][k], d["sign_bearing"][k]]) - zhat
            nu[1] = wrap(nu[1])
            x = x + K @ nu
            x[2] = wrap(x[2])
            P = P - K @ S @ K.T
        xs[k], Ps[k] = x, P
        Xs[k] = sigma_points(x, P)[0]
    return xs, Ps, Xs


# ---------------------------------------------------------------------------
# 2. The referee: the exact belief, by brute force
# ---------------------------------------------------------------------------
def run_cloud(d, sigma_w=ek.SIGMA_W_FILTER, sigma_range=ek.SIGMA_RANGE,
              sigma_theta0=np.radians(5.0), dark_until=0.0, seed=7):
    """N_CLOUD possible cars under the same assumptions as the filters.

    Returns, at every step, the cloud's weighted mean and covariance, plus the
    particles themselves (for drawing). This is a particle filter; see
    pf_tunnel.py for one built to be read.
    """
    rng = np.random.default_rng(seed)
    signs = ek.hd_map()
    x0, P0 = ek.initial_belief(d, sigma_theta0)
    pts = rng.multivariate_normal(x0, P0, N_CLOUD)
    w = np.full(N_CLOUD, 1.0 / N_CLOUD)
    matches = ek.match_rows(d, dark_until)
    n = len(d["t"])
    means, covs, keep = np.zeros((n, 3)), np.zeros((n, 3, 3)), []
    for k in range(n):
        if k > 0:
            v = d["wheel_v"][k - 1] + rng.normal(0, ek.SIGMA_V, N_CLOUD)
            om = d["gyro_w"][k - 1] + rng.normal(0, sigma_w, N_CLOUD)
            pts[:, 0] += v * DT * np.cos(pts[:, 2])
            pts[:, 1] += v * DT * np.sin(pts[:, 2])
            pts[:, 2] = wrap(pts[:, 2] + om * DT)
        if matches[k]:
            z = np.array([d["sign_range"][k], d["sign_bearing"][k]])
            zz = h(pts, signs[int(d["sign_id"][k])])
            e_r = (zz[:, 0] - z[0]) / sigma_range
            e_b = wrap(zz[:, 1] - z[1]) / ek.SIGMA_BEARING
            w = w * np.exp(-0.5 * (e_r**2 + e_b**2))
            w = w / w.sum()
            if 1.0 / np.sum(w**2) < N_CLOUD / 2:       # resample when too few count
                pos = (rng.random() + np.arange(N_CLOUD)) / N_CLOUD
                idx = np.minimum(np.searchsorted(np.cumsum(w), pos), N_CLOUD - 1)
                pts, w = pts[idx], np.full(N_CLOUD, 1.0 / N_CLOUD)
        m, c, _ = mean_and_cov(np.vstack([pts[:1], pts]), np.r_[0.0, w], 2)
        means[k], covs[k] = m, c
        keep.append(pts[::10].copy())
    return means, covs, keep


def compare(d, sigma_theta0, dark_until):
    """Run all three and measure how far each filter is from the exact belief."""
    e_xs, e_Ps = ek.run_ekf(d, sigma_theta0=sigma_theta0, dark_until=dark_until)
    u_xs, u_Ps, u_Xs = run_ukf(d, sigma_theta0=sigma_theta0, dark_until=dark_until)
    c_m, c_P, cloud = run_cloud(d, sigma_theta0=sigma_theta0, dark_until=dark_until)
    big = lambda Ps: np.sqrt(np.array([np.linalg.eigvalsh(P[:2, :2]).max() for P in Ps]))
    off = lambda xs: np.hypot(xs[:, 0] - c_m[:, 0], xs[:, 1] - c_m[:, 1])
    return dict(e_xs=e_xs, e_Ps=e_Ps, u_xs=u_xs, u_Ps=u_Ps, u_Xs=u_Xs, c_m=c_m, c_P=c_P,
                cloud=cloud, e_big=big(e_Ps), u_big=big(u_Ps), c_big=big(c_P),
                e_off=off(e_xs), u_off=off(u_xs))


def report_row(d, r, dark_until):
    """The numbers just before the first sign match after the unlit stretch:
    the moment the uncertainty is largest and the filters disagree most."""
    used = np.where(ek.match_rows(d, dark_until))[0]
    k = max(0, used[0] - 1) if len(used) else len(d["t"]) - 1
    return k, (r["e_off"][k], r["u_off"][k], r["c_big"][k], r["e_big"][k], r["u_big"][k])


# ---------------------------------------------------------------------------
# 3. The live window
# ---------------------------------------------------------------------------

def add_help_panel(fig, sections, rect=(0.745, 0.13, 0.245, 0.80), width=58, fs=8.6):
    """A text panel beside the plots: what each panel shows, and what to try.

    sections is a list of (heading, [bullet, bullet, ...]).
    """
    import textwrap
    from matplotlib.patches import FancyBboxPatch
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.015",
                                transform=ax.transAxes, fc="#F6F7F9", ec="#C6CBD1", lw=1))
    line = fs * 1.3 / (fig.get_figheight() * 72 * rect[3])   # one text line, axes units
    y = 0.975
    for heading, bullets in sections:
        ax.text(0.04, y, heading, transform=ax.transAxes, va="top", fontsize=fs + 0.8,
                fontweight="bold", color="#1E2939")
        y -= line * 1.45
        for b in bullets:
            lines = textwrap.wrap(b, width, initial_indent="\u2022 ", subsequent_indent="   ")
            ax.text(0.05, y, "\n".join(lines), transform=ax.transAxes, va="top",
                    fontsize=fs, color="#1E2939", linespacing=1.3)
            y -= line * len(lines) + line * 0.35
        y -= line * 0.5
    return ax

def build_figure(d, sigma_theta0, dark_until, interactive=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    t = d["t"]
    n = len(t)
    state = {"k": 0, "playing": True, "st": sigma_theta0, "dk": dark_until}
    res = {}
    EKF_C, UKF_C, CLOUD_C = "#D9694A", "#1E9E74", "#8FA6BF"

    fig = plt.figure(figsize=(17.5, 8.4))
    if interactive:
        fig.canvas.manager.set_window_title("UKF and EKF in a curved tunnel")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], hspace=0.42, wspace=0.16,
                          left=0.03, right=0.725, top=0.87, bottom=0.15)
    ax_top = fig.add_subplot(gs[:, 0])
    ax_top.set_title("EKF, UKF and the exact belief", loc="left",
                     fontsize=12, fontweight="bold")
    ek.draw_tunnel(ax_top)
    cloud_sc = ax_top.scatter([], [], s=3, color=CLOUD_C, alpha=0.5, zorder=2,
                              label="exact belief (possible cars)")
    true_dot, = ax_top.plot([], [], "s", color="#1E2939", ms=9, label="true car", zorder=5)
    e_ell, = ax_top.plot([], [], color=EKF_C, lw=2, ls="--", label="EKF", zorder=4)
    e_dot, = ax_top.plot([], [], "o", color=EKF_C, ms=6, zorder=4)
    u_ell, = ax_top.plot([], [], color=UKF_C, lw=2, label="UKF", zorder=4)
    u_dot, = ax_top.plot([], [], "o", color=UKF_C, ms=6, zorder=4)
    u_sp, = ax_top.plot([], [], "+", color=UKF_C, ms=10, mew=2, zorder=4,
                        label="UKF sigma points")
    ax_top.set_aspect("equal"); ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.legend(loc="lower left", fontsize=9, frameon=True)
    info = ax_top.text(0.02, 0.97, "", transform=ax_top.transAxes, va="top", fontsize=10,
                       family="monospace", bbox=dict(fc="white", ec="#C6CBD1", alpha=0.9))

    ax_big = fig.add_subplot(gs[0, 1])
    ax_big.set_title("size of the belief: largest position sigma (m)", loc="left",
                     fontsize=11)
    l_cb, = ax_big.plot([], [], color=CLOUD_C, lw=4, label="exact")
    l_eb, = ax_big.plot([], [], color=EKF_C, lw=2, ls="--", label="EKF")
    l_ub, = ax_big.plot([], [], color=UKF_C, lw=2, label="UKF")
    ax_big.legend(loc="upper right", fontsize=9, frameon=False)
    ax_off = fig.add_subplot(gs[1, 1])
    ax_off.set_title("how far each filter's mean is from the exact one (m)", loc="left",
                     fontsize=11)
    l_eo, = ax_off.plot([], [], color=EKF_C, lw=2, ls="--", label="EKF")
    l_uo, = ax_off.plot([], [], color=UKF_C, lw=2, label="UKF")
    ax_off.legend(loc="upper right", fontsize=9, frameon=False)
    ax_off.set_xlabel("time (s)")
    dark = [a.axvspan(0, 0, color="#E3E6EA", zorder=0) for a in (ax_big, ax_off)]
    nows = [a.axvline(0, color="#1E2939", lw=1) for a in (ax_big, ax_off)]
    for a in (ax_big, ax_off):
        a.set_xlim(t[0], t[-1])
    stats = fig.text(0.38, 0.962, "", ha="center", va="top", fontsize=11.5,
                     fontweight="bold", linespacing=1.5)
    ek.add_help_panel(fig, [
        ("What you see", [
            ("Left: the curved tunnel", [
                "gray dots: the exact belief, 4,000 possible cars",
                "dashed orange ellipse: the EKF",
                "green ellipse: the UKF",
                "green + marks: the UKF's sigma points",
                "dark square: the true car",
            ]),
            ("Top right: the size of each belief", [
                "the largest position sigma: exact, EKF and UKF",
                "gray shading: the unlit stretch, no sign matched",
            ]),
            ("Bottom right: which approximation is right", [
                "how far each filter's mean is from the exact one",
            ]),
        ]),
        ("Try this", [
            ("Raise the heading sigma", [
                "the cloud bends into a banana",
                "the EKF's mean drifts off it (6 m at 25 deg)",
                "the UKF's mean stays on it",
            ]),
            ("Lengthen the unlit stretch", [
                "the car drives further on the gyro alone",
                "the banana grows, and the EKF drifts further",
            ]),
            ("Lower the heading sigma to 3 deg", [
                "the two agree to a few centimeters: the EKF is enough",
            ]),
            ("Watch the first match after the dark", [
                "both snap back, but neither follows the exact belief at once",
                "a 14 m belief meeting a 1 m measurement: the particle "
                "filter's job",
            ]),
        ]),
    ])

    def rerun():
        r = compare(d, state["st"], state["dk"])
        res.update(r)
        for line, key in ((l_cb, "c_big"), (l_eb, "e_big"), (l_ub, "u_big"),
                          (l_eo, "e_off"), (l_uo, "u_off")):
            line.set_data(t, r[key])
        for sp in dark:
            # axvspan returns a Rectangle in recent matplotlib, a Polygon in older ones
            if hasattr(sp, "set_width"):
                sp.set_x(0); sp.set_width(state["dk"])
            else:
                sp.set_xy([[0, 0], [0, 1], [state["dk"], 1], [state["dk"], 0], [0, 0]])
        ax_big.set_ylim(0, 1.15 * max(r["c_big"].max(), r["e_big"].max(), r["u_big"].max()))
        ax_off.set_ylim(0, max(0.5, 1.15 * max(r["e_off"].max(), r["u_off"].max())))
        k, (eo, uo, cb, eb, ub) = report_row(d, r, state["dk"])
        stats.set_text(f"just before the first sign match (t = {t[k]:.1f} s):  mean off the "
                       f"exact belief  EKF {eo:.2f} m,  UKF {uo:.2f} m\n"
                       f"largest sigma  exact {cb:.1f} m,  EKF {eb:.1f} m,  UKF {ub:.1f} m")

    rerun()

    def draw(k):
        k = int(k)
        cx, cy = d["true_x"][k], d["true_y"][k]
        span = max(26.0, 3.0 * res["c_big"][k])
        ax_top.set_xlim(cx - span, cx + span); ax_top.set_ylim(cy - 1.3 * span, cy + 1.3 * span)
        true_dot.set_data([cx], [cy])
        cloud_sc.set_offsets(res["cloud"][k][:, :2])
        for ell, dot, xs, Ps in ((e_ell, e_dot, res["e_xs"], res["e_Ps"]),
                                 (u_ell, u_dot, res["u_xs"], res["u_Ps"])):
            e = ek.ellipse_xy(Ps[k, :2, :2]) + xs[k, :2]
            ell.set_data(e[:, 0], e[:, 1]); dot.set_data([xs[k, 0]], [xs[k, 1]])
        u_sp.set_data(res["u_Xs"][k][1:, 0], res["u_Xs"][k][1:, 1])
        used = np.where(ek.match_rows(d, state["dk"]))[0]
        first = used[0] if len(used) else n
        tag = "no sign matched yet" if k < first else "signs matched"
        info.set_text(f"t {t[k]:5.1f} s   {tag}\n"
                      f"mean off exact:  EKF {res['e_off'][k]:5.2f} m\n"
                      f"                 UKF {res['u_off'][k]:5.2f} m")
        for ln in nows:
            ln.set_xdata([t[k], t[k]])
        return true_dot, cloud_sc, e_ell, e_dot, u_ell, u_dot, u_sp, info, *nows

    if not interactive:
        return fig, draw, n

    ax_play = fig.add_axes([0.04, 0.035, 0.06, 0.045])
    ax_rst = fig.add_axes([0.11, 0.035, 0.06, 0.045])
    ax_st = fig.add_axes([0.40, 0.06, 0.20, 0.025])
    ax_dk = fig.add_axes([0.40, 0.02, 0.20, 0.025])
    b_play = Button(ax_play, "Pause")
    b_rst = Button(ax_rst, "Restart")
    s_st = Slider(ax_st, "heading sigma at the entrance (deg)", 1.0, 30.0,
                  valinit=np.degrees(sigma_theta0))
    s_dk = Slider(ax_dk, "unlit stretch: no sign matched for (s)", 0.0, 10.0,
                  valinit=dark_until, valstep=0.5)

    def on_play(_):
        state["playing"] = not state["playing"]
        b_play.label.set_text("Pause" if state["playing"] else "Play")
    def on_rst(_):
        state["k"] = 0
    def on_change(_):
        state["st"], state["dk"] = np.radians(s_st.val), s_dk.val
        rerun(); draw(state["k"]); fig.canvas.draw_idle()
    b_play.on_clicked(on_play); b_rst.on_clicked(on_rst)
    s_st.on_changed(on_change); s_dk.on_changed(on_change)
    fig._keep = (b_play, b_rst, s_st, s_dk)

    def step(_):
        if state["playing"]:
            state["k"] = (state["k"] + 1) % n
        return draw(state["k"])
    return fig, step, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", default=os.path.join(HERE, "curve_drive.csv"))
    ap.add_argument("--heading-sigma", type=float, default=12.0,
                    help="heading sigma at the entrance (deg)")
    ap.add_argument("--dark", type=float, default=5.0,
                    help="unlit stretch: no sign matched for this many seconds")
    ap.add_argument("--html", help="write the animation to this .html file instead")
    ap.add_argument("--snapshot", help="write one still frame to this .png file")
    ap.add_argument("--frame", type=float, default=5.0, help="time (s) for --snapshot")
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        sys.exit(f"{a.csv} not found. Run: python3 ekf_curve.py --make-csv")
    d = ek.load_csv(a.csv)

    st = np.radians(a.heading_sigma)
    r = compare(d, st, a.dark)
    k, (eo, uo, cb, eb, ub) = report_row(d, r, a.dark)
    print(f"heading sigma {a.heading_sigma:.0f} deg, unlit {a.dark:.1f} s: just before "
          f"the first sign match, t = "
          f"{d['t'][k]:.1f} s the mean is off the exact belief by EKF {eo:.2f} m, "
          f"UKF {uo:.2f} m; largest sigma exact {cb:.2f}, EKF {eb:.2f}, UKF {ub:.2f} m")
    ek.animate_or_save(a, lambda live: build_figure(d, st, a.dark, live),
                       "UKF and EKF in a curved tunnel")


if __name__ == "__main__":
    main()
