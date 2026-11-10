"""Shared pieces for the L10 hands-on: the turn, the bicycle model, the speed profile,
and the cubic and quintic speed changes.

Frame: the turn's local frame, origin at the path's first point, x along the approach,
y to its left (export_curve.py says how it comes from Town03's map frame). Headings
are counterclockwise from x; a positive steering angle turns left.

Nothing here steers or controls the speed. The controller runs in l10_results/ were
computed in advance; you write the controllers in GP4.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WHEELBASE = 2.875        # m, GP4's controller_config.yaml
MAX_STEER = 0.7          # rad, GP4's controller_config.yaml


def load_curve(path=HERE / "l10_curve.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_results(folder=HERE / "l10_results"):
    folder = Path(folder)
    out = {"worked": json.loads((folder / "worked.json").read_text())}
    for name in ("pid_grid", "track_grid", "mpc_candidates"):
        f = folder / f"{name}.npz"
        if not f.exists():
            raise FileNotFoundError(f"missing results file: {f}")
        d = np.load(f)
        out[name] = {k: d[k] for k in d.files}
    return out


# ------------------------------------------------------------------ the bicycle model
def bicycle_step(x, y, theta, v, delta, a, dt, L=WHEELBASE):
    """One step of the kinematic bicycle model, state at the middle of the rear axle.

    x, y: rear axle position (m); theta: heading (rad); v: speed (m/s);
    delta: front wheel angle (rad, positive turns left); a: acceleration (m/s^2).
    """
    return (x + v * np.cos(theta) * dt,
            y + v * np.sin(theta) * dt,
            theta + v / L * np.tan(delta) * dt,
            v + a * dt)


def turning_radius(delta, L=WHEELBASE):
    """Radius of the circle the rear axle drives at a constant steering angle."""
    return L / np.tan(abs(delta))


def drive_constant(x, y, theta, v, delta, T, dt=0.05, L=WHEELBASE):
    """Hold one steering angle and one speed for T seconds; returns the rear axle path."""
    xs, ys, ths = [x], [y], [theta]
    for _ in range(int(round(T / dt))):
        x, y, theta, v = bicycle_step(x, y, theta, v, delta, 0.0, dt, L)
        xs.append(x); ys.append(y); ths.append(theta)
    return np.array(xs), np.array(ys), np.array(ths)


# ------------------------------------------------------------------ speed profile
def speed_profile(s, kappa, v_max=8.0, a_lat=2.0, a_acc=1.5, a_brake=2.0, v_start=8.0):
    """The speed limit, the curvature limit sqrt(a_lat / |kappa|), then a
    backward pass (braking) and a forward pass (accelerating)."""
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-9)))
    ds = np.diff(s)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds[i]))
    v[0] = min(v[0], v_start)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_acc * ds[i - 1]))
    return v


def time_along(s, v):
    """t(s): the time to reach each point, adding ds / (mean speed) segment by segment."""
    vm = 0.5 * (v[1:] + v[:-1])
    return np.concatenate([[0.0], np.cumsum(np.diff(s) / vm)])


# ------------------------------------------------------------------ polynomials in time
def quintic(q0, dq0, ddq0, qT, dqT, ddqT, T):
    """Coefficients a0..a5 of q(t) = sum a_i t^i matching position, speed and
    acceleration at t = 0 and t = T (six conditions, six unknowns)."""
    A = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0],
                  [1, T, T**2, T**3, T**4, T**5],
                  [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                  [0, 0, 2, 6*T, 12*T**2, 20*T**3]], float)
    return np.linalg.solve(A, np.array([q0, dq0, ddq0, qT, dqT, ddqT], float))


def cubic(q0, dq0, qT, dqT, T):
    """Coefficients a0..a3 matching position and speed only at both ends."""
    A = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [1, T, T**2, T**3],
                  [0, 1, 2*T, 3*T**2]], float)
    return np.linalg.solve(A, np.array([q0, dq0, qT, dqT], float))


def poly_eval(c, t, der=0):
    """Value (der=0), speed (1), acceleration (2) or jerk (3) of the polynomial c."""
    p = np.polynomial.Polynomial(c).deriv(der) if der else np.polynomial.Polynomial(c)
    return p(t)


def slow_down(v0, v1, b_max, dt=0.05):
    """Two ways to slow from v0 to v1 (m/s) without passing the braking limit b_max:
    a constant deceleration, and a quintic s(t) with zero acceleration at both ends.
    Returns, for each, the time T, the distance D and the sampled a(t) and jerk(t)."""
    dv = v0 - v1
    out = {}
    T = dv / b_max
    t = np.arange(0, T + 3.0, dt)
    a = np.where(t <= T, -b_max, 0.0)
    a = np.concatenate([[0.0], a])            # cruising before the brake comes on
    out["constant"] = dict(T=T, D=(v0 + v1) / 2 * T, t=np.concatenate([[-dt], t]), a=a,
                           jerk=np.concatenate([[0.0], np.diff(a) / dt]), peak_a=b_max)
    # the quintic's peak deceleration is 1.5 dv / T; choose T so it equals b_max
    Tq = 1.5 * dv / b_max
    Dq = (v0 + v1) / 2 * Tq
    c = quintic(0.0, v0, 0.0, Dq, v1, 0.0, Tq)
    tq = np.linspace(0, Tq, 400)
    out["quintic"] = dict(T=Tq, D=Dq, coef=c, t=tq, v=poly_eval(c, tq, 1),
                          a=poly_eval(c, tq, 2), jerk=poly_eval(c, tq, 3),
                          peak_a=float(-poly_eval(c, tq, 2).min()),
                          peak_jerk=float(np.abs(poly_eval(c, tq, 3)).max()))
    cc = cubic(0.0, v0, out["constant"]["D"], v1, T)
    out["cubic"] = dict(T=T, coef=cc)
    return out


# ------------------------------------------------------------------ drawing
_AV_IMG = None


def draw_av(ax, x, y, theta, scale=1.0, length=4.5, zorder=5):
    """Draw the AV from above (av_top.png, front up) with its base link (the middle of
    the rear axle, 0.9 m in from the rear bumper of the 4.5 m car) on (x, y), turned
    to heading theta (rad)."""
    global _AV_IMG
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    if _AV_IMG is None:
        _AV_IMG = plt.imread(HERE / "av_top.png")
    h, w = _AV_IMG.shape[:2]
    Lc = length * scale
    W = Lc * w / h
    im = ax.imshow(_AV_IMG, extent=(-W / 2, W / 2, -0.9 * scale, Lc - 0.9 * scale),
                   zorder=zorder, interpolation="antialiased")
    im.set_transform(Affine2D().rotate(theta - np.pi / 2).translate(x, y) + ax.transData)
    return im
