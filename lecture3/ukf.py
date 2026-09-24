# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""Script 3. The Unscented Kalman Filter: same car, no Jacobian.

Same state [x, y, theta], same f, same data as ekf.py. Instead of a tangent,
it places 2n + 1 = 7 sigma points on the belief, pushes each through the real
f, and rebuilds a mean and covariance from where they land.

    python3 ukf.py
    python3 ukf.py --heading-error 60     # compare with ekf.py at 60 degrees
"""
import argparse
import numpy as np
from l3common import load, report, plot, wrap, SIGMA_GNSS
from ekf import f, process_noise

N, KAPPA = 3, 0.0              # state size, and the spread knob (n + kappa = 3)


def sigma_points(m, P):
    """Step 1, PLACE: the mean, then one step out and back along each axis."""
    L = np.linalg.cholesky((N + KAPPA) * P)   # columns = axes of the ellipse
    pts = [m] + [m + L[:, i] for i in range(N)] + [m - L[:, i] for i in range(N)]
    W = np.full(2 * N + 1, 1.0 / (2 * (N + KAPPA)))
    W[0] = KAPPA / (N + KAPPA)
    return np.array(pts), W


def mean_of(pts, W):
    """Step 3, REBUILD (the mean). Angles are averaged with sin and cos."""
    m = W @ pts
    m[2] = np.arctan2(W @ np.sin(pts[:, 2]), W @ np.cos(pts[:, 2]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="l3_drive.csv")
    ap.add_argument("--heading-error", type=float, default=0.0)
    args = ap.parse_args()
    d = load(args.data)
    R = SIGMA_GNSS ** 2 * np.eye(2)

    th0 = np.arctan2(d["gnss_y"][10] - d["gnss_y"][0], d["gnss_x"][10] - d["gnss_x"][0])
    s = np.array([d["gnss_x"][0], d["gnss_y"][0], th0 + np.deg2rad(args.heading_error)])
    P = np.diag([SIGMA_GNSS**2, SIGMA_GNSS**2, np.deg2rad(20) ** 2])

    est, nis, nis_t = [], [], []
    for k in range(len(d["t"])):
        if k > 0:
            u = np.array([d["v_meas"][k-1], d["omega_meas"][k-1]])
            pts, W = sigma_points(s, P)
            Y = np.array([f(p, u) for p in pts])          # step 2, PUSH
            s = mean_of(Y, W)                             # step 3, REBUILD
            D = Y - s
            D[:, 2] = wrap(D[:, 2])
            P = (W[:, None] * D).T @ D + process_noise(s, u)
        if not np.isnan(d["gnss_x"][k]):
            # the update runs the same recipe through h(x) = [x, y]
            pts, W = sigma_points(s, P)
            Z = pts[:, :2]                                # h of each point
            zhat = W @ Z
            Dz = Z - zhat
            Dx = pts - s
            Dx[:, 2] = wrap(Dx[:, 2])
            S = (W[:, None] * Dz).T @ Dz + R
            C = (W[:, None] * Dx).T @ Dz                  # how state and reading move together
            K = C @ np.linalg.inv(S)
            nu = np.array([d["gnss_x"][k], d["gnss_y"][k]]) - zhat
            s = s + K @ nu
            s[2] = wrap(s[2])
            P = P - K @ S @ K.T
            nis.append(nu @ np.linalg.inv(S) @ nu)
            nis_t.append(d["t"][k])
        est.append(s[:2].copy())

    est = np.array(est)
    report("UKF", d, est, nis)
    plot("UKF", d, est, nis, nis_t)


if __name__ == "__main__":
    main()
