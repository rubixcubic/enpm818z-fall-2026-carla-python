# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""Script 1. The plain Kalman filter, GNSS only.

The state is [px, py, vx, vy]. The motion model is "keep going at the same
velocity" (constant velocity), which is linear, so a plain Kalman filter
works. It ignores the wheels and the IMU entirely: only GNSS comes in.

    python3 kf_cv.py                  # default tuning
    python3 kf_cv.py --sigma-a 0.1    # Q far too small: overconfident
    python3 kf_cv.py --sigma-a 20     # Q too big: underconfident
"""
import argparse
import numpy as np
from l3common import load, report, plot, DT, SIGMA_GNSS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-a", type=float, default=5.0,
                    help="how hard the car might accelerate (m/s^2). Sets Q.")
    ap.add_argument("--data", default="l3_drive.csv")
    args = ap.parse_args()
    d = load(args.data)

    # Model 1: where the state goes on its own.  x_k = F x_{k-1} + w
    F = np.array([[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], float)
    # Q: how wrong "constant velocity" is over one step, if the car can
    # accelerate by about sigma_a. The standard textbook form.
    q = args.sigma_a ** 2
    Q1 = q * np.array([[DT**4 / 4, DT**3 / 2], [DT**3 / 2, DT**2]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1

    # Model 2: what the GNSS should read.  z = H x + v. It sees position only.
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], float)
    R = SIGMA_GNSS ** 2 * np.eye(2)

    # Start: the first fix, zero velocity, and a big P (we really do not know)
    x = np.array([d["gnss_x"][0], d["gnss_y"][0], 0.0, 0.0])
    P = np.diag([SIGMA_GNSS**2, SIGMA_GNSS**2, 15.0**2, 15.0**2])

    est, nis, nis_t = [], [], []
    for k in range(len(d["t"])):
        if k > 0:
            # PREDICT: move the guess, grow the uncertainty
            x = F @ x
            P = F @ P @ F.T + Q
        if not np.isnan(d["gnss_x"][k]):
            # UPDATE: how surprised are we, and how much do we act on it?
            z = np.array([d["gnss_x"][k], d["gnss_y"][k]])
            nu = z - H @ x                      # the surprise
            S = H @ P @ H.T + R                 # how big we expected it to be
            K = P @ H.T @ np.linalg.inv(S)      # the fraction to act on
            x = x + K @ nu
            P = (np.eye(4) - K @ H) @ P
            if k > 50:                          # skip the start-up transient
                nis.append(nu @ np.linalg.inv(S) @ nu)
                nis_t.append(d["t"][k])
        est.append(x[:2].copy())

    est = np.array(est)
    report("Kalman filter", d, est, nis)
    plot("Kalman filter", d, est, nis, nis_t)


if __name__ == "__main__":
    main()
