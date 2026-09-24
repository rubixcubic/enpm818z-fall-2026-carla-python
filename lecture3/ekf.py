# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""Script 2. The Extended Kalman Filter: the car turns.

The state is [x, y, theta]. The wheels give speed v and the IMU gives turn
rate omega: that is the control input u = [v, omega]. Moving the car needs a
cosine and a sine, so the model is a FUNCTION f, not a matrix, and the EKF
uses its tangent (the Jacobian) wherever a covariance is involved.

    python3 ekf.py                        # the EKF
    python3 ekf.py --check-jacobian       # test the Jacobian first (always!)
    python3 ekf.py --heading-error 60     # start 60 degrees wrong
    python3 ekf.py --data l3_drive_multipath.csv            # 4 bad fixes
    python3 ekf.py --data l3_drive_multipath.csv --gate     # throw them out
"""
import argparse
import numpy as np
from l3common import (load, report, plot, wrap, DT,
                      SIGMA_V, SIGMA_OMEGA, SIGMA_GNSS, NIS_GATE)


def f(s, u):
    """Model 1: one step of driving. Same rule the dataset was made with."""
    x, y, th = s
    v, om = u
    return np.array([x + v * DT * np.cos(th),
                     y + v * DT * np.sin(th),
                     th + om * DT])


def jacobian_f(s, u):
    """The table of slopes of f: nudge input j, how much does output i move?"""
    th = s[2]
    v = u[0]
    return np.array([[1, 0, -v * DT * np.sin(th)],
                     [0, 1,  v * DT * np.cos(th)],
                     [0, 0,  1]])


def check_jacobian():
    """Nudge each input a little and compare with the formula."""
    s, u, eps = np.array([3.0, -2.0, 0.7]), np.array([10.0, 0.1]), 1e-6
    numeric = np.zeros((3, 3))
    for j in range(3):
        ds = np.zeros(3); ds[j] = eps
        numeric[:, j] = (f(s + ds, u) - f(s - ds, u)) / (2 * eps)
    worst = np.abs(numeric - jacobian_f(s, u)).max()
    verdict = "OK" if worst < 1e-6 else "WRONG, fix jacobian_f"
    print(f"Jacobian check: largest difference {worst:.2e}  ->  {verdict}")


def process_noise(s, u):
    """Q: the wheel and IMU noise, pushed through the same tangent."""
    th = s[2]
    G = np.array([[DT * np.cos(th), 0],
                  [DT * np.sin(th), 0],
                  [0, DT]])
    M = np.diag([SIGMA_V ** 2, SIGMA_OMEGA ** 2])
    return G @ M @ G.T + np.diag([0.01, 0.01, 1e-5])   # small floor: f is not perfect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="l3_drive.csv")
    ap.add_argument("--heading-error", type=float, default=0.0,
                    help="add this many degrees of error to the start heading")
    ap.add_argument("--gate", action="store_true",
                    help="skip any fix whose NIS is above 9.21")
    ap.add_argument("--check-jacobian", action="store_true")
    args = ap.parse_args()
    if args.check_jacobian:
        check_jacobian()
        return
    d = load(args.data)

    H = np.array([[1, 0, 0], [0, 1, 0]], float)   # GNSS sees x and y only
    R = SIGMA_GNSS ** 2 * np.eye(2)

    # Start: first fix, heading from the first two fixes, big P
    th0 = np.arctan2(d["gnss_y"][10] - d["gnss_y"][0], d["gnss_x"][10] - d["gnss_x"][0])
    s = np.array([d["gnss_x"][0], d["gnss_y"][0], th0 + np.deg2rad(args.heading_error)])
    P = np.diag([SIGMA_GNSS**2, SIGMA_GNSS**2, np.deg2rad(20) ** 2])

    est, nis, nis_t, rejected = [], [], [], 0
    for k in range(len(d["t"])):
        if k > 0:
            u = np.array([d["v_meas"][k-1], d["omega_meas"][k-1]])
            Fk = jacobian_f(s, u)               # tangent at the current guess
            s = f(s, u)                         # the guess goes through the REAL f
            s[2] = wrap(s[2])
            P = Fk @ P @ Fk.T + process_noise(s, u)
        if not np.isnan(d["gnss_x"][k]):
            z = np.array([d["gnss_x"][k], d["gnss_y"][k]])
            nu = z - H @ s
            S = H @ P @ H.T + R
            eps = nu @ np.linalg.inv(S) @ nu
            if args.gate and eps > NIS_GATE:
                rejected += 1                   # skip it, run on prediction
            else:
                K = P @ H.T @ np.linalg.inv(S)
                s = s + K @ nu
                s[2] = wrap(s[2])               # wrap AFTER the update too
                P = (np.eye(3) - K @ H) @ P
            nis.append(eps)
            nis_t.append(d["t"][k])
        est.append(s[:2].copy())

    est = np.array(est)
    report("EKF", d, est, nis)
    if args.gate:
        print(f"{'':<18} gate rejected {rejected} fixes")
    plot("EKF", d, est, nis, nis_t)


if __name__ == "__main__":
    main()
