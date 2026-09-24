# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""Make the L3 practice dataset: a car driving a curvy road for 60 seconds.

Writes two CSV files next to this script:

    l3_drive.csv            clean GNSS
    l3_drive_multipath.csv  the same drive, but 4 GNSS fixes are 8 to 12 m off

Every row is one 0.1 s step. Columns:

    t                 time (s)
    x, y, theta       where the car REALLY is (m, m, rad). Ground truth.
    v, omega          what the car REALLY did: speed (m/s), turn rate (rad/s)
    v_meas, omega_meas  what the wheels and the IMU SAY it did (noisy)
    gnss_x, gnss_y    the GNSS fix (m). Empty on steps with no fix (1 Hz).

A real car never knows x, y, theta. Your filter must not read them; they
are only there so you can score it afterwards.
"""
import csv
import os
import numpy as np

DT, T_END = 0.1, 60.0
SIGMA_V, SIGMA_OMEGA = 0.3, 0.02   # wheel speed (m/s), IMU yaw rate (rad/s)
SIGMA_GNSS = 2.0                   # GNSS noise (m), each axis
GNSS_EVERY = 10                    # one fix every 10 steps = 1 Hz
SEED = 818


def turn_rate(t):
    """The road: straight, left bend, straight, right bend, gentle S."""
    if 10 <= t < 18:
        return 0.20
    if 25 <= t < 33:
        return -0.18
    if 40 <= t < 55:
        return 0.12 * np.sin(2 * np.pi * (t - 40) / 15)
    return 0.0


def main():
    rng = np.random.default_rng(SEED)
    n = int(round(T_END / DT)) + 1
    t = np.arange(n) * DT
    v = 10.0 + 2.0 * np.sin(0.1 * t)
    omega = np.array([turn_rate(tk) for tk in t])

    x, y, th = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(1, n):
        x[k] = x[k-1] + v[k-1] * DT * np.cos(th[k-1])
        y[k] = y[k-1] + v[k-1] * DT * np.sin(th[k-1])
        th[k] = th[k-1] + omega[k-1] * DT

    v_meas = v + rng.normal(0, SIGMA_V, n)
    omega_meas = omega + rng.normal(0, SIGMA_OMEGA, n)
    has_fix = (np.arange(n) % GNSS_EVERY == 0)
    gx = np.where(has_fix, x + rng.normal(0, SIGMA_GNSS, n), np.nan)
    gy = np.where(has_fix, y + rng.normal(0, SIGMA_GNSS, n), np.nan)

    # multipath: four fixes that arrive on time, look normal, and are wrong
    gx_mp, gy_mp = gx.copy(), gy.copy()
    for k, (dx, dy) in {150: (9, 4), 280: (-8, 7), 410: (10, -6), 520: (-6, -9)}.items():
        gx_mp[k] += dx
        gy_mp[k] += dy

    here = os.path.dirname(os.path.abspath(__file__))
    for name, gxx, gyy in (("l3_drive.csv", gx, gy),
                           ("l3_drive_multipath.csv", gx_mp, gy_mp)):
        path = os.path.join(here, name)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "theta", "v", "omega",
                        "v_meas", "omega_meas", "gnss_x", "gnss_y"])
            for k in range(n):
                fix = (f"{gxx[k]:.3f}", f"{gyy[k]:.3f}") if has_fix[k] else ("", "")
                w.writerow([f"{t[k]:.1f}", f"{x[k]:.3f}", f"{y[k]:.3f}",
                            f"{th[k]:.5f}", f"{v[k]:.3f}", f"{omega[k]:.5f}",
                            f"{v_meas[k]:.3f}", f"{omega_meas[k]:.5f}", *fix])
        print(f"wrote {name}  ({n} rows, {has_fix.sum()} GNSS fixes)")


if __name__ == "__main__":
    main()
