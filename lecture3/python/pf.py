# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""Script 4. The particle filter: a crowd of guesses.

Same car, same data. The belief is N particles, each a full guess
[x, y, theta] with a weight. Every step: PREDICT (move each particle with its
own random kick), WEIGH (score it against the GNSS fix), RESAMPLE (copy the
heavy ones, drop the light ones).

    python3 pf.py
    python3 pf.py --particles 50       # too few: does it collapse?
    python3 pf.py --jitter 0           # no jitter after resampling
    python3 pf.py --global --particles 5000    # start with no idea at all
"""
import argparse
import numpy as np
from l3common import load, report, plot, wrap, DT, SIGMA_V, SIGMA_OMEGA, SIGMA_GNSS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="l3_drive.csv")
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--jitter", type=float, default=0.3,
                    help="random nudge after resampling (m); 0 turns it off")
    ap.add_argument("--global", dest="global_start", action="store_true",
                    help="spread the first particles over the whole map")
    args = ap.parse_args()
    d = load(args.data)
    rng = np.random.default_rng(1)
    n = args.particles

    # Start
    if args.global_start:
        p = np.column_stack([rng.uniform(d["x"].min() - 50, d["x"].max() + 50, n),
                             rng.uniform(d["y"].min() - 50, d["y"].max() + 50, n),
                             rng.uniform(-np.pi, np.pi, n)])
    else:
        th0 = np.arctan2(d["gnss_y"][10] - d["gnss_y"][0], d["gnss_x"][10] - d["gnss_x"][0])
        p = np.column_stack([rng.normal(d["gnss_x"][0], SIGMA_GNSS, n),
                             rng.normal(d["gnss_y"][0], SIGMA_GNSS, n),
                             rng.normal(th0, np.deg2rad(20), n)])
    w = np.full(n, 1.0 / n)

    est, n_eff = [], []
    for k in range(len(d["t"])):
        if k > 0:
            # PREDICT: every particle gets its own noisy copy of the controls
            v = d["v_meas"][k-1] + rng.normal(0, SIGMA_V, n)
            om = d["omega_meas"][k-1] + rng.normal(0, SIGMA_OMEGA, n)
            p[:, 0] += v * DT * np.cos(p[:, 2])
            p[:, 1] += v * DT * np.sin(p[:, 2])
            p[:, 2] = wrap(p[:, 2] + om * DT)
        if not np.isnan(d["gnss_x"][k]):
            # WEIGH: the GNSS bell curve, evaluated at each particle
            gap2 = (p[:, 0] - d["gnss_x"][k]) ** 2 + (p[:, 1] - d["gnss_y"][k]) ** 2
            w = w * np.exp(-0.5 * gap2 / SIGMA_GNSS ** 2)
            w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
            n_eff.append(1.0 / np.sum(w ** 2))
            # RESAMPLE: systematic, only when the weights have become uneven
            if n_eff[-1] < n / 2:
                edges = np.cumsum(w)
                edges[-1] = 1.0
                idx = np.searchsorted(edges, (rng.random() + np.arange(n)) / n)
                p, w = p[idx].copy(), np.full(n, 1.0 / n)
                # jitter the copies a little, so they do not all stay identical
                p[:, :2] += rng.normal(0, args.jitter, (n, 2))
                p[:, 2] = wrap(p[:, 2] + rng.normal(0, args.jitter / 10, n))
        est.append(w @ p[:, :2])       # one number: the weighted average

    est = np.array(est)
    report("Particle filter", d, est, [])
    print(f"{'':<18} effective particles: min {min(n_eff):.0f}, "
          f"median {np.median(n_eff):.0f} of {n}")
    plot("Particle filter", d, est)


if __name__ == "__main__":
    main()
