# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""Small helpers shared by the four filter scripts. Nothing here is a filter."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# An honest filter keeps the NIS of a 2-number GNSS fix inside this band
# 95% of the time (chi-square, 2 degrees of freedom). 9.21 is the 99% gate.
NIS_LO, NIS_HI, NIS_GATE = 0.051, 7.378, 9.21

DT = 0.1
SIGMA_V, SIGMA_OMEGA, SIGMA_GNSS = 0.3, 0.02, 2.0   # as in make_dataset.py


def load(name="l3_drive.csv"):
    """Read the CSV into a dict of numpy arrays. Missing GNSS becomes nan."""
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        raise SystemExit(f"{name} not found. Run:  python3 make_dataset.py")
    raw = np.genfromtxt(path, delimiter=",", names=True)
    return {k: raw[k] for k in raw.dtype.names}


def wrap(a):
    """Put an angle, or an angle difference, into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def report(name, d, est, nis):
    """Print the two numbers that matter: how wrong, and how honest."""
    err = np.hypot(est[:, 0] - d["x"], est[:, 1] - d["y"])
    rmse = np.sqrt(np.mean(err ** 2))
    print(f"{name:<18} position error (RMSE) {rmse:5.2f} m   worst {err.max():5.2f} m")
    if len(nis):
        nis = np.asarray(nis)
        inside = np.mean((nis > NIS_LO) & (nis < NIS_HI))
        print(f"{'':<18} NIS mean {nis.mean():5.2f} (honest: about 2)   "
              f"inside band {100 * inside:3.0f}% (honest: about 95%)")
    return rmse


def plot(name, d, est, nis=(), nis_t=(), out=None):
    """Left: the path. Right: the NIS against its band. Saves a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2 if len(nis) else 1, figsize=(12, 5), squeeze=False)
    ax = axes[0, 0]
    ax.plot(d["x"], d["y"], "k-", lw=2, label="truth")
    ax.plot(d["gnss_x"], d["gnss_y"], ".", color="tab:orange", label="GNSS fixes")
    ax.plot(est[:, 0], est[:, 1], "-", color="tab:blue", lw=1.6, label=name)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(); ax.set_title("Where the filter thinks the car is")
    if len(nis):
        ax = axes[0, 1]
        ax.axhspan(NIS_LO, NIS_HI, color="tab:green", alpha=0.15, label="honest band")
        ax.axhline(NIS_GATE, color="tab:red", ls="--", label="99% gate (9.21)")
        ax.plot(nis_t, nis, "o-", ms=3, color="tab:blue", label="NIS")
        ax.set_yscale("log"); ax.set_xlabel("t (s)"); ax.legend()
        ax.set_title("Is the covariance honest?")
    fig.tight_layout()
    out = out or os.path.join(HERE, name.lower().replace(" ", "_") + ".png")
    fig.savefig(out, dpi=120)
    print(f"{'':<18} saved {os.path.basename(out)}")
