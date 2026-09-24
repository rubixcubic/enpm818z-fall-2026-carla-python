# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""A particle filter for a car that does not know where it is in a tunnel.

The car's computer restarted inside a 700 m tunnel. It knows its speed (the
wheels) but has no idea how far along the tunnel it is. The camera recognizes
two kinds of landmarks and matches them against the HD map:

    lights        one every 25 m, all identical
    SOS niches    emergency phone niches, only five, spaced irregularly

A light match says "you are 30 m before SOME light": that fits 23 places at
once, 25 m apart. A niche match fits five. No bell curve can say that, so the
belief is a crowd of particles, each one a guess at the distance along the
tunnel, s. Every step:

    predict:   move every particle by its own noisy copy of the wheel speed
    weigh:     when the camera matches a landmark, score each particle by how
               well it explains the reading
    resample:  when too few particles carry the weight, redraw the crowd

Watch the crowd split into a comb, then a handful of clusters, then one.

Usage
    python3 pf_tunnel.py --make-csv               # write pf_drive.csv and tunnel_map.csv
    python3 pf_tunnel.py                          # live window
    python3 pf_tunnel.py --html pf_tunnel.html    # web page instead
    python3 pf_tunnel.py --snapshot frame.png --frame 12
    python3 pf_tunnel.py --trials 50              # how often does it lock on correctly?

Needs numpy and matplotlib only.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.1                   # s per step
TUNNEL = 700.0             # m
LIGHT_SPACING = 25.0       # m
NICHES = np.array([95.0, 245.0, 320.0, 470.0, 610.0])   # m, irregular on purpose
MATCH_AT = 30.0            # m: a landmark is matched when it is this far ahead
SIGMA_WHEEL = 0.3          # m/s, wheel speed noise in the data
SIGMA_CAM = 1.0            # m, camera distance to a matched landmark
START = 40.0               # m, where the car really is when the computer restarts
DURATION = 45.0            # s


# ---------------------------------------------------------------------------
# 1. The HD map and the dataset
# ---------------------------------------------------------------------------
def hd_map():
    lights = np.arange(LIGHT_SPACING, TUNNEL, LIGHT_SPACING)
    lights = lights[np.min(np.abs(lights[:, None] - NICHES[None, :]), axis=1) > 5]
    return {"light": lights, "niche": NICHES.copy()}


def make_csv(path, map_path, seed=818):
    rng = np.random.default_rng(seed)
    n = int(round(DURATION / DT)) + 1
    t = np.arange(n) * DT
    speed = 12.0 + 1.5 * np.sin(2 * np.pi * t / 20.0)        # the car really drives
    s = START + np.concatenate([[0.0], np.cumsum(speed[:-1] * DT)])
    wheel = speed + rng.normal(0, SIGMA_WHEEL, n)
    mp = hd_map()
    kind, dist = [""] * n, np.full(n, np.nan)
    done = set()
    for k in range(n):
        for name in ("niche", "light"):
            for j, pos in enumerate(mp[name]):
                if (name, j) in done:
                    continue
                if 0 < pos - s[k] <= MATCH_AT:
                    done.add((name, j))
                    if not kind[k]:                           # one match per frame
                        kind[k] = name
                        dist[k] = pos - s[k] + rng.normal(0, SIGMA_CAM)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "true_s", "wheel_v", "match", "match_dist"])
        for k in range(n):
            w.writerow([f"{t[k]:.1f}", f"{s[k]:.3f}", f"{wheel[k]:.4f}", kind[k],
                        f"{dist[k]:.3f}" if kind[k] else ""])
    with open(map_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "s"])
        for name in ("light", "niche"):
            for pos in mp[name]:
                w.writerow([name, f"{pos:.1f}"])
    nm = sum(1 for x in kind if x)
    print(f"wrote {path}: {n} rows, {nm} matches ({kind.count('niche')} niches); "
          f"wrote {map_path}")


def load(path, map_path):
    d = {"t": [], "true_s": [], "wheel_v": [], "match": [], "match_dist": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            for key in ("t", "true_s", "wheel_v"):
                d[key].append(float(row[key]))
            d["match"].append(row["match"])
            d["match_dist"].append(float(row["match_dist"]) if row["match"] else np.nan)
    d = {k: (np.array(v) if k != "match" else v) for k, v in d.items()}
    mp = {"light": [], "niche": []}
    with open(map_path) as f:
        for row in csv.DictReader(f):
            mp[row["kind"]].append(float(row["s"]))
    return d, {k: np.array(v) for k, v in mp.items()}


# ---------------------------------------------------------------------------
# 2. The particle filter
# ---------------------------------------------------------------------------
def likelihood(parts, kind, dist, mp, sigma_cam):
    """How well each particle explains "a <kind> is <dist> m ahead".

    For a particle at s, every landmark of that kind in the map predicts a
    reading of (landmark - s). The reading could have come from any of them, so
    add up their bell curves. A small floor keeps a particle alive if the
    camera ever mistakes something for a landmark.
    """
    ahead = mp[kind][None, :] - parts[:, None]              # (N, landmarks)
    score = np.exp(-0.5 * ((ahead - dist) / sigma_cam) ** 2).sum(axis=1)
    return score + 1e-3


def clusters(parts, w, width=6.0):
    """Group the crowd into clusters: runs of 2 m bins holding weight.

    Returns a list of (weight, weighted mean) sorted heaviest first.
    """
    bins = np.arange(-20, TUNNEL + 22, 2.0)
    hist, _ = np.histogram(parts, bins=bins, weights=w)
    occupied = hist > 1e-3
    out, k = [], 0
    while k < len(hist):
        if occupied[k]:
            j = k
            while j + 1 < len(hist) and (occupied[j + 1] or
                                         (j + 2 < len(hist) and occupied[j + 2])):
                j += 1
            lo, hi = bins[k], bins[j + 1]
            m = (parts >= lo) & (parts < hi)
            if w[m].sum() > 0:
                out.append((w[m].sum(), np.average(parts[m], weights=w[m])))
            k = j + 1
        k += 1
    return sorted(out, reverse=True)


def run_pf(d, mp, n_particles=2000, sigma_cam=SIGMA_CAM, sigma_wheel=SIGMA_WHEEL,
           seed=1, keep_every=1):
    """Run the particle filter. Returns a dict of histories for plotting."""
    rng = np.random.default_rng(seed)
    N = int(n_particles)
    parts = rng.uniform(0, TUNNEL, N)       # no idea where we are: spread everywhere
    w = np.full(N, 1.0 / N)
    n = len(d["t"])
    hist = {"mean": np.zeros(n), "best": np.zeros(n), "n_clusters": np.zeros(n, int),
            "n_eff": np.zeros(n), "parts": [], "w": []}
    for k in range(n):
        if k > 0:
            # PREDICT: every particle moves by its own draw of the wheel noise.
            v = d["wheel_v"][k - 1] + rng.normal(0, sigma_wheel, N)
            parts = parts + v * DT
        if d["match"][k]:
            # WEIGH: the update. Nothing moves; the crowd gets scored.
            w = w * likelihood(parts, d["match"][k], d["match_dist"][k], mp, sigma_cam)
            w = w / w.sum()
            # RESAMPLE when the effective number of particles gets too small.
            if 1.0 / np.sum(w**2) < N / 2:
                pos = (rng.random() + np.arange(N)) / N          # systematic resampling
                idx = np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)
                parts, w = parts[idx], np.full(N, 1.0 / N)
        cl = clusters(parts, w)
        hist["mean"][k] = np.sum(w * parts)        # the one-bell-curve answer
        hist["best"][k] = cl[0][1] if cl else hist["mean"][k]
        hist["n_clusters"][k] = sum(1 for c in cl if c[0] > 0.02)
        hist["n_eff"][k] = 1.0 / np.sum(w**2)
        if k % keep_every == 0:
            hist["parts"].append(parts.copy()); hist["w"].append(w.copy())
    return hist


def lock_time(d, hist, tol=3.0):
    """First time after which the crowd is in ONE place, within tol of the truth.

    Returns None if it never settles there: either still undecided at the end,
    or, worse, settled on the wrong place (a quiet collapse).
    """
    ok = (np.abs(hist["best"] - d["true_s"]) <= tol) & (hist["n_clusters"] == 1)
    if not ok[-1]:
        return None
    bad = np.where(~ok)[0]
    return d["t"][bad[-1] + 1] if len(bad) else d["t"][0]


# ---------------------------------------------------------------------------
# 3. The live window
# ---------------------------------------------------------------------------

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

def build_figure(d, mp, n_particles, sigma_cam, interactive=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    t = d["t"]
    n = len(t)
    state = {"k": 0, "playing": True, "N": n_particles, "sc": sigma_cam, "seed": 1}
    res = {}
    SHOW = 1500                                    # particles drawn, at most
    rng_draw = np.random.default_rng(0)

    fig = plt.figure(figsize=(17.5, 8.4))
    if interactive:
        fig.canvas.manager.set_window_title("Particle filter in a tunnel")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.3, 1.0], hspace=0.6, wspace=0.18,
                          left=0.04, right=0.725, top=0.86, bottom=0.14)

    # --- top: the whole tunnel, unrolled, with the crowd
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_title("Where am I? The whole 700 m tunnel, unrolled", loc="left",
                     fontsize=12, fontweight="bold", pad=24)
    ax_top.axhspan(-1, 1, color="#E9ECEF", zorder=0)
    ax_top.scatter(mp["light"], np.full(len(mp["light"]), 1.25), marker="|", s=90,
                   color="#7A828C", label="light (all identical)")
    ax_top.scatter(mp["niche"], np.full(len(mp["niche"]), 1.25), marker="s", s=70,
                   color="#1E9E74", label="SOS niche")
    crowd = ax_top.scatter([], [], s=4, color="#2D6CA2", alpha=0.35, label="particles")
    true_dot, = ax_top.plot([], [], "s", color="#1E2939", ms=10, label="true car")
    mean_dot, = ax_top.plot([], [], "v", color="#D9694A", ms=11,
                            label="weighted average (one bell curve)")
    best_dot, = ax_top.plot([], [], "^", color="#2D6CA2", ms=11, label="heaviest cluster")
    ax_top.set_xlim(-5, TUNNEL + 5); ax_top.set_ylim(-1.6, 1.7); ax_top.set_yticks([])
    ax_top.set_xlabel("distance along the tunnel, s (m)")
    ax_top.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=6, frameon=False,
                  fontsize=9, handletextpad=0.3, columnspacing=1.0)
    info = fig.text(0.725, 0.925, "", ha="right", fontsize=10, family="monospace")

    # --- middle: the belief right now, as a histogram
    ax_h = fig.add_subplot(gs[1, :])
    ax_h.set_title("the belief now: how much weight sits at each place", loc="left",
                   fontsize=11)
    bins = np.arange(0, TUNNEL + 2, 2.0)
    bars = ax_h.bar(bins[:-1], np.zeros(len(bins) - 1), width=2.0, align="edge",
                    color="#2D6CA2")
    true_line = ax_h.axvline(0, color="#1E2939", lw=1.5)
    ax_h.set_xlim(-5, TUNNEL + 5); ax_h.set_ylabel("weight")

    # --- bottom: errors, and how many clusters
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.set_title("error (m): heaviest cluster vs weighted average", loc="left", fontsize=11)
    l_best, = ax_e.plot([], [], color="#2D6CA2", lw=2, label="heaviest cluster")
    l_mean, = ax_e.plot([], [], color="#D9694A", lw=2, label="weighted average")
    ax_e.set_yscale("symlog", linthresh=5); ax_e.set_xlim(t[0], t[-1])
    ax_e.legend(fontsize=9, frameon=False); ax_e.set_xlabel("time (s)")
    ax_c = fig.add_subplot(gs[2, 1])
    ax_c.set_title("how many places the crowd is in (clusters)", loc="left", fontsize=11)
    l_cl, = ax_c.step([], [], color="#1E2939", lw=2, where="post")
    ax_c.set_yscale("log"); ax_c.set_xlim(t[0], t[-1]); ax_c.set_xlabel("time (s)")
    for a in (ax_e, ax_c):
        for k in range(n):
            if d["match"][k] == "niche":
                a.axvline(t[k], color="#1E9E74", lw=1.2, alpha=0.6)
    nows = [a.axvline(0, color="#1E2939", lw=1) for a in (ax_e, ax_c)]
    stats = fig.text(0.02, 0.955, "", ha="left", fontsize=12, fontweight="bold")
    add_help_panel(fig, [
        ("What you see", [
            ("Top: the 700 m tunnel, unrolled", [
                "gray ticks: the lights, all identical",
                "green squares: the five SOS niches",
                "blue dots: the particles",
                "dark square: the true car",
                "orange triangle: the weighted average (one bell curve)",
                "blue triangle: the heaviest cluster",
            ]),
            ("Middle: the belief now", [
                "how much weight sits at each place",
                "several tall bars: several places still possible",
            ]),
            ("Bottom left: the error of each answer", [
                "log scale",
                "the average stays far off until the crowd is in one place",
            ]),
            ("Bottom right: how many places", [
                "about 23 at the start, one per light",
                "4 after the first niche, 1 by about 21 s",
                "green lines: niche matches",
            ]),
        ]),
        ("Try this", [
            ("Fewer particles (100)", [
                "press New draw a few times",
                "sometimes it settles on the wrong place, with no warning",
            ]),
            ("More particles (2,000 or 5,000)", [
                "right every time, but each step costs more",
            ]),
            ("Lower R (camera sigma 0.2 m)", [
                "good particles are thrown away; it fails more often",
            ]),
            ("Raise R", [
                "wider clusters, slower to settle, rarely wrong",
            ]),
        ]),
    ])

    def rerun():
        res["h"] = run_pf(d, mp, state["N"], state["sc"], seed=state["seed"])
        h = res["h"]
        l_best.set_data(t, np.abs(h["best"] - d["true_s"]))
        l_mean.set_data(t, np.abs(h["mean"] - d["true_s"]))
        ax_e.set_ylim(0, 400)
        l_cl.set_data(t, np.maximum(h["n_clusters"], 1))
        ax_c.set_ylim(0.8, max(40, h["n_clusters"].max() * 1.3))
        lt = lock_time(d, h)
        if lt is not None:
            lock = f"one place, the right one, from t = {lt:.1f} s"
        elif h["n_clusters"][-1] == 1:
            lock = "ONE PLACE, THE WRONG ONE: a quiet collapse"
        else:
            lock = "still undecided at the end"
        stats.set_text(f"{int(state['N'])} particles, camera sigma {state['sc']:.1f} m: "
                       f"{lock}")
        stats.set_color("#1E2939" if lt is not None else "#D9694A")

    rerun()

    def draw(k):
        k = int(k)
        h = res["h"]
        parts, w = h["parts"][k], h["w"][k]
        show = parts if len(parts) <= SHOW else parts[rng_draw.choice(len(parts), SHOW,
                                                                       replace=False)]
        crowd.set_offsets(np.column_stack([show, rng_draw.uniform(-0.8, 0.8, len(show))]))
        s = d["true_s"][k]
        true_dot.set_data([s], [0]); mean_dot.set_data([h["mean"][k]], [-1.25])
        best_dot.set_data([h["best"][k]], [-1.25])
        hw, _ = np.histogram(parts, bins=bins, weights=w)
        for b, v in zip(bars, hw):
            b.set_height(v)
        ax_h.set_ylim(0, max(0.02, 1.1 * hw.max()))
        true_line.set_xdata([s, s])
        for ln in nows:
            ln.set_xdata([t[k], t[k]])
        m = d["match"][k]
        seen = f"camera: {m} {d['match_dist'][k]:.1f} m ahead" if m else ""
        info.set_text(f"t {t[k]:4.1f} s   clusters {h['n_clusters'][k]:3d}   "
                      f"effective particles {h['n_eff'][k]:6.0f}   {seen}")
        return crowd, true_dot, mean_dot, best_dot, true_line, info, *bars, *nows

    if not interactive:
        return fig, draw, n

    ax_play = fig.add_axes([0.05, 0.03, 0.06, 0.045])
    ax_rst = fig.add_axes([0.12, 0.03, 0.06, 0.045])
    ax_seed = fig.add_axes([0.19, 0.03, 0.08, 0.045])
    ax_n = fig.add_axes([0.50, 0.055, 0.20, 0.025])
    ax_sc = fig.add_axes([0.50, 0.015, 0.20, 0.025])
    b_play = Button(ax_play, "Pause")
    b_rst = Button(ax_rst, "Restart")
    b_seed = Button(ax_seed, "New draw")
    s_n = Slider(ax_n, "particles (log scale)", 1.3, 4.0,
                 valinit=np.log10(n_particles))
    s_sc = Slider(ax_sc, "R: camera sigma (m)", 0.1, 8.0, valinit=sigma_cam)

    def on_play(_):
        state["playing"] = not state["playing"]
        b_play.label.set_text("Pause" if state["playing"] else "Play")
    def on_rst(_):
        state["k"] = 0
    def on_seed(_):
        state["seed"] += 1; state["k"] = 0
        rerun(); draw(0); fig.canvas.draw_idle()
    def on_change(_):
        state["N"], state["sc"] = int(round(10 ** s_n.val)), s_sc.val
        s_n.valtext.set_text(f"{state['N']}")
        rerun(); draw(state["k"]); fig.canvas.draw_idle()
    b_play.on_clicked(on_play); b_rst.on_clicked(on_rst); b_seed.on_clicked(on_seed)
    s_n.on_changed(on_change); s_sc.on_changed(on_change)
    s_n.valtext.set_text(f"{n_particles}")
    fig._keep = (b_play, b_rst, b_seed, s_n, s_sc)

    def step(_):
        if state["playing"]:
            state["k"] = (state["k"] + 1) % n
        return draw(state["k"])
    return fig, step, n


def trials(d, mp, n_particles, sigma_cam, count):
    got, times = 0, []
    for seed in range(1, count + 1):
        lt = lock_time(d, run_pf(d, mp, n_particles, sigma_cam, seed=seed, keep_every=10**9))
        if lt is not None:
            got += 1; times.append(lt)
    med = f", median {np.median(times):.1f} s" if times else ""
    print(f"{n_particles} particles, camera sigma {sigma_cam:.1f} m: settled on the right "
          f"place in {got} of {count} runs{med}")



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
    ap.add_argument("csv", nargs="?", default=os.path.join(HERE, "pf_drive.csv"))
    ap.add_argument("--map", default=os.path.join(HERE, "tunnel_map.csv"))
    ap.add_argument("--make-csv", action="store_true", help="write the data and the map")
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--camera-sigma", type=float, default=SIGMA_CAM, help="R: camera sigma (m)")
    ap.add_argument("--trials", type=int, help="run this many random draws and count successes")
    ap.add_argument("--html", help="write the animation to this .html file instead")
    ap.add_argument("--snapshot", help="write one still frame to this .png file")
    ap.add_argument("--frame", type=float, default=12.0, help="time (s) for --snapshot")
    a = ap.parse_args()

    if a.make_csv:
        make_csv(a.csv, a.map)
        return
    if not (os.path.exists(a.csv) and os.path.exists(a.map)):
        sys.exit("data not found. Run: python3 pf_tunnel.py --make-csv")
    d, mp = load(a.csv, a.map)
    if a.trials:
        trials(d, mp, a.particles, a.camera_sigma, a.trials)
        return
    h = run_pf(d, mp, a.particles, a.camera_sigma, keep_every=10**9)
    lt = lock_time(d, h)
    print(f"{a.particles} particles, camera sigma {a.camera_sigma:.1f} m: "
          + (f"one place, the right one, from t = {lt:.1f} s, final error "
             f"{abs(h['best'][-1] - d['true_s'][-1]):.2f} m" if lt is not None
             else "never settled on the right place"))

    import matplotlib
    if a.html or a.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    if a.snapshot:
        fig, step, n = build_figure(d, mp, a.particles, a.camera_sigma, interactive=True)
        for _ in range(int(round(a.frame / DT))):
            step(None)
        fig.savefig(a.snapshot, dpi=150)
        print("wrote", a.snapshot)
        return
    if a.html:
        fig, draw, n = build_figure(d, mp, a.particles, a.camera_sigma, interactive=False)
        fig.set_dpi(70)
        write_html(fig, draw, range(0, n, 4), a.html, "Particle filter in a tunnel", fps=8)
        return
    fig, step, n = build_figure(d, mp, a.particles, a.camera_sigma, interactive=True)
    anim = FuncAnimation(fig, step, interval=int(DT * 1000), blit=False,
                         cache_frame_data=False)
    fig._anim = anim
    plt.show()


if __name__ == "__main__":
    main()
