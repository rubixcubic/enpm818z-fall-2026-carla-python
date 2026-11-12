"""L10: change the road behavior tree and the road, and see which action the tick picks.

The tree and its four node types come from behavior_tree.py (same folder): a Fallback root
tries its branches from the left (Stop, Pass, Follow), then Lane keep. This window lets you
change the tree and the world, and redraws one tick after every change:

  - the tree: turn a branch on or off, move the selected branch earlier or later, and set
    each condition's threshold (sliders on the left);
  - the road: the stop line, the car ahead (distance, and how long it has driven slowly),
    and a car in the left lane (sliders on the right);
  - the result: each node colored by the status it returned (green Success, red Failure,
    yellow Running, white not ticked), the road from above, and the action that runs.

    python3 behavior_tree_gui.py                    # opens the window
    python3 behavior_tree_gui.py --snapshot bt.png  # saves one picture of the start, no window

Only matplotlib and numpy, as l10_handson.py. Every number is chosen for this example.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from behavior_tree import (FAILURE, RUNNING, SUCCESS, Action, Condition, Fallback,
                           Sequence)

COLOR = {SUCCESS: "#bfe5c4", FAILURE: "#f2b8b0", RUNNING: "#fbe7a1", None: "white"}
FAR = float("inf")
LANE = 3.5                      # m, lane width


# --------------------------------------------- the node types, remembering their status
def traced(cls):
    """A subclass of a node type that remembers the status it returned at the last tick."""
    class Traced(cls):
        status = None

        def tick(self, world):
            self.status = super().tick(world)
            return self.status
    Traced.__name__ = cls.__name__
    return Traced


TCondition, TAction, TSequence, TFallback = (traced(c) for c in
                                             (Condition, Action, Sequence, Fallback))


# -------------------------------------------------------------- the tree, from settings
BRANCHES = ["Stop", "Pass", "Follow"]


def build(order, enabled, th):
    """The root Fallback with the enabled branches in the given order, then Lane keep.
    th holds the thresholds; each branch is a Sequence of its conditions and its action."""
    make = {
        "Stop": lambda: TSequence(
            TCondition(1, f"stop line within {th['stop']:.0f} m",
                       lambda w: w["stop_line_m"] <= th["stop"]),
            TAction("Stop at the line")),
        "Pass": lambda: TSequence(
            TCondition(2, f"car ahead slow for {th['slow']:.1f} s",
                       lambda w: w["slow_for_s"] >= th["slow"]),
            TCondition(3, f"left lane free ({th['behind']:.0f} m behind, {th['ahead']:.0f} m ahead)",
                       lambda w: w["left_free"]),
            TAction("Change lane")),
        "Follow": lambda: TSequence(
            TCondition(4, f"car ahead within {th['lead']:.0f} m",
                       lambda w: w["lead_m"] <= th["lead"]),
            TAction("Follow")),
    }
    branches = [make[b]() for b in order if enabled[b]]
    return TFallback(*branches, TAction("Lane keep"))


def world_from(sv, th):
    """What the AV senses, from the road sliders. A slider at its right end means none."""
    stop = sv["stop"] if sv["stop"] < 100 else FAR
    lead = sv["lead"] if sv["lead"] < 100 else FAR
    left = sv["left"]                             # m along the road, behind < 0
    has_left = sv["has_left"]
    free = not (has_left and -th["behind"] <= left <= th["ahead"])
    return dict(stop_line_m=stop, lead_m=lead, slow_for_s=sv["slow"] if lead < FAR else 0.0,
                left_free=free, checked=[], action=None)


# ------------------------------------------------------------------------ drawing
def node_label(n):
    if isinstance(n, Condition):
        return f"{n.tag}"
    if isinstance(n, Action):
        return n.name
    return "?" if isinstance(n, Fallback) else "→"


def draw_tree(ax, root):
    """The root at the top, its branches below, each branch's children in a row under it."""
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, n, w=0.9, h=0.5):
        if isinstance(n, Condition):
            ax.add_patch(Ellipse((x, y), w, h, fc=COLOR[n.status], ec="0.25", lw=1.2))
        else:
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03",
                                        fc=COLOR[n.status], ec="0.25", lw=1.2))
        ax.text(x, y, node_label(n), ha="center", va="center", fontsize=9 if
                isinstance(n, Action) else 12, fontweight="bold" if
                not isinstance(n, Action) else "normal", wrap=True)

    box(5, 3.7, root, w=0.6)
    kids = root.children
    xs = [10 * (i + 0.5) / len(kids) for i in range(len(kids))]
    for x, k in zip(xs, kids):
        ax.plot([5, x], [3.45, 2.75], color="0.4", lw=1)
        if isinstance(k, Sequence):
            box(x, 2.5, k, w=0.6)
            sub = k.children
            sx = [x + (j - (len(sub) - 1) / 2) * 1.0 for j in range(len(sub))]
            for x2, s in zip(sx, sub):
                ax.plot([x, x2], [2.25, 1.55], color="0.4", lw=1)
                box(x2, 1.3, s, w=0.95 if isinstance(s, Action) else 0.7)
        else:
            box(x, 2.5, k, w=1.2)
    ax.text(0.05, 0.25, "green Success, red Failure, yellow Running, white not ticked",
            fontsize=8, color="0.3")


def draw_road(ax, world, sv, th):
    """The road from above, to scale: the AV (black) at 0 in the right lane, driving right."""
    ax.clear()
    x0, x1 = -45, 105
    ax.add_patch(Rectangle((x0, -LANE / 2), x1 - x0, LANE, fc="#FFF3BF", ec="none"))
    ax.add_patch(Rectangle((x0, LANE / 2), x1 - x0, LANE, fc="#EAE0F5", ec="none"))
    for y, ls in ((-LANE / 2, "-"), (LANE / 2, "--"), (1.5 * LANE, "-")):
        ax.plot([x0, x1], [y, y], color="0.5", ls=ls, lw=1)
    # the stretch condition 3 checks
    ax.add_patch(Rectangle((-th["behind"], LANE / 2 + 0.2), th["behind"] + th["ahead"],
                           LANE - 0.4, fill=False, ec="0.2", ls=(0, (4, 3)), lw=1.2))
    car = dict(width=4.5, height=1.9)
    ax.add_patch(Rectangle((-0.9, -0.95), **car, fc="black"))
    if world["lead_m"] < FAR:
        ax.add_patch(Rectangle((world["lead_m"] - 0.9, -0.95), **car, fc="#2D6CA2"))
    if sv["has_left"]:
        ax.add_patch(Rectangle((sv["left"] - 0.9, LANE - 0.95), **car, fc="#2D6CA2"))
    if world["stop_line_m"] < FAR:
        ax.plot([world["stop_line_m"]] * 2, [-LANE / 2, LANE / 2], color="#B03A2E", lw=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-LANE, 2 * LANE)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xlabel("distance along the road from the AV (m)")
    ax.set_title(f"The tick runs: {world['action']}", fontsize=13, fontweight="bold")


# --------------------------------------------------------------------------- window
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", help="save one picture to this file and exit")
    a = ap.parse_args(argv)
    if a.snapshot:
        plt.switch_backend("Agg")

    fig = plt.figure(figsize=(13, 8))
    fig.canvas.manager.set_window_title("L10 behavior tree")
    ax_tree = fig.add_axes([0.05, 0.50, 0.90, 0.45])
    ax_road = fig.add_axes([0.30, 0.30, 0.68, 0.18])

    state = dict(order=list(BRANCHES), enabled={b: True for b in BRANCHES}, sel="Stop")
    # thresholds (the tree) on the left, the road on the right; starting values: tick 1
    # of the slides (a slow car 25 m ahead for 3 s, the left lane free)
    th_sl = {k: Slider(fig.add_axes([0.15, y, 0.12, 0.025]), lab, lo, hi, valinit=v)
             for k, lab, lo, hi, v, y in [
                 ("stop", "1: stop line (m)", 5, 60, 30, 0.22),
                 ("slow", "2: slow for (s)", 0.5, 8, 3, 0.18),
                 ("behind", "3: behind (m)", 0, 45, 30, 0.14),
                 ("ahead", "3: ahead (m)", 0, 60, 40, 0.10),
                 ("lead", "4: car ahead (m)", 5, 80, 40, 0.06)]}
    road_sl = {k: Slider(fig.add_axes([0.81, y, 0.13, 0.025]), lab, lo, hi, valinit=v)
               for k, lab, lo, hi, v, y in [
                   ("stop", "stop line (m; 100 = none)", 5, 100, 100, 0.22),
                   ("lead", "car ahead (m; 100 = none)", 5, 100, 25, 0.18),
                   ("slow", "it has been slow for (s)", 0, 8, 3, 0.14),
                   ("left", "left-lane car (m)", -40, 60, -20, 0.10)]}
    cb_left = CheckButtons(fig.add_axes([0.70, 0.02, 0.18, 0.05]), ["a car in the left lane"],
                           [False])
    cb_on = CheckButtons(fig.add_axes([0.33, 0.10, 0.08, 0.12]), BRANCHES, [True] * 3)
    rb = RadioButtons(fig.add_axes([0.42, 0.10, 0.08, 0.12]), BRANCHES)
    b_up = Button(fig.add_axes([0.51, 0.17, 0.07, 0.04]), "earlier")
    b_dn = Button(fig.add_axes([0.51, 0.11, 0.07, 0.04]), "later")
    fig.text(0.02, 0.26, "The tree: each condition's threshold", fontweight="bold")
    fig.text(0.33, 0.235, "Branches: on, and order", fontweight="bold")
    fig.text(0.62, 0.26, "The road", fontweight="bold")

    def redraw(_=None):
        th = {k: s.val for k, s in th_sl.items()}
        sv = {k: s.val for k, s in road_sl.items()}
        sv["has_left"] = cb_left.get_status()[0]
        tree = build(state["order"], state["enabled"], th)
        world = world_from(sv, th)
        tree.tick(world)
        draw_tree(ax_tree, tree)
        ax_tree.set_title("Order from the left: " + ", ".join(
            b for b in state["order"] if state["enabled"][b]) + ", Lane keep", fontsize=11)
        draw_road(ax_road, world, sv, th)
        fig.canvas.draw_idle()

    def on_toggle(label):
        state["enabled"][label] = not state["enabled"][label]
        redraw()

    def move(step):
        o, i = state["order"], state["order"].index(state["sel"])
        j = min(max(i + step, 0), len(o) - 1)
        o[i], o[j] = o[j], o[i]
        redraw()

    for s in list(th_sl.values()) + list(road_sl.values()):
        s.on_changed(redraw)
    cb_left.on_clicked(redraw)
    cb_on.on_clicked(on_toggle)
    rb.on_clicked(lambda lab: state.update(sel=lab))
    b_up.on_clicked(lambda _: move(-1))
    b_dn.on_clicked(lambda _: move(+1))
    redraw()
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=110)
        print("wrote", a.snapshot)
    else:
        plt.show()


if __name__ == "__main__":
    main()
