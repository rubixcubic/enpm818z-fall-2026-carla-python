"""Check l9_exercise.edge_cost against values measured with a reference solution.

  python3 l9_check.py

Each check builds a path, calls your edge_cost and compares the result with the
expected cost. It prints PASS or FAIL for each, then what your term changes in the
slides' scene.
"""
import sys

import numpy as np

import l9common as c
import l9_exercise as ex

TOL = 1e-6


def straight(d, n=141):
    """A path that holds one d all the way (141 points: 70 m every 0.5 m)."""
    return {"d": np.full(n, float(d))}


# (what the path is, the path, the expected cost with EDGE_SOFT = 1.5 and W_EDGE = 4)
CASES = [
    ("stays in the right lane, d = 0", straight(0.0), 0.0),
    ("ends 2.0 m left, gap to the edge 2.3 m", straight(2.0), 0.0),
    ("left lane's center, d = 3.5, gap 0.8 m", straight(3.5), 2.8),
    ("ends at d = 4.0, gap 0.3 m", straight(4.0), 4.8),
    ("touches the edge, d = 4.3, gap 0", straight(4.3), 6.0),
]


def main():
    if (ex.EDGE_SOFT, ex.W_EDGE) != (1.5, 4.0):
        print(f"Set EDGE_SOFT back to 1.5 and W_EDGE back to 4.0 to run the checks "
              f"(they are {ex.EDGE_SOFT} and {ex.W_EDGE}).")
        return 1
    try:
        got = [float(ex.edge_cost(p)) for _, p, _ in CASES]
    except NotImplementedError as e:
        print(f"Not done yet: {e}. Write edge_cost in l9_exercise.py, then run this again.")
        return 1
    ok = True
    for (what, _, want), g in zip(CASES, got):
        good = abs(g - want) < TOL
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {what}: expected {want:.2f}, got {g:.2f}")
    if not ok:
        print("Some checks failed: compare your formula with the docstring of l9_exercise.py.")
        return 1
    ref = c.load_scene()
    car = c.parked_car(ref)
    out = c.plan(ref, c.SCENE["av_s"], (0.0, 0.0, 0.0), [car], extra=ex.edge_cost)
    p, sc = out[0]
    print(f"All checks pass. With your term, the cheapest path ends at d = {p['d_end']:.1f} m "
          f"over {p['maneuver']:.0f} m: total {sc['total']:.2f}, of which edge {sc['terms']['extra']:.2f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
