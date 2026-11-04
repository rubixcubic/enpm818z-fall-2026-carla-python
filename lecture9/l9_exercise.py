"""L9 in-class exercise: a fifth cost term that keeps the AV away from oncoming traffic.

The road has two lanes in each direction. The AV drives in the right lane (d = 0);
the left lane (d = 3.5 m) goes the same way. The left lane's left edge, at
d = 3.5 + 3.5 / 2 = 5.25 m, is the road's center line: past it, traffic comes the
other way.

The four cost terms of the slides do not see that line. Write edge_cost(path) so the
planner pays for coming close to it, the same way the clearance term pays for coming
close to the parked car:

    gap to the edge = EDGE_D - (d + half the AV's width), at the path's worst point
    edge cost       = W_EDGE * how far that gap falls under EDGE_SOFT (0 if it does not)

Then:

    python3 l9_check.py                          # pass or fail
    python3 l9_handson.py --part sample --extra  # the table gets an "extra" column

EDGE_SOFT and W_EDGE are chosen for this exercise, like the other weights.
"""
import numpy as np

import l9common as c

EDGE_D = 5.25          # m: the left lane's left edge, the road's center line
EDGE_SOFT = 1.5        # m: a gap to the edge under this costs (as CLEAR_SOFT)
W_EDGE = 4.0           # per m that the gap falls under EDGE_SOFT (as the clearance weight)
HALF_WIDTH = c.AV["width"] / 2   # 0.95 m: from the AV's center line to its left side


def edge_cost(path):
    """The edge cost of one candidate path.

    path["d"] is a numpy array: the AV's d, in meters, at every point of the path
    (every 0.5 m of s). Use the point where the AV comes closest to the edge.
    Return a float, 0.0 when the AV never comes within EDGE_SOFT of the edge.
    """
    # Write your code here: about three lines.
    raise NotImplementedError("edge_cost is not written yet")
