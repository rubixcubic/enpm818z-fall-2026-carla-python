"""Export L10's running example: one right turn in Town03, read from the map file.

  /usr/bin/python3 export_curve.py --xodr /path/to/Town03.xodr

Needs the CARLA Python client (it ships with CARLA 0.9.16) but no CARLA server:
carla.Map(name, xodr_text) reads the OpenDRIVE file offline.

The path: 80 m of road 3, the junction lane road 817 lane 1 (a right turn), then 40 m
of road 19, sampled every 0.25 m along the lane center (lanes 3.5 m wide).
CARLA is left-handed (y to the right); y is flipped so x is east and y is north, and
headings turn counterclockwise, as in lecture6. Those map coordinates are kept as
x_map, y_map, heading_map. The lecture works in a local frame: origin at the path's
first point, x along the approach, y to its left (the map turned by 90 degrees,
because the approach drives north), so every figure reads left to right.

Writes l10_curve.npz: s (m along the path), x, y (m, local frame), heading (rad, local
frame), kappa (1/m, positive to the left), the map-frame copies, junction (True on the
turn's lane), the turn's start and end s, and the road and lane of every point.
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEP = 0.25          # m between path points
BEFORE = 80.0        # m of straight before the turn
AFTER = 40.0         # m of straight after it
TOWN = "Town03"
TURN_ROAD, TURN_LANE = 817, 1


def walk(wp, dist, forward):
    """Waypoints every STEP meters from wp, dist meters forward or backward."""
    out, w = [], wp
    for _ in range(int(round(dist / STEP))):
        nxt = w.next(STEP) if forward else w.previous(STEP)
        if not nxt:
            raise RuntimeError("the lane ends before the requested distance")
        w = nxt[0]
        out.append(w)
    return out


def export(xodr):
    import carla
    m = carla.Map(TOWN, Path(xodr).read_text())
    start = [a for a, _ in m.get_topology() if a.road_id == TURN_ROAD and a.lane_id == TURN_LANE][0]
    turn = [start]
    while True:
        nxt = turn[-1].next(STEP)
        if not nxt or nxt[0].road_id != TURN_ROAD:
            break
        turn.append(nxt[0])
    before = walk(start, BEFORE, forward=False)[::-1]
    after = walk(turn[-1], AFTER, forward=True)
    wps = before + turn + after
    xm = np.array([w.transform.location.x for w in wps])
    ym = -np.array([w.transform.location.y for w in wps])         # right-handed
    hm = np.unwrap(np.radians([-w.transform.rotation.yaw for w in wps]))
    # local frame: origin at the first point, x along the first heading, y to its left
    c, s0 = np.cos(hm[0]), np.sin(hm[0])
    x = c * (xm - xm[0]) + s0 * (ym - ym[0])
    y = -s0 * (xm - xm[0]) + c * (ym - ym[0])
    heading = hm - hm[0]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    kappa = np.gradient(heading, s)
    junction = np.array([w.road_id == TURN_ROAD for w in wps])
    s_turn = (float(s[junction][0]), float(s[junction][-1]))
    roads = np.array([[w.road_id, w.lane_id] for w in wps])
    return dict(s=s, x=x, y=y, heading=heading, kappa=kappa, junction=junction,
                x_map=xm, y_map=ym, heading_map=hm,
                s_turn=np.array(s_turn), roads=roads)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xodr", required=True, help="Town03.xodr (CARLA's OpenDRIVE map)")
    ap.add_argument("--out", default=str(HERE / "l10_curve.npz"))
    a = ap.parse_args(argv)
    d = export(a.xodr)
    np.savez(a.out, **d)
    print(f"{a.out}: {len(d['s'])} points, {d['s'][-1]:.1f} m, turn from s = "
          f"{d['s_turn'][0]:.1f} to {d['s_turn'][1]:.1f} m, "
          f"largest curvature {np.abs(d['kappa']).max():.4f} 1/m")


if __name__ == "__main__":
    main()
