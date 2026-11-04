"""Write candidates.csv: every candidate path of the L9 slides, with what each cost
term needs, and no costs. For the in-class spreadsheet exercise.

  python3 export_candidates.py                  # writes candidates.csv here
  python3 export_candidates.py --out other.csv

One row per candidate from the AV at s = 60 m (the slides' scene): 30 sideways
shifts (10 end offsets times 3 maneuver lengths) and the path that stops behind the
parked car. In a spreadsheet you multiply each column by its weight (the slide "Step 3:
the four cost terms"), add the four terms, and sort. The `dropped` column says which
paths the planner removes before it scores: they get no cost.

numpy only, like l9common.py: no CARLA server.
"""
import argparse
import csv
import sys
from pathlib import Path

import l9common as c

HERE = Path(__file__).resolve().parent
COLUMNS = ["name", "d_end_m", "maneuver_m", "peak_lat_accel_mps2", "smallest_gap_m",
           "lanes_away", "off_center_m", "not_covered_m", "dropped"]


def rows():
    """The candidates of the slides' scene, in the planner's order (cheapest first,
    dropped paths last). Each row holds the raw measures, rounded to 3 decimals."""
    ref = c.load_scene()
    car = c.parked_car(ref)
    out = []
    for p, sc in c.plan(ref, c.SCENE["av_s"], (0.0, 0.0, 0.0), [car]):
        stop = bool(p.get("stop"))
        lanes, off = c.lane_cost_parts(p["d_end"], ref["lane_width"])
        out.append({
            "name": "stop behind" if stop else f"d_end {p['d_end']:.1f} m, {p['maneuver']:.0f} m",
            "d_end_m": round(p["d_end"], 3),
            "maneuver_m": "" if stop else round(p["maneuver"], 3),
            "peak_lat_accel_mps2": round(sc["a_peak"], 3),
            "smallest_gap_m": round(sc["gap_min"], 3),
            "lanes_away": lanes,
            "off_center_m": round(off, 3),
            "not_covered_m": round(c.SCENE["horizon"] - sc["progress"], 3),
            "dropped": sc["dropped"] or "",
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE / "candidates.csv"))
    a = ap.parse_args(argv)
    data = rows()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(data)
    print(f"wrote {a.out}: {len(data)} candidates, "
          f"{sum(1 for r in data if r['dropped'])} dropped")


if __name__ == "__main__":
    main(sys.argv[1:])
