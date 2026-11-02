# L9 hands-on: motion planning

One road from CARLA's Town03 map with a car parked in the AV's lane, and one
parking garage. Needs `numpy` and `matplotlib`; no CARLA server, no ROS, no GPU.

```bash
python3 l9_handson.py --part scene    # 1. the road in the map and in the Frenet frame
python3 l9_handson.py --part sample   # 2. candidate paths, their costs, the cheapest
python3 l9_handson.py --part drive    # 3. re-planning every 0.5 s while passing the car
python3 l9_handson.py --part rrt      # 4. an RRT growing in the garage
python3 l9_handson.py --part search   # 5. grid search and hybrid A* results (read only)
```

Add `--snapshot out.png` to save a still instead of opening a window, or
`--html out.html` for a page (needs `mpld3`). `--seed S` changes the RRT's random
seed (default 4, as on the slides).

## What each part shows

1. **scene**: the stretch of Town03 road 67, from the map, and the same road in
   the Frenet frame, where `s` is the distance along the reference path (the
   right lane's center) and `d` the distance across it, positive to the left.
   Move the point P with the `s` and `d` sliders: in the map it follows the curve;
   in the Frenet frame it moves on straight lines. The title converts P to the
   map and back.
2. **sample**: every candidate path from the AV at `s = 60 m`: 10 end offsets
   (`d` from -0.5 m to 4.0 m, every 0.5 m) times 3 maneuver lengths (20, 30 and
   40 m), plus one that keeps the lane and stops behind the car. Each one is
   checked with three circles around the AV and three around the parked car, and
   scored: comfort (peak sideways acceleration), clearance (a gap under 1.5 m
   costs), lane (lanes away from the route's lane, meters off a lane center) and
   progress (meters of the 70 m horizon not covered). Green: the cheapest; gray:
   kept; rust: collides; purple dashed: too sharp (more than 3 m/s²); black
   dotted: stop behind. The sliders change the weights and where the car is
   parked; the table on the right ranks the nine kept paths.
3. **drive**: the planner runs every 0.5 s. The AV drives the first 0.5 s of the
   cheapest path, then plans again from where it is. Step through the plans with
   the slider: the AV moves into the left lane, passes the car and comes back.
4. **rrt**: a rapidly exploring random tree from the AV's start in the garage to
   the entrance of the free spot. Each new node is a 1.5 m step from the nearest
   node toward a random point; the AV's three circles are checked along the step.
   The slider shows the tree as it grows.
5. **search**: results only, computed in advance and saved in
   `l9_results/garage_search.json`: the toy grid's step numbers from the slides,
   a grid search path through the garage, and the hybrid A* path that reverses
   into the spot (forward solid, reverse dashed), with every pose it expanded.
   This folder has no search code: in GP4 you write A* yourself.

## The files

| file | what it is |
|---|---|
| `l9_handson.py` | the five parts above |
| `l9common.py` | the road, the Frenet frame, the lateral curve, the circles, the costs, the RRT |
| `export_scene.py` | reads `Town03.xodr` with `carla.Map` (offline) and writes `l9_scene.npz` |
| `l9_scene.npz` | the right and left lane centers every 0.5 m, and the town's lanes for the overview |
| `l9_results/garage_search.json` | grid search and hybrid A* results (data only) |
| `tests/` | `python3 -m pytest -q tests` |

Frames: the map frame is in meters, `x` as CARLA's `x` and `y` as minus CARLA's
`y`, so the frame is right-handed (as in L6). Headings are counterclockwise from
`+x`.

## Chosen values

The road is measured from the map. Everything else is chosen for the lecture
and lives at the top of `l9common.py`:

| value | chosen |
|---|---|
| the AV | 4.5 m by 1.9 m, base link 0.9 m in from the rear bumper; wheelbase 2.875 m (chosen for the lecture) and steering limit 0.7 rad (GP4's) |
| the AV's start | `s = 60 m` on the right lane's center, 10 m/s |
| the parked car | 4.5 m by 1.9 m, centered at `s = 118 m`, `d = -0.3 m` |
| horizon | 70 m |
| weights | comfort 2 per m/s², clearance 4 per m, lane 2 per lane and 2 per m off center, progress 0.5 per m |
| limits | a gap under 0.3 m is a collision; more than 3 m/s² sideways is too sharp |
| garage | 36 m by 18 m, one 7 m aisle, spots 2.6 m by 5.5 m |
