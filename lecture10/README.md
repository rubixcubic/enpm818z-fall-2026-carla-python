# L10 hands-on: from a path to steering, throttle and brake

Four parts, one Town03 turn. Needs `numpy` and `matplotlib`; no CARLA, no ROS.

```bash
python3 l10_handson.py --part bicycle   # 1. the bicycle model: one steering angle, one circle
python3 l10_handson.py --part profile   # 2. the speed profile, and two ways to slow down
python3 l10_handson.py --part pid       # 3. speed control: P, PI, PID, windup (precomputed)
python3 l10_handson.py --part track     # 4. steering: Pure Pursuit and Stanley (precomputed)
```

Add `--snapshot out.png` to save a still instead of opening a window, or
`--html out.html` for a page (needs `mpld3`). For `track`, `--run pp_Ld10` (or
`st_k1`, `ppv_kv0.75`, ...) opens that run first.

## What each part shows

1. **bicycle**: hold one steering angle and the rear axle drives a circle of
   radius L / tan(delta). Two sliders set the steering angle and the speed; the
   title prints one 0.05 s step of the model. The right panel plots the radius
   against the steering angle, with the two arcs of our turn marked.
2. **profile**: the turn's speed profile (8 m/s, at most 2.0 m/s^2 sideways in the
   curves, braking at 2.0 m/s^2 and accelerating at 1.5 m/s^2), and two ways to slow
   from 8 m/s to the turn's 3.46 m/s: a constant deceleration and a quintic. The
   bottom panels show the acceleration and the jerk of each.
3. **pid**: the AV at rest, the target speed jumps to 8 m/s. Pick Kp, Ki, Kd and
   whether the integral is clamped; the top panel shows the speed, the bottom one the
   P, I and D terms and the command.
4. **track**: the turn driven from 1 m left of the lane center at 8 m/s, by Pure
   Pursuit with a lookahead of 2 to 12 m, by Pure Pursuit with a lookahead of 0.5,
   0.75 or 1 s times the speed (at least 3 m), or by Stanley with a gain of 0.5 to
   8. Pick a run in the list; the panels show it from above, the rear axle's
   distance from the lane center and the steering angle.

**The pid and track runs were computed in advance** and saved in `l10_results/`.
This folder has no controller code: you write Pure Pursuit and the PID in GP4
Task 2.

## A behavior tree in plain Python

`behavior_tree.py` builds the lecture's road tree from four node types (Condition,
Action, Sequence, Fallback) and ticks it through five moments of a drive. No CARLA, no
ROS, no libraries. Each line prints the action that runs and the conditions the tick
checked, by their number on the slides.

```bash
python3 behavior_tree.py              # four branches: stop, pass, follow, lane keep
python3 behavior_tree.py --no-pass    # the same tree with the pass branch removed
```

## The simulation behind the runs

The kinematic bicycle model at the rear axle, stepped 100 times a second, with the
controllers running 20 times a second. Chosen for the lecture, not CARLA's
dynamics:

| Value | Number |
|---|---|
| Wheelbase | 2.875 m (GP4's `controller_config.yaml`) |
| Largest steering angle | 0.7 rad (GP4) |
| Steering | a 0.1 s delay, then a first-order lag of 0.2 s |
| Throttle | full throttle gives 4 m/s^2, capped at 0.75 (GP4's `max_throttle`) |
| Brake | full brake gives 8 m/s^2 |
| Engine lag | 0.3 s |
| Drag | 0.15 + 0.012 v^2 m/s^2 |
| Speed PID in the track runs | Kp 0.5, Ki 0.3, Kd 0.1, integral clamped at 1 m |

## The data

- `l10_curve.npz`: the path, from `export_curve.py`, which reads `Town03.xodr`
  offline with `carla.Map` (the CARLA 0.9.16 Python client, no server): 80 m of
  road 3, the junction lane road 817 (a right turn), 40 m of road 19, every 0.25 m
  along the lane center. CARLA's y is flipped (x east, y north, as in lecture6),
  then the path is turned so the approach points along x.
- `l10_results/pid_grid.npz`: 72 speed runs (Kp 0.25, 0.5, 1; Ki 0, 0.1, 0.3, 0.6;
  Kd 0, 0.1, 0.3; clamp off or 1 m).
- `l10_results/track_grid.npz`: 16 steering runs.
- `l10_results/mpc_candidates.npz`: the 729 predicted paths of the slides' MPC
  picture.
- `l10_results/worked.json`: the numbers of the slides' worked examples and every
  run's summary.

## Tests

```bash
MPLBACKEND=Agg python3 -m pytest -q tests
```
