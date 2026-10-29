# L8 hands-on: where will the others be?

Four parts, one Town01 junction. Needs `numpy` and `matplotlib`; no CARLA, no ROS.

```bash
python3 l8_handson.py --part scene      # 1. the junction, the AV, car A and car B over time
python3 l8_handson.py --part physics    # 2. CV, CTRV and CTRA for car A, from any "now"
python3 l8_handson.py --part modes      # 3. map-aware prediction: two futures, their probabilities
python3 l8_handson.py --part interact   # 4. car B behind A (IDM), and A's gap to the AV
```

Every part has a **now** slider: drag it to predict from another moment, from
-0.5 s to 3 s (the scene part from -2 s).

| Option | What it does |
|---|---|
| `--now T` | open at T seconds instead of 0 |
| `--snapshot out.png` | save a still instead of opening a window |
| `--html out.html` | save an animation over "now" (open it in a browser) |

## The scene

Junction 54 of Town01, a T-junction at x = 156 m, y = -2 m. The lanes come from
CARLA's `Town01.xodr`, read by `export_scene.py` through `carla.Map` with no CARLA
server, and saved in `l8_scene.npz`. Map frame: x east, y north, in meters (CARLA's
y is flipped).

- **The AV** drives east at 8 m/s, straight on.
- **Car A** drives west, brakes at 1.0 m/s² from 8 m/s and turns left at 5.0 m/s,
  across the AV's lane. It turns only if the AV is at least 4.1 s from the crossing
  when A is 6 m before the junction (its gap); otherwise it waits.
- **Car B** follows car A and goes straight on. The Intelligent Driver Model (IDM)
  drives it.

**t = 0 is "now"**, 2.0 s into A's braking. The predictors read only each car's
**track**: its position every 0.1 s, with a 0.05 m error. Everything after "now" is
the truth they must guess. Run 2 (the check box in part 4) moves the AV 24 m closer,
so A's gap is 3.0 s and A waits. **The motion is simulated, and every number above
is chosen**: CARLA only gives the lanes.

## What each part shows

1. **scene**: the three road users over time and the tracked positions the AV sees.
2. **physics**: car A's future from constant velocity (CV), constant turn rate and
   velocity (CTRV) and constant turn rate and acceleration (CTRA), against the truth,
   and each one's error against the horizon. Move "now" into the turn (about 3 s) and
   see CTRV follow the curve, then keep turning.
3. **modes**: the map gives car A two lanes, straight on and left. Each lane is one
   future; its probability comes from how well it explains A's speed, heading and
   offset from the lane's center (Bayes' rule). Right: P(left) against time.
4. **interact**: left, car A's meters driven against time: the truth, the most likely
   mode, and the same mode with the gap rule. Right, car B's speed: the truth, CV,
   and the IDM behind A's predicted future. Tick **AV 24 m closer** for run 2.

## Files

| File | What it is |
|---|---|
| `l8sim.py` | the scene, the true motion, the track, the fit, the predictors, the scores, the test set |
| `l8_handson.py` | the four parts |
| `vehicle_icons.py` | draws the AV, car A and car B as vector icons, sharp at any zoom |
| `icons/` | the three icons, by Stone from the Noun Project (thenounproject.com) |
| `export_scene.py` | writes `l8_scene.npz` from `Town01.xodr` (needs the `carla` package, no server) |
| `l8_scene.npz` | the lanes of junction 54 |
| `tests/` | `python3 -m pytest -q tests` |

The slides' numbers come from `l8sim.py`, through the slides' figure generator.
