# L3 hands-on: a particle filter in a tunnel

The car's computer restarted inside a 700 m tunnel. It knows its speed from the
wheels but not where it is. The camera matches two kinds of landmarks against
the HD map: **lights**, one every 25 m and all identical, and **SOS niches**,
only five and spaced irregularly. A light match fits 23 places at once; a
niche match fits five. No bell curve can say that, so the belief is a crowd of
particles.

Needs only `numpy` and `matplotlib`.

```bash
python3 pf_tunnel.py --make-csv          # writes pf_drive.csv and tunnel_map.csv
python3 pf_tunnel.py                     # live window
python3 pf_tunnel.py --trials 40         # how often does it settle on the right place?
python3 pf_tunnel.py --html pf.html      # no desktop? a web page
```

## The data

`pf_drive.csv`, 451 rows, one every 0.1 s:

| column | meaning |
|---|---|
| `t` | time (s) |
| `wheel_v` | wheel speed (m/s), noise 0.3 m/s: the control input |
| `match`, `match_dist` | a landmark match: `light` or `niche`, and how far ahead it is (sigma 1 m). Empty on most rows (24 matches, 5 of them niches) |
| `true_s` | the truth, distance along the tunnel. Only for scoring |

`tunnel_map.csv` is the HD map: `kind` (light or niche) and `s`, its position.

## The window

- **Top:** the whole tunnel, unrolled, with the crowd, the true car, the
  weighted average (what one bell curve would report) and the heaviest cluster.
- **Middle:** the belief right now, as weight per 2 m.
- **Bottom:** the error of the heaviest cluster and of the weighted average,
  and how many places the crowd is in. Green lines mark niche matches.
- **Controls:** play/pause, restart, **New draw** (the same data, new random
  numbers), the number of particles, and R (the camera sigma).

## Numbers from the shipped data (40 draws each)

| setting | settled on the right place |
|---|---|
| 5000 particles | 40 of 40 |
| 2000 particles (default) | 40 of 40, from t = 20.9 s |
| 500 particles | 33 of 40 |
| 100 particles | 11 of 40 |
| 30 particles | 1 of 40 |
| camera sigma 0.2 m (R too small) | 31 of 40 |

## Exercises

1. Watch the weighted average until t = 21 s. Where is it, and why is it the
   worst possible answer?
2. Why does the crowd go from 23 places to 4 at t = 2 s, and to 1 at t = 21 s?
3. Set 100 particles and press New draw a few times. When it settles on the
   wrong place, does anything on the screen warn you?
4. Move one niche in `tunnel_map.csv` so two gaps are equal. What happens?
