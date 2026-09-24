# L3 hands-on: a Kalman filter in a tunnel

One script, one CSV, and a live window. The car has lost GNSS in a tunnel. The
IMU gives its acceleration (the control input **u**); every 25 m the camera
matches an emergency exit sign against the HD map, which gives a position (the
measurement **z**). The filter is `run_kf()` in `kf_tunnel.py`: the predict and
update equations from the slides, about 30 lines.

Needs only `numpy` and `matplotlib`.

```bash
python3 kf_tunnel.py --make-csv                        # writes tunnel_drive.csv
python3 kf_tunnel.py tunnel_drive.csv                  # live window
python3 kf_tunnel.py tunnel_drive.csv --html tunnel_kf.html   # no desktop? a web page
```

## The CSV (`tunnel_drive.csv`, 451 rows, one every 0.1 s)

| column | meaning |
|---|---|
| `t` | time (s) |
| `true_x, true_y, true_vx, true_vy` | the truth. The filter never reads it; it is only for scoring |
| `imu_ax, imu_ay` | IMU acceleration (m/s^2): noise 0.3 and a small bias 0.05 on x |
| `sign_x, sign_y, sign_id` | a sign match: position with sigma 1 m along the tunnel and 0.2 m across it (the wall is close). Empty on most rows (19 matches) |

## The window

- **Top:** the tunnel from above, following the car: the true car, the estimate
  with its 1-sigma ellipse (P), the HD map's signs, and each sign match as it
  arrives.
- **Bottom left:** sigma of the position over time: the sawtooth.
- **Bottom right:** the true error against the +/- sigma the filter reports.
- **Top line:** the error, and how often the truth falls inside 1 sigma. An
  honest filter gets about 68%.
- **Controls:** play/pause, restart, and sliders for Q (`sigma_a`) and R
  (`sigma_sign`, along the tunnel; the sideways 0.2 m is fixed). Moving a slider
  reruns the filter.

## Exercises (numbers from the shipped CSV)

| setting | error | inside 1 sigma | what it shows |
|---|---|---|---|
| default: sigma_a 0.5, sigma_sign 1 | 1.06 m | 65% | honest |
| sigma_a 0.01 (Q too small) | 5.46 m | 25% | overconfident, the dangerous case |
| sigma_a 3 (Q too big) | 1.28 m | 84% | underconfident |
| sigma_sign 0.1 (R too small) | 1.46 m | 21% | trusts every sign match |
| sigma_sign 6 (R too big) | 2.95 m | 77% | ignores the signs and drifts |

1. Predict first, then move the slider: what happens to the sawtooth when Q is
   tiny? Why does the error grow while sigma stays small?
2. Where in the drive does the error grow fastest, and what is the car doing
   then? (Hint: the brake at 8 to 12 s.)
3. Delete every second sign match from the CSV. What happens to the sawtooth?
4. Open `run_kf()` and find each equation from the slides.
