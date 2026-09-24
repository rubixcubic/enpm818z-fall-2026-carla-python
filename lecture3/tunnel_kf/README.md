# L3 hands-on: a Kalman filter in a tunnel

One script, one CSV, and a live window. The car has lost GNSS in a tunnel. The
IMU gives its acceleration (the control input **u**); every 25 m the camera
matches an emergency exit sign against the HD map, which gives a position (the
measurement **z**). The filter is `run_kf()` in `kf_tunnel.py`: the predict and
update equations from the slides, about 30 lines.

## Requirements

Python 3 with `numpy` and `matplotlib`. Tested on Ubuntu 24.04 with Python 3.12,
numpy 1.26, and both matplotlib 3.6 (what `apt` installs) and 3.10 (from pip).

```bash
# Ubuntu 24.04 (recommended: system pip is blocked there, PEP 668)
sudo apt install python3-numpy python3-matplotlib python3-tk

# any other system, in a virtual environment
python3 -m venv ~/l3env && source ~/l3env/bin/activate
pip install numpy matplotlib
```

`python3-tk` provides the window matplotlib draws into. Without it the live
window cannot open; use `--html` to get the same animation as a web page.

**Seeing "Unable to import Axes3D ... multiple versions of Matplotlib"?** You
have matplotlib from both `apt` and `pip`. It is harmless here (no script uses
3D), but to clear it keep one copy, for example
`python3 -m pip uninstall matplotlib` to fall back to the `apt` one.

## Running it

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

## The filter's Q

Q comes from the acceleration the IMU gets wrong, pushed through B. Along the
tunnel its sigma is the slider `sigma_a` (default 0.5 m/s^2), which must also
cover the IMU's bias. Across the tunnel it is fixed at 0.3 m/s^2, the IMU's
sideways noise, because a car in its lane has no sideways bias. So predict
stretches the ellipse most along the tunnel. The ellipse figures in the L3
slides (Step 1, 2, 3) are drawn from this script's `run_kf(..., with_prior=True)`.

## The window

- **Top:** the tunnel from above, following the car: the true car, the estimate
  with its 1-sigma ellipse (P), the HD map's signs, and each sign match as it
  arrives. A **zoom** on the right, on a 1 m grid, shows the ellipse's shape: it
  grows along the tunnel between signs and snaps long and thin at each match.
- **Bottom left:** sigma of the position over time: the sawtooth.
- **Bottom right:** the true error against the +/- sigma the filter reports.
- **Top line:** the error, and how often the truth falls inside 1 sigma. An
  honest filter gets about 68%.
- **Controls:** play/pause, restart, sliders for Q (`sigma_a`) and R
  (`sigma_sign`, along the tunnel; the sideways 0.2 m is fixed), and a check box
  **use the IMU (B u)**. Moving a slider or the box reruns the filter.
- **The drive:** the car brakes at 1 m/s^2 from 8 to 12 s and speeds up at
  0.8 m/s^2 from 20 to 24 s; both stretches are shaded on the time plots, and
  the readout shows u, the IMU acceleration, at every step.

## Exercises (numbers from the shipped CSV)

| setting | error | inside 1 sigma | what it shows |
|---|---|---|---|
| default: sigma_a 0.5, sigma_sign 1 | 1.06 m | 65% | honest |
| sigma_a 0.01 (Q too small) | 5.46 m | 25% | overconfident, the dangerous case |
| sigma_a 3 (Q too big) | 1.28 m | 84% | underconfident |
| sigma_sign 0.1 (R too small) | 1.46 m | 21% | trusts every sign match |
| sigma_sign 6 (R too big) | 2.95 m | 77% | ignores the signs and drifts |
| IMU ignored (no B u), sigma_a 0.5 | 3.58 m | 30% | the brake is in w, and Q is too small for it |
| IMU ignored (no B u), sigma_a 2.0 | 2.06 m | 65% | honest again, but twice the error |

1. Predict first, then move the slider: what happens to the sawtooth when Q is
   tiny? Why does the error grow while sigma stays small?
2. Where in the drive does the error grow fastest, and what is the car doing
   then? (Hint: the brake at 8 to 12 s.)
3. Delete every second sign match from the CSV. What happens to the sawtooth?
4. Open `run_kf()` and find each equation from the slides.
