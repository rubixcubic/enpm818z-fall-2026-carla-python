# L3 hands-on: EKF and UKF in a curved tunnel

Same idea as `../tunnel_kf`, one step harder. The tunnel bends 120 degrees to the
left, so the motion has a cosine in it, and the camera now measures the **range
and bearing** to each matched exit sign, a square root and an arctangent. No
matrix F or H can describe either, so the plain Kalman filter cannot be written
down. Two scripts, one CSV:

- `ekf_curve.py`: the extended Kalman filter. The mean goes through the real f
  and h; the covariance goes through their tangents, the Jacobians F_k and H_k.
- `ukf_curve.py`: the unscented Kalman filter next to the EKF, both compared
  with the exact belief, computed by brute force with 4,000 possible cars.

Needs only `numpy` and `matplotlib`.

```bash
python3 ekf_curve.py --make-csv                 # writes curve_drive.csv
python3 ekf_curve.py --check-jacobian           # numerical check of F_k and H_k
python3 ekf_curve.py                            # EKF, live window
python3 ukf_curve.py                            # UKF vs EKF vs exact, live window
python3 ekf_curve.py --html ekf.html            # no desktop? a web page
```

## The CSV (`curve_drive.csv`, 212 rows, one every 0.1 s)

| column | meaning |
|---|---|
| `t` | time (s) |
| `wheel_v`, `gyro_w` | wheel speed (m/s) and gyro turn rate (rad/s): the control input u. Noise 0.2 m/s and 0.02 rad/s, plus a gyro bias of 0.003 rad/s |
| `sign_id`, `sign_range`, `sign_bearing` | a sign match: which sign in the HD map, its range (sigma 1 m) and bearing (sigma 2 deg, positive to the left). Empty on most rows (9 matches) |
| `true_x`, `true_y`, `true_theta` | the truth. The filters never read it; it is only for scoring |

The HD map, the positions of the signs, is computed in `ekf_curve.py`
(`hd_map()`): one every 25 m on the left wall.

## ekf_curve.py

Controls: Q (gyro noise the filter assumes), R (range noise), and a check box
that flips one sign in the Jacobian F_k. Numbers from the shipped CSV:

| setting | position error | heading error | heading inside 1 sigma | what it shows |
|---|---|---|---|---|
| default: gyro 2 deg/s, range 1 m | 0.69 m | 1.4 deg | 65% | honest |
| gyro 0.1 deg/s (Q too small) | 1.18 m | 2.1 deg | 23% | overconfident |
| gyro 8 deg/s (Q too big) | 0.75 m | 1.8 deg | 92% | underconfident |
| range 0.1 m (R too small) | 1.42 m | 1.8 deg | 46% | trusts every match |
| wrong sign in the Jacobian | 1.23 m | 4.0 deg | 30% | runs, looks fine, is wrong |

The default gyro noise, 2 deg/s, is larger than the real 1.1 deg/s on purpose:
Q also has to cover the gyro bias, which the filter does not model.

## ukf_curve.py

Controls: the heading sigma handed over at the tunnel entrance, and an unlit
stretch during which the camera matches no sign. The headline compares each
filter with the exact belief just before the first sign match:

| heading sigma, unlit | mean off the exact belief: EKF | UKF |
|---|---|---|
| 3 deg, 5 s | 0.12 m | 0.04 m |
| 12 deg, 5 s (default) | 1.45 m | 0.03 m |
| 12 deg, 9 s | 1.90 m | 0.03 m |
| 25 deg, 5 s | 6.03 m | 0.01 m |

In the dark the UKF stays on the exact belief and the EKF drifts off it: the
banana from the slides. Then watch the first match. A belief 14 m wide meets a
1 m measurement, and neither filter follows the exact answer at once. The EKF
snaps in and is overconfident; the UKF, whose sigma points straddle the sign,
treats the bend as noise and corrects cautiously. That is where a bell curve,
any bell curve, runs out, and it is what `../pf_tunnel` is for.

## Exercises

1. Run `--check-jacobian`, then change a sign in `jacobian_h` and run it again.
2. Tick "wrong sign in the Jacobian". The filter still runs. How would you know
   it is wrong on a real car, with no truth column?
3. In `ukf_curve.py`, set the heading sigma to 3 deg. Why do the EKF and UKF
   agree now?
4. Find the two places `ekf_curve.py` wraps an angle. Remove one and rerun.
