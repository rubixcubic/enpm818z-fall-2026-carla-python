# L3 Python warm-up: four filters, one dataset

Run every filter from the lecture on the same recorded drive, before touching
CARLA. Needs only `numpy` and `matplotlib`.

```bash
python3 make_dataset.py        # writes l3_drive.csv and l3_drive_multipath.csv
python3 kf_cv.py               # 1. plain Kalman filter, GNSS only
python3 ekf.py                 # 2. EKF: wheels + IMU predict, GNSS updates
python3 ukf.py                 # 3. UKF: same car, no Jacobian
python3 pf.py                  # 4. particle filter: a crowd of guesses
```

Each script prints two things and saves a PNG:

- **position error (RMSE)**: how wrong the filter is, against ground truth;
- **NIS**: whether its covariance is honest. For a GNSS fix (2 numbers) an
  honest filter averages about **2** and keeps about **95%** of fixes inside
  the band 0.051 to 7.378.

## The dataset

`make_dataset.py` drives a car for 60 s on a curvy road: straight, a left
bend, straight, a right bend, a gentle S. One row every 0.1 s.

| column | meaning |
|---|---|
| `t` | time (s) |
| `x, y, theta` | where the car **really** is. Ground truth: never read it inside a filter |
| `v, omega` | what the car **really** did |
| `v_meas, omega_meas` | what the wheels and IMU **say** it did (σ 0.3 m/s, 0.02 rad/s) |
| `gnss_x, gnss_y` | GNSS fix, 1 Hz, σ 2 m. Empty on the other nine rows |

`l3_drive_multipath.csv` is the same drive with four fixes pushed 8 to 12 m
off: on time, normal looking, and wrong.

## Exercises

Reference numbers below come from the shipped dataset (seed 818).

1. **Tune Q.** Run `kf_cv.py --sigma-a 0.1`, then `5`, then `20`. Which one
   is overconfident, which is underconfident, and how does the NIS tell you?
   *(NIS mean about 160, 1.7, 0.8.)*
2. **Why does the EKF win?** `kf_cv.py` is off by about 3.0 m, `ekf.py` by
   about 1.4 m. Both see the same GNSS. What does the EKF have that the plain
   filter does not?
3. **Break the Jacobian.** Flip one sign in `jacobian_f` in `ekf.py`. Does the
   filter still run? Now run `ekf.py --check-jacobian`.
4. **Multipath.** Run `ekf.py --data l3_drive_multipath.csv`, then add
   `--gate`. How many fixes does the gate reject, and what happens to the
   error? *(4 rejected; about 1.60 m becomes 1.40 m.)*
5. **EKF or UKF?** Compare `ekf.py` and `ukf.py` with `--heading-error 60`,
   then `120`. They come out almost the same. Why is the UKF's advantage so
   small here? *(Hint: how often does a GNSS fix arrive to pull the heading
   back?)*
6. **Particles.** Run `pf.py --particles 50 --jitter 0`, then without
   `--jitter 0`. Then `pf.py --global` with 1000 particles, and with
   `--particles 5000`. When does the crowd collapse?

## Files

| file | what it is |
|---|---|
| `make_dataset.py` | makes both CSVs |
| `l3common.py` | loading, angle wrapping, the RMSE and NIS report, plotting |
| `kf_cv.py` | constant-velocity Kalman filter, state `[px, py, vx, vy]` |
| `ekf.py` | EKF, state `[x, y, theta]`, control `[v, omega]`, Jacobian check, gating |
| `ukf.py` | UKF on the same model, 7 sigma points |
| `pf.py` | particle filter, systematic resampling, jitter, global start |
