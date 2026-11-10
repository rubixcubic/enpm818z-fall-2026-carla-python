import numpy as np

import l10common as c


def test_curve_is_the_town03_turn():
    d = c.load_curve()
    assert abs(d["s"][-1] - 137.5) < 0.5
    assert tuple(d["roads"][0]) == (3, 3) and tuple(d["roads"][-1]) == (19, 1)
    # a right turn of 91 degrees: the heading ends near -pi/2
    assert abs(d["heading"][-1] + np.radians(91.4)) < 0.01
    # two arcs: radius 10.00 m and 5.98 m
    k = np.abs(d["kappa"])
    assert abs(1 / np.median(k[(d["s"] > 84) & (d["s"] < 88.5)]) - 10.00) < 0.05
    assert abs(1 / np.median(k[(d["s"] > 90) & (d["s"] < 94.5)]) - 5.98) < 0.05


def test_bicycle_step_turns_on_the_circle():
    delta, v = 0.3, 5.0
    R = c.turning_radius(delta)
    xs, ys, _ = c.drive_constant(0.0, 0.0, 0.0, v, delta, 2.0, dt=0.001)
    # center of the circle is R to the left of the start
    assert np.allclose(np.hypot(xs, ys - R), R, atol=0.02)


def test_speed_profile_respects_its_limits():
    d = c.load_curve()
    v = c.speed_profile(d["s"], d["kappa"])
    assert v.max() <= 8.0 + 1e-9
    assert np.all(v ** 2 * np.abs(d["kappa"]) <= 2.0 + 1e-6)
    dv2 = np.diff(v ** 2) / (2 * np.diff(d["s"]))
    assert dv2.max() <= 1.5 + 1e-6 and dv2.min() >= -2.0 - 1e-6


def test_quintic_meets_its_six_conditions():
    q = c.quintic(0.0, 8.0, 0.0, 20.0, 3.0, 0.0, 3.5)
    T = 3.5
    got = [c.poly_eval(q, 0, 0), c.poly_eval(q, 0, 1), c.poly_eval(q, 0, 2),
           c.poly_eval(q, T, 0), c.poly_eval(q, T, 1), c.poly_eval(q, T, 2)]
    assert np.allclose(got, [0, 8, 0, 20, 3, 0], atol=1e-9)


def test_slow_down_quintic_stays_within_the_brake_limit():
    sd = c.slow_down(8.0, 3.458, 2.0)
    assert abs(sd["quintic"]["peak_a"] - 2.0) < 1e-3
    assert sd["quintic"]["D"] > sd["constant"]["D"]
