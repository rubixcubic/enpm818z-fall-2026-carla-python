import numpy as np
import pytest

import l9common as c


@pytest.fixture(scope="module")
def ref():
    return c.load_scene()


def test_scene_lanes_are_measured(ref):
    # Town03's lanes are 3.5 m wide; the left lane's center sits one lane width left.
    assert ref["lane_width"] == pytest.approx(3.5)
    d = [c.to_frenet(ref, *p)[1] for p in ref["left_xy"][::25]]
    assert np.allclose(d, 3.5, atol=0.01)


def test_frenet_round_trip(ref):
    rng = np.random.default_rng(9)
    for s, d in zip(rng.uniform(5, 230, 50), rng.uniform(-1.75, 5.25, 50)):
        x, y = c.from_frenet(ref, s, d)
        s2, d2, _, _ = c.to_frenet(ref, float(x), float(y))
        assert s2 == pytest.approx(s, abs=0.02) and d2 == pytest.approx(d, abs=0.02)


def test_smooth_step_ends_flat():
    u = np.array([0.0, 0.5, 1.0])
    assert np.allclose(c.smooth_step(u), [0.0, 0.5, 1.0])
    co = c.quintic(0.0, 0.0, 0.0, 3.5, 40.0)
    d, d1, d2 = c.lateral(co, 40.0, np.array([0.0, 40.0]))
    assert np.allclose(d, [0, 3.5]) and np.allclose(d1, 0, atol=1e-12) and np.allclose(d2, 0, atol=1e-12)
    # peak d'' of the smooth step: 10 / sqrt(3) times (change in d) / length^2
    s = np.linspace(0, 40, 40001)
    assert np.max(c.lateral(co, 40.0, s)[2]) == pytest.approx(10 / np.sqrt(3) * 3.5 / 1600, rel=1e-6)


def test_path_curvature_matches_drawn_points():
    th = np.linspace(0, 1.2, 4001)
    R = 50.0
    ref = c.make_reference(R * np.sin(th), R * (1 - np.cos(th)))
    ref["kappa"][:] = 1 / R
    p = c.make_path(ref, 5.0, (0, 0, 0), 3.5, 30.0, 40.0, step=0.01)
    x, y = p["x"], p["y"]
    h = np.unwrap(np.arctan2(np.diff(y), np.diff(x)))
    a = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
    w = 100
    k_num = (h[w:] - h[:-w]) / (a[w:] - a[:-w])
    k_avg = np.convolve(p["kappa"], np.ones(w) / w, "valid")[:len(k_num)]
    assert np.max(np.abs(k_num - k_avg)) < 5e-4
    # a constant offset beside a circle: curvature 1 / (R - d)
    q = c.make_path(ref, 5.0, (3.5, 0, 0), 3.5, 10.0, 20.0)
    assert np.allclose(q["kappa"], 1 / (R - 3.5))


def test_circles_cover_the_body():
    r = c.circle_radius()
    assert r == pytest.approx(np.hypot(0.75, 0.95))
    assert np.allclose(c.circle_offsets(), [-0.15, 1.35, 2.85])
    # every corner of the 4.5 m x 1.9 m body is inside some circle
    for cx in (-0.9, 3.6):
        for cy in (-0.95, 0.95):
            assert min(np.hypot(cx - o, cy) for o in c.circle_offsets()) <= r + 1e-9


def test_the_cheapest_candidate(ref):
    car = c.parked_car(ref)
    out = c.plan(ref, c.SCENE["av_s"], (0, 0, 0), [car])
    best, sc = out[0]
    assert (best["d_end"], best["maneuver"]) == (3.5, 40.0)
    assert sc["total"] == pytest.approx(5.07, abs=0.01)
    assert len(out) == 31
    assert sum(s["dropped"] == "collision" for _, s in out) == 18
    assert sum(s["dropped"] == "too sharp" for _, s in out) == 4
    stop = [s for p, s in out if p.get("stop")][0]
    assert stop["total"] == pytest.approx(11.54, abs=0.01)


def test_drive_passes_and_returns(ref):
    car = c.parked_car(ref)
    states, plans = c.drive(ref, [car], duration=14.0)
    d = np.array([s[1] for s in states])
    assert d.max() > 3.0 and abs(d[-1]) < 0.05
    assert not any(p["stop"] for p in plans)


def test_rrt_is_reproducible_and_collision_free():
    goal = np.array([c.garage_goal()[0], 9.0])
    rects = c.garage_obstacles()
    a = c.rrt(c.GARAGE["start"], goal, rects, goal_bias=0.05, seed=4)
    b = c.rrt(c.GARAGE["start"], goal, rects, goal_bias=0.05, seed=4)
    assert a["iterations"] == b["iterations"] and a["path"] is not None
    P = a["path"]
    for p, q in zip(P[:-1], P[1:]):
        h = np.arctan2(*(q - p)[::-1])
        assert all(c.pose_free(*(p + f * (q - p)), h, rects) for f in np.linspace(0, 1, 6))


def test_goal_pose_is_free():
    assert c.pose_free(*c.garage_goal(), c.garage_obstacles())
    assert c.pose_free(*c.GARAGE["start"], c.garage_obstacles())


def test_rrt_reaches_every_free_spot():
    rects = c.garage_obstacles()
    for spot in c.free_spots():
        assert c.pose_free(*c.garage_goal(spot), rects)
        goal = np.array([c.garage_goal(spot)[0], 9.0])
        res = c.rrt(c.GARAGE["start"], goal, rects, goal_bias=0.05, seed=4)
        assert res["path"] is not None, spot
