"""Hand-checkable tests for l8sim.py and the hands-on (pytest, no CARLA)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import l8sim as m

HERE = Path(__file__).resolve().parents[1]
MEASURED = Path(os.environ.get("L8_MEASURED", Path.home() / "Documents/umd/fall2026/ENPM818Z/latex/"
                               "enpm818z-fall-2026-latex/slides/lectures/lecture8/figures/svg/measured.json"))


def state(p0, v, a=(0.0, 0.0)):
    t = np.round(np.arange(-2.0, 1e-9, 0.1), 6)
    xy = np.array(p0) + np.outer(t, v) + 0.5 * np.outer(t ** 2, a)
    return m.fit_state(t, xy)


def test_fit_recovers_speed_heading_and_braking():
    st = state((10.0, 5.0), (-6.0, 0.0), (1.0, 0.0))        # westbound at 6 m/s, braking at 1 m/s^2
    assert st["speed"] == pytest.approx(6.0, abs=1e-9)
    assert np.degrees(st["heading"]) == pytest.approx(180.0, abs=1e-6)
    assert st["a_along"] == pytest.approx(-1.0, abs=1e-9)
    assert st["omega"] == pytest.approx(0.0, abs=1e-9)


def test_cv_by_hand():
    st = state((0.0, 0.0), (3.0, 4.0))
    p = m.predict_cv(st, np.array([1.0, 2.0]))
    assert np.allclose(p, [[3.0, 4.0], [6.0, 8.0]])


def test_ctrv_stays_on_its_circle():
    st = dict(p0=np.array([0.0, 0.0]), speed=5.0, heading=0.0, omega=0.5, v=np.array([5.0, 0.0]))
    p = m.predict_ctrv(st, np.linspace(0.1, 5, 50))
    center = np.array([0.0, 5.0 / 0.5])                    # radius v / omega = 10 m, to the left
    assert np.allclose(np.hypot(*(p - center).T), 10.0)


def test_ctrv_is_cv_when_it_does_not_turn():
    st = state((0.0, 0.0), (8.0, 0.0))
    h = m.horizons()
    assert np.allclose(m.predict_ctrv(st, h), m.predict_cv(st, h))


def test_ade_fde_and_min_over_modes():
    truth = np.array([[1.0, 0.0], [2.0, 0.0]])
    a = np.array([[1.0, 3.0], [2.0, 4.0]])                 # errors 3 and 4
    b = np.array([[1.0, 1.0], [2.0, 0.0]])                 # errors 1 and 0
    assert m.ade(a, truth) == pytest.approx(3.5)
    assert m.fde(a, truth) == pytest.approx(4.0)
    mi_ade, mi_fde, best = m.min_ade_fde([a, b], truth)
    assert (mi_ade, mi_fde, best) == (pytest.approx(0.5), pytest.approx(0.0), 1)
    # minADE is the ADE of the mode with the smallest FDE, not the smallest ADE
    c = np.array([[1.0, 0.0], [2.0, 2.0]])                 # errors 0 and 2: ADE 1.0, FDE 2.0
    d = np.array([[1.0, 1.5], [2.0, 1.5]])                 # errors 1.5 and 1.5: ADE 1.5, FDE 1.5
    mi_ade, mi_fde, best = m.min_ade_fde([c, d], truth)
    assert (mi_ade, mi_fde, best) == (pytest.approx(1.5), pytest.approx(1.5), 1)


def test_idm_holds_its_equilibrium_gap():
    p = m.IDM
    v = 8.0
    s_eq = (p["s0"] + v * p["T"]) / np.sqrt(1 - (v / p["v0"]) ** p["delta"])
    assert m.idm_accel(v, s_eq, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_mode_probabilities_sum_to_one_and_favor_the_turn():
    sc = m.load_scene()
    o = m.our_scene(sc)
    past = o["t"] <= 1e-9
    st = m.fit_state(o["t"][past], o["A"]["track"][past])
    modes = m.predict_modes(sc, st)
    assert sum(md["prob"] for md in modes) == pytest.approx(1.0)
    assert {md["name"] for md in modes} == {"west_to_west", "west_to_south"}
    assert modes[0]["name"] == "west_to_south"


def test_the_two_runs_share_their_history():
    sc = m.load_scene()
    o1, o2 = m.our_scene(sc), m.our_scene(sc, av_closer=24.0)
    past = o1["t"] <= 1e-9
    assert np.allclose(o1["A"]["track"][past], o2["A"]["track"][past])
    assert o1["meta"]["went"] and not o2["meta"]["went"]


def test_numbers_on_the_slides_match_the_code():
    if not MEASURED.exists():
        pytest.skip("slides' measured.json not found")
    mj = json.load(open(MEASURED))
    sc = m.load_scene()
    o = m.our_scene(sc)
    past = o["t"] <= 1e-9
    st = m.fit_state(o["t"][past], o["A"]["track"][past])
    assert round(st["speed"], 2) == mj["fit_A"]["speed"]
    fut = (o["t"] > 1e-9) & (o["t"] <= m.HORIZON + 1e-9)
    assert round(m.fde(m.predict_cv(st), o["A"]["xy"][fut]), 2) == mj["A_now"]["cv"]["fde"]
    modes = m.predict_modes(sc, st)
    assert round(modes[0]["prob"], 3) == mj["modes_A"]["west_to_south"]["prob"]


@pytest.mark.parametrize("part", ["scene", "physics", "modes", "interact"])
def test_each_part_writes_a_snapshot(part, tmp_path):
    import l8_handson as h
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out)])
    assert out.exists() and out.stat().st_size > 10000


def test_no_group_project_solution_code():
    """L8's public code holds no GP4 or GP3 solution (see the slides' design spec)."""
    banned = ["heapq", "astar", "dijkstra(", "pure_pursuit", "BehaviorFSM", "LANE_FOLLOW",
              "OBSTACLE_AVOIDANCE", "EMERGENCY_STOP", "behavior_node", "rclpy", "jacobian"]
    for f in list(HERE.glob("*.py")):
        text = f.read_text().lower()
        for word in banned:
            assert word.lower() not in text, f"{f.name} contains {word}"
