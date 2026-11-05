"""Shared pieces for the L9 hands-on: the road, the Frenet frame, candidate paths,
the circle check and the costs. numpy only.

Frames. The map frame is in meters: x as CARLA's x, y as minus CARLA's y, so the
frame is right-handed (as in L6). The Frenet frame follows the reference path,
the center of the AV's lane: s is the distance along it, d the distance across
it, positive to the left of the driving direction.

Every number in SCENE, AV, WEIGHTS and GARAGE is chosen for the example; the road
itself is measured (export_scene.py reads it from Town03.xodr).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- chosen values
AV = {"length": 4.5, "width": 1.9, "rear_to_base": 0.9,   # the course's AV drawing
      "wheelbase": 2.875, "max_steer": 0.7}               # wheelbase chosen; the 0.7 rad limit is GP4's
SCENE = {"av_s": 60.0, "av_d": 0.0, "speed": 10.0,        # the AV on the right lane's center
         "car_s": 118.0, "car_d": -0.3,                    # the parked car, in the right lane
         "horizon": 70.0}                                  # meters of path ahead
CANDIDATES = {"d_end": tuple(np.round(np.arange(-0.5, 4.01, 0.5), 2)),
              "maneuver": (20.0, 30.0, 40.0)}
WEIGHTS = {"comfort": 2.0,     # per m/s^2 of peak lateral acceleration
           "clearance": 4.0,   # per m that the gap falls under CLEAR_SOFT
           "lane": 2.0,        # per lane the path ends away from the route's lane
           "center": 2.0,      # per m the path ends away from a lane center
           "progress": 0.5}    # per m of the horizon the AV does not cover
MARGIN = 0.3       # m: a gap under this is a collision (L6's 0.29 m error bound, rounded)
CLEAR_SOFT = 1.5   # m: a gap under this costs
A_LAT_MAX = 3.0    # m/s^2: a path that needs more is not comfortable: dropped
STOP_GAP = 2.0     # m: the stop-behind candidate stops this far behind the car
N_CIRCLES = 3
STEP = 0.5         # m between path points


# ----------------------------------------------------------------- the road
def load_scene(path=None):
    """The exported stretch: the reference (right lane center) with s, heading and
    curvature, and the left lane's center."""
    d = np.load(path or HERE / "l9_scene.npz")
    right, left = d["right"], d["left"]
    ref = make_reference(right[:, 0], right[:, 1])
    ref["left_xy"] = left[:, :2]
    ref["lane_width"] = float(right[0, 3])
    return ref


def make_reference(x, y):
    """A reference path from points: arc length s, heading and curvature at each point."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    hd = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    kappa = np.gradient(hd, s)
    # the map's curves are arcs of constant curvature: smooth the finite-difference noise
    k = 5
    kappa = np.convolve(np.pad(kappa, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return {"x": x, "y": y, "s": s, "heading": hd, "kappa": kappa}


def ref_at(ref, s):
    """Point, heading and curvature of the reference at arc length s (interpolated)."""
    s = np.asarray(s, float)
    return (np.interp(s, ref["s"], ref["x"]), np.interp(s, ref["s"], ref["y"]),
            np.interp(s, ref["s"], ref["heading"]), np.interp(s, ref["s"], ref["kappa"]))


def from_frenet(ref, s, d):
    """Map (x, y) of the point d meters left of the reference at arc length s:
    r(s) + d n(s), with n the unit normal, 90 degrees left of the heading."""
    rx, ry, h, _ = ref_at(ref, s)
    return rx - np.asarray(d) * np.sin(h), ry + np.asarray(d) * np.cos(h)


def to_frenet(ref, x, y):
    """(s, d) of a map point: the nearest point on the reference polyline gives s;
    the signed distance to it gives d (positive to the left). Also returns the
    foot point, for the slides."""
    P = np.array([x, y], float)
    A = np.column_stack([ref["x"][:-1], ref["y"][:-1]])
    B = np.column_stack([ref["x"][1:], ref["y"][1:]])
    AB = B - A
    L2 = np.sum(AB ** 2, axis=1)
    t = np.clip(np.sum((P - A) * AB, axis=1) / L2, 0.0, 1.0)
    F = A + t[:, None] * AB
    i = int(np.argmin(np.sum((F - P) ** 2, axis=1)))
    seglen = np.sqrt(L2[i])
    s = ref["s"][i] + t[i] * seglen
    tx, ty = AB[i] / seglen
    dx, dy = P - F[i]
    d = tx * dy - ty * dx                     # cross product: positive to the left
    return float(s), float(d), (float(F[i, 0]), float(F[i, 1])), i


# ----------------------------------------------------------------- the lateral curve
def smooth_step(u):
    """The lateral curve's shape: 0 at u = 0, 1 at u = 1, with zero slope and zero
    curvature at both ends (a quintic: 10u^3 - 15u^4 + 6u^5)."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def quintic(d0, dp0, dpp0, d1, length):
    """Coefficients of d(s) = sum c_k (s - s0)^k from (d0, d'0, d''0) at s0 to
    (d1, 0, 0) at s0 + length. With d'0 = d''0 = 0 it is d0 + (d1 - d0) smooth_step."""
    L = length
    c0, c1, c2 = d0, dp0, dpp0 / 2
    M = np.array([[L ** 3, L ** 4, L ** 5], [3 * L ** 2, 4 * L ** 3, 5 * L ** 4], [6 * L, 12 * L ** 2, 20 * L ** 3]])
    rhs = np.array([d1 - c0 - c1 * L - c2 * L ** 2, -c1 - 2 * c2 * L, -2 * c2])
    c3, c4, c5 = np.linalg.solve(M, rhs)
    return np.array([c0, c1, c2, c3, c4, c5])


def lateral(c, length, ds):
    """d, d' and d'' at ds meters past the start (constant after the maneuver)."""
    ds = np.minimum(np.asarray(ds, float), length)
    p = np.polynomial.polynomial
    d = p.polyval(ds, c)
    d1 = p.polyval(ds, p.polyder(c))
    d2 = p.polyval(ds, p.polyder(c, 2))
    return d, d1, d2


# ----------------------------------------------------------------- paths
def make_path(ref, s0, d_state, d_end, maneuver, horizon, step=STEP):
    """One candidate: from (s0, d, d', d'') to d_end over `maneuver` meters of s,
    then straight on at d_end until s0 + horizon. Returns s, d and the map points,
    heading and curvature."""
    d0, dp0, dpp0 = d_state
    c = quintic(d0, dp0, dpp0, d_end, maneuver)
    ds = np.arange(0.0, horizon + 1e-9, step)
    if horizon - ds[-1] > 1e-6:
        ds = np.append(ds, horizon)          # end exactly at the horizon
    s = s0 + ds
    d, dp, _ = lateral(c, maneuver, ds)
    dp = np.where(ds <= maneuver, dp, 0.0)
    x, y = from_frenet(ref, s, d)
    _, _, hr, kr = ref_at(ref, s)
    heading = hr + np.arctan2(dp, 1 - kr * d)
    kappa = path_curvature(kr, d, dp, np.where(ds <= maneuver, lateral(c, maneuver, ds)[2], 0.0))
    return {"s": s, "d": d, "x": x, "y": y, "heading": heading, "kappa": kappa,
            "d_end": float(d_end), "maneuver": float(maneuver), "coef": c}


def path_curvature(kr, d, dp, dpp):
    """Curvature of the path d(s) drawn beside a reference of curvature kr, with the
    reference's curvature taken as constant over a short piece (tests/test_l9common.py
    checks it against the curvature of the drawn points). On a straight reference it
    is d'' / (1 + d'^2)^(3/2); for a constant d it is kr / (1 - kr d)."""
    # p(s) = r(s) + d(s) n(s), so p' = (1 - kr d) t + d' n and
    # p'' = -2 kr d' t + ((1 - kr d) kr + d'') n; curvature = (p' x p'') / |p'|^3.
    q = 1 - kr * d
    return (kr * q ** 2 + q * dpp + 2 * kr * dp ** 2) / (q ** 2 + dp ** 2) ** 1.5


def lane_cost_parts(d_end, lane_width):
    """Lanes away from the route's lane (d = 0) and meters away from the nearest lane
    center, for a path that ends at d_end. Lane centers sit at d = 0, w, 2w, ..."""
    k = int(np.round(d_end / lane_width))
    return abs(k), abs(d_end - k * lane_width)


def circle_radius(length=AV["length"], width=AV["width"], n=N_CIRCLES):
    """Radius of n equal circles, centers evenly spaced on the center line, that
    together cover a length x width rectangle: sqrt((length / 2n)^2 + (width / 2)^2)."""
    return float(np.hypot(length / (2 * n), width / 2))


def circle_offsets(length=AV["length"], rear_to_ref=AV["rear_to_base"], n=N_CIRCLES):
    """Where the circle centers sit along the body, measured forward from the reference
    point (the AV: its base link, 0.9 m in from the rear bumper)."""
    return (np.arange(n) + 0.5) * length / n - rear_to_ref


def circles(x, y, heading, offsets):
    """Circle centers for poses (x, y, heading): one row per pose, one column per circle."""
    x, y, h = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, heading))
    cx = x[:, None] + np.cos(h)[:, None] * offsets[None, :]
    cy = y[:, None] + np.sin(h)[:, None] * offsets[None, :]
    return cx, cy


def parked_car(ref, scene=SCENE):
    """The parked car: center, heading and circle centers (it points along the lane)."""
    x, y = from_frenet(ref, scene["car_s"], scene["car_d"])
    _, _, h, _ = ref_at(ref, scene["car_s"])
    off = circle_offsets(rear_to_ref=AV["length"] / 2)    # reference point: its center
    cx, cy = circles(x, y, h, off)
    return {"x": float(x), "y": float(y), "heading": float(h), "cx": cx[0], "cy": cy[0]}


def gaps(path, obstacles, offsets=None, r=None):
    """Smallest gap at each path point between the AV's circles and any obstacle
    circle: center distance minus the two radii. Negative means overlap."""
    offsets = circle_offsets() if offsets is None else offsets
    r = circle_radius() if r is None else r
    cx, cy = circles(path["x"], path["y"], path["heading"], offsets)
    g = np.full(len(path["x"]), np.inf)
    for ob in obstacles:
        dx = cx[:, :, None] - ob["cx"][None, None, :]
        dy = cy[:, :, None] - ob["cy"][None, None, :]
        g = np.minimum(g, np.min(np.hypot(dx, dy), axis=(1, 2)) - 2 * r)
    return g


# ----------------------------------------------------------------- costs
def score(path, obstacles, speed, horizon, weights=WEIGHTS, progress=None, lane_width=3.5):
    """The four cost terms and their sum. A path is dropped (cost inf) if a gap falls
    under MARGIN or its peak lateral acceleration passes A_LAT_MAX."""
    g = gaps(path, obstacles)
    a_lat = speed ** 2 * np.abs(path["kappa"])
    gmin, apk = float(np.min(g)), float(np.max(a_lat))
    prog = horizon if progress is None else progress
    lanes, off = lane_cost_parts(path["d_end"], lane_width)
    terms = {"comfort": weights["comfort"] * apk,
             "clearance": weights["clearance"] * max(0.0, CLEAR_SOFT - gmin),
             "lane": weights["lane"] * lanes + weights["center"] * off,
             "progress": weights["progress"] * (horizon - prog)}
    reason = None
    if gmin < MARGIN:
        reason = "collision"
    elif apk > A_LAT_MAX:
        reason = "too sharp"
    total = float("inf") if reason else float(sum(terms.values()))
    return {"terms": terms, "total": total, "gap_min": gmin, "a_peak": apk,
            "dropped": reason, "progress": prog}


def stop_behind(ref, s0, d_state, car, speed, horizon, scene=SCENE):
    """The candidate that keeps the lane and stops STOP_GAP behind the parked car:
    its path ends where the AV's front bumper is STOP_GAP short of the car's rear."""
    front = AV["length"] - AV["rear_to_base"]
    s_stop = scene["car_s"] - AV["length"] / 2 - STOP_GAP - front
    p = make_path(ref, s0, d_state, 0.0, 30.0, max(s_stop - s0, STEP))
    return p, s_stop - s0


def plan(ref, s0, d_state, obstacles, speed=SCENE["speed"], horizon=SCENE["horizon"],
         weights=WEIGHTS, cands=CANDIDATES, scene=SCENE, with_stop=True):
    """Every candidate, scored; the cheapest one first. The planner the hands-on
    runs: sample lateral end offsets and maneuver lengths, check, score, pick."""
    out = []
    for L in cands["maneuver"]:
        for de in cands["d_end"]:
            p = make_path(ref, s0, d_state, de, L, horizon)
            out.append((p, score(p, obstacles, speed, horizon, weights, lane_width=ref["lane_width"])))
    ahead = scene["car_s"] - s0
    if with_stop and 0.0 < ahead < horizon + AV["length"]:
        p, prog = stop_behind(ref, s0, d_state, None, speed, horizon, scene)
        p["stop"] = True
        out.append((p, score(p, obstacles, speed, horizon, weights, progress=prog,
                             lane_width=ref["lane_width"])))
    out.sort(key=lambda ps: ps[1]["total"])
    return out


def drive(ref, obstacles, s_start=SCENE["av_s"], d_start=SCENE["av_d"], speed=SCENE["speed"],
          period=0.5, duration=12.0, weights=WEIGHTS):
    """Receding horizon: plan, drive the first `period` seconds of the cheapest path,
    plan again from where the AV is. Returns the AV's (s, d) states and each plan."""
    s0, state = s_start, (d_start, 0.0, 0.0)
    states, plans = [], []
    for _ in range(int(round(duration / period))):
        cands = plan(ref, s0, state, obstacles, speed=speed, weights=weights)
        best = cands[0][0]
        plans.append({"s0": s0, "state": state, "best": best, "stop": bool(best.get("stop"))})
        states.append((s0, state[0]))
        if best.get("stop"):
            break
        adv = speed * period
        L = best["maneuver"]
        dd, d1, d2 = lateral(best["coef"], L, adv)
        if adv >= L:
            d1, d2 = 0.0, 0.0
        s0, state = s0 + adv, (float(dd), float(d1), float(d2))
    return states, plans


# ----------------------------------------------------------------- the garage (RRT)
GARAGE = {"size": (36.0, 18.0),          # m: one aisle between two rows of spots (chosen)
          "spot": (2.6, 5.5),            # spot width and depth (chosen)
          "n_spots": 11,
          "start": (3.0, 9.0, 0.0),      # the AV enters the aisle from the left, facing +x
          "goal_spot": 7,                # the free spot in the top row
          "free_bottom": (2, 3)}         # two free spots in the bottom row


def garage_spots():
    """Spot rectangles (x0, y0, x1, y1): one row along the top wall, one along the bottom."""
    W, H = GARAGE["size"]
    w, depth = GARAGE["spot"]
    xs = 6.0 + w * np.arange(GARAGE["n_spots"])
    top = [(x, H - depth, x + w, H) for x in xs]
    bottom = [(x, 0.0, x + w, depth) for x in xs]
    return top, bottom


def garage_obstacles():
    """Parked cars (rectangles) in every spot except the free one, as (x0, y0, x1, y1)."""
    top, bottom = garage_spots()
    cars = []
    for i, (x0, y0, x1, y1) in enumerate(top):
        if i == GARAGE["goal_spot"]:
            continue
        cars.append((x0 + 0.35, y1 - 0.4 - AV["length"], x1 - 0.35, y1 - 0.4))
    for i, (x0, y0, x1, y1) in enumerate(bottom):
        if i in GARAGE["free_bottom"]:
            continue
        cars.append((x0 + 0.35, y0 + 0.4, x1 - 0.35, y0 + 0.4 + AV["length"]))
    return cars


def free_spots():
    """The garage's free spots by name: "top7", the slides' goal, then the two free
    spots of the bottom row ("bottom2", "bottom3")."""
    return [f"top{GARAGE['goal_spot']}"] + [f"bottom{i}" for i in GARAGE["free_bottom"]]


def garage_goal(spot=None):
    """The goal pose: base link in a free spot, the AV reversed in, its rear bumper 0.8 m
    from the wall. spot is one of free_spots(); by default the top one, as on the slides."""
    spot = spot or free_spots()[0]
    if spot not in free_spots():
        raise ValueError(f"{spot} is not a free spot; choose one of {free_spots()}")
    top, bottom = garage_spots()
    if spot.startswith("top"):
        x0, y0, x1, y1 = top[int(spot[3:])]
        return ((x0 + x1) / 2, y1 - 0.8 - AV["rear_to_base"], -np.pi / 2)    # facing down
    x0, y0, x1, y1 = bottom[int(spot[6:])]
    return ((x0 + x1) / 2, y0 + 0.8 + AV["rear_to_base"], np.pi / 2)         # facing up


def pose_free(x, y, h, rects, size=GARAGE["size"], margin=0.1):
    """True if the AV's three circles at pose (x, y, h) clear every rectangle and the walls."""
    r = circle_radius() + margin
    cx, cy = circles(x, y, h, circle_offsets())
    cx, cy = cx[0], cy[0]
    W, H = size
    if np.any(cx - r < 0) or np.any(cx + r > W) or np.any(cy - r < 0) or np.any(cy + r > H):
        return False
    for x0, y0, x1, y1 in rects:
        qx = np.clip(cx, x0, x1)
        qy = np.clip(cy, y0, y1)
        if np.any(np.hypot(cx - qx, cy - qy) < r):
            return False
    return True


def rrt(start, goal_xy, rects, n_iter=1500, step=1.5, goal_bias=0.1, goal_tol=1.5, seed=9):
    """A basic RRT in the plane for the AV's base link. Each edge is a straight step;
    the AV's heading on an edge is the edge's direction, and every point of an edge is
    checked with the three circles. Returns the tree (nodes, parents), the path if the
    goal was reached, and the iteration that reached it."""
    rng = np.random.default_rng(seed)
    W, H = GARAGE["size"]
    nodes = [np.array(start[:2], float)]
    parents = [-1]
    hit = None
    for it in range(1, n_iter + 1):
        q = np.array(goal_xy, float) if rng.random() < goal_bias else rng.uniform([0, 0], [W, H])
        D = np.array(nodes)
        i = int(np.argmin(np.sum((D - q) ** 2, axis=1)))
        v = q - D[i]
        n = np.hypot(*v)
        if n < 1e-9:
            continue
        new = D[i] + v / n * min(step, n)
        h = np.arctan2(v[1], v[0])
        ok = all(pose_free(*(D[i] + f * (new - D[i])), h, rects) for f in np.linspace(0, 1, 4))
        if not ok:
            continue
        nodes.append(new)
        parents.append(i)
        if np.hypot(*(new - goal_xy)) < goal_tol:
            hit = it
            break
    path = None
    if hit is not None:
        k, path = len(nodes) - 1, []
        while k != -1:
            path.append(nodes[k])
            k = parents[k]
        path = np.array(path[::-1])
    return {"nodes": np.array(nodes), "parents": np.array(parents), "path": path, "iterations": hit}
