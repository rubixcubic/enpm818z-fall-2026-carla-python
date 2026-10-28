"""L8 shared code: one Town01 junction, its road users, and the predictors.

Everything here runs with numpy alone. The lanes come from l8_scene.npz, which
export_scene.py wrote from Town01.xodr through carla.Map (no CARLA server).

Map frame: x east, y north, meters (CARLA's y is flipped, as in lecture6).
Time: t = 0 is "now", the moment we predict from. The track before it is the
history the predictors may read; the truth after it is what they must guess.

Every number below is chosen for the lecture, and says so. The lecture's
measured numbers come from running this file (see the slides' measured.json).
"""
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- chosen settings
DT = 0.1          # s between tracked positions (10 Hz, chosen)
HIST = 2.0        # s of track history a predictor may read (chosen)
FIT = 1.5         # s of the newest history used to fit velocity and acceleration (chosen)
HORIZON = 5.0     # s predicted (chosen)
NOISE = 0.05      # m, 1 sigma of each tracked position, per axis, after the tracker (chosen)
CAR_LEN = 4.5     # m, every car (chosen, as in L5 and L6)
V_CRUISE = 8.0    # m/s, cruising speed (chosen; GP4's default target_speed is 8.0)
A_LAT = 2.0       # m/s^2, sideways acceleration a predictor assumes in a turn (chosen)
MISS = 2.0        # m, a prediction misses when its final error is larger (Argoverse's threshold)
SIG_V = 1.0       # m/s, how far a driver's speed may differ from a mode's (chosen)
B_EXP = 1.0       # m/s^2, the braking we expect before a turn (chosen)
SIG_TH = np.radians(8.0)   # rad, same for heading (chosen)
SIG_LAT = 0.7     # m, same for the sideways offset from a lane's center (chosen)
SIG_ACC = 0.5     # m/s per sqrt(s): how fast speed may wander, for CV's uncertainty (chosen)
GAP = 4.1         # s, the shortest gap a left-turning driver accepts (critical gap)
D_DECIDE = 6.0    # m before the junction where car A decides to turn or wait (chosen)

# IDM, the Intelligent Driver Model (Treiber, Hennecke and Helbing 2000). The paper's
# values for T, a, b, delta and s0; v0 is our street's (chosen).
IDM = dict(v0=9.0, T=1.6, a=0.73, b=1.67, delta=4.0, s0=2.0)

# The moves through the junction, named by the direction of travel before and after.
MOVES = {
    "west_to_west": "westbound, straight on",
    "west_to_south": "westbound, left turn",
    "east_to_east": "eastbound, straight on",
    "east_to_south": "eastbound, right turn",
    "north_to_west": "northbound, left turn",
    "north_to_east": "northbound, right turn",
}
APPROACH = {"west": ("west_to_west", "west_to_south"),
            "east": ("east_to_east", "east_to_south"),
            "north": ("north_to_west", "north_to_east")}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------- the map
class Lane:
    """One lane centerline as a polyline with arc length s (m) from its first point."""

    def __init__(self, pts):
        pts = np.asarray(pts, float)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        pts = pts[np.r_[True, seg > 1e-6]]
        self.pts = pts
        self.s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        d = np.gradient(pts, self.s, axis=0)
        self.heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.length = float(self.s[-1])

    def at(self, s):
        s = np.clip(s, 0.0, self.length)
        return np.stack([np.interp(s, self.s, self.pts[:, 0]),
                         np.interp(s, self.s, self.pts[:, 1])], axis=-1)

    def heading_at(self, s):
        return np.interp(np.clip(s, 0.0, self.length), self.s, self.heading)

    def project(self, p):
        """Arc length of the nearest point, and the sideways offset (m, left positive)."""
        p = np.asarray(p, float)
        a, b = self.pts[:-1], self.pts[1:]
        ab = b - a
        u = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        q = a + u[:, None] * ab
        d = np.hypot(*(p - q).T)
        i = int(np.argmin(d))
        s = self.s[i] + u[i] * (self.s[i + 1] - self.s[i])
        side = np.sign(ab[i, 0] * (p[1] - a[i, 1]) - ab[i, 1] * (p[0] - a[i, 0]))
        return float(s), float(side * d[i])


def load_scene(path=HERE / "l8_scene.npz"):
    d = np.load(path)
    lanes = {n[5:]: Lane(d[n]) for n in d.files if n.startswith("path_")}
    info = {}
    for name, ln in lanes.items():
        s_in = ln.project(d[f"entry_{name}"])[0]
        s_out = ln.project(d[f"exit_{name}"])[0]
        turn = float(wrap(ln.heading_at(s_out) - ln.heading_at(s_in)))
        radius = (s_out - s_in) / abs(turn) if abs(turn) > np.radians(30) else np.inf
        info[name] = dict(s_in=s_in, s_out=s_out, turn_deg=float(np.degrees(turn)),
                          inside=float(s_out - s_in), radius=float(radius))
    edges = [d[n] for n in sorted(d.files) if n.startswith("lane_") and n != "lane_in_junction"]
    return dict(lanes=lanes, info=info, edges=edges, edge_in_junction=d["lane_in_junction"], window=d["window"],
                center=d["junction_center"], source=str(d["source"]))


def turn_speed(info, a_lat=A_LAT):
    """Speed through a turn of radius R with sideways acceleration a_lat: sqrt(a_lat R)."""
    return float(np.sqrt(a_lat * info["radius"])) if np.isfinite(info["radius"]) else np.inf


def conflict_point(scene, a="west_to_south", b="east_to_east"):
    """Where two lanes' centerlines cross: arc length on each, and the point."""
    la, lb = scene["lanes"][a], scene["lanes"][b]
    sa = np.arange(0, la.length, 0.05)
    pa = la.at(sa)
    best = (np.inf, 0, 0)
    for i in range(0, len(sa), 4):
        s, off = lb.project(pa[i])
        if abs(off) < best[0]:
            best = (abs(off), sa[i], s)
    return dict(s_a=float(best[1]), s_b=float(best[2]), xy=la.at(best[1]))


# ----------------------------------------------------------------- true motion
def drive(lane, info, s0, v0, t, b=1.0, v_turn=None, a_up=1.0, v_cruise=V_CRUISE,
          wait=None, rng=None, wander=0.0):
    """A driver along one lane, stepped at 0.01 s, sampled at the times t (s).

    Cruise at v_cruise; for a turn, brake at b to reach v_turn 1 m before the junction,
    hold it to the junction's exit, then speed up at a_up. wait = (t_decide_s_before,
    t_go): when the car is that many meters before the junction, it stops 1 m before
    it and waits until time t_go (the gap-acceptance run). wander: 1 sigma (m/s^2) of a
    slow random speed change (a driver never holds a speed exactly)."""
    h = 0.01
    tt = np.arange(t[0], t[-1] + h / 2, h)
    s, v, acc = np.empty(len(tt)), np.empty(len(tt)), np.zeros(len(tt))
    s[0], v[0] = s0, v0
    turning = np.isfinite(info["radius"])
    if v_turn is None:
        v_turn = turn_speed(info) if turning else v_cruise
    stop_at = info["s_in"] - 1.0
    stopping = False
    noise = 0.0
    for k in range(len(tt) - 1):
        sk, vk = s[k], v[k]
        if wait is not None and not stopping and sk >= info["s_in"] - wait[0] and tt[k] < wait[1]:
            stopping = True
        if stopping and tt[k] >= wait[1]:
            stopping = False
        if stopping:
            d = max(stop_at - sk, 0.05)
            a = -min(vk * vk / (2 * d), 6.0) if vk > 0.01 else -vk / h
        elif turning and sk < info["s_out"]:
            target = v_turn
            d = info["s_in"] - 1.0 - sk
            if vk > target and d > 0 and (vk * vk - target * target) / (2 * b) >= d:
                a = -b
            elif d <= 0:
                a = np.clip((target - vk) / 0.5, -b, a_up)
            elif vk < v_cruise and (vk * vk - target * target) / (2 * b) < d - 2.0:
                a = min(a_up, (v_cruise - vk) / 0.5)
            else:
                a = max(-b, min(0.0, (v_cruise - vk) / 0.5))
        else:
            a = np.clip((v_cruise - vk) / 0.5, -b, a_up)
        if wander and rng is not None:
            noise += (-noise / 2.0) * h + wander * np.sqrt(h) * rng.standard_normal()
            a += noise
        acc[k] = a
        v[k + 1] = max(vk + a * h, 0.0)
        s[k + 1] = sk + 0.5 * (vk + v[k + 1]) * h
    idx = np.clip(np.round((t - tt[0]) / h).astype(int), 0, len(tt) - 1)
    return dict(t=t, s=s[idx], v=v[idx], a=acc[idx], xy=lane.at(s[idx]),
                heading=lane.heading_at(s[idx]))


def idm_accel(v, gap, dv, p=IDM):
    """IDM: a [1 - (v/v0)^delta - (s*/s)^2], s* = s0 + v T + v dv / (2 sqrt(a b))."""
    s_star = p["s0"] + max(0.0, v * p["T"] + v * dv / (2 * np.sqrt(p["a"] * p["b"])))
    free = 1 - (v / p["v0"]) ** p["delta"]
    return p["a"] * (free - (s_star / gap) ** 2 if np.isfinite(gap) else free)


def follow(lane, s0, v0, t, leader_xy, p=IDM):
    """A car on `lane` driven by the IDM behind a leader whose positions at times t are
    leader_xy. The leader counts while it is ahead and within 1.5 m of the lane's center."""
    h = 0.01
    tt = np.arange(t[0], t[-1] + h / 2, h)
    lx = np.interp(tt, t, leader_xy[:, 0])
    ly = np.interp(tt, t, leader_xy[:, 1])
    s, v, acc = np.empty(len(tt)), np.empty(len(tt)), np.zeros(len(tt))
    s[0], v[0] = s0, v0
    sl_prev = None
    for k in range(len(tt) - 1):
        sl, off = lane.project((lx[k], ly[k]))
        vl = 0.0 if sl_prev is None else (sl - sl_prev) / h
        sl_prev = sl
        gap = sl - s[k] - CAR_LEN if (abs(off) < 1.5 and sl > s[k]) else np.inf
        a = idm_accel(v[k], max(gap, 0.1), v[k] - vl, p)
        acc[k] = a
        v[k + 1] = max(v[k] + a * h, 0.0)
        s[k + 1] = s[k] + 0.5 * (v[k] + v[k + 1]) * h
    idx = np.clip(np.round((t - tt[0]) / h).astype(int), 0, len(tt) - 1)
    return dict(t=t, s=s[idx], v=v[idx], a=acc[idx], xy=lane.at(s[idx]),
                heading=lane.heading_at(s[idx]))


def times(t0=-HIST, t1=HORIZON + 1.0):
    return np.round(np.arange(t0, t1 + DT / 2, DT), 6)


def our_scene(scene=None, av_closer=0.0, seed=818):
    """The lecture's scene. Car A (westbound) turns left across the AV's lane; car B
    follows A and goes straight on; the AV drives east at 8 m/s.

    Chosen: A cruises at 8 m/s, brakes at 1.0 m/s^2 to 5.0 m/s for the turn, and "now"
    (t = 0) is 2.0 s into that braking. A decides D_DECIDE m before the junction: it turns if the
    AV is at least GAP seconds from where their lanes cross, else it stops and waits.
    The AV is 6.0 s from that point when A decides; av_closer (m) moves it closer, for
    the second run. B starts 24 m behind A at 8 m/s and follows it with the IDM."""
    scene = scene or load_scene()
    la, ia = scene["lanes"]["west_to_south"], scene["info"]["west_to_south"]
    lav, iav = scene["lanes"]["east_to_east"], scene["info"]["east_to_east"]
    lb = scene["lanes"]["west_to_west"]
    t = times(-6.0, HORIZON + 3.0)
    v_turn, b = 5.0, 1.0
    s_brake = ia["s_in"] - 1.0 - (V_CRUISE ** 2 - v_turn ** 2) / (2 * b)
    s_a0 = s_brake - V_CRUISE * (6.0 - 2.0)          # brakes at t = -2.0 s
    a_free = drive(la, ia, s_a0, V_CRUISE, t, b=b, v_turn=v_turn)
    t_dec = float(np.interp(ia["s_in"] - D_DECIDE, a_free["s"], t))
    cp = conflict_point(scene)
    gap = 6.0 - av_closer / V_CRUISE
    s_av0 = cp["s_b"] - V_CRUISE * (t_dec - t[0] + 6.0) + av_closer
    av = dict(t=t, s=s_av0 + V_CRUISE * (t - t[0]), v=np.full_like(t, V_CRUISE), a=np.zeros_like(t))
    av["xy"], av["heading"] = lav.at(av["s"]), lav.heading_at(av["s"])
    t_av_pass = float(np.interp(cp["s_b"] + CAR_LEN, av["s"], t))
    if gap >= GAP:
        a = a_free
        went = True
    else:
        a = drive(la, ia, s_a0, V_CRUISE, t, b=b, v_turn=v_turn, wait=(D_DECIDE, t_av_pass + 1.0))
        went = False
    s_b0 = s_a0 - 24.0 - CAR_LEN
    bcar = follow(lb, s_b0, V_CRUISE, t, a["xy"])
    rng = np.random.default_rng(seed)        # the noise is drawn from t = -6 s on, so the
    for d in (a, bcar):                         # track does not depend on how much is kept
        d["track"] = d["xy"] + NOISE * rng.standard_normal(d["xy"].shape)
    keep = t >= -HIST - 1e-9
    out = {}
    for name, d in (("A", a), ("B", bcar), ("AV", av)):
        out[name] = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    out["t"] = t[keep]
    out["meta"] = dict(t_decide=t_dec, gap=gap, went=went, conflict=cp, t_av_pass=t_av_pass,
                       v_turn=v_turn, brake=b, av_closer=av_closer)
    return out


# ----------------------------------------------------------------- reading the track
def fit_state(t, track, now=0.0, window=FIT):
    """Least-squares fit of p(t) = p0 + v t + a t^2 / 2 (t from now) to the track's last
    `window` seconds. Returns the fitted position, velocity, acceleration, and from them
    the speed, heading, yaw rate and the acceleration along the heading."""
    m = (t <= now + 1e-9) & (t >= now - window - 1e-9)
    tau = t[m] - now
    X = np.c_[np.ones_like(tau), tau, 0.5 * tau ** 2]
    coef, *_ = np.linalg.lstsq(X, track[m], rcond=None)
    p0, v, a = coef[0], coef[1], coef[2]
    speed = float(np.hypot(*v))
    heading = float(np.arctan2(v[1], v[0]))
    omega = float((v[0] * a[1] - v[1] * a[0]) / max(speed ** 2, 1e-6))
    a_along = float(v @ a / max(speed, 1e-6))
    # the velocity's covariance from a straight-line fit, for CV's uncertainty
    X1 = np.c_[np.ones_like(tau), tau]
    cov1 = NOISE ** 2 * np.linalg.inv(X1.T @ X1)
    return dict(p0=p0, v=v, a=a, speed=speed, heading=heading, omega=omega,
                a_along=a_along, cov_pv=cov1, n=int(m.sum()))


def horizons(horizon=HORIZON):
    return np.round(np.arange(DT, horizon + DT / 2, DT), 6)


# ----------------------------------------------------------------- predictors
def predict_cv(st, h=None):
    """Constant velocity: p(h) = p0 + v h."""
    h = horizons() if h is None else h
    return st["p0"] + np.outer(h, st["v"])


def predict_ctrv(st, h=None):
    """Constant turn rate and velocity: an arc of radius v / omega."""
    h = horizons() if h is None else h
    v, th, w = st["speed"], st["heading"], st["omega"]
    if abs(w) < 1e-3:
        return predict_cv(st, h)
    x = st["p0"][0] + v / w * (np.sin(th + w * h) - np.sin(th))
    y = st["p0"][1] + v / w * (np.cos(th) - np.cos(th + w * h))
    return np.c_[x, y]


def predict_ctra(st, h=None):
    """Constant turn rate and acceleration (speed held at 0 once it reaches 0)."""
    h = horizons() if h is None else h
    step = 0.01
    n = int(round(h[-1] / step))
    p, v, th = st["p0"].astype(float).copy(), st["speed"], st["heading"]
    out, k = [], 0
    for i in range(1, n + 1):
        v2 = max(v + st["a_along"] * step, 0.0)
        p = p + 0.5 * (v + v2) * step * np.array([np.cos(th), np.sin(th)])
        th += st["omega"] * step if v2 > 0 else 0.0
        v = v2
        if k < len(h) and abs(i * step - h[k]) < step / 2:
            out.append(p.copy())
            k += 1
    return np.array(out)


def cv_cov(st, h=None, sig_acc=SIG_ACC):
    """CV's position covariance at each horizon (the same for x and y): the fit's
    position and velocity uncertainty carried forward, plus a speed that wanders by
    sig_acc: var = var_p + 2 h cov_pv + h^2 var_v + sig_acc^2 h^3 / 3."""
    h = horizons() if h is None else h
    c = st["cov_pv"]
    return c[0, 0] + 2 * h * c[0, 1] + h ** 2 * c[1, 1] + sig_acc ** 2 * h ** 3 / 3


def candidate_lanes(scene, p, heading, max_off=2.5, max_dth=np.radians(60)):
    """Every lane through the junction that the road user could be on now."""
    out = []
    for name, ln in scene["lanes"].items():
        s, off = ln.project(p)
        if abs(off) <= max_off and abs(wrap(heading - ln.heading_at(s))) <= max_dth:
            out.append(name)
    return out


def mode_plan(lane, info, s0, v0, h, a_lat=A_LAT, a_up=1.0, v_street=V_CRUISE):
    """One mode's future along its lane, as arc length s(h). For a turn: brake evenly to
    sqrt(a_lat R) by the junction, hold that speed through it, then speed up at a_up to
    the street's speed. Straight on: hold the speed. Also returns the braking the mode
    expects now (m/s^2, negative = braking)."""
    if not (np.isfinite(info["radius"]) and s0 < info["s_out"]):
        return s0 + v0 * h, 0.0
    vt = turn_speed(info, a_lat)
    step = 0.01
    n = int(round(h[-1] / step))
    d = info["s_in"] - s0
    b = (v0 ** 2 - vt ** 2) / (2 * d) if (v0 > vt and d > 0.5) else 0.0
    s, v, out, k = s0, v0, [], 0
    for i in range(1, n + 1):
        if s < info["s_in"] and v > vt and b > 0:
            a = -b
        elif s < info["s_out"]:
            a = np.clip((min(v0, vt) - v) / 0.5, -2.0, a_up)
        else:
            a = min(a_up, max(v_street - v, 0.0) / 0.5)
        v2 = max(v + a * step, 0.0)
        s += 0.5 * (v + v2) * step
        v = v2
        if k < len(h) and abs(i * step - h[k]) < step / 2:
            out.append(s)
            k += 1
    return np.array(out), -b


def wait_plan(lane, info, s0, v0, h, t_go, a_up=1.0, a_lat=A_LAT):
    """A left turn that waits: stop 1 m before the junction, wait until t_go (s from
    now), then go through the turn at sqrt(a_lat R)."""
    t = np.r_[0.0, h]
    d = drive(lane, info, s0, v0, t, b=1.0, v_turn=turn_speed(info, a_lat), a_up=a_up,
              wait=(1e9, t_go))
    return d["s"][1:]


def gauss(x, sig):
    return np.exp(-0.5 * (x / sig) ** 2)


def expected_speed(info, s0, v_street=V_CRUISE, b=B_EXP, a_lat=A_LAT):
    """The speed a mode expects here. Straight on: the street's speed. A turn: at most
    sqrt(a_lat R) in the junction, and before it, the speed from which braking at b
    reaches that by the junction: sqrt(v_turn^2 + 2 b d), never above the street's."""
    if not (np.isfinite(info["radius"]) and s0 < info["s_out"]):
        return v_street
    vt = turn_speed(info, a_lat)
    d = max(info["s_in"] - s0, 0.0)
    return float(min(v_street, np.sqrt(vt * vt + 2 * b * d)))


def predict_modes(scene, st, h=None, prior=None):
    """Map-aware prediction: one future per candidate lane, each with a probability.

    Probability (Bayes, prior equal unless given): how well each mode explains what the
    track shows now: its speed against the speed the mode expects here (SIG_V), its
    heading (SIG_TH) and its offset from the lane's center (SIG_LAT)."""
    h = horizons() if h is None else h
    names = candidate_lanes(scene, st["p0"], st["heading"])
    modes = []
    for name in names:
        ln, info = scene["lanes"][name], scene["info"][name]
        s0, off = ln.project(st["p0"])
        s, a_exp = mode_plan(ln, info, s0, st["speed"], h)
        v_exp = expected_speed(info, s0)
        dth = float(wrap(st["heading"] - ln.heading_at(s0)))
        like = dict(speed=float(gauss(st["speed"] - v_exp, SIG_V)),
                    heading=float(gauss(dth, SIG_TH)), offset=float(gauss(off, SIG_LAT)))
        pr = 1.0 / len(names) if prior is None else prior.get(name, 0.0)
        modes.append(dict(name=name, traj=ln.at(s), a_expected=a_exp, v_expected=v_exp, dheading=dth,
                          offset=off, s0=s0, like=like, prior=pr,
                          weight=pr * like["speed"] * like["heading"] * like["offset"]))
    total = sum(m["weight"] for m in modes) or 1.0
    for m in modes:
        m["prob"] = m["weight"] / total
    return sorted(modes, key=lambda m: -m["prob"])


def predict_modes_with_av(scene, st, av_s, av_v, h=None):
    """predict_modes, plus the gap rule for a left turn across the AV's lane: if the AV
    would reach the crossing less than GAP seconds after the car reaches its decision
    point, the left-turn mode waits until the AV has passed the crossing. (Our
    simulated driver waits 1 s longer: the predictor does not know that.)"""
    h = horizons() if h is None else h
    modes = predict_modes(scene, st, h)
    cp = conflict_point(scene)
    for m in modes:
        if m["name"] != "west_to_south":
            continue
        ln, info = scene["lanes"][m["name"]], scene["info"][m["name"]]
        s0, _ = ln.project(st["p0"])
        s_plan, _ = mode_plan(ln, info, s0, st["speed"], h)
        t_reach = float(np.interp(info["s_in"] - D_DECIDE, s_plan, h)) if s0 < info["s_in"] - D_DECIDE else 0.0
        t_av = (cp["s_b"] - av_s) / av_v
        m["gap"] = t_av - t_reach
        if m["gap"] < GAP:
            t_go = (cp["s_b"] + CAR_LEN - av_s) / av_v
            m["traj"] = ln.at(wait_plan(ln, info, s0, st["speed"], h, t_go))
            m["waits"] = True
    return modes


def predict_follower(scene, lane_name, st, leader_traj, h=None, p=IDM):
    """A follower's future from the IDM, behind a leader whose future positions are
    leader_traj (one per horizon step): the behavior model replaces constant speed."""
    h = horizons() if h is None else h
    ln = scene["lanes"][lane_name]
    s0, _ = ln.project(st["p0"])
    t = np.r_[0.0, h]
    lead = np.vstack([leader_traj[0] - (leader_traj[1] - leader_traj[0]), leader_traj])
    d = follow(ln, s0, st["speed"], t, lead, p)
    return d["xy"][1:], d["v"][1:]


# ----------------------------------------------------------------- scores
def ade(pred, truth):
    return float(np.mean(np.hypot(*(pred - truth).T)))


def fde(pred, truth):
    return float(np.hypot(*(pred[-1] - truth[-1])))


def errors_by_horizon(pred, truth):
    return np.hypot(*(pred - truth).T)


def min_ade_fde(trajs, truth):
    """minADE and minFDE as Argoverse 2 defines them: both from the best mode, the one
    whose end is closest to the truth. Returns (minADE, minFDE, index of that mode)."""
    f = [fde(p, truth) for p in trajs]
    best = int(np.argmin(f))
    return ade(trajs[best], truth), f[best], best


# ----------------------------------------------------------------- our test set
def test_set(scene=None, speeds=(6.0, 7.0, 8.0, 9.0, 10.0), seeds=(1, 2, 3, 4), every=0.5):
    """Many simulated drives through the junction, to measure the predictors on more
    than one scene. Every move, five cruising speeds, four random seeds (all chosen).
    Each drive: braking 0.8 to 2.0 m/s^2 and a turn taken at a sideways acceleration of
    1.5 to 2.5 m/s^2 (drawn at random), and a speed that wanders by 0.3 m/s^2 (chosen).
    A prediction starts every `every` seconds that has HIST of history and HORIZON of
    future, while the car is between 50 m before the junction and its exit. Returns one
    row per prediction."""
    scene = scene or load_scene()
    rows = []
    h = horizons()
    for name, ln in scene["lanes"].items():
        info = scene["info"][name]
        for vc in speeds:
            for seed in seeds:
                rng = np.random.default_rng(1000 * seed + int(vc * 10) + zlib.crc32(name.encode()) % 997)
                b = rng.uniform(0.8, 2.0)
                alat = rng.uniform(1.5, 2.5)
                vt = turn_speed(info, alat) if np.isfinite(info["radius"]) else None
                t_end = (ln.length - 5.0) / max(vc, 1.0) + 8.0
                t = np.round(np.arange(0.0, t_end, DT), 6)
                d = drive(ln, info, 2.0, vc, t, b=b, v_turn=vt, v_cruise=vc, rng=rng, wander=0.3)
                ok = d["s"] < ln.length - 1.0
                track = d["xy"] + NOISE * rng.standard_normal(d["xy"].shape)
                for now in np.arange(HIST, t[-1] - HORIZON, every):
                    k = int(round(now / DT))
                    if not (info["s_in"] - 50.0 <= d["s"][k] <= info["s_out"]):
                        continue
                    fut = slice(k + 1, k + 1 + len(h))
                    if not ok[fut].all() or len(t[fut]) < len(h):
                        continue
                    truth = d["xy"][fut]
                    st = fit_state(t[:k + 1], track[:k + 1], now=t[k])
                    turn = abs(float(wrap(d["heading"][fut][-1] - d["heading"][k]))) > np.radians(20)
                    preds = dict(cv=predict_cv(st, h), ctrv=predict_ctrv(st, h), ctra=predict_ctra(st, h))
                    modes = predict_modes(scene, st, h)
                    if modes:
                        preds["map_top"] = modes[0]["traj"]
                    row = dict(move=name, speed=vc, seed=seed, now=float(t[k]), turn=turn,
                               err={n: errors_by_horizon(p, truth) for n, p in preds.items()})
                    if modes:
                        mi = min_ade_fde([m["traj"] for m in modes], truth)
                        row.update(min_ade=mi[0], min_fde=mi[1], n_modes=len(modes),
                                   p_true=float(sum(m["prob"] for m in modes if m["name"] == name)),
                                   mode_probs=[(m["name"] == name, float(m["prob"])) for m in modes])
                    var = cv_cov(st, h)
                    e = preds["cv"] - truth
                    row["cv_inside95"] = (np.sum(e ** 2, axis=1) / var) <= 5.991
                    rows.append(row)
    return rows
