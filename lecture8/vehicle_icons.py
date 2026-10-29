"""Vehicle icons as vector shapes for matplotlib: sharp at any zoom, in a window or a saved file.

    import vehicle_icons as vi
    av = vi.Vehicle(ax, "AV", color="#1E2939")      # icons/top_car.svg, the course's AV
    av.set((x, y), heading)                         # heading in rad, 0 = east

Each icon is read from its SVG file (one or more paths, commands M L H V C S Z) and drawn as
a PathPatch. A car's position is its base link: the middle of the rear axle, 0.9 m in from
the rear bumper of the 4.5 m car (the course's convention). Needs numpy and matplotlib only.

Icons (icons/ next to this file): top_car.svg (the AV), noun_sedancar_6913727.svg (car A) and
noun_sedancar_6913728.svg (car B), by Stone from the Noun Project (thenounproject.com).
"""
import re
from pathlib import Path as _P

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

HERE = _P(__file__).resolve().parent
ICONS = {"AV": "top_car.svg", "A": "noun_sedancar_6913727.svg", "B": "noun_sedancar_6913728.svg"}
CAR_LEN, REAR = 4.5, 0.9
_TOKEN = re.compile(r"[MmLlHhVvCcSsZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _subpaths(d):
    """One SVG path string to a list of closed subpaths, each (vertices, codes), y up."""
    toks = _TOKEN.findall(d)
    i, cmd = 0, None
    cur, start, last_ctrl = np.zeros(2), np.zeros(2), None
    subs, verts, codes = [], [], []

    def num():
        nonlocal i
        i += 1
        return float(toks[i - 1])

    def finish():
        if verts:
            subs.append((np.array(verts), np.array(codes, dtype=np.uint8)))

    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if verts:
                    verts.append(start.copy()); codes.append(Path.CLOSEPOLY)
                    finish(); verts, codes = [], []
                cur, last_ctrl = start.copy(), None
                continue
        rel = cmd.islower()
        c = cmd.upper()
        base = cur if rel else np.zeros(2)
        if c == "M":
            finish(); verts, codes = [], []
            cur = base + (num(), num())
            start = cur.copy()
            verts.append(cur.copy()); codes.append(Path.MOVETO)
            cmd = "l" if rel else "L"          # later pairs are line-tos
            last_ctrl = None
        elif c == "L":
            cur = base + (num(), num())
            verts.append(cur.copy()); codes.append(Path.LINETO); last_ctrl = None
        elif c == "H":
            cur = np.array([(cur[0] if rel else 0) + num(), cur[1]])
            verts.append(cur.copy()); codes.append(Path.LINETO); last_ctrl = None
        elif c == "V":
            cur = np.array([cur[0], (cur[1] if rel else 0) + num()])
            verts.append(cur.copy()); codes.append(Path.LINETO); last_ctrl = None
        elif c in "CS":
            if c == "C":
                c1 = base + (num(), num())
            else:
                c1 = 2 * cur - last_ctrl if last_ctrl is not None else cur.copy()
            c2 = base + (num(), num())
            end = base + (num(), num())
            verts += [c1, c2, end]; codes += [Path.CURVE4] * 3
            cur, last_ctrl = end, c2
        else:
            raise ValueError(f"unsupported SVG path command {cmd!r}")
    finish()
    return [(v * (1, -1), cd) for v, cd in subs]          # SVG y points down


def _area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _inside(pt, v):
    return Path(v).contains_point(pt)


def load_icon(svg):
    """The icon's outline as one Path, front up (+y), its bounding box from x -w/2 to w/2 and
    y 0 (rear) to 1 (front): the length is 1. Holes (windows) are wound against their
    outline, so matplotlib's nonzero fill leaves them empty, as the SVG's even-odd rule does."""
    text = _P(svg).read_text()
    subs = [s for d in re.findall(r'\sd="([^"]*)"', text) for s in _subpaths(d)]
    allv = np.vstack([v for v, _ in subs])
    lo, hi = allv.min(0), allv.max(0)
    scale = 1.0 / (hi[1] - lo[1])
    cx = 0.5 * (lo[0] + hi[0])
    out_v, out_c = [], []
    for k, (v, cd) in enumerate(subs):
        depth = sum(_inside(v[0], w) for j, (w, _) in enumerate(subs) if j != k)
        want_ccw = depth % 2 == 0                        # outlines CCW, holes CW
        if (_area(v) > 0) != want_ccw:
            v, cd = _reverse(v, cd)
        out_v.append((v - (cx, lo[1])) * scale); out_c.append(cd)
    return Path(np.vstack(out_v), np.concatenate(out_c))


def _reverse(v, cd):
    """The same closed subpath, traversed the other way (Bezier segments kept)."""
    segs, i = [], 1                                       # v[0] is the MOVETO
    while i < len(cd):
        n = 3 if cd[i] == Path.CURVE4 else 1
        segs.append((v[i - 1], v[i:i + n], cd[i]))
        i += n
    nv, nc = [segs[-1][1][-1]], [Path.MOVETO]
    for p0, pts, code in reversed(segs):
        if code == Path.CURVE4:
            nv += [pts[1], pts[0], p0]; nc += [Path.CURVE4] * 3
        else:
            nv.append(p0); nc.append(Path.LINETO if code != Path.CLOSEPOLY else Path.LINETO)
    nv.append(nv[0]); nc.append(Path.CLOSEPOLY)
    return np.array(nv), np.array(nc, dtype=np.uint8)


_CACHE = {}


def icon(name, folder=HERE / "icons"):
    """The Path of a vehicle: "AV", "A", "B", or an SVG file name in `folder`."""
    key = (str(folder), name)
    if key not in _CACHE:
        _CACHE[key] = load_icon(_P(folder) / ICONS.get(name, name))
    return _CACHE[key]


class Vehicle:
    """One vehicle icon on `ax`, moved with set(xy, heading)."""

    def __init__(self, ax, name, color="black", alpha=1.0, length=CAR_LEN, rear=REAR, zorder=8,
                 folder=HERE / "icons"):
        self.ax, self.length, self.rear = ax, length, rear
        self.patch = PathPatch(icon(name, folder), facecolor=color, edgecolor="none", alpha=alpha,
                               zorder=zorder)
        ax.add_patch(self.patch)

    def set(self, xy, heading):
        """xy: the base link (m); heading: rad, 0 = east, counterclockwise."""
        L = self.length
        tr = (Affine2D().translate(0, -self.rear / L).scale(L)
              .rotate(heading - np.pi / 2).translate(*xy))
        self.patch.set_transform(tr + self.ax.transData)
        return self.patch


def draw(ax, name, xy, heading, **kw):
    """Draw one vehicle once (for still figures)."""
    v = Vehicle(ax, name, **kw)
    v.set(xy, heading)
    return v.patch


def draw_side(ax, svg, x_front, length=CAR_LEN, color="black", alpha=1.0, zorder=5):
    """A side-view icon (front to the right in the SVG) with its front bumper at x_front and
    its wheels on y = 0. Returns its height (m)."""
    # A side icon's length runs along the SVG's x and its height along y.
    raw = [s for d in re.findall(r'\sd="([^"]*)"', _P(svg).read_text()) for s in _subpaths(d)]
    allv = np.vstack([s[0] for s in raw])
    lo, hi = allv.min(0), allv.max(0)
    sc = length / (hi[0] - lo[0])
    verts, codes = [], []
    for k, (sv, cd) in enumerate(raw):
        depth = sum(_inside(sv[0], w2) for j, (w2, _) in enumerate(raw) if j != k)
        if (_area(sv) > 0) != (depth % 2 == 0):
            sv, cd = _reverse(sv, cd)
        verts.append((sv - (hi[0], lo[1])) * sc + (x_front, 0)); codes.append(cd)
    path = Path(np.vstack(verts), np.concatenate(codes))
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))
    return (hi[1] - lo[1]) * sc
