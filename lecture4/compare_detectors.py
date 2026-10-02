# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""YOLOv8s and RT-DETR-L on the same frames, side by side.

The L4 live demo, and a head start on GP2: both detectors GP2 fine-tunes, run
with their COCO weights on your frames (a CARLA camera image, a folder of them,
or any image). For each frame it draws both sets of boxes next to each other and
prints, per class, how many objects each model found and their mean confidence,
plus how many boxes the two models agree on (same class, IoU of at least 0.5).
At the end it prints the same table over all frames and each model's median time
per frame.

There is no ground truth here, so no precision, recall or mAP: those need
labeled frames. Agreement only says the two models found the same box, not that
it is right.

Usage
    python3 compare_detectors.py                  # the Ultralytics sample bus.jpg
    python3 compare_detectors.py frame.png        # one frame
    python3 compare_detectors.py frames/          # every .png and .jpg in a folder
    python3 compare_detectors.py frames/ --conf 0.5 --save out/
    python3 compare_detectors.py frame.png --list  # also list every detection

Options
    --conf    confidence cut (default 0.25, the Ultralytics default)
    --save    write each side-by-side picture to this folder instead of showing it
    --list    also print every detection: class and confidence

Needs: pip install ultralytics  (downloads yolov8s.pt and rtdetr-l.pt once).
"""
import argparse
import os
import statistics
import time
import warnings

# A machine with matplotlib from both apt and pip warns "Unable to import Axes3D"
# on import. This script never draws in 3D, so the warning is harmless: hide it,
# before matplotlib loads.
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import matplotlib
import matplotlib.patches as mpatches
from PIL import Image

import ultralytics
from ultralytics import YOLO, RTDETR

MODELS = (("YOLOv8s", YOLO, "yolov8s.pt", "#2D6CA2"),
          ("RT-DETR-L", RTDETR, "rtdetr-l.pt", "#1E9E74"))


def frames_from(path):
    """One file, or every image in a folder, sorted by name."""
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path)
                       if n.lower().endswith((".png", ".jpg", ".jpeg")))
        return [os.path.join(path, n) for n in names]
    return [path]


def run(model, frame, conf):
    """One timed inference: the boxes and the wall time in milliseconds."""
    t0 = time.perf_counter()
    result = model(frame, conf=conf, verbose=False)[0]
    ms = 1000 * (time.perf_counter() - t0)
    boxes = [(result.names[int(b.cls)], float(b.conf), [float(v) for v in b.xyxy[0]])
             for b in result.boxes]
    return boxes, ms


def iou(a, b):
    """Intersection over union of two boxes given as (x0, y0, x1, y1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def agreement(boxes_a, boxes_b, thr=0.5):
    """Boxes both models found: same class and IoU >= thr, each box matched once,
    the most overlapping pairs first. Returns (both, only in a, only in b)."""
    pairs = sorted(((iou(ba, bb), i, j) for i, (na, _, ba) in enumerate(boxes_a)
                    for j, (nb, _, bb) in enumerate(boxes_b) if na == nb), reverse=True)
    used_a, used_b = set(), set()
    for v, i, j in pairs:
        if v >= thr and i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j)
    both = len(used_a)
    return both, len(boxes_a) - both, len(boxes_b) - both


def class_table(found, labels, indent="  "):
    """found[label][class] = list of confidences. One row per class: the count and
    the mean confidence for each model."""
    classes = sorted({c for label in labels for c in found[label]},
                     key=lambda c: -sum(len(found[l].get(c, [])) for l in labels))
    print(indent + f"{'class':16s}" + "".join(f"{l:>22s}" for l in labels))
    for c in classes:
        row = ""
        for l in labels:
            confs = found[l].get(c, [])
            cell = f"{len(confs)}  (mean {statistics.mean(confs):.2f})" if confs else "0"
            row += f"{cell:>22s}"
        print(indent + f"{c:16s}" + row)


def draw(ax, image, boxes, title, color):
    ax.imshow(image)
    ax.set_title(title, color=color, fontweight="bold")
    ax.axis("off")
    for name, conf, (x0, y0, x1, y1) in boxes:
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        ec=color, lw=2))
        ax.text(x0 + 3, y0 + 3, f"{name} {conf:.2f}", color="white", fontsize=8,
                va="top", bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none"))


def main():
    default = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", nargs="?", default=default)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    # choose the backend before pyplot loads: OpenCV (pulled in by ultralytics)
    # ships its own Qt plugins, which crash matplotlib's Qt backend
    matplotlib.use("Agg" if a.save else "TkAgg")
    global plt
    import matplotlib.pyplot as plt

    models = [(label, cls(weights), color) for label, cls, weights, color in MODELS]
    frames = frames_from(a.path)
    for _, model, _ in models:                  # warm-up: the first run is slow
        model(frames[0], verbose=False)
    labels = [label for label, _, _ in models]
    times = {label: [] for label in labels}
    total = {label: {} for label in labels}               # class -> confidences, all frames
    agree_total = [0, 0, 0]
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    for frame in frames:
        image = Image.open(frame).convert("RGB")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        print(f"\n{os.path.basename(frame)}")
        found, all_boxes = {}, {}
        for ax, (label, model, color) in zip(axes, models):
            boxes, ms = run(model, frame, a.conf)
            times[label].append(ms)
            all_boxes[label] = boxes
            draw(ax, image, boxes, f"{label}: {len(boxes)} objects, {ms:.1f} ms", color)
            found[label] = {}
            for n, c, _ in boxes:
                found[label].setdefault(n, []).append(c)
                total[label].setdefault(n, []).append(c)
            print(f"  {label:10s} {len(boxes):3d} objects  {ms:6.1f} ms")
            if a.list:
                print("             " + (", ".join(f"{n} {c:.2f}" for n, c, _ in boxes) or "nothing"))
        class_table(found, labels)
        both, only_a, only_b = agreement(all_boxes[labels[0]], all_boxes[labels[1]])
        agree_total = [x + y for x, y in zip(agree_total, (both, only_a, only_b))]
        print(f"  agreement: {both} boxes found by both (same class, IoU >= 0.5); "
              f"{only_a} only by {labels[0]}, {only_b} only by {labels[1]}")
        fig.tight_layout()
        if a.save:
            out = os.path.join(a.save, os.path.splitext(os.path.basename(frame))[0] + "_compare.png")
            fig.savefig(out, dpi=120)
            plt.close(fig)
        else:
            plt.show()

    if len(frames) > 1:                         # with one frame, this repeats its table
        print(f"\nall {len(frames)} frames, confidence cut {a.conf}")
        class_table(total, labels)
        print(f"  agreement: {agree_total[0]} boxes found by both; {agree_total[1]} only by "
              f"{labels[0]}, {agree_total[2]} only by {labels[1]}")
    print("\nmedian time per frame")
    for label, t in times.items():
        n = len(t)
        print(f"  {label:10s} {statistics.median(t):6.1f} ms  over {n} frame{'s' if n > 1 else ''}")


if __name__ == "__main__":
    main()
