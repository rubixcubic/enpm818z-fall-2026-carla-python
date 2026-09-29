# This original version of the code is from fall 2025 and it was modified and
# improved by Anthropic Claude Opus 5.5.
# Signed: Zeid Kootbally

"""YOLOv8s and RT-DETR-L on the same frames, side by side.

The L4 live demo, and a head start on GP2: both detectors GP2 fine-tunes, run
with their COCO weights on your frames (a CARLA camera image, a folder of them,
or any photo). For each frame it draws both sets of boxes next to each other
and prints what each model found. At the end it prints each model's median time
per frame.

Usage
    python3 compare_detectors.py                  # the Ultralytics sample bus.jpg
    python3 compare_detectors.py frame.png        # one frame
    python3 compare_detectors.py frames/          # every .png and .jpg in a folder
    python3 compare_detectors.py frames/ --conf 0.5 --save out/

Options
    --conf    confidence cut (default 0.25, the Ultralytics default)
    --save    write each side-by-side picture to this folder instead of showing it

Needs: pip install ultralytics  (downloads yolov8s.pt and rtdetr-l.pt once).
"""
import argparse
import os
import statistics
import time

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
    times = {label: [] for label, _, _ in models}
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    for frame in frames:
        image = Image.open(frame).convert("RGB")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        print(f"\n{os.path.basename(frame)}")
        for ax, (label, model, color) in zip(axes, models):
            boxes, ms = run(model, frame, a.conf)
            times[label].append(ms)
            draw(ax, image, boxes, f"{label}: {len(boxes)} objects, {ms:.1f} ms", color)
            found = ", ".join(f"{n} {c:.2f}" for n, c, _ in boxes) or "nothing"
            print(f"  {label:10s} {len(boxes):2d} objects  {ms:6.1f} ms   {found}")
        fig.tight_layout()
        if a.save:
            out = os.path.join(a.save, os.path.splitext(os.path.basename(frame))[0] + "_compare.png")
            fig.savefig(out, dpi=120)
            plt.close(fig)
        else:
            plt.show()

    print("\nmedian time per frame")
    for label, t in times.items():
        print(f"  {label:10s} {statistics.median(t):6.1f} ms  over {len(t)} frames")


if __name__ == "__main__":
    main()
