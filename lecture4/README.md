# L4 hands-on: two ways to detect objects

Two notebooks, one per detector family, and one script that runs both side by side.
Everything runs on the photo from the slides (Ultralytics' `bus.jpg`) unless you point
it at your own image, for example a camera frame from GP1.

| File | Approach | What you do |
|---|---|---|
| `yolo_one_stage.ipynb` | one-stage CNN, **YOLOv8s** | open the 84 x 6300 output matrix, decode one cell into a box by hand, write the confidence cut and NMS yourself, sweep the cut, time it |
| `rtdetr_set_prediction.ipynb` | set prediction, **RT-DETR-L** | open the 300 x 84 output matrix, check that no NMS is needed, pair queries with objects with the Hungarian algorithm, compute attention by hand, compare with YOLOv8s |
| `compare_detectors.py` | both | the live demo: both models on the same frames, boxes side by side, median time per frame |

YOLOv8s and RT-DETR-L are the two models you fine-tune in GP2.

## Requirements

Python 3 with `ultralytics` (it installs PyTorch, NumPy and matplotlib) and `scipy`.
Tested on Ubuntu 24.04, Python 3.12, ultralytics 8.3.227, torch 2.9, SciPy 1.16.

```bash
python3 -m venv ~/l4env && source ~/l4env/bin/activate
pip install ultralytics scipy jupyter
```

A GPU is optional: on a CPU every step works, only slower. The weights (`yolov8s.pt`,
22 MB; `rtdetr-l.pt`, 66 MB) download into the folder you run from on first use. Do not
commit them.

## Running

Open a notebook in VS Code (Jupyter extension) or with `jupyter notebook`, and run the
cells from top to bottom. Each notebook ends with exercises.

```bash
python3 compare_detectors.py                 # the slides' photo
python3 compare_detectors.py frame.png       # one CARLA frame
python3 compare_detectors.py frames/         # a folder of frames
```

## What to expect on the slides' photo

| | YOLOv8s | RT-DETR-L |
|---|---|---|
| output matrix | 84 x 6300 (one column per cell) | 300 x 84 (one row per query) |
| above confidence 0.25 | 49 cells, 5 after NMS | 9 queries, no NMS |
| weights | 11.2 M | 33.0 M |
| time, RTX 4060 laptop GPU, PyTorch | about 9 ms | about 29 ms |

Times change a little on every run, and a lot on other hardware.
