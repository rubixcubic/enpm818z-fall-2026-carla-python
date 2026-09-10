#!/usr/bin/env python3
"""Task 3: project LiDAR points into the camera image.

Slides: 'Right to Left', 'The axis trap', 'CARLA Hands-On'.

This is the whole calibration section executed once.  A point measured in the
LiDAR frame L is carried into the camera frame C and then onto the image
plane:

    p_img  ~  K  P  T_{C<-V}  T_{V<-L}  p_L

  T_{V<-L}   LiDAR to vehicle: the mounting transform, exactly
  T_{C<-V}   vehicle to camera: the inverse of the camera's mounting transform
  P          the axis permutation, and the part that costs people an evening
  K          the intrinsic matrix

Run it three ways.  Neither wrong version raises anything, and they fail
differently, which is the lesson:

    python3 lidar_to_camera.py --out out/projection.png
    python3 lidar_to_camera.py --no-permutation --out out/projection_none.png
    python3 lidar_to_camera.py --bad-sign --out out/projection_flipped.png

Omitting P entirely throws every point off the edge of the image, so you get a
blank overlay.  Getting one SIGN of P wrong is the one to show: the points land,
the shape is recognisable, and the scene is upside down.
"""

import argparse
import os
import queue

import numpy as np

import carla
from carla_common import (
    ActorPool,
    DEFAULT_DELTA,
    DEFAULT_HOST,
    DEFAULT_PORT,
    camera_intrinsics,
    connect,
    mounts,
    spawn_ego,
    synchronous,
)

# CARLA (and most robotics) uses x forward, y right, z up.
# K assumes the optical convention: x right, y down, z forward.
# Same three physical directions, different names, so relabel before K sees them.
#
#   x_opt =  y_carla        row 0 = [0, 1,  0]
#   y_opt = -z_carla        row 1 = [0, 0, -1]
#   z_opt =  x_carla        row 2 = [1, 0,  0]
AXIS_PERMUTATION = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

# The same matrix with one sign wrong: y_opt = +z instead of -z. This is the
# mistake that is worth showing, because it is the one that does not look like
# a mistake. The points still land on the image and still form a recognisable
# shape; the shape is flipped top to bottom.
AXIS_PERMUTATION_BAD_SIGN = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
)


def to_matrix(transform: carla.Transform) -> np.ndarray:
    """The 4x4 homogeneous matrix for a carla.Transform.

    CARLA already provides this, and it is worth reading rather than
    reimplementing: get_matrix() takes a point in the transform's own frame and
    returns it in the parent frame, so a sensor's get_matrix() is T_{V<-S} when
    the sensor is attached to the vehicle, and T_{world<-S} for its world pose.
    """
    return np.array(transform.get_matrix())


def lidar_to_array(cloud: carla.LidarMeasurement) -> np.ndarray:
    """(N, 4) array of x, y, z, intensity in the LiDAR's own frame."""
    data = np.frombuffer(cloud.raw_data, dtype=np.float32)
    return np.reshape(data, (-1, 4))


def image_to_array(image: carla.Image) -> np.ndarray:
    """(H, W, 3) uint8 RGB. CARLA delivers BGRA, so drop alpha and flip."""
    data = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = np.reshape(data, (image.height, image.width, 4))
    return bgra[:, :, :3][:, :, ::-1]


def project(
    points_l: np.ndarray,
    t_v_from_l: np.ndarray,
    t_v_from_c: np.ndarray,
    k: np.ndarray,
    permutation: np.ndarray | None = AXIS_PERMUTATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Carry LiDAR points to pixels. Returns (uv, depth) for points in front.

    Every step here matches a line on a slide, in the same order.
    """
    n = points_l.shape[0]
    homogeneous = np.hstack([points_l[:, :3], np.ones((n, 1))]).T  # 4 x N

    # ---- L to V: the mounting transform, used exactly as given --------------
    in_vehicle = t_v_from_l @ homogeneous

    # ---- V to C: the inverse of the camera's mounting transform -------------
    # Read the chain the way you cancel fractions: T_{C<-V} T_{V<-L} leaves
    # C <- L. If the frame letters do not meet in the middle it is backwards.
    t_c_from_v = np.linalg.inv(t_v_from_c)
    in_camera = (t_c_from_v @ in_vehicle)[:3, :]  # 3 x N

    # ---- relabel the axes before K ever sees them ---------------------------
    if permutation is not None:
        in_camera = permutation @ in_camera

    # ---- keep what is in front of the camera --------------------------------
    # Without the permutation this filter is applied to the wrong axis, which
    # is one reason the broken version still produces a plausible-looking
    # picture instead of an error.
    in_front = in_camera[2, :] > 0.1
    in_camera = in_camera[:, in_front]

    # ---- K, then the perspective divide -------------------------------------
    projected = k @ in_camera
    uv = projected[:2, :] / projected[2, :]
    return uv.T, in_camera[2, :]


def draw(
    image: np.ndarray, uv: np.ndarray, depth: np.ndarray, max_range: float = 50.0
) -> np.ndarray:
    """Paint the projected points onto the image, coloured near to far."""
    h, w = image.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    on_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, depth = u[on_image], v[on_image], depth[on_image]

    # near red, far blue: the same colouring the figure in the deck uses
    t = np.clip(depth / max_range, 0.0, 1.0)
    colours = np.stack([(255 * (1 - t)), np.zeros_like(t), 255 * t], axis=1)

    out = image.copy()
    for du in (-1, 0, 1):  # a 3x3 dot, so it is visible
        for dv in (-1, 0, 1):
            uu = np.clip(u + du, 0, w - 1)
            vv = np.clip(v + dv, 0, h - 1)
            out[vv, uu] = colours
    print(f"{on_image.sum()} of {len(on_image)} points land on the image")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out", default="projection.png")
    ap.add_argument(
        "--no-permutation",
        action="store_true",
        help="omit P entirely: every point leaves the image",
    )
    ap.add_argument(
        "--bad-sign",
        action="store_true",
        help="use P with one sign wrong: points land, flipped",
    )
    ap.add_argument(
        "--weather",
        default=None,
        help="a carla.WeatherParameters preset name, e.g. "
        "ClearNoon. Left alone if not given.",
    )
    ap.add_argument(
        "--compare",
        default=None,
        help="write a side-by-side of correct and --bad-sign, "
        "both built from the SAME captured frame",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="steps to let the scene settle before capturing",
    )
    args = ap.parse_args()

    client = connect(args.host, args.port)
    world = client.get_world()
    if args.weather:
        # Worth setting when the output is going into a document: whatever the
        # last script left behind is otherwise baked into the figure.
        world.set_weather(getattr(carla.WeatherParameters, args.weather))

    with synchronous(world), ActorPool(client) as pool:
        vehicle = spawn_ego(world, pool)
        where = mounts(vehicle)
        bp_lib = world.get_blueprint_library()

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        camera = pool.add(
            world.spawn_actor(
                cam_bp,
                where["camera"],
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
        )

        li_bp = bp_lib.find("sensor.lidar.ray_cast")
        li_bp.set_attribute("channels", "64")
        li_bp.set_attribute("range", "60")
        li_bp.set_attribute("points_per_second", "600000")
        li_bp.set_attribute("rotation_frequency", str(1.0 / DEFAULT_DELTA))
        lidar = pool.add(
            world.spawn_actor(
                li_bp,
                where["lidar"],
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
        )

        # In synchronous mode both sensors produce exactly one message per
        # tick, so a queue each is enough to pair them by frame number. This
        # is the cheap version of the time synchronisation slide, and it works
        # only because the simulator hands out a common clock.
        images: queue.Queue = queue.Queue()
        clouds: queue.Queue = queue.Queue()
        camera.listen(images.put)
        lidar.listen(clouds.put)

        vehicle.set_autopilot(True)
        for _ in range(args.warmup):
            world.tick()
            images.get(timeout=5.0)
            clouds.get(timeout=5.0)

        world.tick()
        image = images.get(timeout=5.0)
        cloud = clouds.get(timeout=5.0)
        assert image.frame == cloud.frame, (
            f"frames disagree: {image.frame} vs {cloud.frame}"
        )
        print(f"captured frame {image.frame}")

        # CARLA gives exact extrinsics, so this isolates the maths from
        # calibration error. A real vehicle gives you neither these nor a
        # ground truth to check them against.
        t_v_from_l = to_matrix(where["lidar"])
        t_v_from_c = to_matrix(where["camera"])

        f, cx, cy = camera_intrinsics(cam_bp)
        k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

        points = lidar_to_array(cloud)
        background = image_to_array(image)

        if args.compare:
            # Both panels from the SAME capture. Two separate runs would give
            # two different scenes, and then the picture would be comparing
            # the traffic rather than the maths.
            panels = []
            for name, perm in (
                ("correct", AXIS_PERMUTATION),
                ("one sign wrong", AXIS_PERMUTATION_BAD_SIGN),
            ):
                uv, depth = project(points, t_v_from_l, t_v_from_c, k, perm)
                print(f"  {name}: ", end="")
                panels.append(draw(background, uv, depth))
            gutter = np.full((panels[0].shape[0], 12, 3), 255, dtype=np.uint8)
            overlay = np.hstack([panels[0], gutter, panels[1]])
            out_path = args.compare
        else:
            if args.no_permutation:
                permutation = None
            elif args.bad_sign:
                permutation = AXIS_PERMUTATION_BAD_SIGN
            else:
                permutation = AXIS_PERMUTATION
            uv, depth = project(points, t_v_from_l, t_v_from_c, k, permutation)
            overlay = draw(background, uv, depth)
            out_path = args.out

        directory = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(directory, exist_ok=True)
        try:
            import cv2

            cv2.imwrite(out_path, overlay[:, :, ::-1])
        except ImportError:
            from PIL import Image

            Image.fromarray(overlay).save(out_path)
        print(f"wrote {out_path}")

        if args.no_permutation:
            print(
                "\nNothing raised. K was handed x forward and z up, read them "
                "as x right and\nz along the optical axis, and sent every "
                "point off the edge of the image.\nAn empty overlay, and no "
                "error to tell you why."
            )
        elif args.bad_sign:
            print(
                "\nNothing raised, and this time the points did land. That is "
                "the dangerous\ncase: the overlay looks like a real result. "
                "Compare it against the correct\nrun and the scene is upside "
                "down."
            )


if __name__ == "__main__":
    main()
