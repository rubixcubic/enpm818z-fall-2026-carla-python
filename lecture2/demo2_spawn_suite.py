#!/usr/bin/env python3
"""Demo 2: spawn an ego vehicle and the full sensor suite.

Slides: 'Demo 2: Spawn a Sensor Suite' and 'Placement in CARLA'.

Five sensors, mounted where the coverage discussion says they belong, all
attached Rigid so the sensor-to-vehicle transform stays constant.  The script
reports what each stream actually delivers, because the point of the demo is
that the configured rate and the delivered rate are not the same number.

    python3 demo2_spawn_suite.py --seconds 20 --save-dir out/

Ctrl+C is safe: the ActorPool destroys everything on the way out.
"""

import argparse
import collections
import os

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


def build_suite(world, pool, vehicle, save_dir=None):
    """Attach camera, LiDAR, RADAR, IMU and GNSS, and count what arrives.

    The attachment transform passed here IS the extrinsic calibration from the
    rest of the lecture.  It is hard-coded, which is fine in a simulator that
    also tells you the true answer, and is not fine anywhere else.
    """
    bp_lib = world.get_blueprint_library()
    where = mounts(vehicle)
    counts = collections.Counter()
    sensors = {}

    # ---- forward RGB camera: range and classification -----------------------
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
    f, cx, cy = camera_intrinsics(cam_bp)
    print(f"camera intrinsics: f={f:.1f} px, principal point=({cx:.1f}, {cy:.1f})")

    def on_image(image):
        counts["camera"] += 1
        if save_dir and counts["camera"] % 20 == 0:
            image.save_to_disk(os.path.join(save_dir, f"rgb/{image.frame:06d}.png"))

    camera.listen(on_image)
    sensors["camera"] = camera

    # ---- roof LiDAR: the 360 degree ring ------------------------------------
    li_bp = bp_lib.find("sensor.lidar.ray_cast")
    li_bp.set_attribute("channels", "32")
    li_bp.set_attribute("range", "50")
    li_bp.set_attribute("points_per_second", "300000")
    # one full revolution per simulation step, so a sweep is one frame
    li_bp.set_attribute("rotation_frequency", str(1.0 / DEFAULT_DELTA))
    lidar = pool.add(
        world.spawn_actor(
            li_bp,
            where["lidar"],
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
    )

    def on_cloud(cloud):
        counts["lidar"] += 1
        counts["lidar_points"] += len(cloud)

    lidar.listen(on_cloud)
    sensors["lidar"] = lidar

    # ---- front RADAR: range rate through weather ----------------------------
    ra_bp = bp_lib.find("sensor.other.radar")
    ra_bp.set_attribute("horizontal_fov", "30")
    ra_bp.set_attribute("vertical_fov", "10")
    ra_bp.set_attribute("range", "100")
    radar = pool.add(
        world.spawn_actor(
            ra_bp,
            where["radar"],
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
    )

    def on_radar(measurement):
        counts["radar"] += 1
        counts["radar_detections"] += len(measurement)

    radar.listen(on_radar)
    sensors["radar"] = radar

    # ---- IMU and GNSS: where the vehicle is, and how it is moving -----------
    imu = pool.add(
        world.spawn_actor(
            bp_lib.find("sensor.other.imu"),
            where["imu"],
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
    )
    imu.listen(lambda d: counts.update(imu=1))
    sensors["imu"] = imu

    gnss = pool.add(
        world.spawn_actor(
            bp_lib.find("sensor.other.gnss"),
            where["gnss"],
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
    )
    gnss.listen(lambda d: counts.update(gnss=1))
    sensors["gnss"] = gnss

    return sensors, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument(
        "--save-dir", default=None, help="write every 20th camera frame here"
    )
    ap.add_argument("--no-autopilot", action="store_true")
    args = ap.parse_args()

    client = connect(args.host, args.port)
    world = client.get_world()

    with synchronous(world), ActorPool(client) as pool:
        vehicle = spawn_ego(world, pool)
        e = vehicle.bounding_box.extent
        print(
            f"ego: {vehicle.type_id}, half-extent x={e.x:.2f} y={e.y:.2f} z={e.z:.2f} m"
        )
        for name, tf in mounts(vehicle).items():
            loc = tf.location
            print(f"  {name:7s} at x={loc.x:5.2f} y={loc.y:5.2f} z={loc.z:5.2f}")

        if not args.no_autopilot:
            vehicle.set_autopilot(True)

        sensors, counts = build_suite(world, pool, vehicle, args.save_dir)

        steps = int(args.seconds / DEFAULT_DELTA)
        print(
            f"\nticking {steps} steps at {DEFAULT_DELTA}s "
            f"({args.seconds:.0f}s of simulated time)"
        )
        try:
            for _ in range(steps):
                world.tick()
        except KeyboardInterrupt:
            print("\ninterrupted")

        # The configured rate and the delivered rate are different numbers.
        # That gap is the whole point of the slide.
        print(f"\ndelivered over {args.seconds:.0f}s of simulated time:")
        for name in ("camera", "lidar", "radar", "imu", "gnss"):
            n = counts[name]
            print(f"  {name:7s} {n:5d} messages  ({n / args.seconds:6.1f} Hz)")
        if counts["lidar"]:
            print(
                f"  mean LiDAR points per sweep: "
                f"{counts['lidar_points'] / counts['lidar']:.0f}"
            )
        if counts["radar"]:
            print(
                f"  mean RADAR detections per scan: "
                f"{counts['radar_detections'] / counts['radar']:.0f}"
            )


if __name__ == "__main__":
    main()
