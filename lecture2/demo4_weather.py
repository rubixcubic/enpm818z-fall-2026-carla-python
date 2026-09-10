#!/usr/bin/env python3
"""Demo 4: weather, and what it does to each sensor.

Slide: 'Demo 4: Weather'.

Cycles the presets in the slide and, for each one, reports the mean number of
LiDAR returns and RADAR detections per scan.  Watch the return count, not just
the camera image: the picture getting darker is obvious, the point cloud
thinning is the thing that breaks a perception stack.

    python3 demo4_weather.py --view
    python3 demo4_weather.py --seconds-per-preset 4 --save-dir out/weather

--view opens a window on the camera this script already spawned, so you watch
each preset arrive rather than reading about it afterwards. It needs no second
terminal and no separate viewer.

The table reports all three sensors. For the camera it prints mean brightness
and contrast, both 0 to 255, because the camera fails in two different ways and
one number would hide that.

Note that CARLA has no 'dense fog' preset.  Fog is a set of attributes
(fog_density, fog_distance, fog_falloff) that you apply on top of a preset,
which is why the fog entry below is built by hand.
"""

import argparse
import collections
import os
import queue

import carla
from carla_common import (
    ActorPool,
    DEFAULT_DELTA,
    DEFAULT_HOST,
    DEFAULT_PORT,
    connect,
    mounts,
    spawn_ego,
    synchronous,
)

SETTLE_TICKS = 6  # ticks discarded after each weather change


def dense_fog() -> carla.WeatherParameters:
    """A clear midday sky with heavy fog laid over it.

    Built field by field rather than by editing carla.WeatherParameters
    .ClearNoon: the presets are class attributes, so assigning to one edits it
    for the whole process and every later use of 'clear noon' silently comes
    back foggy.
    """
    return carla.WeatherParameters(
        cloudiness=15.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=75.0,
        fog_density=100.0,  # percent
        fog_distance=5.0,  # metres before the fog starts
        fog_falloff=1.0,  # how fast it thickens with height
        wetness=0.0,
        mie_scattering_scale=0.05,
    )


# the running order from the slide
PRESETS = [
    ("clear noon", carla.WeatherParameters.ClearNoon),
    ("wet noon", carla.WeatherParameters.WetNoon),
    ("hard rain", carla.WeatherParameters.HardRainNoon),
    ("sunset", carla.WeatherParameters.ClearSunset),
    ("dense fog", dense_fog()),
    ("night", carla.WeatherParameters.ClearNight),
]


def _to_rgb(image):
    """carla.Image to an (H, W, 3) RGB array. CARLA delivers BGRA.

    The copy at the end is NOT optional. raw_data is a view onto a buffer the
    server reuses, so an image object kept past its callback stops describing
    the moment it was captured and starts describing whatever arrived most
    recently. Six weather presets then render as six copies of the last one,
    which looks exactly like set_weather having no effect.
    """
    import numpy as np

    buf = np.frombuffer(image.raw_data, dtype=np.uint8)
    return buf.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1].copy()


class _Window:
    """A live view of the camera this script already spawned.

    Deliberately part of the demo rather than a separate viewer: the whole
    point is to see the picture and the numbers move at the same time, and a
    second process watching a second camera would not show that.
    """

    def __init__(self, width: int, height: int):
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ENPM818Z: weather")
        self.font = pygame.font.SysFont("monospace", 22, bold=True)
        self.clock = pygame.time.Clock()

    def draw(self, rgb, preset: str, lidar_points: float) -> bool:
        """Paint one frame. Returns False when the user closes the window."""
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE,
                pygame.K_q,
            ):
                return False

        if rgb is not None:
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            self.screen.blit(surface, (0, 0))

        hud = f"{preset}    LiDAR {lidar_points:5.0f} points/sweep"
        label = self.font.render(hud, True, (255, 255, 255))
        box = pygame.Surface((label.get_width() + 16, label.get_height() + 10))
        box.set_alpha(150)
        box.fill((0, 0, 0))
        self.screen.blit(box, (14, 14))
        self.screen.blit(label, (22, 19))

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self) -> None:
        self.pygame.quit()


def _write_rgb(rgb, path: str) -> None:
    """Write an RGB array to disk, with or without OpenCV."""
    try:
        import cv2

        cv2.imwrite(path, rgb[:, :, ::-1])
    except ImportError:
        from PIL import Image

        Image.fromarray(rgb).save(path)


def _write_contact_sheet(panels, path: str) -> None:
    """All six presets in one image, two rows of three, for the slides."""
    import numpy as np

    try:
        import cv2
    except ImportError:
        cv2 = None

    scaled = []
    for name, rgb in panels:
        h, w = rgb.shape[:2]
        small = rgb[::2, ::2] if cv2 is None else cv2.resize(rgb, (w // 2, h // 2))
        if cv2 is not None:
            cv2.putText(
                small,
                name,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        scaled.append(small)

    while len(scaled) % 3:
        scaled.append(np.zeros_like(scaled[0]))
    rows = [np.hstack(scaled[i : i + 3]) for i in range(0, len(scaled), 3)]
    sheet = np.vstack(rows)

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(path, sheet[:, :, ::-1])
    else:
        from PIL import Image

        Image.fromarray(sheet).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds-per-preset", type=float, default=4.0)
    ap.add_argument(
        "--save-dir", default=None, help="write one camera frame per preset here"
    )
    ap.add_argument(
        "--view",
        action="store_true",
        help="show the camera live in a window while it runs",
    )
    ap.add_argument(
        "--contact-sheet", default=None, help="write all six presets into one image"
    )
    args = ap.parse_args()

    client = connect(args.host, args.port)
    world = client.get_world()

    with synchronous(world), ActorPool(client) as pool:
        vehicle = spawn_ego(world, pool)
        where = mounts(vehicle)
        bp_lib = world.get_blueprint_library()

        width, height = 1280, 720
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        camera = pool.add(
            world.spawn_actor(
                cam_bp,
                where["camera"],
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
        )

        li_bp = bp_lib.find("sensor.lidar.ray_cast")
        li_bp.set_attribute("channels", "32")
        li_bp.set_attribute("range", "50")
        li_bp.set_attribute("points_per_second", "300000")
        li_bp.set_attribute("rotation_frequency", str(1.0 / DEFAULT_DELTA))
        # These two are what make weather visible to the LiDAR at all. Left at
        # zero, CARLA's ray cast is a perfect geometric sensor and fog costs it
        # nothing, which is not the lesson.
        li_bp.set_attribute("atmosphere_attenuation_rate", "0.004")
        li_bp.set_attribute("dropoff_general_rate", "0.10")
        lidar = pool.add(
            world.spawn_actor(
                li_bp,
                where["lidar"],
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
        )

        ra_bp = bp_lib.find("sensor.other.radar")
        ra_bp.set_attribute("horizontal_fov", "30")
        ra_bp.set_attribute("range", "100")
        radar = pool.add(
            world.spawn_actor(
                ra_bp,
                where["radar"],
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
        )

        # One queue per sensor, and exactly one message pulled per tick.
        #
        # Free-running callbacks are the wrong pattern here even though the
        # server is synchronous: world.tick() returns as soon as the server
        # has stepped, while callbacks are delivered on another thread and can
        # arrive later. Accumulate from those and a slow machine attributes
        # frames to the wrong preset, which shows up as night being the
        # brightest row in the table. Pairing one message to one tick makes
        # the attribution exact regardless of how loaded the machine is.
        images: queue.Queue = queue.Queue()
        clouds: queue.Queue = queue.Queue()
        scans: queue.Queue = queue.Queue()
        camera.listen(images.put)
        lidar.listen(clouds.put)
        radar.listen(scans.put)

        stats = collections.Counter()
        latest = {}

        window = _Window(width, height) if args.view else None
        panels = []

        vehicle.set_autopilot(True)
        steps = int(args.seconds_per_preset / DEFAULT_DELTA)

        print(
            f"{'preset':<12} {'LiDAR pts/sweep':>16} {'RADAR det/scan':>16} "
            f"{'cam brightness':>15} {'cam contrast':>13}"
        )
        print("-" * 76)
        for name, weather in PRESETS:
            world.set_weather(weather)
            stats.clear()

            # Drop anything queued under the previous preset.
            for q in (images, clouds, scans):
                while not q.empty():
                    q.get_nowait()

            for step in range(steps):
                world.tick()
                image = images.get(timeout=5.0)
                cloud = clouds.get(timeout=5.0)
                measurement = scans.get(timeout=5.0)

                # A weather change is not applied to the very next frame, so
                # give it a few ticks before anything counts.
                if step >= SETTLE_TICKS:
                    stats["lidar_scans"] += 1
                    stats["lidar_points"] += len(cloud)
                    stats["radar_scans"] += 1
                    stats["radar_dets"] += len(measurement)

                    rgb = _to_rgb(image)
                    latest["rgb"] = rgb
                    # Two numbers for the camera, because it fails in two ways
                    # and one would hide that. Brightness collapses at night.
                    # Contrast collapses in fog while brightness stays high,
                    # which is the surprising one: the image is not dark, it is
                    # uniform, and a detector needs edges rather than photons.
                    # Subsampled, since this runs every tick.
                    small = rgb[::4, ::4]
                    stats["cam_frames"] += 1
                    stats["cam_mean"] += float(small.mean())
                    stats["cam_contrast"] += float(small.std())

                if window is not None:
                    # Live counts, so the numbers and the picture change
                    # together. Reading the table afterwards makes the LiDAR
                    # result look like an artefact of averaging; watching it
                    # sit still while the camera dies does not.
                    n = max(stats["lidar_scans"], 1)
                    if not window.draw(
                        latest.get("rgb"), name, stats["lidar_points"] / n
                    ):
                        print("\nclosed")
                        return

            frames = max(stats["cam_frames"], 1)
            pts = stats["lidar_points"] / max(stats["lidar_scans"], 1)
            det = stats["radar_dets"] / max(stats["radar_scans"], 1)
            bright = stats["cam_mean"] / frames
            contrast = stats["cam_contrast"] / frames
            print(f"{name:<12} {pts:16.0f} {det:16.0f} {bright:15.1f} {contrast:13.1f}")

            if "rgb" in latest:
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    slug = name.replace(" ", "_")
                    _write_rgb(
                        latest["rgb"], os.path.join(args.save_dir, f"{slug}.png")
                    )
                if args.contact_sheet:
                    panels.append((name, latest["rgb"]))

        if args.contact_sheet and panels:
            _write_contact_sheet(panels, args.contact_sheet)
            print(f"wrote {args.contact_sheet}")

        if window is not None:
            window.close()

        print(
            "\nBrightness and contrast are 0 to 255. The camera degrades in "
            "two different ways:\nnight takes the brightness, fog takes the "
            "contrast while leaving the image bright.\nThe LiDAR column "
            "barely moves, and that is a limit of the simulator rather than "
            "a\nfact about LiDAR. Ask what your fusion stage should do when "
            "one sensor degrades\nand another does not."
        )


if __name__ == "__main__":
    main()
