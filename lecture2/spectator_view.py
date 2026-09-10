#!/usr/bin/env python3
"""A window onto the simulation, for use in class.

The server runs with -RenderOffScreen, so it opens no window of its own. That
is not a limitation to work around: it is the only configuration that survives
a map load on an Optimus laptop, and it keeps the whole GPU on the simulation
instead of on a window nobody is looking at.

This is the window instead. It is a plain CARLA client that attaches a camera
and blits the frames into a pygame surface, so it renders on whatever GPU your
desktop already uses and never touches Unreal's display path.

    python3 spectator_view.py                 # follow the ego, or spawn one
    python3 spectator_view.py --no-spawn      # only follow, never spawn

Keys:  TAB  next weather preset      C  next camera position
       R    toggle autopilot         ESC or Q  quit

It is a PASSIVE observer: it never calls world.tick(). Run it alongside a demo
script that owns the clock, or on its own against a free-running server.

It also survives a world reload. load_world destroys every actor in the
world, including this viewer's car and camera, so a demo that loads a town
would otherwise leave the window frozen on its last frame with nothing to say
why. The viewer notices, re-attaches to the new episode, and carries on.
"""

import argparse
import sys

import numpy as np
import pygame

import carla
from carla_common import ActorPool, DEFAULT_HOST, DEFAULT_PORT, connect, spawn_ego
from demo4_weather import PRESETS

# Chase, bonnet, and a high overhead view. Each is (location, rotation).
VIEWS = [
    (
        "chase",
        carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15.0)),
    ),
    ("bonnet", carla.Transform(carla.Location(x=1.4, z=1.4))),
    (
        "overhead",
        carla.Transform(carla.Location(x=-2.0, z=18.0), carla.Rotation(pitch=-80.0)),
    ),
]


def find_ego(world):
    """The vehicle a demo script already spawned, if there is one."""
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "ego":
            return actor
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument(
        "--no-spawn",
        action="store_true",
        help="follow an existing ego only, never spawn one",
    )
    args = ap.parse_args()

    # The window opens FIRST, before a single CARLA call. On a freshly started
    # server the first camera costs a lot of setup, and a viewer that connects
    # before it draws anything looks like a viewer that has hung. Put the
    # window up, say what is happening, and let the wait be visible.
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("ENPM818Z: CARLA")
    font = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()

    def splash(*lines) -> None:
        screen.fill((18, 18, 22))
        for n, line in enumerate(lines):
            colour = (235, 235, 235) if n == 0 else (150, 150, 150)
            screen.blit(font.render(line, True, colour), (28, 28 + 26 * n))
        pygame.display.flip()
        pygame.event.pump()

    splash(f"connecting to {args.host}:{args.port} ...")
    client = connect(args.host, args.port)
    world = client.get_world()

    with ActorPool(client) as pool:
        view = 0
        frame = {}

        def attach():
            """Take hold of the current episode: world, ego vehicle, camera.

            Called once at startup and again after any world reload, which is
            why it re-fetches the world rather than closing over it. A World
            object refers to one episode; after load_world the old one is
            stale and every call through it fails.
            """
            new_world = client.get_world()
            splash(
                f"connected to {new_world.get_map().name}",
                "placing the ego vehicle ...",
            )

            ego = find_ego(new_world)
            if ego is None:
                if args.no_spawn:
                    sys.exit("no ego vehicle in the world, and --no-spawn was given")
                ego = spawn_ego(new_world, pool)
                ego.set_autopilot(True)
                print(f"spawned {ego.type_id} as the ego")
            else:
                print(f"following the existing ego: {ego.type_id}")

            bp = new_world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(args.width))
            bp.set_attribute("image_size_y", str(args.height))

            # SpringArm is right HERE and nowhere else in this course. It
            # smooths the camera's motion, which is what you want for
            # something a person watches and exactly what you must not have on
            # a sensor, because it makes the sensor-to-vehicle transform stop
            # being constant.
            cam = pool.add(
                new_world.spawn_actor(
                    bp,
                    VIEWS[view][1],
                    attach_to=ego,
                    attachment_type=carla.AttachmentType.SpringArmGhost,
                )
            )
            cam.listen(lambda image: frame.__setitem__("image", image))
            frame.clear()
            return new_world, ego, cam

        world, vehicle, camera = attach()

        splash(
            "waiting for the first frame ...",
            "",
            "On a server that has just started this takes a while.",
            "It is one-off setup for the first camera, not a hang.",
        )

        settings = world.get_settings()
        if settings.synchronous_mode:
            print(
                "note: the server is in synchronous mode. This viewer does "
                "not tick it, so\nthe picture only moves while another "
                "script is running."
            )

        weather = 0
        autopilot = True
        waited = 0
        since_check = 0
        running = True
        while running:
            # A world reload destroys every actor, so the camera simply stops
            # delivering. Nothing raises and nothing arrives; without this
            # check the window would sit on its last frame looking hung.
            since_check += 1
            if since_check >= 15:
                since_check = 0
                try:
                    alive = vehicle.is_alive
                except RuntimeError:
                    alive = False
                if not alive:
                    print("the world was reloaded; re-attaching", flush=True)
                    splash(
                        "the world was reloaded", "re-attaching to the new episode ..."
                    )
                    # Detach the old callback first. The sensor it belonged to
                    # went with the episode, and leaving the client dispatching
                    # for it while this thread makes RPC calls is a good way to
                    # sit there forever.
                    try:
                        camera.stop()
                    except RuntimeError:
                        pass
                    pool.forget()
                    world, vehicle, camera = attach()
                    waited = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_TAB:
                        weather = (weather + 1) % len(PRESETS)
                        world.set_weather(PRESETS[weather][1])
                    elif event.key == pygame.K_c:
                        view = (view + 1) % len(VIEWS)
                        camera.set_transform(VIEWS[view][1])
                    elif event.key == pygame.K_r:
                        autopilot = not autopilot
                        vehicle.set_autopilot(autopilot)

            image = frame.get("image")
            if image is not None:
                waited = 0
                # CARLA delivers BGRA; drop alpha, flip to RGB, and transpose
                # because pygame surfaces are indexed (x, y) and images (y, x).
                buf = np.frombuffer(image.raw_data, dtype=np.uint8)
                bgra = buf.reshape((image.height, image.width, 4))
                rgb = bgra[:, :, :3][:, :, ::-1]
                surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
            else:
                waited += 1
                if waited % 30 == 0:
                    splash(
                        "waiting for the first frame ...",
                        f"{waited // 30} s",
                        "",
                        "One-off setup for the first camera on this server.",
                    )
                if waited == 900:
                    print(
                        "still no frames after 30 s. If the server is in "
                        "synchronous mode,\nsomething else has to tick it."
                    )
                clock.tick(30)
                continue

            try:
                speed = vehicle.get_velocity()
                kmh = 3.6 * (speed.x**2 + speed.y**2 + speed.z**2) ** 0.5
            except RuntimeError:
                kmh = float("nan")
            hud = (
                f"{VIEWS[view][0]}  |  {PRESETS[weather][0]}  |  "
                f"{kmh:5.1f} km/h  |  autopilot {'on' if autopilot else 'off'}"
            )
            screen.blit(font.render(hud, True, (255, 255, 255)), (12, 12))
            screen.blit(
                font.render(
                    "TAB weather   C camera   R autopilot   ESC quit",
                    True,
                    (180, 180, 180),
                ),
                (12, 36),
            )

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    main()
