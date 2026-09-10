#!/usr/bin/env python3
"""Demo 1: connect, load a town, inspect the blueprint library.

Slide: 'Demo 1: Connect and Inspect'.

Watch how long load_world takes.  That is a full level load in a game engine,
and it is why the client timeout is set so high.

    python3 demo1_connect.py --town Town03
"""

import argparse

import carla
from carla_common import connect, load_map, DEFAULT_HOST, DEFAULT_PORT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--town", default="Town03")
    args = ap.parse_args()

    client = connect(args.host, args.port)
    print(
        f"client {client.get_client_version()} / server {client.get_server_version()}"
    )
    print("available maps:", [m.split("/")[-1] for m in client.get_available_maps()])

    world = load_map(client, args.town)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    print(f"loaded: {world.get_map().name}")

    bp_lib = world.get_blueprint_library()
    vehicles = bp_lib.filter("vehicle.*")
    sensors = bp_lib.filter("sensor.*")
    print(f"\n{len(vehicles)} vehicle blueprints, {len(sensors)} sensor blueprints")

    # A blueprint is a TEMPLATE.  Nothing exists in the world until you spawn
    # an actor from one.
    print("\nthe sensors this course uses:")
    for wanted in (
        "sensor.camera.rgb",
        "sensor.camera.depth",
        "sensor.camera.semantic_segmentation",
        "sensor.lidar.ray_cast",
        "sensor.other.radar",
        "sensor.other.gnss",
        "sensor.other.imu",
    ):
        bp = bp_lib.find(wanted)
        attrs = [a.id for a in bp if a.is_modifiable]
        print(f"  {wanted:42s} {len(attrs):2d} modifiable attributes")

    print("\nRGB camera attributes you will actually set:")
    cam = bp_lib.find("sensor.camera.rgb")
    for key in ("image_size_x", "image_size_y", "fov", "sensor_tick"):
        # str(ActorAttribute) prints the whole repr, so pull the value out
        attr = cam.get_attribute(key)
        print(f"  {key:16s} default {str(attr).split('value=')[-1].rstrip(')')}")

    print("\nspawn points on this map:", len(world.get_map().get_spawn_points()))


if __name__ == "__main__":
    main()
