"""Shared helpers for the ENPM818Z L2 CARLA demos.

Everything here exists to make a point that is on a slide:

  * synchronous mode is set before anything else happens, and restored on the
    way out, because an asynchronous server hands you whatever happened to be
    ready and no two runs agree;
  * every actor gets destroyed, including on Ctrl+C, because a leaked sensor
    keeps consuming server resources until the map is reloaded;
  * sensor mounting positions are derived from the vehicle's own bounding box
    rather than copied, because every blueprint is a different size.

Tested against CARLA 0.9.16.
"""

from __future__ import annotations

import contextlib
import math
import sys

import carla

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 2000
DEFAULT_TIMEOUT = 20.0          # generous for ordinary calls
LOAD_TIMEOUT = 180.0            # a world load is a full game-engine level load
DEFAULT_DELTA = 0.05            # 20 Hz simulation step


def connect(host: str = DEFAULT_HOST,
            port: int = DEFAULT_PORT,
            timeout: float = DEFAULT_TIMEOUT) -> carla.Client:
    """Connect, and fail with something readable if the server is not up.

    The most common error in this course is a timeout, and it almost always
    means the server has not finished loading rather than that anything is
    wrong with your code.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    try:
        server = client.get_server_version()
    except RuntimeError as exc:
        sys.exit(
            f"Could not reach the CARLA server at {host}:{port} ({exc}).\n"
            "  * Is the server running?\n"
            "  * Did you give it 30-60 s to load the level before connecting?\n"
            "  * Do the client and server versions match?"
        )
    client_ver = client.get_client_version()
    if server != client_ver:
        print(f"WARNING: client {client_ver} against server {server}. "
              "Mismatched versions fail confusingly rather than clearly.")
    return client


def load_map(client: carla.Client, town: str,
             timeout: float = LOAD_TIMEOUT) -> carla.World:
    """Load a town, and explain a failure instead of reporting a bare timeout.

    Two things this gets right that a plain load_world call does not.

    First, the timeout. Loading a level is the one call in the API allowed to
    take a minute or more; everything else should be fast. Running the whole
    session on a 20 s timeout means the load is the only thing that ever trips
    it, and the message you get back blames the network.

    Second, the diagnosis. A timeout here almost never means the server is
    slow. It means the server died partway through the load, and the client is
    still waiting on a process that is no longer there. That is a different
    problem with a different fix, so it gets a different message.
    """
    if not town:
        return client.get_world()

    current = client.get_world().get_map().name.split("/")[-1]
    if current == town:
        print(f"{town} already loaded")
        return client.get_world()

    print(f"loading {town} (this takes a while, it is a full level load)")
    client.set_timeout(timeout)
    try:
        world = client.load_world(town)
    except RuntimeError as exc:
        sys.exit(
            f"Loading {town} failed after {timeout:.0f}s: {exc}\n"
            "\nA timeout HERE usually means the server crashed during the load\n"
            "rather than that it is slow. Check the server before changing any\n"
            "of this code:\n"
            "  * Is the process still alive? A Docker container that exited 139\n"
            "    segfaulted.\n"
            "  * Read its output: docker logs <container>, or the terminal you\n"
            "    started CarlaUE4.sh in.\n"
            "  * A 'MESA' warning in that output means the server is rendering on\n"
            "    the wrong GPU. Start it with -RenderOffScreen so it uses the\n"
            "    NVIDIA device directly instead of going through an X display.\n"
            "  * Try a lighter map first. Town03 is the heaviest of the set;\n"
            "    Town02 and Town10HD_Opt load in a fraction of the memory."
        )
    finally:
        client.set_timeout(DEFAULT_TIMEOUT)
    return world


@contextlib.contextmanager
def synchronous(world: carla.World, delta: float = DEFAULT_DELTA):
    """Run the enclosed block with the server in synchronous mode.

    Slide: 'Run it in synchronous mode, from the first line you write.'
    Outside this context the server free-runs, so sensor callbacks arrive at
    irregular, unrepeatable times and you end up debugging non-determinism you
    created yourself.

    The original settings are restored on exit, including on exception, so a
    crashed demo does not leave the server stepping at a fixed delta with no
    client driving it -- which looks exactly like a frozen simulator.
    """
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)
    try:
        yield
    finally:
        world.apply_settings(original)


class ActorPool:
    """Keeps track of everything spawned so it can all be destroyed.

    Sensors are stopped before destruction: a sensor still listening while its
    actor is torn down can deliver a callback against freed state.
    """

    def __init__(self, client: carla.Client):
        self._client = client
        self._actors: list[carla.Actor] = []

    def add(self, actor: carla.Actor) -> carla.Actor:
        self._actors.append(actor)
        return actor

    def destroy(self) -> None:
        for actor in reversed(self._actors):
            try:
                if isinstance(actor, carla.Sensor) and actor.is_listening:
                    actor.stop()
            except RuntimeError:
                pass
        self._client.apply_batch(
            [carla.command.DestroyActor(a) for a in reversed(self._actors)]
        )
        print(f"destroyed {len(self._actors)} actor(s)")
        self._actors.clear()

    def forget(self) -> None:
        """Drop everything tracked, without trying to destroy it.

        For use after a world reload: load_world destroys every actor in the
        world, so the pool is holding handles to things that no longer exist.
        Destroying them again is at best pointless and at worst an error
        against a fresh episode that has reused their ids.
        """
        self._actors.clear()

    def __enter__(self) -> "ActorPool":
        return self

    def __exit__(self, *exc) -> None:
        self.destroy()


def spawn_ego(world: carla.World, pool: ActorPool,
              blueprint: str = "vehicle.tesla.model3",
              spawn_index: int | None = None) -> carla.Vehicle:
    """Spawn the ego vehicle at the first spawn point that is actually free.

    try_spawn_actor returns None instead of raising when a spawn point is
    occupied, which matters the moment there is traffic on the map.
    """
    bp = world.get_blueprint_library().find(blueprint)
    bp.set_attribute("role_name", "ego")
    points = world.get_map().get_spawn_points()
    if spawn_index is not None:
        vehicle = world.try_spawn_actor(bp, points[spawn_index])
        if vehicle is None:
            sys.exit(f"spawn point {spawn_index} is occupied")
        return pool.add(vehicle)
    for point in points:
        vehicle = world.try_spawn_actor(bp, point)
        if vehicle is not None:
            return pool.add(vehicle)
    sys.exit("every spawn point on this map is occupied")


def mounts(vehicle: carla.Vehicle) -> dict[str, carla.Transform]:
    """Sensor mounting transforms, derived from this vehicle's actual size.

    Slide: 'Do not copy those numbers, derive them.'  bounding_box.extent is a
    HALF-size, so the roof is near 2 * extent.z and the front bumper near
    extent.x.  Treat it as a full size and you mount the camera inside the
    bodywork, which returns a perfectly valid image of upholstery.

    Note these are relative to the vehicle actor's origin, which CARLA places
    near the body centre -- NOT at the centre of the rear axle, which is the
    convention the lecture uses for the vehicle frame V.
    """
    e = vehicle.bounding_box.extent
    roof_z = 2.0 * e.z
    return {
        # behind the windscreen, looking forward
        "camera": carla.Transform(carla.Location(x=0.5 * e.x, z=roof_z * 0.95)),
        # on the roof, clear line of sight all round
        "lidar": carla.Transform(carla.Location(z=roof_z + 0.10)),
        # front bumper, at bumper height, looking forward
        "radar": carla.Transform(carla.Location(x=e.x, z=0.5 * e.z),
                                 carla.Rotation(pitch=5.0)),
        # position sensors sit wherever; put them at the origin
        "imu": carla.Transform(),
        "gnss": carla.Transform(),
    }


def camera_intrinsics(camera_bp) -> tuple[float, float, float]:
    """(f, cx, cy) in pixels for a CARLA RGB camera blueprint.

    CARLA cameras are an ideal pinhole: square pixels, principal point exactly
    at the image centre, no distortion.  Real cameras are none of those things,
    which is why the calibration section exists.
    """
    w = int(camera_bp.get_attribute("image_size_x"))
    h = int(camera_bp.get_attribute("image_size_y"))
    fov = float(camera_bp.get_attribute("fov"))
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return f, w / 2.0, h / 2.0
