"""L10: a behavior tree for an AV on city streets, built and run in plain Python.

No CARLA, no ROS, no libraries: four node types and one tree. Each tick, the root
asks its children for a status, from the left, and the tick ends at the first
branch that does not fail. The action that runs is printed with the conditions the
tick checked, by number (a condition the tick never reached is not listed).

    python3 behavior_tree.py              # the tree with four branches
    python3 behavior_tree.py --no-pass    # the same tree with the pass branch removed

Every threshold is chosen for this example, as on the slides.
"""

import argparse

SUCCESS, FAILURE, RUNNING = "Success", "Failure", "Running"


# ------------------------------------------------------------- the four node types
class Condition:
    """Checks a fact about the world: Success if it holds, Failure if not.
    tag is the condition's number on the slides."""

    def __init__(self, tag, name, check):
        self.tag, self.name, self.check = tag, name, check

    def tick(self, world):
        holds = self.check(world)
        world["checked"].append(f"{self.tag} {'yes' if holds else 'no'}")
        return SUCCESS if holds else FAILURE


class Action:
    """Runs a behavior. A driving behavior is never done, so it returns Running."""

    def __init__(self, name):
        self.name = name

    def tick(self, world):
        world["action"] = self.name
        return RUNNING


class Sequence:
    """Stops at the first child that does not succeed."""

    def __init__(self, *children):
        self.children = children

    def tick(self, world):
        for child in self.children:
            status = child.tick(world)
            if status != SUCCESS:
                return status
        return SUCCESS


class Fallback:
    """Stops at the first child that does not fail."""

    def __init__(self, *children):
        self.children = children

    def tick(self, world):
        for child in self.children:
            status = child.tick(world)
            if status != FAILURE:
                return status
        return FAILURE


# -------------------------------------------------------------------- the tree
def build_tree(allow_pass=True):
    stop = Sequence(
        Condition(1, "stop line within 30 m",
                  lambda w: w["stop_line_m"] <= 30),
        Action("Stop at the line"))
    pass_car = Sequence(
        Condition(2, "car ahead under 6 m/s for 3 s",
                  lambda w: w["slow_for_s"] >= 3),
        Condition(3, "left lane free",
                  lambda w: w["left_free"]),
        Action("Change lane"))
    follow = Sequence(
        Condition(4, "car ahead within 40 m",
                  lambda w: w["lead_m"] <= 40),
        Action("Follow"))
    branches = [stop, pass_car, follow]
    if not allow_pass:
        branches.remove(pass_car)
    return Fallback(*branches, Action("Lane keep"))


# ------------------------------------------------- what the AV senses, tick by tick
FAR = float("inf")
SCENE = [   # (time in s, what the AV senses then); chosen for this example
    (0.0, dict(stop_line_m=FAR, lead_m=FAR, slow_for_s=0.0, left_free=True)),
    (2.0, dict(stop_line_m=FAR, lead_m=35.0, slow_for_s=1.0, left_free=True)),
    (4.0, dict(stop_line_m=FAR, lead_m=25.0, slow_for_s=3.0, left_free=True)),
    (4.1, dict(stop_line_m=FAR, lead_m=25.0, slow_for_s=3.1, left_free=False)),
    (9.0, dict(stop_line_m=25.0, lead_m=FAR, slow_for_s=0.0, left_free=True)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pass", action="store_true", help="remove the pass branch")
    tree = build_tree(allow_pass=not parser.parse_args().no_pass)
    for t, sensed in SCENE:
        world = dict(sensed, checked=[], action=None)
        tree.tick(world)
        print(f"t = {t:3.1f} s  {world['action']:<17} checked: " + ", ".join(world["checked"]))


if __name__ == "__main__":
    main()
