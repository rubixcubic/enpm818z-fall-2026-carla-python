import behavior_tree as bt


def run(tree):
    out = []
    for _, sensed in bt.SCENE:
        world = dict(sensed, checked=[], action=None)
        assert tree.tick(world) == bt.RUNNING
        out.append((world["action"], world["checked"]))
    return out


def test_the_scene_matches_the_slides():
    actions = [a for a, _ in run(bt.build_tree())]
    assert actions == ["Lane keep", "Follow", "Change lane", "Follow", "Stop at the line"]


def test_a_tick_stops_at_the_first_branch_that_does_not_fail():
    checked = [c for _, c in run(bt.build_tree())]
    assert checked[4] == ["1 yes"]                       # the stop line wins; nothing else is checked
    assert checked[3] == ["1 no", "2 yes", "3 no", "4 yes"]


def test_removing_the_pass_branch_changes_nothing_else():
    actions = [a for a, _ in run(bt.build_tree(allow_pass=False))]
    assert actions == ["Lane keep", "Follow", "Follow", "Follow", "Stop at the line"]


def test_sequence_and_fallback_on_their_own():
    yes = bt.Condition(0, "yes", lambda w: True)
    no = bt.Condition(0, "no", lambda w: False)
    w = {"checked": []}
    assert bt.Sequence(yes, yes).tick(w) == bt.SUCCESS
    assert bt.Sequence(yes, no, yes).tick(w) == bt.FAILURE
    assert bt.Fallback(no, no).tick(w) == bt.FAILURE
    assert bt.Fallback(no, yes, no).tick(w) == bt.SUCCESS
