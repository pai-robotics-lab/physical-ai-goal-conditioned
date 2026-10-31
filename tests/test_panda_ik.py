import numpy as np

from pai_goal_conditioned.panda_ik import TARGETS, quaternion_error_deg


def test_five_unique_targets_are_defined() -> None:
    assert len(TARGETS) == 5
    assert len({target for _, target in TARGETS}) == 5


def test_quaternion_error_uses_shortest_rotation() -> None:
    target = np.array([0.0, 1.0, 0.0, 0.0])
    assert quaternion_error_deg(target, target) == 0.0
    assert quaternion_error_deg(target, -target) == 0.0

