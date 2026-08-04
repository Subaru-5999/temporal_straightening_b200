"""env/__init__.py must import without the MuJoCo stack.

It used to do `from .pointmaze import U_MAZE, MEDIUM_MAZE` eagerly, which chains
through maze_model.py into `gym.envs.mujoco` and `d4rl`. That made mujoco-py
(compiled from source) and d4rl (installed from git) hard requirements for
*every* environment, including PushT, which needs neither.

These tests run without gym installed by stubbing the one symbol env/__init__.py
imports, so they verify the structure rather than the runtime.

Run:  pytest tests/test_env_registration.py -q
"""

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT = os.path.join(ROOT, "env", "__init__.py")
SRC = open(INIT, encoding="utf-8").read()


def test_pointmaze_import_is_guarded():
    """The import must be inside a try/except, not at module top level."""
    m = re.search(r"try:\s*\n\s*from \.pointmaze import", SRC)
    assert m, "the .pointmaze import is not inside a try block"


def test_import_failure_is_recorded_not_raised():
    assert "_POINTMAZE_AVAILABLE = False" in SRC
    assert "except Exception" in SRC


def test_maze_registrations_are_conditional():
    """point_maze and point_maze_medium reference U_MAZE / MEDIUM_MAZE, so they
    must only register when the import succeeded."""
    guard = SRC.index("if _POINTMAZE_AVAILABLE:")
    for env_id in ("id='point_maze'", 'id="point_maze_medium"'):
        assert SRC.index(env_id) > guard, f"{env_id} registers outside the guard"


@pytest.mark.parametrize("env_id", [
    'id="pusht"', 'id="wall"', 'id="deformable_env"',
])
def test_mujoco_free_envs_register_unconditionally(env_id):
    """PushT / Wall / deformable need no MuJoCo, so they must always register."""
    idx = SRC.index(env_id)
    guard = SRC.index("if _POINTMAZE_AVAILABLE:")
    # either before the guard, or after it at zero indentation
    if idx > guard:
        line_start = SRC.rfind("\n", 0, SRC.rfind("register(", 0, idx)) + 1
        assert not SRC[line_start:].startswith(" "), f"{env_id} is inside the guard"


def test_failure_message_points_at_the_fix():
    assert "setup_planning.sh" in SRC
    assert "PushT" in SRC


def test_module_imports_with_pointmaze_broken(monkeypatch, tmp_path):
    """Simulate a missing MuJoCo stack: env/__init__.py must still import and
    still register pusht."""
    gym = pytest.importorskip("gym", reason="gym not installed in this env")

    registered = []

    def fake_register(**kwargs):
        registered.append(kwargs.get("id"))

    monkeypatch.setattr("gym.envs.registration.register", fake_register)
    # make `from .pointmaze import ...` fail the way a missing d4rl would
    monkeypatch.setitem(sys.modules, "env.pointmaze", None)
    for mod in [m for m in list(sys.modules) if m == "env" or m.startswith("env.")]:
        if mod != "env.pointmaze":
            monkeypatch.delitem(sys.modules, mod, raising=False)

    sys.path.insert(0, ROOT)
    with pytest.warns(UserWarning, match="PointMaze environments unavailable"):
        import env  # noqa: F401

    assert "pusht" in registered
    assert "point_maze" not in registered
