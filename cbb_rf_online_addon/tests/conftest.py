"""Pytest configuration for the RF Online addon tests.

These tests run against a real Blender runtime provided by the ``bpy`` Python
module (``pip install bpy``), so the exact Action/Slot behavior under test is
Blender's, not a mock's.

They make the sibling ``cbb_rf_online_addon`` package importable and give each
test an empty scene.
"""

import os
import sys
from pathlib import Path

import pytest

# <repo>/addons/cbb_rf_online_addon/tests/conftest.py -> <repo>/addons
# Override with CBB_ADDON_DIR to run the suite against another addon checkout.
_ADDONS_DIR = Path(os.environ.get("CBB_ADDON_DIR", Path(__file__).resolve().parents[2]))
if str(_ADDONS_DIR) not in sys.path:
    sys.path.insert(0, str(_ADDONS_DIR))

import bpy  # noqa: E402  (must come after sys.path setup)


def _reset_data() -> None:
    """Remove all data-blocks so each test starts from an empty scene.

    We deliberately avoid ``read_factory_settings`` here because that also
    resets Blender's registered Operator classes.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


@pytest.fixture(scope="session", autouse=True)
def _runtime():
    """Start from an empty file and register the ANI operators once."""
    bpy.ops.wm.read_factory_settings(use_empty=True)

    from cbb_rf_online_addon import ani

    ani.register()
    yield


@pytest.fixture(autouse=True)
def fresh_scene():
    """Give every test an empty scene."""
    _reset_data()
    return bpy.context.scene
