"""Tests for the ANI importer, in particular multi-file imports.

Regression: dropping several ``.ANI`` files onto Blender ran the operator's
``invoke``, which unconditionally opened the file selector. The selector was
pre-pointed at a single file, so confirming it imported only that one file and
created a single Action. A drop must import every dropped file.
"""

import glob
import math
import os

import bpy
from bpy_extras.io_utils import ImportHelper
from mathutils import Quaternion

from cbb_rf_online_addon import ani


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

CLIP_NAMES = (
    "ISIS_AMY_ATTACK_03_00",
    "ISIS_AMY_ATTACK_03_01",
    "ISIS_AMY_ATTACK_03_02",
    "ISIS_AMY_CORPSE_01_00",
    "ISIS_AMY_CRITICAL_01_00",
)


def _make_armature(name, bones):
    arm_data = bpy.data.armatures.new(name + "Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone_name, head, tail in bones:
        edit_bone = arm_data.edit_bones.new(bone_name)
        edit_bone.head = head
        edit_bone.tail = tail
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _new_action(arm_obj, name):
    action = bpy.data.actions.new(name)
    slot = action.slots.new(id_type="OBJECT", name=arm_obj.name)
    adt = arm_obj.animation_data or arm_obj.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    return action


def _key_bone(arm_obj, bone, frames=(1, 5)):
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    pose_bone = arm_obj.pose.bones[bone]
    pose_bone.rotation_mode = "QUATERNION"
    for frame in frames:
        pose_bone.rotation_quaternion = Quaternion(
            (math.cos(frame * 0.1), math.sin(frame * 0.1), 0.0, 0.0))
        pose_bone.keyframe_insert("rotation_quaternion", frame=frame)
    bpy.ops.object.mode_set(mode="OBJECT")


def _make_ani_files(arm_obj, tmp_path):
    """Export one .ANI per clip name; return the file names."""
    for name in CLIP_NAMES:
        _new_action(arm_obj, name)
        _key_bone(arm_obj, "A")
    bpy.ops.cbb.ani_export("EXEC_DEFAULT", filepath=str(tmp_path / "out.ANI"))
    return sorted(os.path.basename(path) for path in glob.glob(str(tmp_path / "*.ANI")))


def _clear_actions(arm_obj):
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    if arm_obj.animation_data:
        arm_obj.animation_data.action = None


def _select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_export_writes_one_file_per_clip(fresh_scene, tmp_path):
    arm = _make_armature("Rig", [("A", (0, 0, 0), (0, 0, 1))])
    files = _make_ani_files(arm, tmp_path)
    assert files == [f"{name}.ANI" for name in sorted(CLIP_NAMES)]


def test_import_creates_one_action_per_file(fresh_scene, tmp_path):
    arm = _make_armature("Rig", [("A", (0, 0, 0), (0, 0, 1))])
    files = _make_ani_files(arm, tmp_path)
    _clear_actions(arm)
    _select_only(arm)

    result = bpy.ops.cbb.ani_import(
        "EXEC_DEFAULT",
        directory=str(tmp_path),
        files=[{"name": name} for name in files],
        apply_to_selected_objects=True,
    )

    assert result == {"FINISHED"}
    assert sorted(action.name for action in bpy.data.actions) == sorted(CLIP_NAMES)


# --------------------------------------------------------------------------- #
# The invoke path (what a file drop hits)
# --------------------------------------------------------------------------- #

class _FakeProperties:
    def __init__(self, set_names):
        self._set = set(set_names)

    def is_property_set(self, name):
        return name in self._set


class _FakeWindowManager:
    def __init__(self):
        self.fileselect_add_calls = 0
        self.props_dialog_calls = []

    def fileselect_add(self, operator):
        self.fileselect_add_calls += 1

    def invoke_props_dialog(self, operator, confirm_text="", title="", translate=False):
        self.props_dialog_calls.append({"title": title, "confirm_text": confirm_text})
        return {"RUNNING_MODAL"}


class _FakeContext:
    def __init__(self):
        self.window_manager = _FakeWindowManager()


class _FakeOperator:
    bl_label = "Import ani"

    def __init__(self, set_names, filepath="", files=()):
        self.properties = _FakeProperties(set_names)
        self.filepath = filepath
        self.files = list(files)
        self.execute_calls = 0

    def execute(self, context):
        self.execute_calls += 1
        return {"FINISHED"}

    def invoke_popup(self, context, confirm_text=""):
        # Delegate to Blender's real implementation so the test exercises the
        # same code path the operator uses at runtime.
        return ImportHelper.invoke_popup(self, context, confirm_text)


def test_invoke_shows_the_options_dialog_for_dropped_files():
    """A drop must show the import options instead of importing blindly."""
    context = _FakeContext()
    dropped = _FakeOperator(
        {"filepath"},
        filepath="/tmp/ISIS_AMY_ATTACK_03_00.ANI",
        files=["a", "b", "c", "d", "e"],
    )

    result = ani.CBB_OT_ImportAni.invoke(dropped, context, None)

    assert result == {"RUNNING_MODAL"}
    assert dropped.execute_calls == 0                        # waits for confirmation
    assert context.window_manager.fileselect_add_calls == 0   # no file selector
    assert context.window_manager.props_dialog_calls == [
        {"title": "Import 5 files", "confirm_text": "Import ani"}
    ]


def test_invoke_opens_the_file_selector_when_no_filepath_is_set():
    """The menu path (no filepath yet) must still open the file selector."""
    context = _FakeContext()
    manual = _FakeOperator(set())

    result = ani.CBB_OT_ImportAni.invoke(manual, context, None)

    assert result == {"RUNNING_MODAL"}
    assert manual.execute_calls == 0
    assert context.window_manager.fileselect_add_calls == 1
    assert context.window_manager.props_dialog_calls == []
