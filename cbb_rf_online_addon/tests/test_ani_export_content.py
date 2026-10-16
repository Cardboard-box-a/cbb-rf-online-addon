"""Content-level tests for the ANI exporter.

Motivation: a user reported that an export produced a single, larger ``.ANI``
that looked like two animations "joined", instead of one file per animation.
These tests pin down what the exporter actually does:

* one ``.ANI`` file per Action, named after the Action;
* each file contains only that Action's objects (no cross-Action / cross-object
  contamination);
* each object carries only its own animation - two Actions' data is never
  merged, and two bones never share one another's curves;
* the frame and keyframe counts match the Action's frame range.

They run against the real ``bpy`` runtime, so the per-frame sampling, slot
handling and coordinate conversions under test are Blender's real behaviour.
"""

import glob
import math
import os
import struct

import bpy
from mathutils import Quaternion


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_armature(name, bones):
    """Create an armature. ``bones`` is a list of ``(name, head, tail)``."""
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


def _make_cube(name):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    return obj


def _new_action(arm_obj, name, slot_name=None):
    """Create a one-slot action and assign it to ``arm_obj``."""
    action = bpy.data.actions.new(name)
    slot = action.slots.new(id_type="OBJECT", name=slot_name or arm_obj.name)
    adt = arm_obj.animation_data or arm_obj.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    return action


def _key_bone_rotation(arm_obj, bone, quats_by_frame):
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    pose_bone = arm_obj.pose.bones[bone]
    pose_bone.rotation_mode = "QUATERNION"
    for frame, quat in quats_by_frame.items():
        pose_bone.rotation_quaternion = quat
        pose_bone.keyframe_insert("rotation_quaternion", frame=frame)
    bpy.ops.object.mode_set(mode="OBJECT")


def _key_bone_location(arm_obj, bone, locs_by_frame):
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    pose_bone = arm_obj.pose.bones[bone]
    for frame, loc in locs_by_frame.items():
        pose_bone.location = loc
        pose_bone.keyframe_insert("location", frame=frame)
    bpy.ops.object.mode_set(mode="OBJECT")


def _rot(degrees, axis):
    half = math.radians(degrees) * 0.5
    s = math.sin(half)
    if axis == "x":
        return Quaternion((math.cos(half), s, 0.0, 0.0))
    if axis == "y":
        return Quaternion((math.cos(half), 0.0, s, 0.0))
    return Quaternion((math.cos(half), 0.0, 0.0, s))


def _export(tmp_path):
    bpy.ops.cbb.ani_export("EXEC_DEFAULT", filepath=str(tmp_path / "out.ANI"))
    return sorted(glob.glob(str(tmp_path / "*.ANI")))


def _parse_ani(path):
    """Parse an .ANI file into ``{object_name: {...}}``.

    Layout (little endian), per object: char[100] name, ushort frame_amount,
    ushort frame_count, byte[36], then rotation/position/scale/unknown
    sections (each: ushort count followed by count * element_size bytes).
    """
    with open(path, "rb") as handle:
        data = handle.read()

    offset = 0
    (count,) = struct.unpack_from("<H", data, offset)
    offset += 2

    objects = {}
    for _ in range(count):
        name = data[offset:offset + 100].split(b"\0", 1)[0].decode("euc-kr", "ignore")
        offset += 100
        frame_amount, frame_count = struct.unpack_from("<HH", data, offset)
        offset += 4
        offset += 36

        sections = {}
        for section, element_size in (("rotations", 20), ("positions", 16),
                                      ("scales", 16), ("unknown", 8)):
            (key_count,) = struct.unpack_from("<H", data, offset)
            offset += 2
            values = []
            for _ in range(key_count):
                if section == "rotations":
                    quat = struct.unpack_from("<4f", data, offset)
                    (frame,) = struct.unpack_from("<I", data, offset + 16)
                    values.append((quat, frame))
                elif section in ("positions", "scales"):
                    vec = struct.unpack_from("<3f", data, offset)
                    (frame,) = struct.unpack_from("<I", data, offset + 12)
                    values.append((vec, frame))
                else:
                    (frame,) = struct.unpack_from("<I", data, offset + 4)
                    values.append((frame,))
                offset += element_size
            sections[section] = values

        objects[name] = {
            "frame_amount": frame_amount,
            "frame_count": frame_count,
            **sections,
        }
    return objects


def _distinct_rotations(entry, skip_bind_pose=True):
    """Distinct rotation values for one object entry.

    The first keyframe is the rest/bind pose, which is written for every bone
    regardless of whether it is animated, so it is skipped by default.
    """
    rotations = entry["rotations"][1:] if skip_bind_pose else entry["rotations"]
    return {tuple(round(component, 4) for component in quat) for quat, _frame in rotations}


# --------------------------------------------------------------------------- #
# One file per Action, never merged
# --------------------------------------------------------------------------- #

def test_two_actions_export_to_two_separate_files(fresh_scene, tmp_path):
    arm = _make_armature("Rig", [("A", (0, 0, 0), (0, 0, 1)), ("B", (0, 0, 1), (0, 0, 2))])

    # Clip A animates bone A only; Clip B animates bone B only.
    _new_action(arm, "ClipA")
    _key_bone_rotation(arm, "A", {frame: _rot(frame * 10, "x") for frame in (1, 10, 20)})
    _new_action(arm, "ClipB")
    _key_bone_rotation(arm, "B", {frame: _rot(frame * 10, "y") for frame in (1, 10, 20)})

    files = _export(tmp_path)
    assert [os.path.basename(path) for path in files] == ["ClipA.ANI", "ClipB.ANI"]

    # The two files must not be identical (that would be a "joined" export).
    with open(tmp_path / "ClipA.ANI", "rb") as a, open(tmp_path / "ClipB.ANI", "rb") as b:
        assert a.read() != b.read()

    clip_a = _parse_ani(str(tmp_path / "ClipA.ANI"))
    clip_b = _parse_ani(str(tmp_path / "ClipB.ANI"))

    # Both describe the same skeleton...
    assert set(clip_a) == set(clip_b) == {"A", "B"}

    # ...but each carries only its own animation.
    assert len(_distinct_rotations(clip_a["A"])) > 1   # A animated in ClipA
    assert len(_distinct_rotations(clip_a["B"])) == 1  # B static in ClipA
    assert len(_distinct_rotations(clip_b["B"])) > 1   # B animated in ClipB
    assert len(_distinct_rotations(clip_b["A"])) == 1  # A static in ClipB

    # The bind pose (first keyframe) is the skeleton's rest pose in every file.
    assert clip_a["A"]["rotations"][0] == clip_b["A"]["rotations"][0]
    assert clip_a["B"]["rotations"][0] == clip_b["B"]["rotations"][0]


def test_each_file_contains_only_its_own_objects(fresh_scene, tmp_path):
    rig_one = _make_armature("RigOne", [("one_a", (0, 0, 0), (0, 0, 1)), ("one_b", (0, 0, 1), (0, 0, 2))])
    rig_two = _make_armature("RigTwo", [("two_a", (0, 0, 0), (0, 0, 1)), ("two_b", (0, 0, 1), (0, 0, 2))])

    _new_action(rig_one, "ClipOne")
    _key_bone_rotation(rig_one, "one_a", {frame: _rot(frame * 10, "x") for frame in (1, 5)})
    _new_action(rig_two, "ClipTwo")
    _key_bone_rotation(rig_two, "two_a", {frame: _rot(frame * 10, "y") for frame in (1, 5)})

    _export(tmp_path)
    clip_one = _parse_ani(str(tmp_path / "ClipOne.ANI"))
    clip_two = _parse_ani(str(tmp_path / "ClipTwo.ANI"))

    assert set(clip_one) == {"one_a", "one_b"}
    assert set(clip_two) == {"two_a", "two_b"}


def test_exported_frame_and_keyframe_counts_match_action(fresh_scene, tmp_path):
    arm = _make_armature("Rig", [("A", (0, 0, 0), (0, 0, 1))])
    _new_action(arm, "Clip")
    _key_bone_rotation(arm, "A", {1: Quaternion((1, 0, 0, 0)), 24: _rot(90, "x")})

    _export(tmp_path)
    entry = _parse_ani(str(tmp_path / "Clip.ANI"))["A"]

    assert entry["frame_amount"] == 24          # last keyed frame
    assert len(entry["rotations"]) == 25        # bind pose + frames 1..24
    assert len(entry["positions"]) == 25
    assert len(entry["scales"]) == 25


def test_exported_rotation_is_constant_for_an_unkeyed_bone(fresh_scene, tmp_path):
    """A bone with no keys must come out as one repeated value, not extra motion."""
    arm = _make_armature("Rig", [("A", (0, 0, 0), (0, 0, 1)), ("B", (0, 0, 1), (0, 0, 2))])
    _new_action(arm, "Clip")
    _key_bone_rotation(arm, "A", {frame: _rot(frame * 15, "z") for frame in (1, 10, 20)})

    _export(tmp_path)
    data = _parse_ani(str(tmp_path / "Clip.ANI"))

    assert len(_distinct_rotations(data["B"])) == 1
    assert len(_distinct_rotations(data["A"])) > 1


def test_keyed_translation_is_exported(fresh_scene, tmp_path):
    """A keyed bone translation must reach the file as varying positions."""
    arm = _make_armature("Rig", [("A", (0, 0, 0), (0, 0, 1))])
    _new_action(arm, "Clip")
    _key_bone_location(arm, "A", {1: (0.0, 0.0, 0.0), 5: (0.0, 0.0, 1.0), 10: (0.0, 0.0, 2.0)})

    _export(tmp_path)
    entry = _parse_ani(str(tmp_path / "Clip.ANI"))["A"]

    positions = {tuple(round(c, 4) for c in vec) for vec, _frame in entry["positions"][1:]}
    assert len(positions) > 1


def test_objects_sharing_the_action_slot_are_exported_together(fresh_scene, tmp_path):
    """An object that selects the same Action + slot is written into the same file.

    This is the current contract: the exporter groups by Action, and every
    object using one of the Action's slots belongs to that file.
    """
    arm = _make_armature("Rig", [("A", (0, 0, 0), (0, 0, 1))])
    cube = _make_cube("Cube")

    action = _new_action(arm, "Shared")
    _key_bone_rotation(arm, "A", {frame: _rot(frame * 10, "x") for frame in (1, 5)})

    cube.animation_data_create()
    cube.animation_data.action = action
    cube.animation_data.action_slot = action.slots[0]

    _export(tmp_path)
    data = _parse_ani(str(tmp_path / "Shared.ANI"))

    assert "A" in data
    assert "Cube" in data
