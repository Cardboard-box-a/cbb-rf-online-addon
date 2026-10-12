"""Tests for the ANI exporter's Action-Slot handling.

Regression context
------------------
The exporter used to decide which objects an Action animates by *name-matching
the Action's slot identifiers against object names* (``slot.identifier`` is
``"OB" + slot name``). That silently fails for skeletal/armature animations:
when a slot is created from the slot selector while an armature is selected,
Blender default-names the slot ``"Slot"`` (identifier ``"OBSlot"``) instead of
after the armature, so no object named ``"Slot"`` exists and the Action is
dropped ("has NO linked objects -> skipped").

The fix derives the objects from the slot's own users and, for each object,
remembers the slot it selected so evaluation uses the right channelbag.

These tests run against the real ``bpy`` module, so ``slot.users()``, the
default slot naming, and animation evaluation are Blender's real behavior.
"""

import struct

import bpy

from cbb_rf_online_addon import ani


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_armature(name="WING", bone="Root"):
    arm_data = bpy.data.armatures.new(name + "Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bone = arm_data.edit_bones.new(bone)
    edit_bone.head = (0.0, 0.0, 0.0)
    edit_bone.tail = (0.0, 0.0, 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _key_bone(arm_obj, bone="Root", frames=(1, 24)):
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    pose_bone = arm_obj.pose.bones[bone]
    pose_bone.rotation_mode = "QUATERNION"
    for frame in frames:
        pose_bone.rotation_quaternion = (0.7071, 0.7071, 0.0, 0.0)
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    bpy.ops.object.mode_set(mode="OBJECT")


def _make_cube(name="Cube"):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    return obj


def _parse_ani_object_names(path):
    """Parse just the animated-object names out of an .ANI file.

    Layout (little endian): ushort object_count, then per object:
      char[100] name, ushort frame_amount, ushort frame_count, byte[36],
      rotation/position/scale/unknown sections, each: ushort count then
      count * element_size bytes.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    offset = 0
    (count,) = struct.unpack_from("<H", data, offset)
    offset += 2

    names = []
    for _ in range(count):
        name = data[offset:offset + 100].split(b"\0", 1)[0].decode("euc-kr", "ignore")
        names.append(name)
        offset += 100
        offset += 2  # frame amount
        offset += 2  # frame count
        offset += 36
        for element_size in (20, 16, 16, 8):  # rotation, position, scale, unknown
            (key_count,) = struct.unpack_from("<H", data, offset)
            offset += 2
            offset += key_count * element_size

    return names


# --------------------------------------------------------------------------- #
# The core regression: an armature whose slot is default-named "Slot"
# --------------------------------------------------------------------------- #

def test_collect_finds_armature_with_default_named_slot(fresh_scene):
    arm = _make_armature("WING")
    action = bpy.data.actions.new("TEST_WING")
    # This is exactly what Blender produces for an armature-created slot: the
    # display name is "Slot", the identifier is "OBSlot".
    slot = action.slots.new(id_type="OBJECT", name="Slot")
    assert slot.identifier == "OBSlot"

    adt = arm.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    _key_bone(arm)
    assert adt.action_slot.identifier == "OBSlot"

    targets = ani.collect_action_targets(action)

    assert [(obj.name, s.identifier) for obj, s in targets] == [("WING", "OBSlot")]


def test_armature_is_found_even_when_a_slot_named_object_exists(fresh_scene):
    """Guard against the old name-based heuristic.

    The old code looked for an object literally named after the slot ("Slot").
    Here such an object exists as a decoy; the fix must still return the
    armature that is actually animated by the slot.
    """
    _make_cube("Slot")  # decoy object whose name matches the slot name

    arm = _make_armature("WING")
    action = bpy.data.actions.new("TEST_WING")
    slot = action.slots.new(id_type="OBJECT", name="Slot")
    adt = arm.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    _key_bone(arm)

    targets = ani.collect_action_targets(action)

    assert [(obj.name, s.identifier) for obj, s in targets] == [("WING", "OBSlot")]


def test_collect_uses_each_objects_own_selected_slot(fresh_scene):
    """One action, two slots, two objects each using a different slot."""
    cube = _make_cube("Cube")
    arm = _make_armature("Rig")

    action = bpy.data.actions.new("Shared")
    cube_slot = action.slots.new(id_type="OBJECT", name="Cube")
    arm_slot = action.slots.new(id_type="OBJECT", name="Rig")

    cube.animation_data_create()
    cube.animation_data.action = action
    cube.animation_data.action_slot = cube_slot
    cube.keyframe_insert("location", frame=1)

    arm.animation_data_create()
    arm.animation_data.action = action
    arm.animation_data.action_slot = arm_slot
    _key_bone(arm)

    got = {obj.name: slot.identifier for obj, slot in ani.collect_action_targets(action)}

    assert got == {"Cube": "OBCube", "Rig": "OBRig"}


def test_collect_finds_nla_stashed_action_with_strip_slot(fresh_scene):
    """A stashed (not active) action is found through its NLA strip's slot."""
    cube = _make_cube("Cube")
    action = bpy.data.actions.new("Stashed")
    slot = action.slots.new(id_type="OBJECT", name="ArbitraryName")

    adt = cube.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    cube.location = (1.0, 2.0, 3.0)
    cube.keyframe_insert("location", frame=1)

    # Stash the action into an NLA track and clear the active action.
    track = adt.nla_tracks.new()
    strip = track.strips.new(action.name, int(action.frame_range[0]), action)
    strip.action_slot = slot
    adt.action = None

    targets = ani.collect_action_targets(action)

    assert [(obj.name, s.identifier if s else None) for obj, s in targets] == [
        ("Cube", "OBArbitraryName")
    ]


# --------------------------------------------------------------------------- #
# End-to-end: run the real export operator
# --------------------------------------------------------------------------- #

def test_export_operator_writes_armature_animation(fresh_scene, tmp_path):
    arm = _make_armature("WING")
    action = bpy.data.actions.new("TEST_WING")
    slot = action.slots.new(id_type="OBJECT", name="Slot")
    adt = arm.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    _key_bone(arm, frames=(1, 24))

    out = tmp_path / "TEST_WING.ANI"
    result = bpy.ops.cbb.ani_export("EXEC_DEFAULT", filepath=str(out))

    assert result == {"FINISHED"}
    assert out.exists(), "the armature animation must be exported, not skipped"
    assert "Root" in _parse_ani_object_names(out)


def test_export_operator_leaves_action_and_slot_unchanged(fresh_scene, tmp_path):
    arm = _make_armature("WING")
    action = bpy.data.actions.new("TEST_WING")
    slot = action.slots.new(id_type="OBJECT", name="Slot")
    adt = arm.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    _key_bone(arm)

    bpy.ops.cbb.ani_export("EXEC_DEFAULT", filepath=str(tmp_path / "TEST_WING.ANI"))

    assert arm.animation_data.action is action
    assert arm.animation_data.action_slot.identifier == "OBSlot"


def test_export_operator_does_not_export_unrelated_actions(fresh_scene, tmp_path):
    """An action with no users must not produce files or crash the export."""
    arm = _make_armature("WING")
    action = bpy.data.actions.new("TEST_WING")
    slot = action.slots.new(id_type="OBJECT", name="Slot")
    adt = arm.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    _key_bone(arm)

    # An unrelated action nobody uses.
    bpy.data.actions.new("Orphan")

    out = tmp_path / "TEST_WING.ANI"
    bpy.ops.cbb.ani_export("EXEC_DEFAULT", filepath=str(out))

    assert (tmp_path / "TEST_WING.ANI").exists()
    assert not (tmp_path / "Orphan.ANI").exists()


def test_export_operator_writes_mesh_animation(fresh_scene, tmp_path):
    """Plain-object animation keeps working end to end."""
    cube = _make_cube("Cube")
    action = bpy.data.actions.new("CubeAnim")
    slot = action.slots.new(id_type="OBJECT", name="Cube")
    adt = cube.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    for frame in (1, 24):
        cube.location = (float(frame), 0.0, 0.0)
        cube.keyframe_insert("location", frame=frame)

    out = tmp_path / "CubeAnim.ANI"
    result = bpy.ops.cbb.ani_export("EXEC_DEFAULT", filepath=str(out))

    assert result == {"FINISHED"}
    assert out.exists()
    assert "Cube" in _parse_ani_object_names(out)


# --------------------------------------------------------------------------- #
# Filename sanitization (illegal characters in Action names)
# --------------------------------------------------------------------------- #

def test_sanitize_filename_replaces_illegal_characters():
    assert ani.sanitize_filename("Rig|Idle") == "Rig_Idle"
    assert ani.sanitize_filename("Armature|clip|Base Layer") == "Armature_clip_Base Layer"
    assert ani.sanitize_filename('a:b*c?d"e<f>g\\h/i') == "a_b_c_d_e_f_g_h_i"
    assert ani.sanitize_filename("trailing. ") == "trailing"
    assert ani.sanitize_filename("") == "animation"


def test_export_operator_handles_action_name_with_illegal_chars(fresh_scene, tmp_path):
    """FBX-imported action names contain '|' (e.g. "Rig|Idle"); export must not
    crash with OSError [Errno 22] and must write a sanitized file name."""
    arm = _make_armature("Rig")
    action = bpy.data.actions.new("Rig|pl_bepo_orig02_skill_a_end|Base Layer")
    slot = action.slots.new(id_type="OBJECT", name="Slot")
    adt = arm.animation_data_create()
    adt.action = action
    adt.action_slot = slot
    _key_bone(arm)

    result = bpy.ops.cbb.ani_export("EXEC_DEFAULT", filepath=str(tmp_path / "out.ANI"))

    assert result == {"FINISHED"}
    expected = tmp_path / "Rig_pl_bepo_orig02_skill_a_end_Base Layer.ANI"
    assert expected.exists()
    assert "Root" in _parse_ani_object_names(expected)
