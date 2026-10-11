# RF Online addon — tests

These tests exercise the addon against a **real, headless Blender runtime**
provided by the [`bpy`](https://pypi.org/project/bpy/) Python module, so the
Action/Slot behavior under test is Blender's actual behavior, not a mock's.

## Setup

`bpy` wheels are version-locked to a single Python version. The addon targets
Blender 5.1+, whose `bpy` build is CPython 3.13:

```bash
cd <repo>
python3.13 -m venv .venv-bpy
. .venv-bpy/bin/activate
pip install "bpy==5.2.2"
pip install pytest
# The addon also imports numpy / Pillow / wand at package import time:
pip install addons/cbb_rf_online_addon/wheels/numpy-*manylinux*.whl
pip install addons/cbb_rf_online_addon/wheels/pillow-*manylinux*.whl
pip install addons/cbb_rf_online_addon/wheels/wand-*.whl
```

(`bpy` bundles numpy, so the numpy wheel is usually unnecessary.)

## Running

```bash
cd addons/cbb_rf_online_addon
python -m pytest tests -v
```

## What is covered (`test_ani_slots.py`)

Regression for the bug where skeletal/armature animations were silently dropped
on export because the exporter inferred an Action's objects from its **slot
names** (`slot.identifier == "OB" + object_name`). Blender default-names a slot
created for an armature `"Slot"`, so no matching object existed and the action
was skipped.

- `test_collect_finds_armature_with_default_named_slot` — the core repro: an
  armature animated through a slot named `"Slot"` is found via `slot.users()`.
- `test_armature_is_found_even_when_a_slot_named_object_exists` — guards against
  the old name-based heuristic (a decoy object named `"Slot"` must be ignored).
- `test_collect_uses_each_objects_own_selected_slot` — one action, two slots,
  two objects, each evaluated through its **own** selected slot.
- `test_collect_finds_nla_stashed_action_with_strip_slot` — an action stashed in
  an NLA track is found with the strip's slot.
- `test_export_operator_writes_armature_animation` — end-to-end: the export
  operator produces an `.ANI` containing the bone.
- `test_export_operator_leaves_action_and_slot_unchanged` — the exporter saves
  and restores each object's action **and** slot (no scene mutation).
- `test_export_operator_does_not_export_unrelated_actions` — orphan actions are
  ignored.
- `test_export_operator_writes_mesh_animation` — plain-object export regression.

The `.ANI` reader helper in the test parses just the animated-object names from
the exported file to assert the expected object/bone is present.
