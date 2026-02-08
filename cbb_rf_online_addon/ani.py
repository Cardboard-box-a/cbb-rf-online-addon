import bpy
import struct
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Context, Event, Operator, Action
from bpy.props import CollectionProperty, StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
import ntpath
import mathutils
import math
import traceback
from .utils import Utils, CoordsSys
import xml.etree.ElementTree as ET
from mathutils import Vector, Quaternion, Matrix
from .bn_skeleton import SkeletonData
from pathlib import Path
import os

FRAME_SCALE = 160

class CBB_OT_ImportAni(Operator, ImportHelper):
    bl_idname = "cbb.ani_import"
    bl_label = "Import ani"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".ANI"

    filter_glob: StringProperty(default="*.ANI", options={"HIDDEN"}) # type: ignore

    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"}
    ) # type: ignore

    directory: StringProperty(subtype="FILE_PATH") # type: ignore

    apply_to_selected_objects: BoolProperty(
        name="Apply to Selected Objects",
        description="Enabling this option will apply the animation to the currently selected objects. If false, a collection with the same base name as the animation is searched for and its objects used",
        default=False
    ) # type: ignore
    
    ignore_not_found: BoolProperty(
        name="Ignore Not Found Objects",
        description="Enabling this option will make the operator not raise an error in case an animated object or bone is not found, which is useful if the goal is to import the animation only to an armature and not to the whole mesh+skeleton package. It will still cause errors if the armature found is incompatible",
        default=True
    ) # type: ignore

    debug: BoolProperty(
        name="Debug",
        description="Enabling this option will print debug data to console",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.import_animations_from_files(context)

    def import_animations_from_files(self, context):
        
        msg_handler = Utils.MessageHandler(self.debug, self.report)
        
        for file in self.files:
            if file.name.casefold().endswith(".ani"):
                filepath: str = os.path.join(self.directory, file.name)
                
                skeleton_data = None
                target_armature = None
                
                animated_object_count = 0
                animated_object_names: list[str] = []
                
                frame_amount = []
                frame_counts = []
                
                rotation_keyframe_counts = []
                rotation_frames: list[list[tuple[Quaternion, int]]] = []
                
                position_keyframe_counts = []
                position_frames: list[list[tuple[Vector, int]]] = []
                
                scale_keyframe_counts = []
                scale_frames: list[list[tuple[Vector, int]]] = []
                
                unknown_keyframe_counts = []
                unknown_frames: list[list[tuple[float, int]]] = []
                
                
                co_conv = Utils.CoordinatesConverter(CoordsSys._3DSMax, CoordsSys.Blender)
                
                file_base_name = Path(file.name).stem.split("_")[0]
                
                try:
                    with open(filepath, "rb") as f:
                        reader = Utils.Serializer(f, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                        animated_object_count = reader.read_ushort()
                        for i in range(animated_object_count):
                            animated_object_names.append(reader.read_fixed_string(100, "euc-kr"))
                            # If set to 0 animation is not considered
                            frame_amount.append(reader.read_ushort())
                            # If set higher than the amount of keyframes that are registered, it affects looping animations, which does indicate this is the maximum frame.
                            # If set lower, the highest keyframe set (along the animation's keyframes) defines the maximum frame. In this case, this number is ignored.
                            # The reason why this value is not set in seconds or frames or whatever else is unknown. However, a single frame has a value of 160 for this number.
                            # Since the value is an u16, around 409 or so frames the number would overflow and the effect of this is also unknown.
                            frame_counts.append(reader.read_ushort())
                            
                            f.seek(36,1)
                            
                            def __read_rotation_frames(reader: Utils.Serializer):
                                return (reader.read_converted_quaternion(), reader.read_uint())
                            
                            def __read_position_frames(reader: Utils.Serializer):
                                return (reader.read_converted_vector3f(), reader.read_uint())
                            
                            def __read_scale_frames(reader: Utils.Serializer):
                                return (reader.read_vector3f(), reader.read_uint())
                            
                            def __read_unknown_frames(reader: Utils.Serializer):
                                return (reader.read_float(), reader.read_uint())
                            
                            rotation_keyframe_counts.append(reader.read_ushort())
                            
                            rotation_frames.append([__read_rotation_frames(reader) for _ in range(rotation_keyframe_counts[i])])
                            
                            position_keyframe_counts.append(reader.read_ushort())
                            position_frames.append([__read_position_frames(reader) for _ in range(position_keyframe_counts[i])])
                            
                            scale_keyframe_counts.append(reader.read_ushort())
                            scale_frames.append([__read_scale_frames(reader) for _ in range(scale_keyframe_counts[i])])
                            
                            unknown_keyframe_counts.append(reader.read_ushort()) # What is being animated with a single float in the range of 0 to 1???
                            unknown_frames.append([__read_unknown_frames(reader) for _ in range(unknown_keyframe_counts[i])])

                except Exception as e:
                    msg_handler.report("ERROR", f"Failed to read file at [{filepath}]: {e}")
                    traceback.print_exc()
                    continue

                found_objects: list[tuple[bpy.types.Object, int]] = []
                found_bones: list[tuple[str, int]] = []
                objects_collection = None
                
                if self.apply_to_selected_objects == True:
                    objects_collection = bpy.context.selected_objects
                else:
                    matching_collection = None
                    for collection in bpy.data.collections:
                        if collection.name.casefold() == file_base_name.casefold():
                            matching_collection = collection
                            break
                    
                    if matching_collection:
                        msg_handler.debug_print(f"Name [{file_base_name}] found within collections")
                        objects_collection = matching_collection.objects
                    else:
                        msg_handler.report("ERROR", f"No collection with the same base name [{file_base_name}] of the animation could be found in the scene.")
                        continue
                
                armature_objects = [obj for obj in objects_collection if obj.type == 'ARMATURE']
                if len(armature_objects) > 1:
                    msg_handler.report("ERROR", "More than one armature is selected. Please select only one armature.")
                    return {"CANCELLED"}
                
                if armature_objects:
                    target_armature = armature_objects[0]
                    msg_handler.debug_print(f"Target armature name: {target_armature.name}")
                    skeleton_data = SkeletonData(msg_handler)
                    try:
                        skeleton_data.build_skeleton_from_armature(target_armature, False)
                    except Exception as e:
                        msg_handler.report("ERROR", f"Armature [{target_armature}] which is the target of the imported animation has been found not valid. Aborting. Reason: {e}")
                        continue
                
                for i, object_name in enumerate(animated_object_names):
                    found = False
                    
                    for obj in objects_collection:
                        if obj.name == object_name:
                            found_objects.append((obj, i))
                            found = True
                            break

                    if not found and target_armature:
                        for bone in target_armature.data.bones:
                            if bone.name == object_name:
                                found_bones.append((bone.name, i))
                                found = True
                                break

                    if not found and self.ignore_not_found == False:
                        msg_handler.report("ERROR", f"Object or bone with name '{object_name}' not found in the selection.")
                        return {"CANCELLED"}

                msg_handler.debug_print(f"Animation Data: ")
                msg_handler.debug_print(f" Amount of animated objects: {animated_object_count}")
                msg_handler.debug_print(f" Amount of frames: {frame_amount[0]}")
                
                try:
                    # ACTION SLOTS IMPLEMENTATION:
                    # Create ONE action for the entire animation - all objects/bones will share this action
                    # but each will have its own slot with isolated data paths.
                    # This is only possible on newer Blender versions
                    animation_name = Path(file.name).stem
                    action = bpy.data.actions.new(name=animation_name)
                    action.use_fake_user = True  # Prevent deletion on save
                    highest_frame = 0
                    
                    # Import object-level animations (MESH and EMPTY objects)
                    for obj, index in found_objects:
                        # Assign the SAME action to this object
                        # Blender will automatically create a separate slot for this object
                        if obj.animation_data is None:
                            obj.animation_data_create()
                        obj.animation_data.action = action
                        
                        # Keyframe the object - these keyframes go into this object's slot
                        # No data path pollution because each object has its own slot
                        for count in range(rotation_keyframe_counts[index]):
                            animated_rotation, scaled_frame = rotation_frames[index][count]
                            frame = scaled_frame / FRAME_SCALE

                            if highest_frame < frame:
                                highest_frame = frame

                            rot = animated_rotation.conjugated()
                            obj.rotation_quaternion = rot
                            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

                        for count in range(position_keyframe_counts[index]):
                            animated_position, scaled_frame = position_frames[index][count]
                            frame = scaled_frame / FRAME_SCALE

                            if highest_frame < frame:
                                highest_frame = frame

                            obj.location = animated_position
                            obj.keyframe_insert(data_path="location", frame=frame)
                        
                        for count in range(scale_keyframe_counts[index]):
                            animated_scale, scaled_frame = scale_frames[index][count]
                            frame = scaled_frame / FRAME_SCALE

                            if highest_frame < frame:
                                highest_frame = frame

                            obj.scale = animated_scale
                            obj.keyframe_insert(data_path="scale", frame=frame)
                    
                    # Import bone animations (armature)
                    if target_armature and found_bones:
                        # Assign the SAME action to the armature
                        # The armature gets its own slot within this action
                        if target_armature.animation_data is None:
                            target_armature.animation_data_create()
                        target_armature.animation_data.action = action
                        
                        for bone_name, index in found_bones:
                            bone_id = skeleton_data.bone_name_to_id[bone_name]
                            
                            if rotation_keyframe_counts[index] == 0:
                                target_armature.pose.bones[bone_name].rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=0)
                            for count in range(rotation_keyframe_counts[index]):
                                animated_rotation, scaled_frame = rotation_frames[index][count]
                                frame = scaled_frame/FRAME_SCALE
                                
                                if highest_frame < frame:
                                    highest_frame = frame
                                    
                                rot = Utils.get_local_rotation(skeleton_data.bone_local_rotations[bone_id], Quaternion((-animated_rotation.w, animated_rotation.x, animated_rotation.y, animated_rotation.z)))
                                
                                target_armature.pose.bones[bone_name].rotation_quaternion = rot
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=frame)
                                
                            if position_keyframe_counts[index] == 0:
                                target_armature.pose.bones[bone_name].location = Vector((0.0, 0.0, 0.0))
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="location", frame=0)
                            for count in range(position_keyframe_counts[index]):
                                animated_position, scaled_frame = position_frames[index][count]
                                frame = scaled_frame/FRAME_SCALE
                                
                                if highest_frame < frame:
                                    highest_frame = frame
                                
                                loc = Utils.get_local_position(skeleton_data.bone_local_positions[bone_id], skeleton_data.bone_local_rotations[bone_id], animated_position)

                                target_armature.pose.bones[bone_name].location = loc
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="location", frame=frame)
                            
                            if scale_keyframe_counts[index] == 0:
                                target_armature.pose.bones[bone_name].scale = Vector((1.0, 1.0, 1.0))
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="scale", frame=0)
                            for count in range(scale_keyframe_counts[index]):
                                animated_scale, scaled_frame = scale_frames[index][count]
                                frame = scaled_frame/FRAME_SCALE
                                
                                if highest_frame < frame:
                                    highest_frame = frame

                                target_armature.pose.bones[bone_name].scale = animated_scale
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="scale", frame=frame)

                    # Set animation frames range
                    action.frame_range = (0, highest_frame)

                except Exception as e:
                    animation_name = Path(file.name).stem
                    msg_handler.report("ERROR", f"Failed to create animation {animation_name}: {e}")
                    traceback.print_exc()
                    return {"CANCELLED"}
                
        
        return {"FINISHED"}

    def invoke(self, context: Context, event: Event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class CBB_FH_ImportAni(bpy.types.FileHandler):
    bl_idname = "CBB_FH_import_ani"
    bl_label = "File handler for ANI import"
    bl_import_operator = "cbb.ani_import"
    bl_file_extensions = ".ANI"

    @classmethod
    def poll_drop(cls, context):
        return (context.area and context.area.type == 'VIEW_3D')


class CBB_OT_ExportAni(Operator, ExportHelper):
    bl_idname = "cbb.ani_export"
    bl_label = "Export ANI"
    bl_options = {"PRESET"}

    filename_ext = ".ANI"

    filter_glob: StringProperty(default="*.ANI", options={"HIDDEN"}) # type: ignore
    
    debug: BoolProperty(
        name="Debug",
        description="Enabling this option will print debug data to console",
        default=False
    ) # type: ignore
    
    def execute(self, context):
    
        msg_handler = Utils.MessageHandler(self.debug, self.report)
        msg_handler.debug_print("Starting ANI export execution...")
        
        directory = os.path.dirname(self.filepath)
        msg_handler.debug_print(f"Export directory: {directory}")
        
        # ACTION SLOTS IMPLEMENTATION:
        # Instead of exporting per-object actions, we export per-Action
        # Each Action represents one complete animation (like one .ANI file)
        
        actions_to_export = {}
        msg_handler.debug_print("Collecting actions with linked objects...")
        
        total_actions_scanned = 0
        actions_with_objects = 0
        
        for action in bpy.data.actions:
            if not action:
                continue
                
            total_actions_scanned += 1
            msg_handler.debug_print(f"  Processing action: '{action.name}' (users: {action.users})")
            
            linked_objects = []
            
            # Primary method: Action slots
            if hasattr(action, "slots") and action.slots:
                msg_handler.debug_print(f"    Action has {len(action.slots)} slot(s)")
                
                for slot in action.slots:
                    identifier = slot.identifier
                    
                    # Strip Blender's internal ID type prefix (OB = Object, AR = Armature, etc.)
                    # The slot identifier uses format like "OBObjectName" but obj.name is just "ObjectName"
                    if identifier.startswith("OB"):
                        identifier = identifier[2:]
                    
                    msg_handler.debug_print(f"      Slot identifier: '{slot.identifier}' → object name: '{identifier}'")
                    
                    # Try to find matching object
                    found = False
                    for obj in bpy.data.objects:
                        if obj.name == identifier and obj not in linked_objects:
                            linked_objects.append(obj)
                            msg_handler.debug_print(f"        → Found object: '{obj.name}' ({obj.type})")
                            found = True
                            break
                    
                    if not found:
                        msg_handler.debug_print(f"      No object found for slot identifier: '{identifier}'")
            
            # Fallback: Check NLA tracks for stashed/pushed actions
            nla_found_count = 0
            for obj in bpy.data.objects:
                if not (obj.animation_data and obj.animation_data.nla_tracks):
                    continue
                    
                nla_actions = Utils.get_actions_from_nla_tracks(obj)
                if action in nla_actions and obj not in linked_objects:
                    linked_objects.append(obj)
                    msg_handler.debug_print(f"      NLA fallback → Added object: '{obj.name}' ({obj.type}) from NLA track")
                    nla_found_count += 1
            
            if nla_found_count > 0:
                msg_handler.debug_print(f"    NLA fallback added {nla_found_count} object(s)")
            
            # Final decision for this action
            if linked_objects:
                actions_to_export[action] = linked_objects
                actions_with_objects += 1
                msg_handler.debug_print(f"  Action '{action.name}' will be exported — linked to {len(linked_objects)} object(s):")
                for obj in linked_objects:
                    msg_handler.debug_print(f"     • {obj.name} ({obj.type})")
            else:
                msg_handler.debug_print(f"  Action '{action.name}' has NO linked objects → skipped")
        
        msg_handler.debug_print(f"Summary: Found {actions_with_objects} exportable actions out of {total_actions_scanned} total actions")
        msg_handler.debug_print(f"Actions to export: {len(actions_to_export)}")
        
        if not actions_to_export:
            msg_handler.debug_print("No actions with linked objects found. Nothing to export.")
            return {"FINISHED"}
        
        # Actually export each collected action
        exported_count = 0
        for action, objects in actions_to_export.items():
            msg_handler.debug_print(f"Exporting action: '{action.name}' ({len(objects)} objects)")
            self.export_action(action, objects, directory, msg_handler)
            exported_count += 1
        
        msg_handler.debug_print(f"Export finished — processed {exported_count} action(s)")
        
        return {"FINISHED"}
        
    def export_action(self, action: Action, action_objects: list[bpy.types.Object], directory: str, msg_handler: Utils.MessageHandler):
        print(f"Exporting action {action.name}")
        
        # Save current state to restore later
        old_active_object = bpy.context.view_layer.objects.active
        old_active_selected = old_active_object.select_get() if old_active_object else False
        old_active_mode = old_active_object.mode if old_active_object else 'OBJECT'
        old_selection = [obj for obj in bpy.context.selected_objects]
        old_frame = bpy.context.scene.frame_current
        old_actions = {}
        
        # Save all objects' current actions
        for obj in action_objects:
            if obj.animation_data:
                old_actions[obj] = obj.animation_data.action
        
        # Switch to object mode for evaluation
        if old_active_object:
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # ACTION SLOTS MANUAL EVALUATION:
        # Instead of baking, we assign the action to all objects and manually step through frames
        # This preserves Action Slots and properly evaluates constraints/drivers
        # Baking the actions to NLA data was not working very well with the new slots implementation.
        
        for obj in action_objects:
            # Ensure the object has the action assigned
            if obj.animation_data is None:
                obj.animation_data_create()
            obj.animation_data.action = action
        
        filepath = bpy.path.ensure_ext(directory + "/" + action.name, self.filename_ext)
        
        initial_frame = action.frame_range[0]
        last_frame = action.frame_range[1]
        
        # FRAME COUNT CALCULATION:
        # We export frames 0 through last_frame (inclusive)
        # Total keyframes = last_frame + 1
        # Example: frame_range (0, 69) → export frames 0-69 = 70 keyframes
        total_frames = int(last_frame)  # For the frame count field
        total_export_frames = int(last_frame) + 1  # Actual number of keyframes exported
        
        export_object_names = []
        export_unique_keyframe_counts = []
        export_maximum_frames = []
        export_rotation_keyframe_counts = []
        export_rotation_keyframes = {}
        export_position_keyframe_counts = []
        export_position_keyframes = {}
        export_scale_keyframe_counts = []
        export_scale_keyframes = {}
        export_unknown_keyframe_counts = []
        export_unknown_keyframes = {}
        
        msg_handler.debug_print(f"Animation [{action.name}] frame range: {int(action.frame_range[0])} - {int(action.frame_range[1])}")
        
        # ACTION SLOTS: Each object in action_objects has its own slot within the same action
        # We can iterate through them and export each slot's data
        for object in action_objects:
            
            if object.type in {"MESH", "EMPTY"}:
                def add_object_animation_data(_object_name, _total_frames, _total_export_frames, _export_rotation_keyframes, _export_position_keyframes, _export_scale_keyframes):
                    nonlocal export_object_names
                    nonlocal export_unique_keyframe_counts
                    nonlocal export_maximum_frames
                    nonlocal export_rotation_keyframe_counts
                    nonlocal export_rotation_keyframes
                    nonlocal export_position_keyframe_counts
                    nonlocal export_position_keyframes
                    nonlocal export_scale_keyframe_counts
                    nonlocal export_scale_keyframes
                    nonlocal export_unknown_keyframe_counts
                    nonlocal export_unknown_keyframes
                    
                    index = len(export_object_names)
                    
                    export_object_names.append(_object_name)
                    export_unique_keyframe_counts.append(_total_frames)
                    export_maximum_frames.append(_total_frames*FRAME_SCALE)
                    export_rotation_keyframe_counts.append(_total_export_frames)
                    export_position_keyframe_counts.append(_total_export_frames)
                    export_scale_keyframe_counts.append(_total_export_frames)
                    export_unknown_keyframe_counts.append(0)
                    
                    export_rotation_keyframes[index] = _export_rotation_keyframes
                    export_position_keyframes[index] = _export_position_keyframes
                    export_scale_keyframes[index] = _export_scale_keyframes
                
                object_name = object.name
                temp_rotation_keyframes = []
                temp_position_keyframes = []
                temp_scale_keyframes = []
                
                # SIMPLIFIED OBJECT EXPORT:
                # For objects, matrix_basis IS the local transform (what the animator keyframes).
                # Unlike bones, objects don't have a separate "rest pose"
                # We simply export matrix_basis for all frames including frame 0.
                
                depsgraph = bpy.context.evaluated_depsgraph_get()
                
                # Export all frames starting from 0
                for frame in range(0, int(action.frame_range[1]+1)):
                    # Set the scene to this frame
                    bpy.context.scene.frame_set(frame)
                    
                    # Force update
                    depsgraph.update()
                    
                    # Get the evaluated object (this includes constraints, drivers, etc.)
                    object_eval = object.evaluated_get(depsgraph)
                    
                    # Read matrix_basis - this is the object's local transform (independent of parent)
                    # This is what the animator keyframes and what the ANI format expects
                    local_matrix = object_eval.matrix_basis
                    
                    obj_animated_rotation = local_matrix.to_quaternion()
                    obj_animated_position = local_matrix.to_translation()
                    obj_animated_scale = local_matrix.to_scale()
                    
                    # Convert to export format
                    temp_rotation_keyframes.append(Quaternion((-obj_animated_rotation.w, obj_animated_rotation.x, obj_animated_rotation.y, obj_animated_rotation.z)))
                    temp_position_keyframes.append(obj_animated_position)
                    temp_scale_keyframes.append(obj_animated_scale)
                    
                if object.type == "MESH":
                    mesh: bpy.types.Mesh = object.data
                    mesh_polygons = mesh.polygons
                    
                    # Group polygons by material (Logic must match msh.py)
                    material_polygon_counts = {}
                    
                    if not mesh_polygons:
                        pass
                    else:
                        for poly in mesh_polygons:
                            mat_idx = poly.material_index if poly.material_index < len(object.material_slots) else 0
                            
                            # Calculate indices for this polygon (triangulated)
                            # 3 -> 3 indices
                            # 4 -> 6 indices
                            # n -> (n-2)*3 indices
                            poly_indices_count = (len(poly.loop_indices) - 2) * 3
                            
                            if mat_idx not in material_polygon_counts:
                                material_polygon_counts[mat_idx] = 0
                            material_polygon_counts[mat_idx] += poly_indices_count

                    # Iterate through groups and apply splitting logic
                    # Sort keys to ensure deterministic order (though dicts are ordered in modern Python, explicit sort is safer for file stability)
                    sorted_mat_indices = sorted(material_polygon_counts.keys())
                    
                    for mat_idx in sorted_mat_indices:
                        indices_count = material_polygon_counts[mat_idx]
                        sub_object_base_name = f"{object_name}_{mat_idx}"
                        
                        if indices_count <= 65535:
                            add_object_animation_data(sub_object_base_name, total_frames, total_export_frames, temp_rotation_keyframes, temp_position_keyframes, temp_scale_keyframes)
                        else:
                            print(f"Material Group {mat_idx} too large ({indices_count} indices). Splitting...")
                            maximum_split_amount = math.ceil(indices_count / 65535.0)
                            
                            for split_number in range(0, maximum_split_amount):
                                split_object_name = f"{sub_object_base_name}_{split_number}"
                                add_object_animation_data(split_object_name, total_frames, total_export_frames, temp_rotation_keyframes, temp_position_keyframes, temp_scale_keyframes)
                elif object.type == "EMPTY":
                    # EMPTY objects export normally without splitting
                    add_object_animation_data(object_name, total_frames, total_export_frames, temp_rotation_keyframes, temp_position_keyframes, temp_scale_keyframes)
            
            # Export armature bone animations from this armature's slot in the action
            if object.type == "ARMATURE":
                skeleton_data = SkeletonData(msg_handler)
                skeleton_data.build_skeleton_from_armature(object, False)
                
                for bone_name in skeleton_data.bone_names:
                    index = len(export_object_names)
                    export_object_names.append(bone_name)
                    export_unique_keyframe_counts.append(total_frames)
                    export_maximum_frames.append(total_frames*FRAME_SCALE)
                    export_rotation_keyframe_counts.append(total_export_frames)
                    export_position_keyframe_counts.append(total_export_frames)
                    export_scale_keyframe_counts.append(total_export_frames)
                    export_unknown_keyframe_counts.append(0)
                    
                    bone_id = skeleton_data.bone_name_to_id[bone_name]
                    pose_bone = object.pose.bones[bone_name]
                    
                    temp_rotation_keyframes = []
                    temp_position_keyframes = []
                    temp_scale_keyframes = []
                    
                    # Add binding pose as first keyframe (frame 0 in ANI format)
                    # For bones, we always use the skeleton's rest pose data
                    obj_local_rot = skeleton_data.bone_local_rotations[bone_id]
                    temp_rotation_keyframes.append(Quaternion((-obj_local_rot.w, obj_local_rot.x, obj_local_rot.y, obj_local_rot.z)))
                    temp_position_keyframes.append(skeleton_data.bone_local_positions[bone_id])
                    temp_scale_keyframes.append(skeleton_data.bone_absolute_scales[bone_id])
                    
                    # MANUAL FRAME EVALUATION FOR BONES:
                    # Export animation frames starting from frame 1
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    
                    for frame in range(1, int(action.frame_range[1]+1)):
                        # Set the scene to this frame
                        bpy.context.scene.frame_set(frame)
                        
                        # Force update
                        depsgraph.update()
                        
                        # Get the evaluated armature (includes constraints, drivers, etc.)
                        object_eval = object.evaluated_get(depsgraph)
                        pose_bone_eval = object_eval.pose.bones[bone_name]
                        
                        pose_bone_rotation = pose_bone_eval.rotation_quaternion.copy()
                        local_animated_rotation = Utils.get_world_rotation(skeleton_data.bone_local_rotations[bone_id], pose_bone_rotation)
                        temp_rotation_keyframes.append(Quaternion((-local_animated_rotation.w, local_animated_rotation.x, local_animated_rotation.y, local_animated_rotation.z)))
                        
                        pose_bone_position = pose_bone_eval.location.copy()
                        local_animated_position = Utils.get_world_position(skeleton_data.bone_local_positions[bone_id], skeleton_data.bone_local_rotations[bone_id], pose_bone_position)
                        temp_position_keyframes.append(local_animated_position)
                        
                        obj_animated_scale = pose_bone_eval.scale.copy()
                        temp_scale_keyframes.append(obj_animated_scale)
                        
                    export_rotation_keyframes[index] = temp_rotation_keyframes
                    export_position_keyframes[index] = temp_position_keyframes
                    export_scale_keyframes[index] = temp_scale_keyframes
                    
        
        co_conv = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys._3DSMax)
        with open(filepath, 'wb') as file:
            writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
            writer.write_ushort(len(export_object_names))
            for index, object_name in enumerate(export_object_names):
                writer.write_fixed_string(100, "euc-kr", object_name)
                writer.write_ushort(export_unique_keyframe_counts[index])
                writer.write_ushort(export_maximum_frames[index])
                file.write(bytearray(36))
                
                exporting_rotation_keyframes = export_rotation_keyframes.get(index)
                writer.write_ushort(export_rotation_keyframe_counts[index])
                if exporting_rotation_keyframes is not None:
                    for frame_number, rotation in enumerate(exporting_rotation_keyframes):
                        writer.write_converted_quaternion(rotation)
                        writer.write_uint(frame_number*FRAME_SCALE)
                
                exporting_position_keyframes = export_position_keyframes.get(index)
                writer.write_ushort(export_position_keyframe_counts[index])
                if exporting_position_keyframes is not None:
                    for frame_number, position in enumerate(exporting_position_keyframes):
                        writer.write_converted_vector3f(position)
                        writer.write_uint(frame_number*FRAME_SCALE)
                
                exporting_scale_keyframes = export_scale_keyframes.get(index)
                writer.write_ushort(export_scale_keyframe_counts[index])
                if exporting_scale_keyframes is not None:
                    for frame_number, scale in enumerate(exporting_scale_keyframes):
                        writer.write_vector3f(scale)
                        writer.write_uint(frame_number*FRAME_SCALE)
                
                exporting_unknown_keyframes = export_unknown_keyframes.get(index)
                writer.write_ushort(export_unknown_keyframe_counts[index])
                if exporting_unknown_keyframes is not None:
                    for frame_number, unknown in enumerate(exporting_unknown_keyframes):
                        writer.write_float(unknown)
                        writer.write_uint(frame_number*FRAME_SCALE)
        
        # Restore previous state
        bpy.context.scene.frame_set(old_frame)
        
        # Restore old actions
        for obj, old_action in old_actions.items():
            if obj.animation_data:
                obj.animation_data.action = old_action
        
        # Restore selection and mode
        bpy.ops.object.select_all(action='DESELECT')
        
        if old_active_object:
            bpy.context.view_layer.objects.active = old_active_object
            old_active_object.select_set(old_active_selected)
        
        for obj in old_selection:
            obj.select_set(True)
            
        if old_active_object and old_active_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=old_active_mode)
        
        msg_handler.debug_print(f"Successfully exported animation '{action.name}' to {filepath}")


def menu_func_import(self, context):
    self.layout.operator(CBB_OT_ImportAni.bl_idname, text="ANI (.ANI)")

def menu_func_export(self, context):
    self.layout.operator(CBB_OT_ExportAni.bl_idname, text="ANI (.ANI)")



def register():
    bpy.utils.register_class(CBB_OT_ImportAni)
    bpy.utils.register_class(CBB_FH_ImportAni)
    bpy.utils.register_class(CBB_OT_ExportAni)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(CBB_OT_ImportAni)
    bpy.utils.unregister_class(CBB_FH_ImportAni)
    bpy.utils.unregister_class(CBB_OT_ExportAni)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()