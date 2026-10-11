import bpy
import struct
import traceback
import mathutils
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Context, Event, Operator
from bpy.props import CollectionProperty, StringProperty, BoolProperty
from mathutils import Vector, Quaternion, Matrix
import math
from .utils import Utils, CoordsSys
import os
from pathlib import Path

# Very small bones get deleted automatically by Blender, so we need a minimum length to ensure bones aren't deleted while importing.
MIN_BONE_LENGTH = 0.05

class CBB_OT_ImportBNSkeleton(Operator, ImportHelper):
    bl_idname = "cbb.bn_skeleton_import"
    bl_label = "Import BN Skeleton"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".bn"

    filter_glob: StringProperty(default="*.bn", options={"HIDDEN"}) # type: ignore

    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"}
    ) # type: ignore

    directory: StringProperty(subtype="FILE_PATH") # type: ignore

    debug: BoolProperty(
        name="Debug",
        description="Enabling this option will print debug data to console",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.import_skeletons_from_files(context)

    def import_skeletons_from_files(self: "CBB_OT_ImportBNSkeleton", context: bpy.types.Context):
        
        msg_handler = Utils.MessageHandler(self.debug, self.report)
        
        for file in self.files:
            if file.name.casefold().endswith(".bn"):
                filepath = os.path.join(self.directory, file.name)

                msg_handler.debug_print(f"Importing skeleton from: {filepath}")
                
                skeleton_data = SkeletonData(msg_handler)
                try:
                    skeleton_data.read_skeleton_data(filepath)
                except Exception as e:
                    msg_handler.report("ERROR", f"Error while trying to read skeleton data from file [{filepath}]: {e}")
                    continue
                try:
                    file_base_name = Path(file.name).stem
                    
                    new_collection = None
                    if file_base_name in bpy.data.collections:
                        new_collection = bpy.data.collections[file_base_name]
                    else:
                        new_collection = bpy.data.collections.new(file_base_name)
                        bpy.context.scene.collection.children.link(new_collection)
                    
                    bone_shapes_collection = bpy.data.collections.new(name="Bone Shapes")
                    new_collection.children.link(bone_shapes_collection)
                    
                    # Create armature and enter edit mode
                    armature = bpy.data.armatures.new(skeleton_data.skeleton_name)
                    armature_obj = bpy.data.objects.new(skeleton_data.skeleton_name, armature)
                    new_collection.objects.link(armature_obj)
                    
                    bpy.context.view_layer.objects.active = armature_obj
                    bpy.ops.object.mode_set(mode="EDIT")

                    edit_bones = armature_obj.data.edit_bones
                    bones = []
                    bone_lengths: list[float] = []
                    
                    
                    for i in range(skeleton_data.bone_count):
                        bone = edit_bones.new(skeleton_data.bone_names[i])
                        bones.append(bone)
                        bone_lengths.append(9999)
                    
                    msg_handler.debug_print(f"Created [{len(edit_bones)}] bones in Blender armature.")

                    root_bones = [i for i in range(skeleton_data.bone_count) if skeleton_data.bone_parent_names[i] == SkeletonData.INVALID_NAME]
                    
                    msg_handler.debug_print(f"Detected [{len(root_bones)}] bones without a parent.")
                    
                    def __calculate_bone_length(cur_bone_id):
                        def pick_bone_length():
                            child_locs = []
                            for child_id in [idx for idx, pid in enumerate(skeleton_data.bone_parent_ids) if pid == cur_bone_id]:
                                child_locs.append(skeleton_data.bone_local_matrices[child_id].to_translation())
                                
                            # If the bone has children, return the min of the children's position length
                            if child_locs:
                                min_length = min((loc.length for loc in child_locs))
                                return max(min_length, MIN_BONE_LENGTH)
                                
                            # If the bone is not a root bone and has no children, return the parent's length
                            if skeleton_data.bone_parent_ids[cur_bone_id] != SkeletonData.NO_PARENT:
                                parent_bone_length = bone_lengths[skeleton_data.bone_parent_ids[cur_bone_id]]
                                return max(parent_bone_length, MIN_BONE_LENGTH)

                            return 1
                        
                        bone_lengths[cur_bone_id] = pick_bone_length()
                        for child_id in [idx for idx, pid in enumerate(skeleton_data.bone_parent_ids) if pid == cur_bone_id]:
                            __calculate_bone_length(child_id)

                    for root_bone_id in root_bones:
                        __calculate_bone_length(root_bone_id)
                    
                    for i in range(skeleton_data.bone_count):
                        bones[i].length = bone_lengths[i]
                        edit_bone = armature_obj.data.edit_bones[skeleton_data.bone_names[i]]
                        edit_bone.matrix = skeleton_data.bone_absolute_matrices[i]
                        msg_handler.debug_print(f"Length of bone [{bones[i].name}]: {bones[i].length}")
                        
                        if skeleton_data.bone_parent_ids[i] != SkeletonData.NO_PARENT:
                            bones[i].parent = bones[skeleton_data.bone_parent_ids[i]]
                            
                        


                    bpy.context.view_layer.update()
                    bpy.ops.object.mode_set(mode="POSE")
                    
                    for bone_index, bone_name in enumerate(skeleton_data.bone_names):
                        # Bones exported without a custom shape carry no vertex data. Skip creating a
                        # bone shape object for them, otherwise the importer would assign an empty
                        # mesh as the bone's custom shape, making it invisible in Pose mode.
                        if len(skeleton_data.bone_vertices[bone_index]) == 0:
                            msg_handler.debug_print(f"Skipping bone shape object for [{bone_name}]: no vertex data.")
                            continue
                        
                        mesh_name = f"{bone_name}_Mesh"
                        mesh_data = bpy.data.meshes.new(name=mesh_name)
                        mesh_obj = bpy.data.objects.new(name=mesh_name, object_data=mesh_data)

                        bone_shapes_collection.objects.link(mesh_obj)

                        mesh_data.from_pydata(skeleton_data.bone_vertices[bone_index], [], skeleton_data.bone_faces[bone_index])
                        mesh_data.update()
                        if mesh_obj is not None:
                            pose_bone = armature_obj.pose.bones.get(bone_name)
                            if pose_bone:
                                pose_bone.custom_shape = mesh_obj
                                pose_bone.use_custom_shape_bone_size = False
                                
                    bpy.ops.object.mode_set(mode="OBJECT")

                except Exception as e:
                    msg_handler.report("ERROR", f"Failed to create blender armature for skeleton at [{filepath}]: {e}")
                    traceback.print_exc()
                    continue

        return {"FINISHED"}

    def invoke(self, context: Context, event: Event):
        return self.invoke_popup(context)

class CBB_FH_ImportBNSkeleton(bpy.types.FileHandler):
    bl_idname = "CBB_FH_bn_skeleton_import"
    bl_label = "File handler for skeleton imports"
    bl_import_operator = CBB_OT_ImportBNSkeleton.bl_idname
    bl_file_extensions = CBB_OT_ImportBNSkeleton.filename_ext

    @classmethod
    def poll_drop(cls, context):
        return (context.area and context.area.type == "VIEW_3D")
    
class CBB_OT_ExportBNSkeleton(Operator, ExportHelper):
    bl_idname = "cbb.bn_skeleton_export"
    bl_label = "Export BN Skeleton"
    bl_options = {"PRESET"}

    filename_ext = CBB_OT_ImportBNSkeleton.filename_ext

    filter_glob: StringProperty(default="*.bn", options={"HIDDEN"}) # type: ignore

    directory: StringProperty(subtype="FILE_PATH") # type: ignore

    debug: BoolProperty(
        name="Debug export",
        description="Enabling this option will make the exporter print debug data to console",
        default=False
    ) # type: ignore
    
    export_only_selected: BoolProperty(
        name="Export only selected",
        description="Leave this option checked if you wish export only skeletons among currently selected objects",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.export_skeletons(context, self.directory)

    def export_skeletons(self, context: bpy.types.Context, directory: str):
        
        msg_handler = Utils.MessageHandler(self.debug, self.report)
        
        objects_for_exportation = None
        if self.export_only_selected == True:
            objects_for_exportation: list[bpy.types.Object] = [obj for obj in context.selected_objects if obj.type == "ARMATURE"]
        else:
            objects_for_exportation = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]

        if not objects_for_exportation:
            if self.export_only_selected == True:
                msg_handler.report("ERROR", f"There are no objects of type ARMATURE among currently selected objects. Aborting exportation.")
            else:
                msg_handler.report("ERROR", f"There are no objects of type ARMATURE among scene objects. Aborting exportation.")
            return {"CANCELLED"}
        
        for armature_object_export in objects_for_exportation:
            def export_skeleton(armature_object: bpy.types.Object):
                armature_export_name = Utils.get_immediate_parent_collection(armature_object).name
                filepath: str = bpy.path.ensure_ext(directory + armature_export_name, self.filename_ext)
                msg_handler.debug_print(f"Exporting armature [{armature_object.name}] to file at [{filepath}]")
                
                skeleton_data = SkeletonData(msg_handler)
                try:
                    skeleton_data.build_skeleton_from_armature(armature_object, True)
                except Exception as e:
                    msg_handler.report("ERROR", f"Failed to build skeleton from armature [{armature_object.name}]: {e}")
                    traceback.print_exc()
                    return
                    
                
                try:
                    skeleton_data.write_skeleton_data(filepath)
                except Exception as e:
                    msg_handler.report("ERROR", f"Failed to export skeleton: {e}")
                    return
            
            export_skeleton(armature_object_export)

        
        return {"FINISHED"}

class SkeletonData:
    """
    Class that holds convenient skeleton information. Do note that absolute in the name of transform variables refers to them being 
    referent to the armature only, as if the armature transform was the center of the world.
    """
    
    
    INVALID_NAME: str = "NULL"
    NO_PARENT:int = -1
    def __init__(self, msg_handler: Utils.MessageHandler):
        self.skeleton_name: str = ""
        self.skeleton_hit_box_max: Vector = Vector((0.0, 0.0, 0.0))
        self.skeleton_hit_box_min: Vector = Vector((0.0, 0.0, 0.0))
        self.bone_name_to_id: dict[str, int] = {}
        self.bone_count: int = 0
        self.bone_names: str = []
        self.bone_parent_names: str = []
        self.bone_parent_ids: list[int] = []
        self.bone_absolute_matrices: list[Matrix] = []
        self.bone_local_matrices: list[Matrix] = []
        self.bone_parent_inverse_matrices: list[Matrix] = []
        self.bone_absolute_positions: list[Vector] = []
        self.bone_absolute_scales: list[Vector] = []
        self.bone_absolute_rotations: list[Quaternion] = []
        self.bone_local_positions: list[Vector] = []
        self.bone_local_rotations: list[Quaternion] = []
        self.bone_hit_boxes_max: list[Vector] = []
        self.bone_hit_boxes_min: list[Vector] = []
        self.bone_vertices: list[list[Vector]] = []
        self.bone_normals: list[list[Vector]] = []
        self.bone_faces: list[list[int]] = []
        self.msg_handler: Utils.MessageHandler = msg_handler
    
    def read_skeleton_data(self, filepath: str):
        """
        Populates skeleton data from data in a file.
        """
        bbx_filepath = os.path.splitext(filepath)[0]+".bbx"
        co_conv = Utils.CoordinatesConverter(CoordsSys._3DSMaxInverseY, CoordsSys.Blender)
        
        try:
            with open(bbx_filepath, "rb") as f:
                reader = Utils.Serializer(f, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                try:
                    self.skeleton_name = reader.read_fixed_string(256, "ascii")
                except Exception as e:
                    self.msg_handler.report("WARNING",f"Skeleton's name in bbx file could not be read or decoded as ascii correctly, using bn file name. Error: {e}")
                self.skeleton_hit_box_max = reader.read_converted_vector3f()
                self.skeleton_hit_box_min = reader.read_converted_vector3f()

        except FileNotFoundError:
            self.msg_handler.report("WARNING", f"BBX file was not found at path [{bbx_filepath}]. This is not vital, but the skeleton name in the bbx file will be ignored.")
        except Exception as e:
            self.msg_handler.report("WARNING", f"Unknown error, failed to read bbx data(this is not vital, but the skeleton name in the bbx file will be ignored). Error: {e} ")
            traceback.print_exc()
        
        if self.skeleton_name == "":
            self.skeleton_name = Path(filepath).stem
        
        co_conv = Utils.CoordinatesConverter(CoordsSys._3DSMax, CoordsSys.Blender)
        
        try:
            with open(filepath, "rb") as f:
                reader = Utils.Serializer(f, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                self.bone_count = reader.read_ushort()
                self.msg_handler.debug_print(f"Bone count from source skeleton: {self.bone_count}")

                for count in range(self.bone_count):
                    self.bone_names.append(reader.read_fixed_string(100, "ascii"))
                    self.bone_parent_names.append(reader.read_fixed_string(100, "ascii"))
                    self.bone_name_to_id[self.bone_names[count]] = count
                    
                    self.msg_handler.debug_print(f"Processing bone number [{count}], name: {self.bone_names[count]}")
                    
                    world_matrix = reader.read_matrix()
                    translation, rotation, scale = Utils.decompose_matrix_position_rotation_scale(world_matrix)
                    
                    rotation_needs_adjustment = False
                    if scale[0] < 0:
                        scale = Vector((-scale[0], scale[1], scale[2]))
                        rotation_needs_adjustment = True
                    if scale[1] < 0:
                        scale = Vector((scale[0], -scale[1], scale[2]))
                        rotation_needs_adjustment = True
                    if scale[2] < 0:
                        scale = Vector((scale[0], scale[1], -scale[2]))
                        rotation_needs_adjustment = True
                    
                    if rotation_needs_adjustment:
                        # Magic fix
                        # Maybe if the scale is -1 only in certain axes this would need to be changed, but I haven't found such bone yet
                        self.msg_handler.debug_print(f" Bone needs rotation fix")
                        rotation = Quaternion((-rotation.w, -rotation.x, -rotation.y, -rotation.z))
                        world_matrix = Matrix.Translation(translation) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
                    
                    
                    
                    self.bone_absolute_matrices.append(co_conv.convert_matrix(world_matrix))

                    self.msg_handler.debug_print(f" Bone converted and fixed world matrix:")
                    self.msg_handler.debug_print(f"{self.bone_absolute_matrices[count]}")
                    
                    self.bone_local_matrices.append(reader.read_converted_matrix())
                    
                    self.msg_handler.debug_print(f" Bone converted local matrix:")
                    self.msg_handler.debug_print(f"{self.bone_local_matrices[count]}")
                    
                    # Not 100% sure about this one, everything works just fine without it
                    self.bone_parent_inverse_matrices.append(reader.read_converted_matrix())
                    
                    self.msg_handler.debug_print(f" Bone converted parent inverse matrix:")
                    self.msg_handler.debug_print(f"{self.bone_parent_inverse_matrices[count]}")
                    
                    shape_vertices_amount = reader.read_ushort()
                    shape_faces_amount = reader.read_ushort()
                    unknown_number = reader.read_ushort()
                    
                    self.msg_handler.debug_print(f" Vertices amount [{shape_vertices_amount}]")
                    self.msg_handler.debug_print(f" Faces amount [{shape_faces_amount}]")
                    self.msg_handler.debug_print(f" Unknown amount [{unknown_number}]")
                    
                    f.seek(204, 1)
                    
                    # Not sure either, values are way too large for small bones sometimes
                    self.bone_hit_boxes_max.append(reader.read_converted_vector3f())
                    self.bone_hit_boxes_min.append(reader.read_converted_vector3f())
                    
                    unknown_flags = reader.read_values("16I", 64)
                    has_face_terminator_indices = unknown_flags[4] != 0
                    f.seek(3, 1)
                    
                    boneVertices = []
                    boneNormals = []
                    
                    for i in range(shape_vertices_amount):
                        read_vertex = reader.read_vector3f()
                        if rotation_needs_adjustment:
                            read_vertex = Vector((-read_vertex.x, -read_vertex.y, -read_vertex.z))
                        boneVertices.append(co_conv.convert_vector3f(read_vertex))
                        f.seek(4,1)
                        boneNormals.append(reader.read_vector3f())
                    
                    boneFaces = []
                    firstFaceIndex = reader.read_uint() if shape_faces_amount > 0 or has_face_terminator_indices else 0
                    
                    for i in range(shape_faces_amount):
                        
                        lastFaceIndices = reader.read_values("2I", 8)
                        
                        boneFaces.append((firstFaceIndex, lastFaceIndices[0], lastFaceIndices[1]))
                        
                        f.seek(76, 1)
                        
                        if has_face_terminator_indices or i != shape_faces_amount - 1:
                            firstFaceIndex = reader.read_uint()
                        
                    self.bone_vertices.append(boneVertices)
                    self.bone_faces.append(boneFaces)
                    self.bone_normals.append(boneNormals)
                    
                    if unknown_number > 0:
                        f.seek(100+40*unknown_number, 1)
                        
                for count in range(self.bone_count):
                    parent_name = self.bone_parent_names[count]
                    if parent_name != SkeletonData.INVALID_NAME:
                        self.bone_parent_ids.append(self.bone_name_to_id[parent_name])
                    else:
                        self.bone_parent_ids.append(SkeletonData.NO_PARENT)
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"BN file not found at: {filepath}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Error decoding a string for the ascii encoding: {e}")
        except Exception as e:
            raise RuntimeError(f"Unknown error, failed to read skeleton data: {e}")
    
    def write_skeleton_data(self, filepath: str):
        co_conv = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys._3DSMax)
        try:
            with open(filepath, "wb") as file:
                writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                try:
                    writer.write_ushort(self.bone_count)
                    for bone_index in range(self.bone_count):
                        writer.write_fixed_string(100, "ascii", self.bone_names[bone_index])
                        if self.bone_parent_ids[bone_index] == SkeletonData.NO_PARENT:
                            writer.write_fixed_string(100, "ascii", "NULL")
                        else:
                            writer.write_fixed_string(100, "ascii", self.bone_parent_names[bone_index])
                        writer.write_converted_matrix(self.bone_absolute_matrices[bone_index])
                        writer.write_converted_matrix(self.bone_local_matrices[bone_index])
                        writer.write_matrix(self.bone_parent_inverse_matrices[bone_index])
                        writer.write_ushort(len(self.bone_vertices[bone_index]))
                        writer.write_ushort(len(self.bone_faces[bone_index]))
                        file.write(bytearray(238))
                        writer.write_uint(1)
                        writer.write_uint(0)
                        writer.write_uint(1)
                        file.write(bytearray(47))
                        for bone_vertex_index in range(len(self.bone_vertices[bone_index])):
                            writer.write_converted_vector3f(self.bone_vertices[bone_index][bone_vertex_index])
                            writer.write_float(1.0)
                            writer.write_converted_vector3f(self.bone_normals[bone_index][bone_vertex_index])
                        
                        if self.bone_faces[bone_index]:
                            writer.write_uint(self.bone_faces[bone_index][0][0])
                            for face_index, face in enumerate(self.bone_faces[bone_index]):
                                writer.write_uint(face[1])
                                writer.write_uint(face[2])
                                file.write(bytearray(72))
                                writer.write_int(-1)
                                # If not the last face
                                if face_index != len(self.bone_faces[bone_index])-1:
                                    writer.write_uint(self.bone_faces[bone_index][face_index+1][0])
                                else:
                                    writer.write_uint(0)
                                
                        else:
                            writer.write_uint(0)    
                        
                except Exception as e:
                    file.close()
                    os.remove(filepath)
                    traceback.print_exc()
                    raise Exception(f"Exception while writing to file at [{filepath}]: {e}")
        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Could not open file for writing at [{filepath}]: {e}")
        
        self.msg_handler.debug_print(f"Skeleton written successfully to: [{filepath}]")
        
        co_conv = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys._3DSMaxInverseY)
        
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
        max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
        has_position_data = False

        for bone_position in self.bone_absolute_positions:
            if not bone_position:
                continue
            has_position_data = True

            min_x = min(min_x, bone_position.x)
            min_y = min(min_y, bone_position.y)
            min_z = min(min_z, bone_position.z)
            max_x = max(max_x, bone_position.x)
            max_y = max(max_y, bone_position.y)
            max_z = max(max_z, bone_position.z)

        if not has_position_data:
            self.msg_handler.debug_print("Warning: No position for bones found to calculate skeleton bounding box. BBX will have zero bounds.")
            min_x, min_y, min_z = 0.0, 0.0, 0.0
            max_x, max_y, max_z = 0.0, 0.0, 0.0

        skeleton_hit_box_min_target_space = mathutils.Vector((min_x, min_y, min_z))
        skeleton_hit_box_max_target_space = mathutils.Vector((max_x, max_y, max_z))

        bbx_filepath = os.path.splitext(filepath)[0] + ".bbx"
        try:
            with open(bbx_filepath, "wb") as bbx_file:
                bbx_writer = Utils.Serializer(bbx_file, Utils.Serializer.Endianness.Little,
                                              Utils.Serializer.Quaternion_Order.XYZW,
                                              Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)

                # Write skeletonName (char[256])
                name_bytes = self.skeleton_name.encode('ascii', errors='ignore')
                padded_name_bytes = name_bytes[:255]
                bbx_file.write(padded_name_bytes)
                bbx_file.write(b'\0' * (256 - len(padded_name_bytes)))
                bbx_writer.write_converted_vector3f(skeleton_hit_box_max_target_space)
                bbx_writer.write_converted_vector3f(skeleton_hit_box_min_target_space)

        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Could not open or write BBX file at [{bbx_filepath}]: {e}")

        self.msg_handler.debug_print(f"BBX file written successfully to: [{bbx_filepath}]")
        
        return True
    
    
    def build_skeleton_from_armature(self, armature_object: bpy.types.Object, check_for_exportation: bool):
        
        """
            Function returns a SkeletonData class built from a Blender armature.
        """
        
        bones: list[bpy.types.Bone] = armature_object.data.bones
        
        self.skeleton_name = armature_object.name
        self.bone_count = len(bones)
        self.bone_names = [""] * self.bone_count
        self.bone_parent_names = [""] * self.bone_count
        self.bone_parent_ids = [SkeletonData.NO_PARENT] * self.bone_count
        self.bone_absolute_matrices: list[Matrix] = [Matrix.Identity(4)] * self.bone_count
        self.bone_local_matrices: list[Matrix] = [Matrix.Identity(4)] * self.bone_count
        self.bone_parent_inverse_matrices = [Matrix.Identity(4)] * self.bone_count
        self.bone_absolute_positions = [Vector((0.0, 0.0, 0.0))] * self.bone_count
        self.bone_absolute_scales = [Vector((0.0, 0.0, 0.0))] * self.bone_count
        self.bone_absolute_rotations = [Quaternion((0.0, 0.0, 0.0, 0.0))] * self.bone_count
        self.bone_local_positions = [Vector((0.0, 0.0, 0.0))] * self.bone_count
        self.bone_local_rotations = [Quaternion((0.0, 0.0, 0.0, 0.0))] * self.bone_count
        
        self.bone_vertices = [[] for _ in range(self.bone_count)]
        self.bone_normals = [[] for _ in range(self.bone_count)]
        self.bone_faces = [[] for _ in range(self.bone_count)]
        existing_bone_ids = set()
        
        
        for i, bone in enumerate(bones):
            self.bone_name_to_id[bone.name] = i
            
        for i, bone in enumerate(bones):
            bone_name = bone.name
            bone_id = i
            existing_bone_ids.add(bone_id)
            
            self.msg_handler.debug_print(f"Bone data for: {bone_name}, bone_id: {bone_id}")
            
            self.bone_names[bone_id] = bone_name
            
            self.bone_absolute_matrices[bone_id] = bone.matrix_local
            edit_bone_position , edit_bone_rotation, edit_bone_scale = Utils.decompose_matrix_position_rotation_scale(bone.matrix_local)
            self.bone_absolute_positions[bone_id] = edit_bone_position
            self.bone_absolute_rotations[bone_id] = edit_bone_rotation
            self.bone_absolute_scales[bone_id] = edit_bone_scale
            
            bone_parent = bone.parent
                
            self.msg_handler.debug_print(f" edit position: {edit_bone_position}")
            self.msg_handler.debug_print(f" edit rotation: {edit_bone_rotation}")
            self.msg_handler.debug_print(f" parent: {bone_parent}")
            
            if bone_parent is not None:
                self.bone_parent_names[bone_id] = bone_parent.name
                self.bone_parent_ids[bone_id] = self.bone_name_to_id[bone_parent.name]
                parent_edit_bone_position , parent_edit_bone_rotation = Utils.decompose_blender_matrix_position_rotation(bone_parent.matrix_local)
                self.bone_local_matrices[bone_id] = bone_parent.matrix_local.inverted() @ bone.matrix_local
                self.bone_local_positions[bone_id] = Utils.get_local_position(parent_edit_bone_position, parent_edit_bone_rotation, edit_bone_position)
                self.bone_local_rotations[bone_id] = Utils.get_local_rotation(parent_edit_bone_rotation, edit_bone_rotation)
                self.msg_handler.debug_print(f" local position: {self.bone_local_positions[bone_id]}")
                self.msg_handler.debug_print(f" local rotation: {self.bone_local_rotations[bone_id]}")
            else:
                self.bone_parent_ids[bone_id] = SkeletonData.NO_PARENT
                self.bone_parent_names[bone_id] = SkeletonData.INVALID_NAME
                self.bone_local_matrices[bone_id] = bone.matrix_local
                self.bone_local_positions[bone_id] = edit_bone_position
                self.bone_local_rotations[bone_id] = edit_bone_rotation
            
            pose_bone = armature_object.pose.bones.get(bone.name)
            if pose_bone and pose_bone.custom_shape:
                mesh = pose_bone.custom_shape.data
                vertices = [v.co for v in mesh.vertices]
                faces = [list(p.vertices) for p in mesh.polygons]
                normals = [v.normal for v in mesh.vertices]
                
                self.bone_vertices[i] = vertices
                self.bone_faces[i] = faces
                self.bone_normals[i] = normals
        
                
        min_corner = Vector((float('inf'), float('inf'), float('inf')))
        max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))

        for position in self.bone_absolute_positions:
            min_corner.x = min(min_corner.x, position.x)
            min_corner.y = min(min_corner.y, position.y)
            min_corner.z = min(min_corner.z, position.z)
            
            max_corner.x = max(max_corner.x, position.x)
            max_corner.y = max(max_corner.y, position.y)
            max_corner.z = max(max_corner.z, position.z)

        self.skeleton_hit_box_min = min_corner
        self.skeleton_hit_box_max = max_corner
           

def menu_func_import(self, context):
    self.layout.operator(CBB_OT_ImportBNSkeleton.bl_idname, text="BN Skeleton (.bn)")

def menu_func_export(self, context):
    self.layout.operator(CBB_OT_ExportBNSkeleton.bl_idname, text="BN Skeleton (.bn)")

def register():
    bpy.utils.register_class(CBB_OT_ImportBNSkeleton)
    bpy.utils.register_class(CBB_FH_ImportBNSkeleton)
    bpy.utils.register_class(CBB_OT_ExportBNSkeleton)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(CBB_OT_ImportBNSkeleton)
    bpy.utils.unregister_class(CBB_FH_ImportBNSkeleton)
    bpy.utils.unregister_class(CBB_OT_ExportBNSkeleton)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
