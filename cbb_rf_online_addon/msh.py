import bpy
import struct
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Context, Event, Operator
from bpy.props import CollectionProperty, StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
import ntpath
import mathutils
import tempfile
from mathutils import Vector, Quaternion, Matrix
import math
import traceback
from .utils import Utils, CoordsSys
import os
import xml.etree.ElementTree as ET
from .bn_skeleton import SkeletonData
import random
from pathlib import Path
import bmesh
from .rf_shared import RFShared

class CBB_OT_ImportMSH(Operator, ImportHelper):
    bl_idname = "cbb.msh_import"
    bl_label = "Import MSH"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".msh"

    filter_glob: StringProperty(default="*.msh",options={"HIDDEN"}) # type: ignore
    
    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"}
    ) # type: ignore

    directory: StringProperty(subtype="DIR_PATH") # type: ignore

    apply_to_armature_in_selected: BoolProperty(
        name="Apply to Armature in Selected",
        description="Enabling this option will make the import of the animation to target any armature present between currently selected objects.",
        default=False
    ) # type: ignore

    debug: BoolProperty(
        name="Debug",
        description="Enabling this option will print debug data to console",
        default=False
    ) # type: ignore
    
    preserve_parenting_relationships: BoolProperty(
        name="Preserve Parenting",
        description="Enabling this option make the importer preserve parent-child relationships across all objects, even object to bone relations. Turn this off if you wish to export the imported object to the .FBX format later. Keep it on if you pretend to use the imported object for anything else, such as reimporting it to RF again.",
        default=True
    ) # type: ignore
    
    EXTRACTION_FOLDER = "cbb_extract"
    WEIGHT_TOLERANCE = 1e-05
    @staticmethod
    def find_target_directory(start_path, target_dir, max_levels):
        if start_path == "":
            return ""
        
        current_path = start_path
        for _ in range(max_levels):
            current_path = os.path.abspath(os.path.join(current_path, os.pardir))
            if os.path.exists(os.path.join(current_path, *target_dir)):
                return os.path.join(current_path, *target_dir)
            
        return None
    
    def find_texture_in_directory(target_directory, mesh_name, possible_extensions=[".png", ".jpg", ".jpeg", ".bmp", ".tga", ".dds"]):
        for root, dirs, files in os.walk(target_directory):
            for file in files:
                for ext in possible_extensions:
                    if file.casefold() == (mesh_name + ext).casefold():
                        return os.path.join(root, file)
        return None
    
    
    def get_texture_as_image(self, mesh_file_path, texture_name: str, target_dir, max_levels) -> bpy.types.Image:
        # Check if the image is already loaded in Blender
        if texture_name in bpy.data.images:
            return bpy.data.images[texture_name]

        # Find the target directory
        #target_directory = CBB_OT_ImportMSH.find_target_directory(mesh_file_path, os.path.split(target_dir), max_levels)
        #if target_directory:
            #texture_path = CBB_OT_ImportMSH.find_texture_in_directory(target_directory, texture_name.split(".")[0])
            #if texture_path:
                #return bpy.data.images.load(texture_path, check_existing=True)

        # Normalize mesh file path
        mesh_file_path = os.path.normpath(mesh_file_path)

        # Traverse up to max_levels to find the "mesh" directory
        for _ in range(max_levels):
            if os.path.basename(mesh_file_path).casefold() == "mesh":
                break
            parent_path = os.path.dirname(mesh_file_path)
            if parent_path == mesh_file_path:  # Root directory reached
                break
            mesh_file_path = parent_path

        # Determine textures folder path
        textures_folder = os.path.normpath(os.path.join(mesh_file_path, "../Tex/"))
        if not os.path.exists(textures_folder):
            self.report({"INFO"}, f"Textures folder does not exist: {textures_folder}")
            return None

        target_filename_stem = os.path.splitext(texture_name)[0]
        target_filename_stem_lower = target_filename_stem.casefold()


        for root, _, files_in_dir in os.walk(textures_folder):
            for found_file in files_in_dir:
                found_file_stem, found_file_ext = os.path.splitext(found_file)
                if found_file_stem.casefold() == target_filename_stem_lower:
                    if found_file_ext.casefold() == ".dds":
                        full_texture_path = os.path.join(root, found_file)
                        try:
                            blender_image = bpy.data.images.load(full_texture_path, check_existing=True)
                            blender_image.pack()
                            return blender_image
                        except RuntimeError as e:
                            self.report({"WARNING"}, f"Found texture file '{full_texture_path}' but failed to load: {e}")
                        except Exception as e:
                            self.report({"ERROR"}, f"Unexpected error loading loose texture '{full_texture_path}': {e}")
                            traceback.print_exc()
                            return None # Or just 'continue' to try RFS files next
        
        # List all .rfs files in the textures folder
        rfs_files = [file for file in os.listdir(textures_folder) if file.casefold().endswith('.rfs')]

        for file_path in rfs_files:
            try:
                full_file_path = os.path.join(textures_folder, file_path)
                with open(full_file_path, "rb") as opened_file:
                    file_count = struct.unpack("<I", opened_file.read(4))[0]

                    for _ in range(file_count):
                        file_name = opened_file.read(56).split(b"\x00")[0].decode("euc-kr")
                        file_offset = struct.unpack("<I", opened_file.read(4))[0]
                        file_size = struct.unpack("<I", opened_file.read(4))[0]
                        if os.path.splitext(file_name)[0].casefold() == os.path.splitext(texture_name)[0].casefold():
                            # Old approach extracted the image and loaded it, the new approach packs the image within the blend file
                            #output_folder = os.path.join(textures_folder, ImportMSH.EXTRACTION_FOLDER, texture_name)

                            opened_file.seek(file_offset, 0)
                            dds_header = bytearray(opened_file.read(128))

                            if dds_header[:4] != b'DDS ':
                                dds_header = RFShared.unlock_dds(list(struct.unpack('<32I', dds_header)))
                                dds_header = struct.pack('<32I', *dds_header)

                            texture_data = opened_file.read(file_size - 128)

                            #os.makedirs(os.path.dirname(output_folder), exist_ok=True)
                            #with open(output_folder, 'wb') as dds_file:
                            #    dds_file.write(dds_header)
                            #    dds_file.write(texture_data)

                            
                            # new
                            temp_file = tempfile.NamedTemporaryFile(suffix=".dds", delete=False)
                            temp_file.write(dds_header)
                            temp_file.write(texture_data)
                            temp_file.flush()
                            blender_image = bpy.data.images.load(temp_file.name)
                            blender_image.name = texture_name
                            blender_image.pack()
                            temp_file.close()
                            os.remove(temp_file.name)
                            # new
                            #return bpy.data.images.load(output_folder)
                            return blender_image

            except (OSError, IOError) as e:
                self.report({"ERROR"}, f"Error while opening file at [{file_path}]: {e}")
                traceback.print_exc()
                return None

        return None
    
    @staticmethod
    def apply_texture_to_mesh(mesh_obj, texture_image):
        mat = bpy.data.materials.new(name=mesh_obj.name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]

        tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_image.image = texture_image

        mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_image.outputs["Color"])

        specular_value = mat.node_tree.nodes.new(type="ShaderNodeValue")
        specular_value.outputs[0].default_value = 0.0
        mat.node_tree.links.new(specular_value.outputs[0], bsdf.inputs[13])
        
        mat.node_tree.links.new(bsdf.inputs["Alpha"], tex_image.outputs["Alpha"])

        mat.blend_method = 'CLIP'
        
        mat.show_transparent_back = False
        
        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = mat
        else:
            mesh_obj.data.materials.append(mat)
    
    def execute(self, context):
        return self.import_meshes(context)

    def import_meshes(self, context):
        msg_handler = Utils.MessageHandler(self.debug, self.report)
        
        for file in self.files:
            if file.name.casefold().endswith(".msh"):
                co_conv = Utils.CoordinatesConverter(CoordsSys._3DSMax, CoordsSys.Blender)
                
                def import_msh(file):
                    
                    filepath: str = os.path.join(self.directory, file.name)
                    
                    target_armature: bpy.types.Object = None
                    
                    if self.apply_to_armature_in_selected == True:
                        for obj in bpy.context.selected_objects:
                            if obj.type == "ARMATURE":
                                if target_armature is None:
                                    target_armature = obj
                                else:
                                    msg_handler.report("ERROR", f"More than one armature has been found in the current selection. The imported mesh can only be assigned to one armature at a time.")
                                    return
                        
                    
                    with open(filepath, "rb") as opened_file:
                        try:
                            reader = Utils.Serializer(opened_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                            list_of_bones_used = set()
                            
                            is_mesh08 = False

                            mesh_type = opened_file.read(6).decode("ascii", "ignore")
                            if mesh_type == "MESH08":
                                is_mesh08 = True
                            else:
                                opened_file.seek(0, 0)

                            object_amount = reader.read_ushort()
                            
                            file_base_name = Path(file.name).stem
                            new_collection = None
                            if file_base_name in bpy.data.collections:
                                new_collection = bpy.data.collections[file_base_name]
                            else:
                                new_collection = bpy.data.collections.new(file_base_name)
                                bpy.context.scene.collection.children.link(new_collection)
                            
                            created_objects: list[bpy.types.Object] = []
                            object_parent_names: list[str] = []

                            msg_handler.debug_print(f"Importing object from: {filepath} // Is MESH08 type: {is_mesh08} // Object amount: {object_amount}")
                            bpy.context.window_manager.progress_begin(0, object_amount)
                            
                            # If there is no target armature yet, search for any armature that has all used bones
                            if target_armature is None:
                                for obj in bpy.context.scene.objects:
                                    if obj.type == "ARMATURE":
                                        bone_names = {bone.name for bone in obj.data.bones}
                                        
                                        # Check if the armature has all the bones in list_of_bones_used
                                        if list_of_bones_used.issubset(bone_names):
                                            target_armature = obj
                                            break
                                if target_armature is None:
                                    msg_handler.report("INFO", f"No compatible armature could be found for file at: {filepath}. Information about parenting of objects to bones will be written in custom properties.")
                            else:
                                bone_names = {bone.name for bone in target_armature.data.bones}
                                        
                                # Check if the armature has all the bones in list_of_bones_used
                                if list_of_bones_used.issubset(bone_names) == False:
                                    target_armature = None
                                    msg_handler.report("INFO", f"Selected armature is not compatible with the imported mesh. Information about parenting of objects to bones will be written in custom properties.")
                            
                            if target_armature is not None:
                                bone_names_list = [bone.name for bone in target_armature.data.bones]
                            
                            for object_num in range(object_amount):
                                
                                bpy.context.window_manager.progress_update(object_num)
                                
                                msg_handler.debug_print(f" Processing object number: {object_num}")
                                object_name = reader.read_fixed_string(100, "euc-kr")
                                parent_name = reader.read_fixed_string(100, "euc-kr")
                                
                                
                                
                                msg_handler.debug_print(f"  Object name: {object_name}")
                                msg_handler.debug_print(f"  Object parent name: {parent_name}")
                                
                                object_world_matrix = reader.read_converted_matrix()
                                
                                msg_handler.debug_print(f"  Object converted matrix: {object_world_matrix}")
                                
                                # Skip local and third matrices, as they are unused
                                opened_file.seek(128,1)
                                
                                vertex_amount = reader.read_ushort()
                                triangle_amount = reader.read_ushort()
                                weight_amount = reader.read_ushort()
                                
                                # Force no parent for weighted meshes
                                if weight_amount != 0:
                                    parent_name = SkeletonData.INVALID_NAME
                                
                                force_parent_as_weights = False
                                if target_armature is not None and vertex_amount != 0 and parent_name in bone_names_list and self.preserve_parenting_relationships == False:
                                    force_parent_as_weights = True
                                
                                msg_handler.debug_print(f"  Vertex amount: {vertex_amount}")
                                msg_handler.debug_print(f"  Triangle amount: {triangle_amount}")
                                msg_handler.debug_print(f"  Weight amount: {weight_amount}")
                                
                                texture_path = reader.read_fixed_string(100, "euc-kr")
                                effect_path = reader.read_fixed_string(100, "euc-kr")
                                
                                msg_handler.debug_print(f"  Texture path: {texture_path}")
                                msg_handler.debug_print(f"  Effect texture path: {effect_path}")
                                
                                # I'm not actually sure about this. The values are way too high sometimes for meshes that are quite small
                                bounding_box_max = reader.read_converted_vector3f()
                                bounding_box_min = reader.read_converted_vector3f()
                                
                                unknown_float3_1 = reader.read_vector3f()
                                
                                unknown_flags_1 = reader.read_values("2I", 8)
                                
                                # Only useful for non MESH08 meshes
                                weight_model_type = reader.read_uint()
                                
                                msg_handler.debug_print(f"  Weight model type: {weight_model_type}")
                                
                                unknown_float3_2 = reader.read_vector3f()
                                
                                unknown_float_1 = reader.read_float()
                                
                                opened_file.seek(31, 1)
                                
                                vertices = []
                                normals = []
                                uvs = []
                                weights = []
                                weight_bones = []
                                triangles = []
                                
                                if is_mesh08:
                                    vertex_amount = reader.read_ushort()
                                    
                                    msg_handler.debug_print(f"  MESH08 vertex amount: {vertex_amount}")
                                    
                                    bone_indices = []
                                    for i in range(vertex_amount):
                                        vertices.append(reader.read_converted_vector3f())
                                        
                                        read_weights = reader.read_values("3f", 12)
                                        if weight_amount > 0:
                                            final_weights = list(read_weights)
                                            s = sum(final_weights)
                                            if s < (1.0 - CBB_OT_ImportMSH.WEIGHT_TOLERANCE):
                                                final_weights.append(1.0 - s)
                                            while len(final_weights) < 4:
                                                final_weights.append(0.0)
                                            if sum(final_weights) < 1e-6:
                                                final_weights = [1.0, 0.0, 0.0, 0.0]
                                            weights.append(final_weights)
                                        
                                        bone_indices.append(reader.read_values("4H", 8))
                                        
                                        normals.append(reader.read_converted_vector3f())
                                        uvs.append(reader.read_values("2f", 8))
                                        uvs[i] = (uvs[i][0], -uvs[i][1])
                                        # Binormals, perhaps?
                                        opened_file.seek(12, 1)
                                        
                                    triangle_amount = reader.read_ushort()
                                    
                                    msg_handler.debug_print(f"  MESH08 triangle indices amount: {triangle_amount}")
                                    
                                    for i in range(int(triangle_amount/3)):
                                        triangles.append(reader.read_values("3H", 6))
                                    
                                    bone_group_amount = reader.read_ushort()
                                    unique_bone_names = []
                                    
                                    msg_handler.debug_print(f"  MESH08 bone group amount: {bone_group_amount}")
                                    
                                    for i in range(bone_group_amount):
                                        current_group_bone_amount = reader.read_uint()
                                        bone_names = [reader.read_fixed_string(100, "euc-kr") for i in range(current_group_bone_amount)]
                                        
                                        for bone_name in bone_names:
                                            if bone_name not in unique_bone_names:
                                                unique_bone_names.append(bone_name)
                                        
                                        opened_file.seek((4-current_group_bone_amount)*100, 1)
                                    
                                    msg_handler.debug_print(f"  Successfully read MESH08 data. Organizing bone weights")
                                    
                                    for vertex_index, weight_data in enumerate(weights):
                                        bone_names = []
                                        for weight_count in range(len(weight_data)):
                                            bone_names.append(unique_bone_names[bone_indices[vertex_index][weight_count]])
                                        weight_bones.append(bone_names)
                                    
                                else:
                                    base_vertices = []
                                    base_vertices_normals = []
                                    for _ in range(vertex_amount):
                                        base_vertices.append(reader.read_converted_vector3f())
                                        # What might be this? It's always 1.0
                                        opened_file.seek(4, 1)
                                        base_vertices_normals.append(reader.read_converted_vector3f())
                                        
                                    msg_handler.debug_print(f"  Default mesh successfully read vertice data")
                                    
                                    base_triangles = []
                                    base_triangle_normals = []
                                    base_triangle_uvs = []
                                    for triangle_count in range(triangle_amount):
                                        def __read_uv (file):
                                            impure_uv = reader.read_values("3f", 12)
                                            return Vector((impure_uv[0], impure_uv[1]))
                                        
                                        base_triangles.append(reader.read_values("3I", 12))
                                        base_triangle_normals.append((reader.read_converted_vector3f(), reader.read_converted_vector3f(), reader.read_converted_vector3f()))
                                        base_triangle_uvs.append((__read_uv(opened_file), 
                                                                __read_uv(opened_file), 
                                                                __read_uv(opened_file)))
                                        opened_file.seek(4, 1)
                                    
                                    msg_handler.debug_print(f"  Default mesh successfully read triangle data")
                                    
                                    base_vertices_weights = {}
                                    if weight_model_type == 1:
                                        bone_amount = reader.read_uint()
                                        
                                        bone_names_for_assignment = []
                                        for _ in range(bone_amount):
                                            bone_names_for_assignment.append(reader.read_fixed_string(100, "euc-kr"))
                                        
                                        for _ in range(weight_amount):
                                            vertex_index, amount_of_weights, bone0_index, bone1_index, bone2_index, bone3_index = struct.unpack("<IIiiii", opened_file.read(24))
                                            bone_indices = (bone0_index, bone1_index, bone2_index, bone3_index)
                                            
                                            read_weights = reader.read_values("4f", 16)
                                            bone_names = [
                                                bone_names_for_assignment[bone_indices[i]] if bone_indices[i] != -1 else SkeletonData.INVALID_NAME
                                                for i in range(4)
                                            ]
                                            base_vertices_weights[vertex_index] = (bone_names, read_weights)
                                    else:
                                        for _ in range(weight_amount):
                                            vertex_index, amount_of_weights = struct.unpack("<II", opened_file.read(8))
                                            bone_names = [reader.read_fixed_string(100, "euc-kr") for _ in range(4)]
                                            read_weights = reader.read_values("4f", 16)
                                            base_vertices_weights[vertex_index] = (bone_names, read_weights)
                                        
                                    msg_handler.debug_print(f"  Default mesh successfully read weight data")
                                    
                                    for i, tri in enumerate(base_triangles):
                                        vertices_len = len(vertices)
                                        for n, vertex_index in  enumerate(tri):
                                            vertices.append(base_vertices[vertex_index])
                                            normals.append(base_triangle_normals[i][n])
                                            uvs.append(base_triangle_uvs[i][n])
                                            
                                            if base_vertices_weights:
                                                vertices_weight_data = base_vertices_weights[vertex_index]
                                                bone_names = []
                                                bone_weights = []
                                                for (bone_name, weight) in zip(vertices_weight_data[0], vertices_weight_data[1]):
                                                    if bone_name != SkeletonData.INVALID_NAME:
                                                        bone_names.append(bone_name)
                                                        bone_weights.append(weight)
                                                weights.append(bone_weights)
                                                weight_bones.append(bone_names)
                                                        
                                                
                                        triangles.append((vertices_len, vertices_len+1, vertices_len+2))
                                    
                                    msg_handler.debug_print(f"  Default mesh successfully reconstructed vertices and triangles")
                                    
                                    msg_handler.debug_print(f"  Default mesh type vertex amount: {len(vertices)}")
                                    msg_handler.debug_print(f"  Default mesh type triangle amount: {len(triangles)}")
                                
                                msg_handler.debug_print(f"  Data from file read successfully")
                                
                                if vertex_amount != 0:
                                    mesh = bpy.data.meshes.new(object_name)
                                else: 
                                    mesh = None
                                
                                obj = bpy.data.objects.new(object_name, mesh)
                                
                                obj.matrix_world = object_world_matrix
                                new_collection.objects.link(obj)
                                created_objects.append(obj)
                                
                                msg_handler.debug_print(f"  Object created in Blender")
                                
                                if vertex_amount != 0:
                                    mesh.from_pydata(vertices, [], triangles, False)
                                    mesh.update()
                                
                                msg_handler.debug_print(f"  Mesh Data assigned")
                                
                                if uvs and mesh is not None:
                                    mesh.uv_layers.new(name="UVMap")
                                    uv_layer = mesh.uv_layers.active.data
                                    for poly in mesh.polygons:
                                        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                                            uv = uvs[mesh.loops[loop_index].vertex_index]
                                            uv_layer[loop_index].uv = uv
                                    msg_handler.debug_print(f"  UV data assigned")
                                else:
                                    msg_handler.debug_print(f"  No UV data to assign")
                                
                                
                                if weights and mesh is not None:
                                    vertex_groups = {}
                                    for group in obj.vertex_groups:
                                        vertex_groups[group.name] = group
                                    
                                    for vertex_index, (weight_values, bone_names) in enumerate(zip(weights, weight_bones)):
                                        for weight_value, bone_name in zip(weight_values, bone_names):
                                            list_of_bones_used.add(bone_name)
                                            
                                            if bone_name not in vertex_groups:
                                                vertex_groups[bone_name] = obj.vertex_groups.new(name=bone_name)
                                            group = vertex_groups[bone_name]
                                            
                                            group.add([vertex_index], weight_value, "ADD")
                                    
                                    msg_handler.debug_print(f"  Weight data assigned")
                                else:
                                    msg_handler.debug_print(f"  No weight data to assign")
                                
                                # Set object-bone parenting as direct weight parenting.
                                if force_parent_as_weights == True:
                                    main_group = obj.vertex_groups.new(name=parent_name)
                                    for vertex_index in range(len(vertices)):
                                        main_group.add([vertex_index], 1.0, "ADD")
                                    parent_name = SkeletonData.INVALID_NAME
                                
                                object_parent_names.append(parent_name)
                                
                                if texture_path:
                                    texture_path = ntpath.basename(texture_path)
                                    texture = self.get_texture_as_image(self.directory, texture_path, CBB_OT_ImportMSH.EXTRACTION_FOLDER, 5) 
                                    if texture is not None:
                                        CBB_OT_ImportMSH.apply_texture_to_mesh(obj, texture)
                                        msg_handler.debug_print(f"  Texture data assigned")
                                    else:
                                        msg_handler.report("INFO", f"Could not find texture: {texture_path}")
                                        msg_handler.debug_print(f"  Texture data assignment failed")
                                else:
                                    msg_handler.debug_print(f"  Object has no texture path")
                                    
                                if vertex_amount != 0:
                                    bm = bmesh.new()
                                    bm.from_mesh(mesh)

                                    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)

                                    bm.to_mesh(mesh)
                                    mesh.update()

                                    bm.free()
                            
                            
                            for created_object, parent_name in zip(created_objects, object_parent_names):
                                if target_armature is not None and created_object.type == "MESH":
                                    armature_modifier = created_object.modifiers.new(name="Armature", type="ARMATURE")
                                    armature_modifier.object = target_armature
                                    
                                if parent_name != SkeletonData.INVALID_NAME:
                                    if parent_name in bpy.context.scene.objects:
                                        created_object.parent = bpy.context.scene.objects[parent_name]
                                        created_object.matrix_parent_inverse = bpy.context.scene.objects[parent_name].matrix_world.inverted()
                                        
                                    elif target_armature is not None and parent_name in target_armature.data.bones:
                                        created_object.parent = target_armature
                                        created_object.parent_type = "BONE"
                                        created_object.parent_bone = parent_name
                                        bone: bpy.types.PoseBone = target_armature.pose.bones[parent_name]
                                        
                                        vec = bone.head - bone.tail
                                        trans = Matrix.Translation(vec)
                                        created_object.matrix_parent_inverse = (target_armature.matrix_world @ bone.matrix).inverted_safe() @ trans
                                    else:
                                        created_object["intended_parent_name"] = parent_name
                            
                            bpy.context.view_layer.update()
                                

                        except UnicodeDecodeError as e:
                            msg_handler.report("ERROR", f"Unicode decode error while opening file at [{filepath}]: {e}")
                            traceback.print_exc()
                            return
                        
                        except Exception as e:
                            msg_handler.report("ERROR", f"Unexpected error while opening file at [{filepath}]: {e}")
                            traceback.print_exc()
                            return

                import_msh(file)

        return {"FINISHED"}

    def invoke(self, context: Context, event: Event):
        return self.invoke_popup(context)

class CBB_FH_ImportMSH(bpy.types.FileHandler):
    bl_idname = "CBB_FH_msh_import"
    bl_label = "File handler for msh imports"
    bl_import_operator = CBB_OT_ImportMSH.bl_idname
    bl_file_extensions = CBB_OT_ImportMSH.filename_ext

    @classmethod
    def poll_drop(cls, context):
        return (context.area and context.area.type == "VIEW_3D")

class CBB_OT_ExportMSH(Operator, ExportHelper):
    bl_idname = "cbb.msh_export"
    bl_label = "Export MSH"
    bl_options = {"PRESET"}

    filename_ext = CBB_OT_ImportMSH.filename_ext

    filter_glob: StringProperty(default="*.msh",options={"HIDDEN"}) # type: ignore

    directory: StringProperty(
        name="Directory",
        description="Directory to export files to",
        subtype="DIR_PATH",
        default=""
    ) # type: ignore

    debug: BoolProperty(
        name="Debug export",
        description="Enabling this option will make the exporter print debug data to console",
        default=False
    ) # type: ignore
    
    collection_export_option: bpy.props.EnumProperty(
        name="Collection Type to Export",
        description="Choose what to export",
        items=[
            ("SELECTED", "Selected Objects", "Export only selected objects (exported .msh file will have the name defined in the export window)"),
            ("ACTIVE_COLLECTION", "Active Collection", "Export all objects in the active collection (usually the collection of the currently active object, context sensitive)"),
            ("ALL_COLLECTIONS", "All Collections", "Export all objects in all collections that contain meshes"),
        ],
        default='ALL_COLLECTIONS'
    ) # type: ignore
    
    msh_format_option: bpy.props.EnumProperty(
        name="Mesh Format to Export",
        description="Choose which type to export",
        items=[
            ("STANDARD", "Standard", "Will export .msh file with the standard format"),
            ("MESH08", "MESH 08", "Will export .msh file with the MESH08 format"),
        ],
        default="STANDARD"
    ) # type: ignore

    def execute(self, context):
        return self.export_meshes(context, self.directory)

    def export_meshes(self, context, directory):
        
        msg_handler = Utils.MessageHandler(self.debug, self.report)
        
        objects_for_exportation: list[list[bpy.types.Object], str] = []

        if self.collection_export_option == "SELECTED":
            valid_objects = [
                obj for obj in context.selected_objects 
                if obj.type in {"MESH", "EMPTY"}
            ]
            objects_for_exportation.append((valid_objects, bpy.path.ensure_ext(self.filepath, ".msh")))
        

        elif self.collection_export_option == "ACTIVE_COLLECTION":
            active_collection = bpy.context.view_layer.active_layer_collection.collection
            if active_collection.name.casefold() != "bone shapes":
                valid_objects = [
                    obj for obj in active_collection.objects 
                    if obj.type in {"MESH", "EMPTY"}
                ]
                if valid_objects:
                    objects_for_exportation.append((valid_objects, bpy.path.ensure_ext(os.path.join(self.directory, active_collection.name), ".msh")))

        elif self.collection_export_option == "ALL_COLLECTIONS":
            for collection in bpy.data.collections:
                if collection.name.casefold() != "bone shapes":
                    # Check if the collection has any objects of type MESH or EMPTY directly inside it
                    valid_objects = [
                        obj for obj in collection.objects 
                        if obj.type in {"MESH", "EMPTY"}
                    ]
                    if valid_objects:
                        objects_for_exportation.append((valid_objects, bpy.path.ensure_ext(os.path.join(self.directory, collection.name), ".msh")))

        if not objects_for_exportation:
            if self.collection_export_option == 'SELECTED':
                msg_handler.report("ERROR", "There are no objects of type MESH or EMPTY among currently selected objects. Aborting exportation.")
            elif self.collection_export_option == 'ACTIVE_COLLECTION':
                msg_handler.report("ERROR", f"There are no objects of type MESH or EMPTY in the active collection '{active_collection.name}'. Aborting exportation.")
            elif self.collection_export_option == 'ALL_COLLECTIONS':
                msg_handler.report("ERROR", "There are no objects of type MESH or EMPTY in any of the collections. Aborting exportation.")
            return {"CANCELLED"}
        
        co_conv = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys._3DSMax)
        
        for collection_objects, collection_file_path in objects_for_exportation:
            def __export_mesh(objects: list[bpy.types.Object], export_file_path, mesh_export_format):
                object_amount = 0
                object_names = []
                object_parent_names = []
                object_world_matrices = []
                object_local_matrices = []
                object_inverse_parent_matrices = []
                object_vertex_amounts = []
                object_face_amounts = []
                object_weight_amounts = []
                object_texture_paths = []
                object_effect_paths = []
                object_unique_bones_lists = {}
                object_vertice_data = {}
                object_face_data = {}
                object_weight_data = {}
                
                try:
                    msg_handler.debug_print(f"Exporting objects to file at [{export_file_path}]")
                    
                    def __add_object_data(object_index: int , name: str, parent_name: str, world_matrix: Matrix, local_matrix: Matrix, exporting_vertices, exporting_normals, exporting_uvs, exporting_polygons, exporting_weights, exporting_unique_bones_list, texture_path, effect_path):
                        nonlocal object_amount
                        nonlocal object_names
                        nonlocal object_parent_names
                        nonlocal object_world_matrices
                        nonlocal object_local_matrices
                        nonlocal object_inverse_parent_matrices
                        nonlocal object_vertex_amounts
                        nonlocal object_face_amounts
                        nonlocal object_weight_amounts
                        nonlocal object_texture_paths
                        nonlocal object_effect_paths
                        nonlocal object_unique_bones_lists
                        nonlocal object_vertice_data
                        nonlocal object_face_data
                        nonlocal object_weight_data
                        
                        object_amount += 1
                        object_names.append(name)
                        object_parent_names.append(parent_name)
                        object_world_matrices.append(world_matrix)
                        object_local_matrices.append(local_matrix)
                        object_inverse_parent_matrices.append(Matrix.Identity(4))
                        vertex_data = []
                        face_data = []
                        weight_data = []
                        
                        if mesh_export_format == "MESH08":
                            for vertex_index in range(len(exporting_vertices)):
                                vertex_position = exporting_vertices[vertex_index]
                                weight_values = (0.0, 0.0, 0.0)
                                unique_bone_indices = (0, 0, 0, 0)
                                vertex_normal = exporting_normals[vertex_index]
                                vertex_uv = exporting_uvs[vertex_index]
                                vertex_binormal = (0.0, 0.0, 0.0)

                                # Populate weight values and bone indices (limit to 4)
                                if exporting_weights:
                                    weight_values = exporting_weights[vertex_index][0]
                                    unique_bone_indices = exporting_weights[vertex_index][1]

                                # Append all data as a tuple to the vertex data collection
                                vertex_data.append((vertex_position, weight_values, unique_bone_indices, vertex_normal, vertex_uv, vertex_binormal))

                            # Build face data collection
                            
                            for tri in exporting_polygons:
                                face_data.extend(tri)

                            for i in range(0, len(exporting_unique_bones_list), 4):
                                bone_group = exporting_unique_bones_list[i:i+4]
                                weight_data.append((len(bone_group), bone_group))
                        else:
                            for vertex_index in range(len(exporting_vertices)):
                                vertex_position = exporting_vertices[vertex_index]
                                vertex_normal = exporting_normals[vertex_index]

                                # Append all data as a tuple to the vertex data collection
                                vertex_data.append((vertex_position, 1.0, vertex_normal))
                                valid_bone_amount = 0
                                
                                
                                exporter_weight = exporting_weights[vertex_index]
                                
                                for bone_index in exporter_weight[1]:
                                    if bone_index != -1:
                                        valid_bone_amount += 1
                                current_exporting_weights = exporter_weight[0]
                                
                                weight_sum = current_exporting_weights[0] + current_exporting_weights[1] + current_exporting_weights[2]
                                
                                if weight_sum < 1.0-CBB_OT_ImportMSH.WEIGHT_TOLERANCE:
                                    current_exporting_weights = (current_exporting_weights[0], current_exporting_weights[1], current_exporting_weights[2], 1.0-weight_sum)
                                else:
                                    current_exporting_weights = (current_exporting_weights[0], current_exporting_weights[1], current_exporting_weights[2], 0.0)
                                weight_data.append((vertex_index, valid_bone_amount, exporter_weight[1], current_exporting_weights))
                                    
                                
                            for tri in exporting_polygons:
                                def __get_extended_uv(uv):
                                    return (uv[0], uv[1], 0.0)
                                
                                # Ensure you are accessing the UVs and normals using the indices in tri
                                face_normals_to_export = (exporting_normals[tri[0]], exporting_normals[tri[1]], exporting_normals[tri[2]])
                                face_uvs_to_export = (
                                    __get_extended_uv(exporting_uvs[tri[0]]), 
                                    __get_extended_uv(exporting_uvs[tri[1]]), 
                                    __get_extended_uv(exporting_uvs[tri[2]])
                                )
                                
                                # Append the flattened data to face_data
                                face_data.append((tri, face_normals_to_export, face_uvs_to_export, 0))
                                
                        object_vertex_amounts.append(len(exporting_vertices))
                        object_face_amounts.append(len(exporting_polygons))
                        object_weight_amounts.append(len(exporting_vertices))
                        object_texture_paths.append(texture_path)
                        object_effect_paths.append(effect_path)
                        object_unique_bones_lists[object_index] = exporting_unique_bones_list
                        
                        object_vertice_data[object_index] = vertex_data
                        object_face_data[object_index] = face_data
                        object_weight_data[object_index] = weight_data
                    
                    
                    for object in objects:
                        object_name = object.name
                        object_parent_name = ""
                        
                        exporter_vertices:list[Vector] = []
                        exporter_normals = []
                        exporter_uvs = []
                        exporter_polygons = []
                        exporter_weights = []
                        object_texture_path = ""
                        unique_bones_list: list[str] = []
                        polygon_indices_amount = 0
                        
                        parent_matrix: Matrix = None
                        if object.parent:
                            if object.parent_type == "BONE":
                                object_parent_name = object.parent_bone
                                bone = object.parent.pose.bones[object.parent_bone]
                                # If the parent is a bone, object.parent refers to the armature itself, so we need to transform the bone matrix relative to the armature for the correct world matrix of the bone
                                parent_matrix = object.parent.matrix_basis @ bone.matrix
                            else:
                                object_parent_name= object.parent.name
                                parent_matrix = object.parent.matrix_basis
                        else:
                            parent_matrix = object.matrix_basis
                            object_parent_name = SkeletonData.INVALID_NAME
                        
                        object_world_matrix = object.matrix_basis
                        object_local_matrix = parent_matrix.inverted() @ object.matrix_basis
                        
                        material = None
                        
                        if object.material_slots:
                            material = object.material_slots[0].material
                            
                        albedo_texture: bpy.types.Image = None
                        if material and material.use_nodes:
                            node_tree = material.node_tree
                            for node in node_tree.nodes:
                                if node.type == 'BSDF_PRINCIPLED':
                                    albedo_texture = Utils.find_image_texture_for_input(node, 'Base Color')
                        
                        if albedo_texture:
                            object_texture_path = f"D:\\{albedo_texture.name}"
                        
                        if object.type == "MESH":
                            mesh: bpy.types.Mesh = object.data
                            mesh_vertices = mesh.vertices
                            mesh_polygons = mesh.polygons
                            mesh_loops = mesh.loops
                            mesh_uvs = mesh.uv_layers.active.data if mesh.uv_layers.active else None

                            msg_handler.debug_print(f"Object [{object.name}]'s vertex amount: [{len(mesh_vertices)}]")
                            msg_handler.debug_print(f"Object [{object.name}]'s polygon amount: [{len(mesh_polygons)}]")
                            msg_handler.debug_print(f"Object [{object.name}]'s loop amount: [{len(mesh_loops)}]")
                            if mesh_uvs is not None:
                                msg_handler.debug_print(f"Object [{object.name}]'s uv amount: [{len(mesh_uvs)}]")

                            # Step 1: Get original vertices
                            original_mesh_vertices = [v.co for v in mesh_vertices]

                            # Step 2: Construct collection of vertex index and UVs from loops
                            # loop_data holds a collection of tuples that each contain the index of a vertex in mesh_vertices and raw uv and normal data for the vertex.
                            loop_data = []
                            for poly in mesh_polygons:
                                for loop_index in poly.loop_indices:
                                    loop = mesh_loops[loop_index]
                                    uv = tuple((mesh_uvs[loop.index].uv)) if mesh_uvs else (0.0, 0.0)
                                    normal = tuple(loop.normal)
                                    loop_data.append((loop.vertex_index, uv, normal))

                            # Step 4: Create unique vertices collection
                            # Unique as in this combination of vertex + uv + normal at n index is not repeated anywhere in these collections
                            unique_vertex_indices = []
                            unique_uvs = []
                            unique_normals = []
                            seen_pairs = set()

                            for vertex_index, uv, normal in loop_data:
                                if (vertex_index, uv, normal) not in seen_pairs:
                                    unique_vertex_indices.append(vertex_index)
                                    unique_uvs.append(tuple(uv))
                                    unique_normals.append(tuple(normal))
                                    seen_pairs.add((vertex_index, uv, normal))

                            # Step 5: Build exporter vertex collection
                            
                            mesh_vertex_to_export_vertex_map = {}
                            unique_bones_used = set()
                            unique_bones_used_indices = {}
                            for i, (vertex_index, uv, normal) in enumerate(zip(unique_vertex_indices, unique_uvs, unique_normals)):
                                exporter_vertices.append(original_mesh_vertices[vertex_index])
                                exporter_normals.append(normal)
                                exporter_uvs.append(uv)

                                mesh_vertex_to_export_vertex_map[(vertex_index, uv, normal)] = i

                                # Get bone weights
                                mesh_vertex = mesh_vertices[vertex_index]
                                groups = mesh_vertex.groups
                                
                                bone_weight_pairs = []
                                for group in groups:
                                    bone_name = object.vertex_groups[group.group].name
                                    if bone_name not in unique_bones_used:
                                        unique_bones_used_indices[bone_name] = len(unique_bones_list)
                                        unique_bones_used.add(bone_name)
                                        unique_bones_list.append(bone_name)
                                    
                                    bone_index = unique_bones_used_indices[bone_name]
                                    bone_weight_pairs.append((bone_index, group.weight))
                                
                                # Sort the bone-weight pairs by weight in descending order
                                bone_weight_pairs.sort(key=lambda x: x[1], reverse=True)

                                # Keep the top 4 bones
                                bone_weight_pairs = bone_weight_pairs[:4]

                                # Split the bone indices and weights
                                bone_indices, weights = zip(*bone_weight_pairs) if bone_weight_pairs else ([], [])

                                if len(weights) > 3:
                                    weights = weights[:3]
                                else:
                                    weights = list(weights) + [0.0] * (3 - len(bone_indices))

                                # Ensure exactly 4 bone indices, padding with 0 if necessary
                                bone_indices = list(bone_indices) + [-1] * (4 - len(bone_indices))

                                exporter_weights.append((weights, bone_indices))
                                
                            # Rebuild polygon indices
                            for poly in mesh_polygons:
                                poly_indices = []
                                for loop_index in poly.loop_indices:
                                    loop = mesh_loops[loop_index]
                                    uv = tuple(mesh_uvs[loop.index].uv) if mesh_uvs else (0.0, 0.0)
                                    normal = tuple(loop.normal)
                                    poly_indices.append(mesh_vertex_to_export_vertex_map[(loop.vertex_index, uv, normal)])

                                amount_of_polys = len(poly_indices)
                                
                                if amount_of_polys == 3:
                                    # If the polygon is already a triangle, just add it directly
                                    exporter_polygons.append(poly_indices)
                                elif amount_of_polys == 4:
                                    # If the polygon is a quad, split it into two triangles
                                    exporter_polygons.append([poly_indices[0], poly_indices[1], poly_indices[2]])
                                    exporter_polygons.append([poly_indices[0], poly_indices[2], poly_indices[3]])
                                else:
                                    # Handle polygons with more than 4 vertices (n-gons)
                                    v0 = poly_indices[0]
                                    for i in range(1, amount_of_polys - 1):
                                        exporter_polygons.append([v0, poly_indices[i], poly_indices[i + 1]])
                            
                            # Polygons are already triangulated, final indice amount will be it's length * 3
                            polygon_indices_amount = len(exporter_polygons)*3
                            
                            msg_handler.debug_print(f"Object [{object.name}]'s vertex amount for export: [{len(exporter_vertices)}]")
                            msg_handler.debug_print(f"Object [{object.name}]'s polygon amount for export: [{len(exporter_polygons)}]")
                        
                        if polygon_indices_amount <= 65535:
                            __add_object_data(object_amount, object_name, object_parent_name, object_world_matrix, object_local_matrix, exporter_vertices, exporter_normals, exporter_uvs, exporter_polygons, exporter_weights, unique_bones_list, object_texture_path, "")
                        else:
                            Utils.debug_print("Object has to be split in chunks.")
                            maximum_split_amount = math.ceil(polygon_indices_amount / 65535.0)
                            for object_number in range(0, maximum_split_amount):
                                Utils.debug_print(f"Exporting split number: {object_number}")
                                split_object_name = f"{object_name}_{object_number}"
                                split_exporter_vertices:list[Vector] = []
                                split_exporter_normals = []
                                split_exporter_uvs = []
                                split_exporter_polygons = []
                                split_exporter_weights = []
                                
                                exporter_vertex_to_split_vertex_map = {}
                                for poly_to_split_index in range (21845*object_number, 21845*(object_number+1)):
                                    if poly_to_split_index >= len(exporter_polygons):
                                        break
                                    
                                    poly_indices = []
                                    for vertex_index in exporter_polygons[poly_to_split_index]:
                                        if vertex_index not in exporter_vertex_to_split_vertex_map:
                                            split_exporter_vertices.append(exporter_vertices[vertex_index])
                                            split_exporter_normals.append(exporter_normals[vertex_index])
                                            split_exporter_uvs.append(exporter_uvs[vertex_index])
                                            split_exporter_weights.append(exporter_weights[vertex_index])
                                            exporter_vertex_to_split_vertex_map[vertex_index] = len(split_exporter_vertices)-1
                                        poly_indices.append(exporter_vertex_to_split_vertex_map[vertex_index])
                                    split_exporter_polygons.append([poly_indices[0], poly_indices[1], poly_indices[2]])
                                        
                                
                                __add_object_data(object_amount, split_object_name, object_parent_name, object_world_matrix, object_local_matrix, split_exporter_vertices, split_exporter_normals, split_exporter_uvs, split_exporter_polygons, split_exporter_weights, unique_bones_list, object_texture_path, "")
                            
                            
                        
                except Exception as e:
                    msg_handler.report("ERROR", f"Exception while trying to remap vertices for object [{object.name}]: {e}")
                    traceback.print_exc()
                    return
                try:
                    # Write to file
                    with open(export_file_path, "wb") as file:
                        writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                        try:
                            if mesh_export_format == "MESH08":
                                writer.write_fixed_string(6, "ascii", "MESH08")
                            writer.write_ushort(object_amount)
                            for object_index in range(len(object_names)):
                                writer.write_fixed_string(100, "euc-kr", object_names[object_index])
                                writer.write_fixed_string(100, "euc-kr", object_parent_names[object_index])
                                writer.write_converted_matrix(object_world_matrices[object_index])
                                writer.write_converted_matrix(object_local_matrices[object_index])
                                writer.write_matrix(object_inverse_parent_matrices[object_index])
                                writer.write_ushort(object_vertex_amounts[object_index])
                                writer.write_ushort(object_face_amounts[object_index])
                                if object_parent_names[object_index] == SkeletonData.INVALID_NAME:
                                    writer.write_ushort(object_weight_amounts[object_index])
                                else:
                                    writer.write_ushort(0)
                                writer.write_fixed_string(100, "euc-kr", object_texture_paths[object_index])
                                writer.write_fixed_string(100, "euc-kr", object_effect_paths[object_index])
                                file.write(bytearray(36))
                                writer.write_uint(1)
                                writer.write_uint(256)
                                writer.write_uint(1)
                                file.write(bytearray(47))
                                if mesh_export_format == "MESH08":
                                    writer.write_ushort(object_vertex_amounts[object_index])
                                    for vertex_position, weight_values, unique_bone_indices, vertex_normal, vertex_uv, vertex_binormal in object_vertice_data[object_index]:
                                        writer.write_converted_vector3f(vertex_position)
                                        writer.write_values("3f", weight_values)
                                        if object_parent_names[object_index] == SkeletonData.INVALID_NAME:
                                            writer.write_values("4h", unique_bone_indices)
                                        else:
                                            # For some reason, if a certain object holds no weight data, having the fourth index as not 0 causes the mesh to go invisible.
                                            writer.write_values("4h", (-1, -1, -1, 0))
                                        writer.write_converted_vector3f(Vector(vertex_normal))
                                        writer.write_values("2f", (vertex_uv[0], -vertex_uv[1]))
                                        writer.write_converted_vector3f(Vector(vertex_binormal))
                                    
                                    writer.write_ushort(object_face_amounts[object_index]*3)
                                    for face_index in object_face_data[object_index]:
                                        writer.write_ushort(face_index)
                                    
                                    writer.write_ushort(len(object_weight_data[object_index]))
                                    for bone_amount, bone_names in object_weight_data[object_index]:
                                        writer.write_uint(bone_amount)
                                        for bone_name in bone_names:
                                            writer.write_fixed_string(100, "euc-kr", bone_name)
                                        file.write(bytearray(100*(4 -bone_amount)))
                                else:
                                    for vertex_position, unknown, vertex_normal in object_vertice_data[object_index]:
                                        writer.write_converted_vector3f(vertex_position)
                                        writer.write_float(unknown)
                                        #writer.write_converted_vector3f(Vector(vertex_normal))
                                        writer.write_vector3f(Vector(vertex_normal))
                                    for tri, face_normals_to_export, face_uvs_to_export, unknown in object_face_data[object_index]:
                                        writer.write_values("3I", tri)
                                        for face_normal in face_normals_to_export:
                                            writer.write_converted_vector3f(Vector(face_normal))
                                        for face_uv in face_uvs_to_export:
                                            writer.write_vector3f(face_uv)
                                        writer.write_uint(unknown)
                                    
                                    writer.write_uint(len(object_unique_bones_lists[object_index]))
                                    for unique_bone in object_unique_bones_lists[object_index]:
                                        writer.write_fixed_string(100, "euc-kr", unique_bone)
                                    if object_parent_names[object_index] == SkeletonData.INVALID_NAME:
                                        for vertex_index, valid_bone_amount, export_bone_indices, export_weight_values in object_weight_data[object_index]:
                                            writer.write_uint(vertex_index)
                                            writer.write_uint(valid_bone_amount)
                                            for bone_index in export_bone_indices:
                                                writer.write_int(bone_index)
                                            for weight_value in export_weight_values:
                                                writer.write_float(weight_value)
                                            
                                    
                        except Exception as e:
                            file.close()
                            os.remove(export_file_path)
                            msg_handler.report("ERROR", f"Exception while writing to file at [{export_file_path}]: {e}")
                            traceback.print_exc()
                            return
                except Exception as e:
                    msg_handler.report("ERROR", f"Could not open file for writing at [{export_file_path}]: {e}")
                    traceback.print_exc()
                    return

            __export_mesh(collection_objects, collection_file_path, self.msh_format_option)

        return {"FINISHED"}

def menu_func_import(self, context):
    self.layout.operator(CBB_OT_ImportMSH.bl_idname, text="MSH (.msh)")

def menu_func_export(self, context):
    self.layout.operator(CBB_OT_ExportMSH.bl_idname, text="MSH (.msh)")

def register():
    bpy.utils.register_class(CBB_OT_ImportMSH)
    bpy.utils.register_class(CBB_FH_ImportMSH)
    bpy.utils.register_class(CBB_OT_ExportMSH)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(CBB_OT_ImportMSH)
    bpy.utils.unregister_class(CBB_FH_ImportMSH)
    bpy.utils.unregister_class(CBB_OT_ExportMSH)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_export)

if __name__ == "__main__":
    register()
