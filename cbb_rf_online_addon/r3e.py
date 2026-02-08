import bpy
import struct
import ntpath
import traceback
import mathutils
from PIL import Image
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Context, Event, Operator
from bpy.props import CollectionProperty, StringProperty, BoolProperty, FloatProperty
from mathutils import Vector, Quaternion, Matrix
from .utils import Utils, CoordsSys
import os
from enum import Enum
from .rf_shared import RFShared, SCALE_FACTOR, ReadFaceStruct, MaterialGroup, AnimatedObject, LayerFlag, R3MMaterial
from io import BufferedReader



class ImportR3E(Operator, ImportHelper):
    bl_idname = "cbb.r3e_import"
    bl_label = "Import R3E"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".R3E"

    filter_glob: StringProperty(default="*.R3E", options={"HIDDEN"}) # type: ignore

    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"}
    ) # type: ignore

    directory: StringProperty(subtype="FILE_PATH") # type: ignore

    debug: BoolProperty(
        name="Debug import",
        description="Enabling this option will make the importer print debug data to console.",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.import_r3e_from_files(context)
    
    
    

    def import_r3e_from_files(self, context: bpy.types.Context):
        for import_file in self.files:
            if import_file.name.casefold().endswith(".r3e".casefold()):
                filepath = self.directory + import_file.name
                Utils.debug_print(self, f"Importing r3e from: {filepath}")
                file_stem = os.path.splitext(import_file.name)[0]
                
                r3m_materials = RFShared.get_materials_from_r3m_file(self.directory, file_stem)
                
                texture_dictionary = RFShared.get_color_texture_dictionary_from_r3t_file(self.directory, file_stem)

                
                r3e_filepath = os.path.join(self.directory, import_file.name)
                
                
                
                with open(r3e_filepath, 'rb') as r3e_file:
                    ImportR3E.import_r3e_entity_from_opened_file(r3e_file, r3m_materials, texture_dictionary, file_stem, context)

        return {"FINISHED"}
    
    @staticmethod
    def import_r3e_entity_from_opened_file(r3e_file: BufferedReader, r3m_materials: list[R3MMaterial], texture_dictionary, entity_name, context: bpy.types.Context, collection = None) -> bpy.types.Object:
        co_conv_unity_blender = Utils.CoordinatesConverter(CoordsSys.Unity, CoordsSys.Blender)
        co_conv_blender_unity = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys.Unity)
        reader = Utils.Serializer(r3e_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv_unity_blender)
                    
        version = reader.read_uint()
        identity = reader.read_uint()
        
        if version != 113:
            print(f"Warning: R3E file version [{version}] is different than the version [113] this addon was built in mind with.")
        
        # Read the header
        header_data = reader.read_values("20I", 80)
        
        CompHeader_offset, CompHeader_size = header_data[0], header_data[1]
        Vertex_offset, Vertex_size = header_data[2], header_data[3]
        VColor_offset, VColor_size = header_data[4], header_data[5]
        UV_offset, UV_size = header_data[6], header_data[7]
        Face_offset, Face_size = header_data[8], header_data[9]
        FaceId_offset, FaceId_size = header_data[10], header_data[11]
        VertexId_offset, VertexId_size = header_data[12], header_data[13]
        MatGroup_offset, MatGroup_size = header_data[14], header_data[15]
        Object_offset, Object_size = header_data[16], header_data[17]
        Track_offset, Track_size = header_data[18], header_data[19]

        vector_data_type = reader.read_ushort()
        r3e_file.seek(12, 1)
        r3e_pos = reader.read_vector3f()
        r3e_scale = reader.read_float()
        r3e_uv_min = reader.read_float()
        r3e_uv_max = reader.read_float()
        
        r3e_uv_scale = (r3e_uv_max-r3e_uv_min)/2.0
        r3e_uv_pos = r3e_uv_min+r3e_uv_scale
        
        read_vertices = []
        
        if vector_data_type == 0x8000:
            read_vertices = [co_conv_unity_blender.convert_vector3f(RFShared.convert_vector3c_to_f(reader.read_values("3b", 3), r3e_scale, r3e_pos)) * SCALE_FACTOR for _ in range(Vertex_size // 3)]
        elif vector_data_type == 0x4000:
            read_vertices = [co_conv_unity_blender.convert_vector3f(RFShared.convert_vector3s_to_f(reader.read_values("3h", 6), r3e_scale, r3e_pos)) * SCALE_FACTOR for _ in range(Vertex_size // 6)]
        else:
            read_vertices = [reader.read_converted_vector3f() * SCALE_FACTOR for _ in range(Vertex_size // 12)]
        
        vertices_colors = [reader.read_uint() for _ in range(VColor_size // 4)]
        
        uvs = [Vector(((reader.read_short()/32767.0*r3e_uv_scale)+r3e_uv_pos, 1.0-((reader.read_short()/32767.0*r3e_uv_scale)+r3e_uv_pos))) for _ in range(UV_size // 4)]
        
        entity_faces = [ReadFaceStruct(reader.read_ushort(), reader.read_uint()) for _ in range(Face_size // 6)]
        
        faces_ids = [reader.read_ushort() for _ in range(FaceId_size // 2)]
        
        vertices_ids = [reader.read_ushort() for _ in range(VertexId_size // 2)]
        
        material_groups = [MaterialGroup.r3e_material_from_unpacked_bytes(reader.read_values('H I h H 3h 3h', 22)) for _ in range(MatGroup_size // 22)]
        
        animated_objects: list[AnimatedObject] = []
        for _ in range(Object_size // 88):
            animated_object = AnimatedObject()
            animated_object.flag = reader.read_ushort()
            animated_object.parent = reader.read_ushort()
            animated_object.frames = reader.read_int()
            animated_object.pos_count = reader.read_int()
            animated_object.rot_count = reader.read_int()
            animated_object.scale_count = reader.read_int()
            animated_object.scale = reader.read_vector3f()
            animated_object.scale_rot = reader.read_quaternion()
            
            pos = reader.read_vector3f()*SCALE_FACTOR
            rot = reader.read_quaternion()
            
            #pos = Vector((-pos[0], -pos[1], pos[2]))
                
            animated_object.pos = pos
            animated_object.quat = rot
            animated_object.pos_offset = reader.read_uint()
            animated_object.rot_offset = reader.read_uint()
            animated_object.scale_offset = reader.read_uint()
            animated_objects.append(animated_object)
        
        tracks = r3e_file.read(Track_size)
        
        static_mesh = bpy.data.meshes.new(name=f"Entity_{entity_name}Mesh_")
        static_object = bpy.data.objects.new(name=f"Entity_{entity_name}", object_data=static_mesh)
        if collection is None:
            bpy.context.collection.objects.link(static_object)
        else:
            collection.objects.link(static_object)
        
        all_vertices = []
        all_faces = []
        all_uvs = []
        material_indices = []  # To store material index for each face
        vertex_index_map = {}  # To map original vertex indices to new indices
        
        created_objects = []  # Include static object in created objects list
        
        
        # Create Entity Action (One per entity)
        entity_action = None
        if any(animated_object.frames > 0 for animated_object in animated_objects):
            entity_action = bpy.data.actions.new(name=f"Anim_{entity_name}")
            entity_action.use_fake_user = True

        for material_group in material_groups:
            if material_group.material_id != -1:
                material_id: int = material_group.material_id
                material_name = r3m_materials[material_id].name
                
                material = bpy.data.materials.new(material_name)
                material.use_nodes = True
                nodes = material.node_tree.nodes
                links = material.node_tree.links
                bsdf = nodes.get('Principled BSDF')
                
                RFShared.process_texture_layers(r3m_materials[material_id], material, nodes, links, bsdf, texture_dictionary, context, f"Entity_{entity_name}Mesh_")
                    
                if material_group.animated_object_id == 0:
                    # Add material to static object
                    static_object.data.materials.append(material)
                    current_material_index = len(static_object.data.materials) - 1
                    
                    # Process vertices and faces for static mesh
                    face_start_id = material_group.starting_face_id
                    
                    for i in range(material_group.number_of_faces):
                        face_struct = entity_faces[faces_ids[face_start_id + i]]
                        vertex_start_id = face_struct.vertex_start_id
                        vertex_amount = face_struct.vertex_amount
                        
                        face_vertices = []
                        for j in range(vertex_amount):
                            vertex_index = vertices_ids[vertex_start_id + j]
                            new_vertex_index = len(all_vertices)
                            vertex_index_map[vertex_start_id + j] = new_vertex_index
                            
                            all_vertices.append(read_vertices[vertex_index])
                            all_uvs.append(uvs[vertex_start_id + j])
                            face_vertices.append(new_vertex_index)
                        
                        all_faces.append(face_vertices)
                        material_indices.append(current_material_index)
                else:
                    # Create separate object for animated parts
                    animated_mesh = bpy.data.meshes.new(name=f"Mesh_{material_name}")
                    animated_obj = bpy.data.objects.new(name=material_name, object_data=animated_mesh)
                    if collection is None:
                        bpy.context.collection.objects.link(animated_obj)
                    else:
                        collection.objects.link(animated_obj)
                    created_objects.append(animated_obj)
                    
                    vertices = []
                    faces = []
                    animated_uvs = []
                    vertex_index_to_read_index = []
                    
                    face_start_id = material_group.starting_face_id
                    for i in range(material_group.number_of_faces):
                        face_struct = entity_faces[faces_ids[face_start_id + i]]
                        vertex_start_id = face_struct.vertex_start_id
                        vertex_amount = face_struct.vertex_amount
                        
                        face_vertices = []
                        for j in range(vertex_amount):
                            vertex_index = vertices_ids[vertex_start_id + j]
                            vertex_index_to_read_index.append(vertex_start_id + j)
                            face_vertices.append(len(vertices))
                            
                            three_ds_max_vertex = co_conv_blender_unity.convert_vector3f(read_vertices[vertex_index])
                            vertices.append(three_ds_max_vertex)
                            animated_uvs.append(uvs[vertex_start_id + j])
                        
                        faces.append(face_vertices)
                    
                    animated_mesh.from_pydata(vertices, [], faces)
                    animated_obj.data.materials.append(material)
                    
                    if not animated_mesh.uv_layers:
                        animated_mesh.uv_layers.new(name="UVMap")
                    
                    uv_layer = animated_mesh.uv_layers.active.data
                    for poly in animated_mesh.polygons:
                        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                            vertex_index = vertex_index_to_read_index[animated_mesh.loops[loop_index].vertex_index]
                            uv_layer[loop_index].uv = uvs[vertex_index]
                    
                    # Handle animation
                    animated_object = animated_objects[material_group.animated_object_id-1]
                    if animated_object.parent != 0:
                        animated_obj.parent = created_objects[animated_object.parent-1]
                    animated_obj.location = animated_object.pos
                    animated_obj.rotation_mode = "QUATERNION"
                    animated_obj.rotation_quaternion = animated_object.quat
                    animated_obj.scale = animated_object.scale
                    
                    if animated_object.frames > 0 and entity_action:
                        if animated_obj.animation_data is None:
                            animated_obj.animation_data_create()
                        animated_obj.animation_data.action = entity_action

                        # Add keyframes for position
                        for i in range(animated_object.pos_count):
                            offset = animated_object.pos_offset + i * 16
                            frame, pos_x, pos_y, pos_z = struct.unpack_from("<f3f", tracks, offset)
                            
                            converted_position = Vector((pos_x, pos_y, pos_z))
                            animated_obj.location = converted_position * SCALE_FACTOR
                            animated_obj.keyframe_insert(data_path="location", frame=frame)

                        # Add keyframes for rotation
                        for i in range(animated_object.rot_count):
                            offset = animated_object.rot_offset + i * 20
                            frame, rot_x, rot_y, rot_z, rot_w = struct.unpack_from("<f4f", tracks, offset)
                            
                            converted_rotation = Quaternion((rot_w, rot_x, rot_y, -rot_z))
                            animated_obj.rotation_quaternion = converted_rotation
                            animated_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

                        # Add keyframes for scale
                        for i in range(animated_object.scale_count):
                            offset = animated_object.scale_offset + i * 32
                            frame, scale_x, scale_y, scale_z, scale_axis_x, scale_axis_y, scale_axis_z, scale_axis_w = struct.unpack_from("<f3f4f", tracks, offset)
                            scale_quat = Quaternion((scale_axis_w, scale_axis_x, scale_axis_y, scale_axis_z))
                            scale_vec = Vector((scale_x, scale_y, scale_z))
                            scale_vec.rotate(scale_quat)
                            
                            animated_obj.scale = scale_vec
                            animated_obj.keyframe_insert(data_path="scale", frame=frame)

                    if animated_obj.parent is None:
                        animated_obj.parent = static_object
        
        # Create the final static mesh
        if all_vertices:
            static_mesh.from_pydata(all_vertices, [], all_faces)
            
            # Assign material indices
            for poly, mat_idx in zip(static_mesh.polygons, material_indices):
                poly.material_index = mat_idx
            
            # Setup UV mapping for static mesh
            if not static_mesh.uv_layers:
                static_mesh.uv_layers.new(name="UVMap")
            
            uv_layer = static_mesh.uv_layers.active.data
            for poly in static_mesh.polygons:
                for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    vertex_index = static_mesh.loops[loop_index].vertex_index
                    uv_layer[loop_index].uv = all_uvs[vertex_index]
                    
        created_objects.append(static_object)
        # Final cleanup
        bpy.ops.object.select_all(action='DESELECT')
        for obj in created_objects:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.object.mode_set(mode='OBJECT')
        for obj in created_objects:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        
        bpy.ops.object.shade_smooth()
        bpy.ops.object.select_all(action='DESELECT')
        
        return static_object
    
    

    def add_uv_warp_modifier(obj):
        if not obj.modifiers.get('UVWarp'):
            uv_warp = obj.modifiers.new(name='UVWarp', type='UV_WARP')
            return uv_warp
        return obj.modifiers['UVWarp']
   
    
    
    def invoke(self, context: Context, event: Event):
        return self.invoke_popup(context)

class CBB_FH_ImportR3E(bpy.types.FileHandler):
    bl_idname = "CBB_FH_r3e_import"
    bl_label = "File handler for r3e imports"
    bl_import_operator = ImportR3E.bl_idname
    bl_file_extensions = ImportR3E.filename_ext

    @classmethod
    def poll_drop(cls, context):
        return (context.area and context.area.type == "VIEW_3D")


def menu_func_import(self, context):
    self.layout.operator(ImportR3E.bl_idname, text="R3E (.R3E)")

#def menu_func_export(self, context):
   # self.layout.operator(ExportSkeleton.bl_idname, text="Skeleton (.Skeleton)")

def register():
    bpy.utils.register_class(ImportR3E)
    bpy.utils.register_class(CBB_FH_ImportR3E)
    #bpy.utils.register_class(ExportSkeleton)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    #bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ImportR3E)
    bpy.utils.unregister_class(CBB_FH_ImportR3E)
    #bpy.utils.unregister_class(ExportSkeleton)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    #bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
