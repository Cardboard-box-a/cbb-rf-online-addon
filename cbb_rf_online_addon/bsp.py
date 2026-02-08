import bpy
import struct
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Context, Event, Operator
from bpy.props import CollectionProperty, StringProperty, BoolProperty
from mathutils import Vector, Quaternion, Matrix, Euler
import mathutils
from .utils import Utils, CoordsSys, Vector3Int
import os
from .rf_shared import RFShared, SCALE_FACTOR, ReadFaceStruct, MaterialGroup, AnimatedObject, LayerFlag, ReadEntityStruct, EntityStruct, EntityRPKIndices, R3MMaterial, MaterialProperties, TextureLayer, BlendMethod
from .r3e import ImportR3E
from math import radians
import traceback
from collections import namedtuple, deque
import sys
from typing import List, Dict, Set, Tuple
import math
from dataclasses import dataclass
import sys
from . import texture_utils
import numpy as np
import bmesh
import datetime
import time

class BSPNode:
    def __init__(self, plane_id: int, distance: float, front_id: int, back_id: int, bounding_box_min: Vector, bounding_box_max: Vector):
        self.plane_id = plane_id
        self.distance = distance
        self.front_id = front_id
        self.back_id = back_id
        self.bounding_box_min = bounding_box_min
        self.bounding_box_max = bounding_box_max

class BSPLeaf:
    def __init__(self, type: int, face_amount: int, face_start_id: int, material_group_amount: int, material_group_start_id: int, bounding_box_min: Vector, bounding_box_max: Vector):
        self.type = type
        self.face_amount = face_amount
        self.face_start_id = face_start_id
        self.material_group_amount = material_group_amount
        self.material_group_start_id = material_group_start_id
        self.bounding_box_min = bounding_box_min
        self.bounding_box_max = bounding_box_max

BSP_MAP_COLLECTION_NAME = "BSP_MAP"
class ImportBSP(Operator, ImportHelper):
    bl_idname = "cbb.bsp_import"
    bl_label = "Import BSP"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".BSP"

    filter_glob: StringProperty(default="*.BSP", options={"HIDDEN"}) # type: ignore

    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"}
    ) # type: ignore

    directory: StringProperty(subtype="FILE_PATH") # type: ignore

    debug: BoolProperty(
        name="Debug import",
        description="Enabling this option will make the importer print debug data to console",
        default=False
    ) # type: ignore
    
    visualize_bsp_data: BoolProperty(
        name="Visualize BSP Data (Slow)",
        description="Outputs info from the imported BSP in the console and also creates the BSP structure as boxes in the scene. It will make the scene very slow to render",
        default=False
    ) # type: ignore
    
    import_and_show_light_maps: BoolProperty(
        name="Import And Show Light Maps",
        description="This option will import light maps and show them in the imported material groups",
        default=False
    ) # type: ignore
    
    import_spt_entities: BoolProperty(
        name="Import SPT Entities",
        description="This option, if true, will import SPT entities along common R3E ones. SPT entities are incomplete, however, and will not work as expected.",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.import_bsp_from_files(context)

    def import_bsp_from_files(self: "ImportBSP", context: bpy.types.Context):
        for import_file in self.files:
            if import_file.name.casefold().endswith(".bsp".casefold()):
                filepath = self.directory + import_file.name
                Utils.debug_print(self, f"Importing bsp from: {filepath}")
                
                file_stem = os.path.splitext(import_file.name)[0]
                
                r3m_materials = RFShared.get_materials_from_r3m_file(self.directory, file_stem)
                
                color_texture_dictionary = RFShared.get_color_texture_dictionary_from_r3t_file(self.directory, file_stem)
                
                light_texture_dictionary = RFShared.get_light_texture_dictionary_from_r3t_file(self.directory, file_stem)
                
                bsp_filepath = os.path.join(self.directory, import_file.name)
                
                co_conv_unity_blender = Utils.CoordinatesConverter(CoordsSys.Unity, CoordsSys.Blender)
                co_conv_blender_unity = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys.Unity)
                
                with open(bsp_filepath, 'rb') as ebp_file:
                    reader = Utils.Serializer(ebp_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv_unity_blender)
                    
                    version = reader.read_uint()
                    
                    if version != 39:
                        print(f"Warning: BSP file version [{version}] is different than the version [39] this addon was built in mind with.")
                    
                    
                    header_format = "170I"
                    header_data = reader.read_values(header_format, 680)
                    
                    CPlanes_offset, CPlanes_size = header_data[0], header_data[1]
                    CFaceId_offset, CFaceId_size = header_data[2], header_data[3]
                    Node_offset, Node_size = header_data[4], header_data[5]
                    Leaf_offset, Leaf_size = header_data[6], header_data[7]
                    MatListInLeaf_offset, MatListInLeaf_size = header_data[8], header_data[9]
                    Object_offset, Object_size = header_data[10], header_data[11]
                    Track_offset, Track_size = header_data[12], header_data[13]
                    EventObjectID_offset, EventObjectID_size = header_data[14], header_data[15]

                    FVertex_offset, FVertex_size = header_data[90], header_data[91]
                    VertexColor_offset, VertexColor_size = header_data[92], header_data[93]
                    UV_offset, UV_size = header_data[94], header_data[95]
                    LgtUV_offset, LgtUV_size = header_data[96], header_data[97]
                    Face_offset, Face_size = header_data[98], header_data[99]
                    FaceId_offset, FaceId_size = header_data[100], header_data[101]
                    VertexId_offset, VertexId_size = header_data[102], header_data[103]
                    ReadMatGroup_offset, ReadMatGroup_size = header_data[104], header_data[105]

                    ebp_file.seek(CPlanes_offset)
                    collision_planes = [reader.read_vector3f() for _ in range(CPlanes_size // 12)]

                    collision_face_ids = [reader.read_uint() for _ in range(CFaceId_size // 4)]

                    #bsp_nodes = [reader.read_values("I f h h 3h 3h", 24) for _ in range(Node_size // 24)]

                    #bsp_leafs = [reader.read_values('B H I H I 3h 3h', 25) for _ in range(Leaf_size // 25)]

                    bsp_nodes = [
                        {
                            'f_normal_id': values[0],
                            'd': values[1],
                            'front': values[2],
                            'back': values[3],
                            'bb_min': (values[4], values[5], values[6]),
                            'bb_max': (values[7], values[8], values[9]),
                        }
                        for values in (reader.read_values("I f h h 3h 3h", 24) for _ in range(Node_size // 24))
                    ]

                    bsp_leafs = [
                        {
                            'type': values[0],
                            'face_num': values[1],
                            'face_start_id': values[2],
                            'm_group_num': values[3],
                            'm_group_start_id': values[4],
                            'bb_min': (values[5], values[6], values[7]),
                            'bb_max': (values[8], values[9], values[10]),
                        }
                        for values in (reader.read_values("B H I H I 3h 3h", 25) for _ in range(Leaf_size // 25))
                    ]
                                        
                    mat_list_in_leaf = [reader.read_ushort() for _ in range(MatListInLeaf_size // 2)]

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
                            
                        animated_object.pos = pos
                        animated_object.quat = rot
                        animated_object.pos_offset = reader.read_uint()
                        animated_object.rot_offset = reader.read_uint()
                        animated_object.scale_offset = reader.read_uint()
                        animated_objects.append(animated_object)

                    tracks = ebp_file.read(Track_size)

                    event_object_ids = [reader.read_ushort() for _ in range(EventObjectID_size // 2)]

                    ebp_file.seek(FVertex_offset)
                    read_vertices = [reader.read_converted_vector3f() * SCALE_FACTOR for _ in range(FVertex_size // 12)]

                    bsp_vertex_colors = [reader.read_uint() for _ in range(VertexColor_size // 4)]

                    uvs = [Vector((reader.read_float(), 1.0-reader.read_float())) for _ in range(UV_size // 8)]

                    bsp_light_uvs = [Vector(((reader.read_short()/32767.0), 1.0-(reader.read_short()/32767.0))) for _ in range(LgtUV_size // 4)]
                    
                    face_pointers = [ReadFaceStruct(reader.read_ushort(), reader.read_uint()) for _ in range(Face_size // 6)]
                    
                    faces_ids = [reader.read_uint() for _ in range(FaceId_size // 4)]

                    vertices_ids = [reader.read_uint() for _ in range(VertexId_size // 4)]
                    
                    material_groups = [MaterialGroup.bsp_material_from_unpacked_bytes(reader.read_values('H H I h h 3h 3h 3f f H', 42)) for _ in range(ReadMatGroup_size // 42)]
                    
                    def create_combined_leaf_object(leaf, collision_face_ids, face_pointers, vertices_ids, read_vertices, name):
                        """Create a combined object for a BSP leaf containing its bounding box and mesh."""
                        # Create leaf bounding box
                        bb_min, bb_max = leaf['bb_min'], leaf['bb_max']
                        verts = [
                            (bb_min[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),
                            (bb_min[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),
                            (bb_min[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR),
                            (bb_min[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR),
                            (bb_max[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),
                            (bb_max[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),
                            (bb_max[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR),
                            (bb_max[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR)
                        ]
                        edges = [
                            (0, 1), (1, 3), (3, 2), (2, 0),
                            (4, 5), (5, 7), (7, 6), (6, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]

                        # Create the mesh for the leaf
                        faces = []
                        mesh_verts = []
                        vert_map = {}
                        for i in range(leaf['face_num']):
                            face_id = collision_face_ids[leaf['face_start_id'] + i]
                            face_data = face_pointers[face_id]
                            face_vertices = []
                            for j in range(face_data.vertex_amount):
                                vert_id = vertices_ids[face_data.vertex_start_id + j]
                                if vert_id not in vert_map:
                                    vert_map[vert_id] = len(mesh_verts)
                                    mesh_verts.append(read_vertices[vert_id])
                                face_vertices.append(vert_map[vert_id])
                            faces.append(face_vertices)

                        # Combine bounding box and mesh
                        combined_verts = verts + [(v[0] * SCALE_FACTOR, v[1] * SCALE_FACTOR, v[2] * SCALE_FACTOR) for v in mesh_verts]
                        combined_faces = [[i for i in range(len(verts))]] + [[len(verts) + vi for vi in f] for f in faces]

                        # Create the Blender object
                        mesh = bpy.data.meshes.new(f"{name}_combined_mesh")
                        obj = bpy.data.objects.new(f"{name}_combined", mesh)
                        bpy.context.scene.collection.children['Leaf BSPs'].objects.link(obj)
                        mesh.from_pydata(combined_verts, edges, combined_faces)
                        return obj

                    leaf_collection = bpy.data.collections.new("BSP_LEAVES")
                    bpy.context.scene.collection.children.link(leaf_collection)
                    
                    node_collection = bpy.data.collections.new("BSP_NODES")
                    bpy.context.scene.collection.children.link(node_collection)
                    
                    def create_bbox(bb_min, bb_max, name, collection):
                        """Create a bounding box."""
                        verts = [
                            (bb_min[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),  # Swapped Y and Z
                            (bb_min[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR),  # Swapped Y and Z
                            (bb_min[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),  # Swapped Y and Z
                            (bb_min[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR),  # Swapped Y and Z
                            (bb_max[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),  # Swapped Y and Z
                            (bb_max[0] * SCALE_FACTOR, bb_min[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR),  # Swapped Y and Z
                            (bb_max[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_min[1] * SCALE_FACTOR),  # Swapped Y and Z
                            (bb_max[0] * SCALE_FACTOR, bb_max[2] * SCALE_FACTOR, bb_max[1] * SCALE_FACTOR)   # Swapped Y and Z
                        ]
                        edges = [
                            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
                            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
                            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
                        ]
                        mesh = bpy.data.meshes.new(f"{name}_bbox")
                        obj = bpy.data.objects.new(f"{name}_bbox", mesh)
                        collection.objects.link(obj)
                        mesh.from_pydata(verts, edges, [])
                        return obj

                    def create_leaf_mesh(leaf, collision_face_ids, face_pointers: list[ReadFaceStruct], vertices_ids, read_vertices, collection):
                        """Create a mesh for a BSP leaf."""
                        faces = []
                        verts = []
                        vert_map = {}  # Map original vertices to new indices
                        
                        # Process faces
                        for i in range(leaf['face_num']):
                            face_id = collision_face_ids[leaf['face_start_id'] + i]
                            face_data = face_pointers[face_id]
                            face_vertices = []
                            
                            for j in range(face_data.vertex_amount):
                                vert_id = vertices_ids[face_data.vertex_start_id + j]
                                if vert_id not in vert_map:
                                    vert_map[vert_id] = len(verts)
                                    verts.append(read_vertices[vert_id])
                                face_vertices.append(vert_map[vert_id])
                            face_vertices.reverse()
                            faces.append(face_vertices)
                        
                        # Create the mesh
                        mesh = bpy.data.meshes.new(f"Leaf_{leaf['face_start_id']}_mesh")
                        obj = bpy.data.objects.new(f"Leaf_{leaf['face_start_id']}_mesh", mesh)
                        collection.objects.link(obj)
                        mesh.from_pydata(verts, [], faces)
                        return obj

                    def visualize_bsp(bsp_nodes, bsp_leafs, collision_face_ids, face_pointers, vertices_ids, read_vertices):
                        """Main function to visualize BSP nodes and leaves."""
                        total_bsp_nodes = len(bsp_nodes)
                        for i, node in enumerate(bsp_nodes):
                            sys.stdout.write(f"\rProcessing bsp_node {i}/{total_bsp_nodes}...")
                            sys.stdout.flush()
                            create_bbox(node['bb_min'], node['bb_max'], f"Node_{i}", node_collection)
                        
                        total_bsp_leaves = len(bsp_leafs)
                        for i, leaf in enumerate(bsp_leafs):
                            sys.stdout.write(f"\rProcessing bsp_leaf {i}/{total_bsp_leaves}...")
                            sys.stdout.flush()
                            bbox_obj = create_bbox(leaf['bb_min'], leaf['bb_max'], f"Leaf_{i}", leaf_collection)
                            leaf_mesh = create_leaf_mesh(leaf, collision_face_ids, face_pointers, vertices_ids, read_vertices, leaf_collection)
                            # Optionally parent the mesh to the bounding box for better organization
                            leaf_mesh.parent = bbox_obj

                    # Run visualization
                    if self.visualize_bsp_data:
                        visualize_bsp(bsp_nodes, bsp_leafs, collision_face_ids, face_pointers, vertices_ids, read_vertices)
                    
                    def print_property_extremes(bsp_nodes, bsp_leafs):
                        def find_extremes(collection, name):
                            # Find all keys in the dictionaries
                            keys = collection[0].keys()
                            print(f"\n{name}:")
                            for key in keys:
                                # Extract values for the current key
                                values = [item[key] for item in collection]
                                if isinstance(values[0], tuple):  # Handle tuple properties like bb_min and bb_max
                                    min_value = tuple(min(sub[key] for sub in values) for key in range(len(values[0])))
                                    max_value = tuple(max(sub[key] for sub in values) for key in range(len(values[0])))
                                    print(f"  {key} -> Min: {min_value}, Max: {max_value}")
                                else:  # Handle scalar properties
                                    min_value = min(values)
                                    max_value = max(values)
                                    print(f"  {key} -> Min: {min_value}, Max: {max_value}")

                        if bsp_nodes:
                            find_extremes(bsp_nodes, "BSP Nodes")

                        if bsp_leafs:
                            find_extremes(bsp_leafs, "BSP Leafs")


                    # Example usage:
                    print_property_extremes(bsp_nodes, bsp_leafs)
                    
                    def process_bsp_leafs(bsp_leafs, mat_list_in_leaf, collision_face_ids, material_groups):
                        for leaf_idx, leaf in enumerate(bsp_leafs):
                            # Step 2: Check if the leaf type is not 0
                            if leaf['type'] != 0:
                                print(f"Leaf at ID {leaf_idx} has type {leaf['type']}")

                            # Step 3: Get all the group ids of the current leaf
                            group_ids = mat_list_in_leaf[leaf['m_group_start_id']:leaf['m_group_start_id'] + leaf['m_group_num']]

                            # Step 4: Get all the face ids of the current leaf
                            face_ids = collision_face_ids[leaf['face_start_id']:leaf['face_start_id'] + leaf['face_num']]

                            # Step 5: Sort the face ids and separate them by material group
                            sorted_face_ids = sorted(face_ids)
                            material_group_ids_from_faces = set()

                            for face_id in sorted_face_ids:
                                for b, mat_group in enumerate(material_groups):
                                    if mat_group.starting_face_id <= face_id < mat_group.starting_face_id + mat_group.number_of_faces:
                                        material_group_ids_from_faces.add(b)
                                        break

                            # Step 6: Print the leaf index, material group ids from faces, and group ids from mat_list_in_leaf
                            print(f"Leaf Index: {leaf_idx} || Face_amount[{leaf['face_num']}] || Face_start_id[{leaf['face_start_id'] }] || Group_num[{leaf['m_group_num']}] || group_start_id[{leaf['m_group_start_id']}]")
                            print(f"Material Group IDs from Faces: {material_group_ids_from_faces}")
                            print(f"Group IDs from mat_list_in_leaf: {group_ids}")
                            print(f"Face IDs from collision_face_ids: {face_ids}")
                            print("-" * 40)

                    if self.visualize_bsp_data:
                        process_bsp_leafs(bsp_leafs, mat_list_in_leaf, collision_face_ids, material_groups)
                    
                    created_objects = []
                    
                    entity_collection = bpy.data.collections.new("BSP_ENTITIES")
                    bpy.context.scene.collection.children.link(entity_collection)
                    
                    entity_template_collection = bpy.data.collections.new("BSP_ENTITIES_TEMPLATES")
                    bpy.context.scene.collection.children.link(entity_template_collection)
                    
                    map_collection = bpy.data.collections.new(BSP_MAP_COLLECTION_NAME)
                    bpy.context.scene.collection.children.link(map_collection)
                    
                    for progress, material_group in enumerate(material_groups, start=1):
                        try: 
                            if material_group.material_id != -1: 
                                material_id: int = material_group.material_id

                                material_name = r3m_materials[material_id].name
                                mesh = bpy.data.meshes.new(name=f"Mesh_{material_name}")
                                obj = bpy.data.objects.new(name=f"{progress-1}-Object_{material_name}", object_data=mesh)
                                map_collection.objects.link(obj)
                                created_objects.append(obj)
                                

                                print(f"\r[{progress}/{len(material_groups)}] {(progress/len(material_groups)*100.0):.2f}% Processing object: {material_name} | ", end=f"")
                                
                                vertices = []
                                faces_indices = []

                                vertex_index_to_read_index = []

                                face_start_id = material_group.starting_face_id
                                for i in range(material_group.number_of_faces):
                                    face_struct = face_pointers[faces_ids[face_start_id + i]]
                                    vertex_start_id = face_struct.vertex_start_id
                                    vertex_amount = face_struct.vertex_amount

                                    
                                    for j in range(vertex_amount - 2):
                                        face_vertices = []
                                        
                                        vertex_index_0 = vertices_ids[vertex_start_id + 0]
                                        vertex_index_to_read_index.append(vertex_start_id + 0)
                                        face_vertices.append(len(vertices))
                                        if material_group.animated_object_id != 0:
                                            three_ds_max_vertex_0 = co_conv_blender_unity.convert_vector3f(read_vertices[vertex_index_0])
                                            vertices.append(three_ds_max_vertex_0)
                                        else:
                                            vertices.append(read_vertices[vertex_index_0])
                                        
                                        vertex_index_1 = vertices_ids[vertex_start_id + j + 1]
                                        vertex_index_to_read_index.append(vertex_start_id + j + 1)
                                        face_vertices.append(len(vertices))
                                        if material_group.animated_object_id != 0:
                                            three_ds_max_vertex_1 = co_conv_blender_unity.convert_vector3f(read_vertices[vertex_index_1])
                                            vertices.append(three_ds_max_vertex_1)
                                        else:
                                            vertices.append(read_vertices[vertex_index_1])
                                        
                                        vertex_index_2 = vertices_ids[vertex_start_id + j + 2]
                                        vertex_index_to_read_index.append(vertex_start_id + j + 2)
                                        face_vertices.append(len(vertices))
                                        if material_group.animated_object_id != 0:
                                            three_ds_max_vertex_2 = co_conv_blender_unity.convert_vector3f(read_vertices[vertex_index_2])
                                            vertices.append(three_ds_max_vertex_2)
                                        else:
                                            vertices.append(read_vertices[vertex_index_2])
                                                                            
                                        face_vertices.reverse()
                                        faces_indices.append(face_vertices)
                                       

                                mesh.from_pydata(vertices, [], faces_indices)
                               
                                # While some material groups do indeed share r3m materials, every one of them can have a different light texture, so we cannot share materials inside Blender. They should be cumpled together, if possible, later when exporting, as we can merge r3m materials that have no difference.
                                #if material_name not in bpy.data.materials:
                                material = bpy.data.materials.new(material_name)
                                material.use_nodes = True
                                nodes = material.node_tree.nodes
                                links = material.node_tree.links
                                bsdf = nodes.get('Principled BSDF')
                                obj.data.materials.append(material)
                                    
                                RFShared.process_texture_layers(r3m_materials[material_id], material, nodes, links, bsdf, color_texture_dictionary, context)
                                
                                
                                if not mesh.uv_layers:
                                    mesh.uv_layers.new(name="UVMap")
                                    
                                if self.import_and_show_light_maps == True:
                                    if not mesh.uv_layers.get("LightUVMap"):
                                        light_uv_layer = mesh.uv_layers.new(name="LightUVMap")
                                    else:
                                        light_uv_layer = mesh.uv_layers["LightUVMap"]
                                    uv_light_layer = mesh.uv_layers.get("LightUVMap").data
                                    light_uv_layer.active = True
                                
                                uv_layer = mesh.uv_layers.get("UVMap").data
                                
                                for poly in mesh.polygons:
                                    for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                                        vertex_index = vertex_index_to_read_index[mesh.loops[loop_index].vertex_index]
                                        uv_layer[loop_index].uv = uvs[vertex_index]
                                        if self.import_and_show_light_maps == True:
                                            uv_light_layer[loop_index].uv = bsp_light_uvs[vertex_index]
                                        
                                if self.import_and_show_light_maps == True:
                                    if material_group.light_id != -1:
                                        uv_map_node = nodes.new(type='ShaderNodeUVMap')
                                        uv_map_node.location = (-400, 200)
                                        uv_map_node.uv_map = "LightUVMap"

                                        image_texture_node = nodes.new(type='ShaderNodeTexImage')
                                        image_texture_node.location = (-200, 200)
                                        image_texture_node.image = bpy.data.images.get(light_texture_dictionary[material_group.light_id])
                                        if not image_texture_node.image:
                                            raise ValueError(f"Image for light ID {material_group.light_id} not found in Blender.")
                                        image_texture_node.image.colorspace_settings.name = 'Non-Color'
                                        
                                        links.new(uv_map_node.outputs["UV"], image_texture_node.inputs["Vector"])
                                        
                                        current_base_color_input = None
                                        for link in links:
                                            if link.to_node == bsdf and link.to_socket.name == "Base Color":
                                                current_base_color_input = link.from_node
                                                links.remove(link)  # Remove the existing connection
                                                break

                                        if not current_base_color_input:
                                            raise ValueError("No input found for BSDF's Base Color.")

                                        vector_math_node = nodes.new(type="ShaderNodeVectorMath")
                                        vector_math_node.operation = "MULTIPLY"
                                        vector_math_node.inputs[1].default_value = (2.0, 2.0, 2.0)
                                        links.new(image_texture_node.outputs["Color"], vector_math_node.inputs[0])
                                        
                                        mix_node = nodes.new(type='ShaderNodeMix')
                                        mix_node.data_type = 'RGBA'
                                        mix_node.location = (0, 200)
                                        mix_node.blend_type = 'MULTIPLY'  # You can change this to 'ADD', 'MULTIPLY', etc., as needed.
                                        mix_node.inputs['Factor'].default_value = 1.0  # Set to full influence
                                        
                                        links.new(current_base_color_input.outputs[0], mix_node.inputs['A'])
                                        
                                        links.new(vector_math_node.outputs['Vector'], mix_node.inputs['B'])
                                        
                                        links.new(mix_node.outputs["Result"], bsdf.inputs['Base Color'])
                                
                                
                                
                                if material_group.animated_object_id != 0:
                                    animated_object: AnimatedObject = animated_objects[material_group.animated_object_id-1]
                                    if animated_object.parent != 0:
                                        obj.parent = created_objects[animated_object.parent-1]
                                    obj.location = animated_object.pos
                                    obj.rotation_mode = "QUATERNION"
                                    obj.rotation_quaternion = animated_object.quat
                                    obj.scale = animated_object.scale
                                    if animated_object.frames > 0:
                                        if obj.animation_data is None:
                                            obj.animation_data_create()
                                        action = bpy.data.actions.new(name=f"Anim_BSP_{material_name}")
                                        obj.animation_data.action = action

                                        for i in range(animated_object.pos_count):
                                            offset = animated_object.pos_offset + i * 16
                                            frame, pos_x, pos_y, pos_z = struct.unpack_from("<f3f", tracks, offset)
                                            #converted_position = co_conv_3ds_blend.convert_vector3f(Vector((pos_x, pos_y, pos_z)))
                                            converted_position = Vector((pos_x, pos_y, pos_z))
                                            
                                            obj.location = converted_position * SCALE_FACTOR
                                            obj.keyframe_insert(data_path="location", frame=frame)

                                        for i in range(animated_object.rot_count):
                                            offset = animated_object.rot_offset + i * 20
                                            
                                            frame, rot_x, rot_y, rot_z, rot_w = struct.unpack_from("<f4f", tracks, offset)
                                            
                                            converted_rotation = Quaternion((rot_w, rot_x, rot_y, -rot_z))
                                            obj.rotation_quaternion = converted_rotation
                                            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

                                        for i in range(animated_object.scale_count):
                                            offset = animated_object.scale_offset + i * 32
                                            frame, scale_x, scale_y, scale_z, scale_axis_x, scale_axis_y, scale_axis_z, scale_axis_w = struct.unpack_from("<f3f4f", tracks, offset)
                                            scale_quat = Quaternion((scale_axis_w, scale_axis_x, scale_axis_y, scale_axis_z))
                                            scale_vec = Vector((scale_x, scale_y, scale_z))
                                            scale_vec.rotate(scale_quat)
                                            
                                            obj.scale = scale_vec
                                            obj.keyframe_insert(data_path="scale", frame=frame)
                                
                        except Exception as e:
                            print(f"Exception while reading material number [{progress-1}]: {e}")
                            traceback.print_exc()
                            
                    
                    print()
                
                print("BSP file read!")
                
                ebp_filepath = os.path.join(self.directory, f"{file_stem}.ebp")
                
                with open(ebp_filepath, 'rb') as ebp_file:
                    reader = Utils.Serializer(ebp_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv_unity_blender)

                    version = reader.read_uint()
                    
                    if version != 20:
                        print(f"Warning: EBP file version [{version}] is different than the version [20] this addon was built in mind with.")
                    
                    header_data = reader.read_values("18I", 72)
                    
                    CFVertex_offset, CFVertex_size = header_data[0], header_data[1]
                    CFLine_offset, CFLine_size = header_data[2], header_data[3]
                    CFLineId_offset, CFLineId_size = header_data[4], header_data[5]
                    CFLeaf_offset, CFLeaf_size = header_data[6], header_data[7]
                    EntityList_offset, EntityList_size = header_data[8], header_data[9]
                    EntityID_offset, EntityID_size = header_data[10], header_data[11]
                    LeafEntityList_offset, LeafEntityList_size = header_data[12], header_data[13]
                    SoundEntityID_offset, SoundEntityID_size = header_data[14], header_data[15]
                    LeafSoundEntityList_offset, LeafSoundEntityList_size = header_data[16], header_data[17]
                    ebp_file.seek(144, 1)
                    
                    header_data = reader.read_values("6I", 24)
                    
                    CFLine = namedtuple('CFLine', ['attr', 'start_v', 'end_v', 'height', 'front', 'back'])
                    CFLeaf = namedtuple('CFLeaf', ['start_id', 'line_num'])
                    
                    # Seek to CFVertex data and read vertices
                    ebp_file.seek(CFVertex_offset)
                    num_vertices = CFVertex_size // 12  # Each vertex is 12 bytes (3 floats)
                    vertices = [reader.read_converted_vector3f()*SCALE_FACTOR for _ in range(num_vertices)]
                    print(f"Number of vertices: {len(vertices)}")

                    # Seek to CFLine data and read CFLine structs
                    ebp_file.seek(CFLine_offset)
                    num_cflines = CFLine_size // 16  # Each CFLine is 16 bytes (4 + 2 + 2 + 4 + 2 + 2)
                    cflines = [CFLine(
                        reader.read_uint(),       # attr
                        reader.read_ushort(),    # start_v
                        reader.read_ushort(),    # end_v
                        reader.read_float(),     # height
                        reader.read_ushort(),    # front
                        reader.read_ushort()     # back
                    ) for _ in range(num_cflines)]
                    print(f"Number of CFLines: {len(cflines)}")

                    if cflines:
                        # Calculate maximum values for CFLine variables
                        max_attr = max(line.attr for line in cflines)
                        max_start_v = max(line.start_v for line in cflines)
                        max_end_v = max(line.end_v for line in cflines)
                        max_height = max(line.height for line in cflines)
                        max_front = max(line.front for line in cflines)
                        max_back = max(line.back for line in cflines)
                        print(f"Max CFLine values - attr: {max_attr}, start_v: {max_start_v}, end_v: {max_end_v}, height: {max_height}, front: {max_front}, back: {max_back}")

                    # Seek to CFLineId data and read CFLineIds (ushorts)
                    ebp_file.seek(CFLineId_offset)
                    num_cflineids = CFLineId_size // 2  # Each CFLineId is 2 bytes (ushort)
                    cflineids = [reader.read_ushort() for _ in range(num_cflineids)]
                    print(f"Number of CFLineIds: {len(cflineids)}")

                    # Calculate maximum value for CFLineId
                    if cflineids:
                        max_cflineid = max(cflineids)
                        print(f"Max CFLineId value: {max_cflineid}")

                    # Seek to CFLeaf data and read CFLeaf structs
                    ebp_file.seek(CFLeaf_offset)
                    num_cfleaves = CFLeaf_size // 6  # Each CFLeaf is 6 bytes (4 + 2)
                    cfleaves = [CFLeaf(
                        reader.read_uint(),       # start_id
                        reader.read_ushort()      # line_num
                    ) for _ in range(num_cfleaves)]
                    print(f"Number of CFLeaves: {len(cfleaves)}")

                    # Calculate maximum values for CFLeaf variables
                    if cfleaves:
                        max_start_id = max(leaf.start_id for leaf in cfleaves)
                        max_line_num = max(leaf.line_num for leaf in cfleaves)
                        print(f"Max CFLeaf values - start_id: {max_start_id}, line_num: {max_line_num}")
                    
                    mesh = bpy.data.meshes.new("000_COLLISION_Mesh")
                    collision_object = bpy.data.objects.new("000_COLLISION", mesh)
                    bpy.context.collection.objects.link(collision_object)

                    # Initialize lists for mesh data
                    mesh_vertices = []
                    mesh_faces = []

                    # Loop through each CFLine and create faces
                    for line in cflines:
                        # Get the downward vertices
                        start_vertex = Vector(vertices[line.start_v])
                        end_vertex = Vector(vertices[line.end_v])
                        
                        # Calculate the upward vertices
                        start_vertex_up = start_vertex + Vector((0, 0, line.height*SCALE_FACTOR))
                        end_vertex_up = end_vertex + Vector((0, 0, line.height*SCALE_FACTOR))
                        
                        # Add the vertices to the mesh list
                        start_index = len(mesh_vertices)  # Current count of vertices as the start index
                        mesh_vertices.extend([
                            start_vertex,
                            end_vertex,
                            end_vertex_up,
                            start_vertex_up
                        ])
                        
                        # Define the face using indices of the vertices
                        # The face is defined in a clockwise or counter-clockwise order
                        mesh_faces.append([
                            start_index,         # Bottom-left
                            start_index + 1,     # Bottom-right
                            start_index + 2,     # Top-right
                            start_index + 3      # Top-left
                        ])

                    # Create the mesh
                    if mesh_vertices and mesh_faces:
                        mesh.from_pydata(mesh_vertices, [], mesh_faces)
                        mesh.update()

                        print(f"Created mesh '000_COLLISION' with {len(mesh_vertices)} vertices and {len(mesh_faces)} faces.")
                    
                    #------------------------------------------------------------------------------------------------
                    
                    MapEntitiesList_offset, MapEntitiesList_size = header_data[0], header_data[1]
                    SoundEntityList_offset, SoundEntityList_size = header_data[2], header_data[3]
                    SoundEntitiesList_offset, SoundEntitiesList_size = header_data[4], header_data[5]
                    
                    ebp_file.seek(EntityList_offset)
                    
                    # The factors here are used only when shader_id is either 1 or 2. It's used for grass rendering, and the factors are frequency and amplitude of grass movement.
                    entities_list = [EntityStruct(reader.read_ubyte(), reader.read_ubyte(), reader.read_fixed_string(62, "euc-kr").casefold(), reader.read_float(), reader.read_float(), reader.read_ushort(), reader.read_ushort(), reader.read_values("2f", 8)) for _ in range(EntityList_size // 84)]
                    
                    
                    entities_file_paths = [read_entity.file_path for read_entity in entities_list]
                    
                    
                    parent_directory = os.path.dirname(self.directory)
                    
                    while parent_directory != "":
                        parent_directory = os.path.dirname(parent_directory)
                        if os.path.basename(parent_directory).casefold() == "map":
                            break

                    rpk_filepaths = []
                    
                    
                    entity_directory = os.path.join(parent_directory, "entity")
                    
                    if os.path.isdir(entity_directory):
                        rpk_filepaths = [
                            os.path.join(entity_directory, file)
                            for file in os.listdir(entity_directory)
                            if file.casefold().endswith(".rpk")
                        ]
                    
                    instantiated_entities = [None]*len(entities_list)
                    
                    entity_database = {}
                    
                    for rpk_filepath in rpk_filepaths:
                        with open(rpk_filepath, 'rb') as rpk_file:
                            rpk_reader = Utils.Serializer(rpk_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor)
                            version = rpk_reader.read_float()
                            if version != 1.0:
                                print(f"Warning: RPK file version [{version}] is different than the version [1.0] this addon was built in mind with.")
                            
                            rpk_file_amount = rpk_reader.read_uint()
                            rpk_file_offset_indices = [rpk_reader.read_uint() for _ in range(rpk_file_amount)]
                            rpk_file_offsets = [rpk_reader.read_int() for _ in range(rpk_file_amount)]
                            
                            
                            entries_name = []
                            entries_file_size = []
                            entries_file_amount = []
                            entries_offset_indices_index = []
                            
                            for i in range (rpk_file_amount):
                                entries_name.append(rpk_reader.read_fixed_string(52, "euc-kr").lstrip(f"."))
                                entries_file_size.append(rpk_reader.read_int())
                                # Skip name length, not used
                                rpk_file.seek(2, 1)
                                entries_file_amount.append(rpk_reader.read_ushort())
                                entries_offset_indices_index.append(rpk_reader.read_uint())
                            
                            rpk_start_offset = rpk_file.tell()
                            
                            file_structure_stack = []
                            for i in range (rpk_file_amount):
                                entry_name = entries_name[i]
                                #print(f"Entry: {entry_name}")
                                entry_file_size = entries_file_size[i]
                                entry_file_amount = entries_file_amount[i]
                                
                                # Folder
                                if os.path.splitext(entry_name)[1] == "": 
                                    file_structure_stack.append((entry_name, entry_file_amount))
                                else:
                                    # File
                                    
                                    entity_data = {}
                                    entity_data["rpk_file_path"] = rpk_filepath
                                    paths = [t[0] for t in file_structure_stack]
                                    # rpk_start_offset here is the offset in the RPK after having read all RPK header data.
                                    entity_data["entity_offset_in_rpk"] = rpk_file_offsets[i]+rpk_start_offset
                                    if entry_file_size >= 0:
                                        entity_data["entity_size_in_rpk"] = entry_file_size
                                    else:
                                        next_valid_offset = 0
                                        for next_file_index in range(i+1, rpk_file_amount, 1):
                                            next_entry_name = entries_name[next_file_index]
                                            if os.path.splitext(next_entry_name)[1] == "" and entries_file_size[next_file_index] > 0:
                                                next_valid_offset = entries_file_size[next_file_index]
                                                break
                                                
                                        entity_data["entity_size_in_rpk"] = entry_file_size+next_valid_offset
                                    
                                    entity_relative_path = os.path.join(f"\\".join(paths), (f"\\{entry_name}")).casefold().replace(f"\\\\", f"\\")
                                    #print(f"Entity Relative Path: {entity_relative_path}")
                                    entity_database[entity_relative_path] = entity_data
                                    file_structure_stack[-1] = (file_structure_stack[-1][0], file_structure_stack[-1][1]-1)
                                    

                                if file_structure_stack[-1][1] <= 0:
                                    file_structure_stack.pop()
                                    
                                    while True:
                                        if len(file_structure_stack) > 0:
                                            file_structure_stack[-1] = (file_structure_stack[-1][0], file_structure_stack[-1][1]-1)
                                            if file_structure_stack[-1][1] <= 0:
                                                file_structure_stack.pop()
                                            else: 
                                                break
                                        else: 
                                            break
                                    
                                    
                    
                    
                    
                    for i, entity_file_path in enumerate(entities_file_paths):
                        spt_data = None
                        spt_lines = None
                        r3e_path = entity_file_path
                        entity_name = os.path.basename(entity_file_path)
                        if entity_file_path.endswith(".spt"):
                            if self.import_spt_entities == False:
                                continue
                            
                            entity_data = entity_database.get(entity_file_path)
                            if entity_data is not None:
                                with open(entity_data["rpk_file_path"], 'rb') as rpk_file:
                                    rpk_file.seek(entity_data["entity_offset_in_rpk"])
                                    spt_data = rpk_file.read(entity_data["entity_size_in_rpk"]).decode(encoding="euc-kr")
                                    spt_lines = spt_data.splitlines()
                                    for line in spt_lines:
                                        #print(f"Line in spt_data for entity {entity_name}: {line}")
                                        line = line.strip()
                                        split_line = line.split()
                                        if len(split_line) > 0 and split_line[0] == "entity_file":
                                            #print(f"Found raw path to r3e entity for SPT: {split_line[1]}")
                                            r3e_path = split_line[1].casefold().removeprefix(f".\\map\\entity")
                        
                        r3m_path = f"{os.path.splitext(r3e_path)[0]}.r3m"
                        r3t_path = f"{os.path.splitext(r3e_path)[0]}.r3t"
                        
                        
                        
                        r3m_materials = None
                        texture_dictionary = None
                        
                        r3m_data = entity_database.get(r3m_path, None)
                        if r3m_data is not None:
                            with open(r3m_data["rpk_file_path"], 'rb') as r3m_rpk_file:
                                r3m_rpk_file.seek(r3m_data["entity_offset_in_rpk"])
                                r3m_materials = RFShared.get_materials_from_r3m_filestream(r3m_rpk_file)
                        else:
                            print(f"r3m_data not found for path {r3m_path}")
                        
                        r3t_data = entity_database.get(r3t_path, None)
                        if r3t_data is not None:
                            with open(r3t_data["rpk_file_path"], 'rb') as r3t_rpk_file:
                                r3t_rpk_file.seek(r3t_data["entity_offset_in_rpk"])
                                texture_dictionary = RFShared.get_color_texture_dictionary_from_r3t_filestream(r3t_rpk_file)
                        else:
                            print(f"r3t_data not found for path {r3t_path}")
                        
                        r3e_data = entity_database.get(r3e_path, None)
                        if r3e_data is not None and r3m_data is not None and r3t_data is not None:
                            with open(r3e_data["rpk_file_path"], 'rb') as r3e_rpk_file:
                                r3e_rpk_file.seek(r3e_data["entity_offset_in_rpk"])
                                instantiated_entities[i] = ImportR3E.import_r3e_entity_from_opened_file(r3e_rpk_file, r3m_materials, texture_dictionary, entity_name, context, entity_template_collection)
                        else:
                            print(f"Not all necessary data was found for r3e entity: {r3e_path}")
                    
                    

                    
                    
                    
                    ebp_file.seek(MapEntitiesList_offset)
                    
                    read_entities = [ReadEntityStruct(reader.read_ushort(), reader.read_float(), reader.read_converted_vector3f()*SCALE_FACTOR, reader.read_float(), reader.read_float(), Vector((float(reader.read_short()), float(reader.read_short()), float(reader.read_short()))), Vector((float(reader.read_short()), float(reader.read_short()), float(reader.read_short())))) for _ in range(MapEntitiesList_size // 38)]
                    
                    for progress, read_entity in enumerate(read_entities, start=1):
                        print(f"\rProcessing entities. [{progress}/{len(read_entities)}] {(progress/len(read_entities)*100.0):.2f}%", end="")
                        try:
                            original_entity = instantiated_entities[read_entity.id]
                            if original_entity is not None:
                                new_entity = ImportBSP.duplicate_object_with_children(original_entity, entity_collection)
                                
                                new_entity.scale = (read_entity.scale, read_entity.scale, read_entity.scale)
                                new_entity.location = read_entity.position
                                new_entity.rotation_mode = "QUATERNION"
                                euler_rotation = Euler((radians(read_entity.rot_x), radians(read_entity.rot_y), 0.0), 'XYZ')
                                quaternion_rotation = euler_rotation.to_quaternion()
                                quaternion_rotation = co_conv_unity_blender.convert_quaternion(quaternion_rotation)
                                new_entity.rotation_quaternion = quaternion_rotation
                            
                        except IndexError as e:
                            print(f"Index [{read_entity.id}] caused: {e}")

                    print()
                    
                
                
                bpy.ops.object.select_all(action='DESELECT')
                for obj in created_objects:
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.remove_doubles(threshold=0.0001)  # Adjust threshold as needed
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.shade_smooth()
                bpy.ops.object.select_all(action='DESELECT')

        return {"FINISHED"}

    def add_uv_warp_modifier(obj):
        if not obj.modifiers.get('UVWarp'):
            uv_warp = obj.modifiers.new(name='UVWarp', type='UV_WARP')
            return uv_warp
        return obj.modifiers['UVWarp']

    def invoke(self, context: Context, event: Event):
        return self.invoke_popup(context)

    @staticmethod
    def duplicate_object_with_children(obj, collection)-> bpy.types.Object:
        """
        Duplicate an object and all its children recursively, ensuring data is copied as well.
        Links the duplicated objects to the specified collection.
        
        Parameters:
            obj: bpy.types.Object - The object to duplicate.
            collection: bpy.types.Collection - The collection to link the duplicated objects to.
        
        Returns:
            bpy.types.Object - The root object of the duplicated hierarchy.
        """
        # Create a copy of the object
        new_obj = obj.copy()
        new_obj.data = obj.data.copy() if obj.data else None
        new_obj.animation_data_clear()  # Clear animation data references
        if obj.animation_data and obj.animation_data.action:
            new_obj.animation_data_create()
            new_obj.animation_data.action = obj.animation_data.action.copy()
        
        # Link the new object to the collection
        collection.objects.link(new_obj)
        
        # Recursively copy children
        for child in obj.children:
            new_child = ImportBSP.duplicate_object_with_children(child, collection)
            new_child.parent = new_obj  # Set the parent of the new child to the duplicated object
        
        return new_obj
    
class CBB_FH_ImportBSP(bpy.types.FileHandler):
    bl_idname = "CBB_FH_bsp_import"
    bl_label = "File handler for bsp imports"
    bl_import_operator = ImportBSP.bl_idname
    bl_file_extensions = ImportBSP.filename_ext

    @classmethod
    def poll_drop(cls, context):
        return (context.area and context.area.type == "VIEW_3D")

class ExportBSP(Operator, ExportHelper):
    bl_idname = "cbb.bsp_export"
    bl_label = "Export BSP"
    bl_options = {"PRESET"}

    filename_ext = ImportBSP.filename_ext

    filter_glob: StringProperty(default="*.BSP",options={"HIDDEN"}) # type: ignore

    directory: StringProperty(
        name="Directory",
        description="Directory to export files to",
        subtype="DIR_PATH",
        default=""
    ) # type: ignore
    
    filepath: StringProperty(subtype="FILE_PATH") # type: ignore

    debug: BoolProperty(
        name="Debug export",
        description="Enabling this option will make the exporter print debug data to console",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.export_bsp(context, self.directory)
    
    def export_bsp(self, context, directory):
        collection = bpy.data.collections.get(BSP_MAP_COLLECTION_NAME)

        if not collection:
            print(f"Collection [{BSP_MAP_COLLECTION_NAME}] not found.")
            return "CANCELLED"
        
        objects_in_collection = collection.objects
        
        result = self.new_bake_lightmaps(objects_in_collection)
        
        if result:
            objects_data, lightmap_images = result
        else:
            return {"FINISHED"}
        
                
        bsp_vertex_data: dict[Vector, tuple[int, int]] = {} #fvertices, key: position, value: (id, color)
        bsp_uvs: list[Vector] = []
        bsp_light_uvs: list[Vector] = []
        bsp_vertex_ids: list[int] = []
        bsp_face_pointers: list[ReadFaceStruct] = []
        bsp_face_ids: list[int] = []
        
        material_groups: list[MaterialGroup] = []
        
        r3m_materials: dict[str, tuple[R3MMaterial, int]] = {} # R3M file
        
        texture_dictionary: dict[str, int] = {} # R3T file
        
        
        for (light_texture_index, blender_material), material_data in objects_data.items():
            
            material_vertices: list[Vector] = material_data["vertices"]
            material_uvs: list[tuple[float, float]] = material_data["uvs"]
            material_light_uvs: list[tuple[float, float]] = material_data["light_uvs"]
            material_polygons: list[list[int]] = material_data["polygons"]
            
            
            number_of_faces = len(material_polygons)
            starting_face_id = len(bsp_face_pointers)
            material_id = ExportBSP.process_blender_material(blender_material, r3m_materials, texture_dictionary)
            bb_pos_data = ExportBSP.calculate_bounding_box_and_middle(material_vertices)
            bounding_box_min = bb_pos_data[0]
            bounding_box_max = bb_pos_data[1]
            material_position = bb_pos_data[2]
            
            material_group = MaterialGroup()
            material_group.number_of_faces = number_of_faces
            material_group.starting_face_id = starting_face_id
            material_group.material_id = material_id
            material_group.light_id = light_texture_index
            material_group.bounding_box_min = bounding_box_min
            material_group.bounding_box_max = bounding_box_max
            material_group.position = material_position
            
            material_groups.append(material_group)
            
            for polygon in material_polygons:
                bsp_face_ids.append(len(bsp_face_ids))
                bsp_face_pointers.append(ReadFaceStruct(len(polygon), len(bsp_vertex_ids), len(material_groups)-1))
                for vertex_id in polygon:
                    polygon_vertex = material_vertices[vertex_id]
                    polygon_vertex_tuple = tuple(polygon_vertex)
                    polygon_vertex_uv = material_uvs[vertex_id]
                    if polygon_vertex_tuple not in bsp_vertex_data:
                        bsp_vertex_data[polygon_vertex_tuple] = (len(bsp_vertex_data), 0xFFFFFFFF)
                    polygon_vertex_id = bsp_vertex_data[polygon_vertex_tuple][0]
                    bsp_vertex_ids.append(polygon_vertex_id)
                    bsp_uvs.append(polygon_vertex_uv)
                    bsp_light_uvs.append(material_light_uvs[vertex_id])
        
        
        bsp_nodes, bsp_leaves, collision_face_ids, material_list_in_leaf_ids, splitting_planes = ExportBSP.create_bsp_structure(bsp_vertex_data, bsp_vertex_ids, bsp_face_pointers)
        
        file_base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        
        r3t_filepath = os.path.join(self.directory, f"{file_base_name}.r3t")
        r3t_light_filepath = os.path.join(self.directory, f"{file_base_name}Lgt.r3t")
        r3m_filepath = os.path.join(self.directory, f"{file_base_name}.r3m")
        bsp_filepath = os.path.join(self.directory, f"{file_base_name}.bsp")
        ebp_filepath = os.path.join(self.directory, f"{file_base_name}.ebp")
        
        co_conv = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys.Unity)
        
        # Writing R3T
        try:
            with open(r3t_filepath, "wb") as file:
                writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, 
                                        Utils.Serializer.Quaternion_Order.XYZW, 
                                        Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                
                writer.write_float(1.2)  # Version
                writer.write_uint(len(texture_dictionary))
                
                for texture_name in texture_dictionary:
                    dds_name = os.path.splitext(texture_name)[0] + ".dds"
                    writer.write_fixed_string(128, "euc-kr", f".\\{dds_name}")
                
                for i, texture_name in enumerate(texture_dictionary, start=1):
                    sys.stdout.write(f"\rConverting textures {i}/{len(texture_dictionary)}...")
                    sys.stdout.flush()
                    image = bpy.data.images.get(texture_name)
                    if not image:
                        raise texture_utils.TextureProcessingError(f"Image not found: {texture_name}")
                    
                    dds_data = texture_utils.convert_to_dds(image)
                    if not dds_data:
                        raise texture_utils.TextureProcessingError(
                            f"Failed to convert {texture_name} to DDS format")
                    
                    writer.write_uint(len(dds_data))
                    file.write(dds_data)
                
        except texture_utils.TextureProcessingError as e:
            self.report({"ERROR"}, str(e))
            print(e)
            if os.path.exists(r3t_filepath):
                os.remove(r3t_filepath)
            
        except Exception as e:
            self.report({"ERROR"}, f"Unexpected error writing R3T file: {str(e)}")
            print(e)
            if os.path.exists(r3t_filepath):
                os.remove(r3t_filepath)
        
        # Writing R3T_light
        try:
            with open(r3t_light_filepath, "wb") as file:
                writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, 
                                        Utils.Serializer.Quaternion_Order.XYZW, 
                                        Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                
                writer.write_float(1.1)  # Version
                writer.write_uint(len(lightmap_images))
                
                for i, texture_image in enumerate(lightmap_images, start=1):
                    sys.stdout.write(f"\rConverting light textures {i}/{len(lightmap_images)}...")
                    sys.stdout.flush()
                    
                    dds_data = texture_utils.convert_to_dds_with_format(texture_image, texture_utils.D3DFormat.R5G6B5)
                    if not dds_data:
                        raise texture_utils.TextureProcessingError(
                            f"Failed to convert {texture_name} to DDS format")
                    
                    writer.write_uint(len(dds_data))
                    file.write(dds_data)
                    
                    bpy.data.images.remove(texture_image)
                
        except texture_utils.TextureProcessingError as e:
            self.report({"ERROR"}, str(e))
            print(e)
            if os.path.exists(r3t_light_filepath):
                os.remove(r3t_light_filepath)
            
        except Exception as e:
            self.report({"ERROR"}, f"Unexpected error writing R3T light file: {str(e)}")
            print(e)
            if os.path.exists(r3t_light_filepath):
                os.remove(r3t_light_filepath)
        
        # Writing R3M
        try:
            with open(r3m_filepath, "wb") as file:
                writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                try:
                    writer.write_float(1.1)
                    writer.write_uint(len(r3m_materials))
                    for material_name, (material, material_id) in r3m_materials.items():
                        writer.write_uint(material.layer_num)
                        writer.write_uint(material.flag)
                        writer.write_int(material.detail_surface)
                        writer.write_float(material.detail_scale)
                        writer.write_fixed_string(128, "cp949", material.name)
                        for texture_layer in material.texture_layers:
                            writer.write_short(texture_layer.iTileAniTexNum)
                            writer.write_int(texture_layer.texture_id)
                            writer.write_uint(texture_layer.alpha_type)
                            a = int(texture_layer.argb_color[0] * 255)
                            r = int(texture_layer.argb_color[1] * 255)
                            g = int(texture_layer.argb_color[2] * 255)
                            b = int(texture_layer.argb_color[3] * 255)
                            argb = (a << 24) | (r << 16) | (g << 8) | b
                            writer.write_uint(argb)
                            writer.write_uint(texture_layer.flags)
                            
                            writer.write_short(texture_layer.lava_wave_effect_rate)
                            writer.write_short(texture_layer.lava_wave_effect_speed)
                            
                            writer.write_short(texture_layer.scroll_u)
                            writer.write_short(-texture_layer.scroll_v)
                            writer.write_short(-texture_layer.uv_rotation)
                            writer.write_short(texture_layer.uv_starting_scale)
                            writer.write_short(texture_layer.uv_ending_scale)
                            writer.write_short(texture_layer.uv_scale_speed)
                            writer.write_short(texture_layer.metal_effect_size)
                            writer.write_short(texture_layer.alpha_flicker_rate)
                            writer.write_ushort(texture_layer.alpha_flicker_animation)
                            writer.write_short(texture_layer.animated_texture_frame)
                            writer.write_short(texture_layer.animated_texture_speed)
                            writer.write_short(texture_layer.gradient_alpha)
                            
                except Exception as e:
                    file.close()
                    os.remove(r3m_filepath)
                    self.report({"ERROR"}, f"Exception while writing to file at [{r3m_filepath}]: {e}")
                    traceback.print_exc()
                    return
                
        except Exception as e:
            self.report({"ERROR"}, f"Could not open file for writing at [{r3m_filepath}]: {e}")
            traceback.print_exc()
            return
        
        # Writing BSP
        try:
            with open(bsp_filepath, "wb") as file:
                writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                try:
                    writer.write_uint(39)
                    offset = 684
                    def __write_offset_and_size(size):
                        nonlocal offset
                        writer.write_uint(offset)
                        writer.write_uint(size)
                        offset += size
                    __write_offset_and_size(len(splitting_planes)*12)
                    __write_offset_and_size(len(collision_face_ids)*4)
                    __write_offset_and_size(len(bsp_nodes)*24)
                    __write_offset_and_size(len(bsp_leaves)*25)
                    __write_offset_and_size(len(material_list_in_leaf_ids)*2)
                    __write_offset_and_size(0)# animated object
                    __write_offset_and_size(0)# track
                    __write_offset_and_size(0)# eventobjectid
                    
                    for _ in range(37):
                        __write_offset_and_size(0) # Write spares and unused (last two include B and W vertex, which I have yet to see being used for BSPs)
                    
                    __write_offset_and_size(len(bsp_vertex_data)*12)
                    __write_offset_and_size(len(bsp_vertex_data)*4)
                    __write_offset_and_size(len(bsp_uvs)*8)
                    __write_offset_and_size(len(bsp_light_uvs)*4)
                    __write_offset_and_size(len(bsp_face_pointers)*6)
                    __write_offset_and_size(len(bsp_face_ids)*4)
                    __write_offset_and_size(len(bsp_vertex_ids)*4)
                    __write_offset_and_size(len(material_groups)*42)
                    
                    for _ in range(32):
                        __write_offset_and_size(0) # Write spares
                    for (splitting_plane) in splitting_planes:
                        writer.write_converted_vector3f(Vector(splitting_plane))
                        
                    for bsp_partitioned_face_id in collision_face_ids:
                        writer.write_uint(bsp_partitioned_face_id)
                        
                    for bsp_node in bsp_nodes:
                        writer.write_uint(bsp_node.plane_id)
                        writer.write_float(bsp_node.distance)
                        writer.write_short(bsp_node.front_id)
                        writer.write_short(bsp_node.back_id)
                        bb_min = ExportBSP.vector_to_shorts(co_conv.convert_vector3f(bsp_node.bounding_box_min))
                        bb_max = ExportBSP.vector_to_shorts(co_conv.convert_vector3f(bsp_node.bounding_box_max))
                        for bb_element in bb_min + bb_max:
                            writer.write_short(bb_element)
                    
                    for bsp_leaf in bsp_leaves:
                        bsp_leaf: BSPLeaf
                        writer.write_ubyte(bsp_leaf.type)
                        writer.write_ushort(bsp_leaf.face_amount)
                        writer.write_uint(bsp_leaf.face_start_id)
                        writer.write_ushort(bsp_leaf.material_group_amount)
                        writer.write_uint(bsp_leaf.material_group_start_id)
                        bb_min = ExportBSP.vector_to_shorts(co_conv.convert_vector3f(bsp_leaf.bounding_box_min))
                        bb_max = ExportBSP.vector_to_shorts(co_conv.convert_vector3f(bsp_leaf.bounding_box_max))
                        for bb_element in bb_min + bb_max:
                            writer.write_short(bb_element)
                            
                    for material_id_in_leaves in material_list_in_leaf_ids:
                        writer.write_ushort(material_id_in_leaves)
                    
                    extracted_vertex_data = [(id_, vec, color) for vec, (id_, color) in bsp_vertex_data.items()]
                    sorted_vertex_data = sorted(extracted_vertex_data, key=lambda x: x[0])
                    
                    for (vertex_id, vertex_position, vertex_color) in sorted_vertex_data:
                        writer.write_converted_vector3f(Vector(vertex_position))
                    
                    for _ in range(len(extracted_vertex_data)):
                        writer.write_uint(0xFFFFFFFF)
                    
                    for uv in bsp_uvs:
                        writer.write_float(uv[0])
                        writer.write_float(1.0-uv[1])
                    
                    for light_uv in bsp_light_uvs:
                        clamped_u = max(0.0, min(1.0, light_uv[0]))
                        clamped_v = max(0.0, min(1.0, light_uv[1]))

                        scaled_u = int(clamped_u * 32767.0)
                        scaled_v = int((1.0 - clamped_v) * 32767.0)
                        
                        scaled_u = max(-32768, min(32767, scaled_u))
                        scaled_v = max(-32768, min(32767, scaled_v))

                        writer.write_short(scaled_u)
                        writer.write_short(scaled_v)
                        
                    for face_pointer in bsp_face_pointers:
                        writer.write_ushort(face_pointer.vertex_amount)
                        writer.write_uint(face_pointer.vertex_start_id)
                    
                    for face_id in bsp_face_ids:
                        writer.write_uint(face_id)
                    
                    for vertex_id in bsp_vertex_ids:
                        writer.write_uint(vertex_id)
                    
                    for material_group in material_groups:
                        writer.write_ushort(material_group.attribute)
                        writer.write_ushort(material_group.number_of_faces)
                        writer.write_uint(material_group.starting_face_id)
                        writer.write_short(material_group.material_id)
                        writer.write_short(material_group.light_id)
                        bb_min = ExportBSP.vector_to_shorts(co_conv.convert_vector3f(Vector((material_group.bounding_box_min.x, material_group.bounding_box_min.y, material_group.bounding_box_min.z))))
                        bb_max = ExportBSP.vector_to_shorts(co_conv.convert_vector3f(Vector((material_group.bounding_box_max.x, material_group.bounding_box_max.y, material_group.bounding_box_max.z))))
                        for bb_element in bb_min + bb_max:
                            writer.write_short(bb_element)
                        writer.write_converted_vector3f(material_group.position)
                        writer.write_float(material_group.scale)
                        writer.write_ushort(0)
                    
                except Exception as e:
                    file.close()
                    os.remove(bsp_filepath)
                    self.report({"ERROR"}, f"Exception while writing to file at [{bsp_filepath}]: {e}")
                    traceback.print_exc()
                    return
                
        except Exception as e:
            self.report({"ERROR"}, f"Could not open file for writing at [{bsp_filepath}]: {e}")
            traceback.print_exc()
            return
        
        # Writing EBP
        try:
            with open(ebp_filepath, "wb") as file:
                writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                try:
                    offset = 388
                    def __write_offset_and_size(size):
                        nonlocal offset
                        writer.write_uint(offset)
                        writer.write_uint(size)
                        offset += size
                    writer.write_uint(20)
                    for _ in range(3):
                        __write_offset_and_size(0)
                        
                    __write_offset_and_size(len(bsp_leaves)*6)
                    
                    for _ in range(44):
                        __write_offset_and_size(0)
                        
                    file.write(bytes(len(bsp_leaves)*6))
                    
                except Exception as e:
                    file.close()
                    os.remove(ebp_filepath)
                    self.report({"ERROR"}, f"Exception while writing to file at [{ebp_filepath}]: {e}")
                    traceback.print_exc()
                    return
                
        except Exception as e:
            self.report({"ERROR"}, f"Could not open file for writing at [{ebp_filepath}]: {e}")
            traceback.print_exc()
            return
        
        return {'FINISHED'}  # Indicate success

    def adjust_lightness(image_name, lightness=-50):  # -50 is -50% like in Paint.NET
        """
        Adjust image lightness using Paint.NET's algorithm.
        :param lightness: Lightness adjustment (-100 to 100)
        """
        import numpy as np
        
        # Get the image
        image = bpy.data.images.get(image_name)
        if not image or not image.pixels:
            print(f"Image '{image_name}' not found or has no pixel data.")
            return
        
        # Convert pixel data to a NumPy array
        pixels = np.array(image.pixels[:])
        pixels = pixels.reshape(-1, 4)  # Reshape to (num_pixels, 4)
        
        # Convert -100 to 100 range to -1 to 1
        factor = lightness / 100.0
        
        # Paint.NET's lightness adjustment
        if factor < 0:
            # For negative lightness (darkening)
            factor = factor + 1  # Convert -1..0 to 0..1
            
            # Apply to RGB channels
            pixels[:, 0:3] *= factor  # Multiply RGB by factor, keeping alpha
        else:
            # For positive lightness (brightening)
            pixels[:, 0:3] += (1 - pixels[:, 0:3]) * factor
        
        # Write the modified pixel data back to the image
        image.pixels = pixels.flatten()
        image.update()
    
    def setup_compositor_for_denoising():
        """Set up the compositor for denoising the baked texture."""
        # Enable compositing and use nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        tree.nodes.clear()  # Clear existing nodes

        # Add nodes
        image_node = tree.nodes.new(type='CompositorNodeImage')
        
        denoise_node = tree.nodes.new(type='CompositorNodeDenoise')
        viewer_node = tree.nodes.new(type='CompositorNodeViewer')

        # Link nodes
        links = tree.links
        links.new(image_node.outputs['Image'], denoise_node.inputs['Image'])
        links.new(denoise_node.outputs['Image'], viewer_node.inputs['Image'])

        # Return the nodes for cleanup
        return tree, image_node, denoise_node, viewer_node

    def apply_denoising_and_save(texture_name, tree, image_node, denoise_node, viewer_node):
        image_node.image = bpy.data.images.get(texture_name)
        if not image_node.image:
            raise ValueError(f"Image '{texture_name}' not found in Blender's data block.")
        

        bpy.ops.render.render(write_still=False)

        viewer_result = bpy.data.images.get("Viewer Node")
        if viewer_result:
            original_texture = bpy.data.images.get(texture_name)
            if original_texture:
                original_texture.pixels = viewer_result.pixels[:]
                print(f"Updated original texture '{texture_name}' with denoised result.")
                original_texture.update()
                print(f"Packed denoised texture '{texture_name}' into the .blend file.")
            else:
                print(f"Error: Could not find original texture '{texture_name}'.")
        else:
            print("Error: Could not find denoised result in the Viewer Node.")
    
    def adjust_lightmap_for_d3d8(image_name):
        """
        Adjust lightmap for D3D8 engine compatibility. It multiplies the image's values by 0.5, since R3Engine will multiply it by 2 later.
        """
        image = bpy.data.images.get(image_name)
        if not image or not image.pixels:
            return
            
        pixels = np.array(image.pixels[:])
        pixels = pixels.reshape(-1, 4)
        
        # Since D3D8 will multiply by 2, pre-divide by 2
        # Modify this later, but let's try no division of the final result first.
        pixels[:, 0:3] *= 1
        
        image.pixels = pixels.flatten()
        image.update()
     
     
     
    def calc_face_area(face, obj):
        # Get the face's vertices in world space
        verts = [obj.matrix_world @ v.co for v in face.verts]
        num_verts = len(verts)

        if num_verts == 3:
            return mathutils.geometry.area_tri(*verts[:3])
        
        elif num_verts == 4:
            # Split quad into two triangles (0-1-2 and 0-2-3)
            tri1 = mathutils.geometry.area_tri(verts[0], verts[1], verts[2])
            tri2 = mathutils.geometry.area_tri(verts[0], verts[2], verts[3])
            return tri1 + tri2
        
        elif num_verts > 4:
            # Triangulate by fanning from the first vertex
            total_area = 0.0
            for i in range(1, num_verts - 1):
                total_area += mathutils.geometry.area_tri(verts[0], verts[i], verts[i + 1])
            return total_area
        
        # Handle degenerate cases (0, 1, or 2 vertices)
        else:
            return 0.0

    def flood_fill_with_limit(start_face: bmesh.types.BMFace, unmarked_faces: set[bmesh.types.BMFace], max_texels: int, obj: bpy.types.Object, units_per_texel: float):
        connected_faces: set[bmesh.types.BMFace] = set()
        island_faces: set[bmesh.types.BMFace] = set()
        connected_faces_islands: list[set[bmesh.types.BMFace]] = []
        queue = deque([start_face])
        total_texels = 0
        current_group_texels = 0
        min_group_texels = 16384  # 8x8 texels
        
        group_count = 0
        
        # Continue until we've used all unmarked faces or reached the texel limit
        while len(unmarked_faces) > 0 and total_texels < max_texels:
            # Check if current group is done and we need to start a new group
            if len(queue) == 0:
                # Add the current group's clamped texel count to the total
                clamped_group_texels = max(current_group_texels, min_group_texels)
                total_texels += clamped_group_texels
                
                # Start a new group
                if len(unmarked_faces) > 0 and total_texels < max_texels:
                    queue.append(list(unmarked_faces)[0])
                    current_group_texels = 0
                    connected_faces_islands.append(island_faces.copy())
                    island_faces.clear()
                group_count +=1
            
            # If queue is still not empty, process the next face
            if len(queue) > 0:
                face = queue.popleft()
                if face in unmarked_faces:
                    area = ExportBSP.calc_face_area(face, obj)
                    texels = area / (units_per_texel ** 2)
                    current_group_texels += texels
                    
                    connected_faces.add(face)
                    island_faces.add(face)
                    unmarked_faces.remove(face)
                    
                    # Add connected faces to the queue
                    for edge in face.edges:
                        for linked_face in edge.link_faces:
                            if linked_face in unmarked_faces:
                                queue.append(linked_face)
        
        # Handle the last group before returning
        if current_group_texels > 0:
            clamped_group_texels = max(current_group_texels, min_group_texels)
            total_texels += clamped_group_texels
            
        return connected_faces, connected_faces_islands
   
    def flood_fill_with_limit_indices(start_face: int, unmarked_faces: set[int], max_texels: int, obj: bpy.types.Object, bm: bmesh.types.BMesh, units_per_texel: float, max_face_amount_for_group = 20000):
        connected_faces: set[int] = set()
        island_faces: set[int] = set()
        connected_faces_islands: list[set[int]] = []
        queue = deque([start_face])
        total_texels = 0
        current_group_texels = 0
        min_group_texels = 16384  # 8x8 texels
        
        group_count = 0
        
        # Continue until we've used all unmarked faces or reached the texel limit
        while len(unmarked_faces) > 0 and total_texels < max_texels and len(connected_faces) < max_face_amount_for_group:
            # Check if current group is done and we need to start a new group
            if len(queue) == 0:
                # Add the current group's clamped texel count to the total
                clamped_group_texels = max(current_group_texels, min_group_texels)
                total_texels += clamped_group_texels
                
                connected_faces_islands.append(island_faces.copy())
                island_faces.clear()
                
                
                # Start a new group
                if len(unmarked_faces) > 0 and total_texels < max_texels:
                    queue.append(next(iter(unmarked_faces)))
                    current_group_texels = 0
                    
                group_count +=1
            
            # If queue is still not empty, process the next face
            if len(queue) > 0:
                face_index = queue.popleft()
                if face_index in unmarked_faces:
                    real_face = bm.faces[face_index]
                    area = ExportBSP.calc_face_area(real_face, obj)
                    texels = area / (units_per_texel ** 2)
                    current_group_texels += texels
                    
                    connected_faces.add(face_index)
                    island_faces.add(face_index)
                    unmarked_faces.remove(face_index)
                    
                    # Add connected faces to the queue
                    for edge in real_face.edges:
                        for linked_face in edge.link_faces:
                            if linked_face.index in unmarked_faces:
                                queue.append(linked_face.index)
        
        # Handle the last group before returning
        if current_group_texels > 0:
            clamped_group_texels = max(current_group_texels, min_group_texels)
            total_texels += clamped_group_texels
            connected_faces_islands.append(island_faces.copy())
            
        return connected_faces, connected_faces_islands
    
    def process_material_in_light_group(faces: list[bmesh.types.BMFace], bm: bmesh.types.BMesh, mesh: bpy.types.Mesh) -> dict:
        """
        Process a list of faces (all using the same material) within a lightmap group.
        Returns vertices, polygons, UVs, and light UVs, with reversed face windings.
        """
        exporter_vertices = []
        exporter_uvs = []
        exporter_light_uvs = []
        exporter_polygons = []

        # Get UV layers from the BMesh
        uv_layer = bm.loops.layers.uv["UVMap"]
        light_uv_layer = bm.loops.layers.uv["LightUVMap"]

        # Map to track unique vertex+UV+light UV combinations
        vertex_map = {}  # Key: (vertex_index, uv, light_uv), Value: index in exporter lists
        vertex_counter = 0

        for face in faces:
            poly_indices = []
            for loop in face.loops:
                vert = loop.vert
                uv = tuple(loop[uv_layer].uv)
                light_uv = tuple(loop[light_uv_layer].uv)
                key = (vert.index, uv, light_uv)

                if key not in vertex_map:
                    vertex_map[key] = vertex_counter
                    exporter_vertices.append(vert.co / SCALE_FACTOR)
                    exporter_uvs.append(uv)
                    exporter_light_uvs.append(light_uv)
                    vertex_counter += 1

                poly_indices.append(vertex_map[key])

            # Reverse polygon winding, matching process_mesh_object
            amount_of_polys = len(poly_indices)
            if amount_of_polys == 3:
                exporter_polygons.append([poly_indices[2], poly_indices[1], poly_indices[0]])
            elif amount_of_polys == 4:
                exporter_polygons.append([poly_indices[2], poly_indices[1], poly_indices[0]])
                exporter_polygons.append([poly_indices[3], poly_indices[2], poly_indices[0]])
            else:
                v0 = poly_indices[0]
                for i in range(1, amount_of_polys - 1):
                    exporter_polygons.append([poly_indices[i + 1], poly_indices[i], v0])

        return {
            "vertices": exporter_vertices,
            "uvs": exporter_uvs,
            "light_uvs": exporter_light_uvs,
            "polygons": exporter_polygons
        }
    
    def add_udim_tiles(lightmap_udim, texture_size, start_index, end_index):
        """
        Add UDIM tiles to an existing tiled image using modern Blender 4.0+ context override.
        
        Args:
            lightmap_udim: Reference to the Blender Image object
            texture_size: Size of each tile (width/height in pixels)
            start_index: First tile index to create
            end_index: Last tile index to create (exclusive)
        """
        
        # Store the current context state
        original_area_type = None
        temp_area = None
        original_active_object = bpy.context.active_object
        original_mode = None
        if original_active_object:
            original_mode = original_active_object.mode
        
        try:
            # Exit edit mode if we're in it (can interfere with context)
            if original_active_object and original_mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            
            # Find or create Image Editor area
            for area in bpy.context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    temp_area = area
                    original_area_type = 'IMAGE_EDITOR'
                    break
                    
            if not temp_area:
                # If no image editor exists, find an area to temporarily convert
                for area in bpy.context.screen.areas:
                    if area.type not in {'PROPERTIES', 'INFO'}:  # Avoid these areas
                        temp_area = area
                        original_area_type = area.type
                        temp_area.type = 'IMAGE_EDITOR'
                        break
            
            # Set the UDIM texture as the active image in the image editor
            if temp_area:
                temp_area.spaces.active.image = lightmap_udim
                
                # Create the tiles using modern context override
                for i in range(start_index, end_index):
                    tile_number = 1001 + i  # UDIM tiles start at 1001
                    
                    # Use the modern context override method
                    with bpy.context.temp_override(area=temp_area):
                        bpy.ops.image.tile_add(
                            width=texture_size,
                            height=texture_size,
                            float=True,
                            generated_type='BLANK',
                            number=tile_number
                        )
                    
        except Exception as e:
            print(f"Error adding UDIM tiles: {e}")
            raise
            
        finally:
            # Restore original context state
            if temp_area and original_area_type and temp_area.type != original_area_type:
                temp_area.type = original_area_type
                
            # Restore original object mode
            if original_active_object and original_mode:
                bpy.context.view_layer.objects.active = original_active_object
                if original_mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode=original_mode)
    
    # Define the connectivity limit based on Blender's source
    STD_UV_CONNECT_LIMIT = 0.0001
    STD_UV_CONNECT_LIMIT_SQ = STD_UV_CONNECT_LIMIT**2 # Use squared distance for efficiency
    
    def find_uv_islands_bm(
    bm: bmesh.types.BMesh,
    group_face_indices: set[int],
    uv_layer: bmesh.types.BMLayerItem
    ) -> list[set[int]]:
        """
        Identifies distinct UV islands using pure BMesh traversal, mimicking
        Blender's internal UV connectivity logic by pre-grouping UVs at vertices.
        You can think of this function as Blender's select linked operator's C code being unwrapped to work directly for our purposes, instead of relying on their call with overhead.

        Preconditions:
            - Input 'bm' reflects mesh state AFTER unwrapping.
            - 'uv_layer' is the active BMUVLayer.
            - No specific selection state required.

        :param bm: The BMesh object.
        :param group_face_indices: Indices of faces to search within.
        :param uv_layer: The active BMUVLayer.
        :return: List of sets, each set contains face indices of one UV island.
        """
        if not bm or not bm.is_valid or not uv_layer:
            print("[ERROR] find_uv_islands_bm: Invalid BMesh or UV layer.")
            return []
        if not group_face_indices: return []

        # --- Performance Timing Start ---
        func_start_time = time.perf_counter()
        time_build_map = 0.0
        time_bfs_total = 0.0

        # --- 1. Build UV Connectivity Map (Mimic UvVertMap Creation) ---
        t0 = time.perf_counter()
        # Structure: { BMVert -> [ (representative_uv, [loop1, loop2, ...]), ... ] }
        # Stores, for each vertex, a list of distinct UV groups. Each group has a
        # representative UV coordinate and a list of loops sharing that coordinate (within tolerance).
        vert_uv_groups = {}
        loops_processed_for_map = set() # Track loops added to the map

        bm.verts.ensure_lookup_table() # Needed for vert iteration/access if not done prior
        bm.faces.ensure_lookup_table()

        # Iterate through faces first to only consider relevant loops/verts
        relevant_loops = []
        for face_idx in group_face_indices:
            try:
                face = bm.faces[face_idx]
                if face.is_valid:
                    relevant_loops.extend(face.loops)
            except IndexError: continue # Skip invalid faces


        for loop in relevant_loops:
            if not loop.is_valid or loop in loops_processed_for_map: continue

            vert = loop.vert
            loop_uv = loop[uv_layer].uv

            if vert not in vert_uv_groups:
                vert_uv_groups[vert] = []

            found_group = False
            # Check if this loop's UV matches an existing group for this vertex
            for i, (rep_uv, loop_list) in enumerate(vert_uv_groups[vert]):
                if (loop_uv - rep_uv).length_squared < ExportBSP.STD_UV_CONNECT_LIMIT_SQ:
                    # Matches existing group, add loop to it
                    loop_list.append(loop)
                    loops_processed_for_map.add(loop)
                    found_group = True
                    break # Found the group for this loop

            if not found_group:
                # No matching group found, create a new one for this UV coordinate
                new_group_list = [loop]
                vert_uv_groups[vert].append((loop_uv.copy(), new_group_list)) # Store copy of UV as rep
                loops_processed_for_map.add(loop)


        # --- Optional: Map loop -> uv_group_id for faster lookup during BFS? ---
        # Structure: { BMLoop -> (BMVert, uv_group_index) }
        loop_to_uv_group_id = {}
        for vert, groups in vert_uv_groups.items():
            for group_index, (rep_uv, loop_list) in enumerate(groups):
                group_id = (vert, group_index) # Unique ID for this specific UV loc at this vert
                for loop in loop_list:
                    loop_to_uv_group_id[loop] = group_id

        t1 = time.perf_counter(); time_build_map = t1 - t0

        # --- 2. Perform BFS Traversal Using the Connectivity Map ---
        visited_loops = set()
        actual_uv_islands_faces = []
        all_loops_in_group = loops_processed_for_map # Use the set we already built

        print(f"[INFO] Starting BMesh UV island detection (Pre-Grouped) for {len(all_loops_in_group)} loops...")

        while all_loops_in_group:
            start_loop = all_loops_in_group.pop()
            if start_loop in visited_loops:
                continue

            t_bfs_start = time.perf_counter()
            current_island_loops = set()
            queue = deque([start_loop])
            processed_in_this_bfs = {start_loop}

            while queue:
                current_loop = queue.popleft()

                current_island_loops.add(current_loop)
                visited_loops.add(current_loop)

                # Find the UV group ID for the current loop
                current_group_id = loop_to_uv_group_id.get(current_loop)
                if not current_group_id: continue # Should not happen if map is built correctly

                # Traverse neighbors:
                # a) Around the face: Next/Prev loops MUST be connected
                neighbors_around_face = [current_loop.link_loop_next, current_loop.link_loop_prev]
                for next_loop in neighbors_around_face:
                    if (next_loop.is_valid and
                        next_loop.face.index in group_face_indices and
                        next_loop not in visited_loops and
                        next_loop not in processed_in_this_bfs):
                        queue.append(next_loop)
                        processed_in_this_bfs.add(next_loop)


                # b) Across the vertex: Other loops sharing the SAME UV group ID at this vertex
                vert = current_loop.vert
                # We already grouped them, find all loops in the same group
                # This lookup should be fast using the precomputed map
                loops_in_same_uv_group = []
                if vert in vert_uv_groups:
                    for group_index, (rep_uv, loop_list) in enumerate(vert_uv_groups[vert]):
                        if (vert, group_index) == current_group_id:
                            loops_in_same_uv_group = loop_list
                            break

                for other_loop in loops_in_same_uv_group:
                    if other_loop is not current_loop: # Don't re-add self
                        if (other_loop.is_valid and
                            other_loop.face.index in group_face_indices and
                            other_loop not in visited_loops and
                            other_loop not in processed_in_this_bfs):
                            queue.append(other_loop)
                            processed_in_this_bfs.add(other_loop)


            # --- BFS for one island finished ---
            t_bfs_end = time.perf_counter(); time_bfs_total += (t_bfs_end - t_bfs_start)

            if current_island_loops:
                island_face_indices = {loop.face.index for loop in current_island_loops}
                actual_uv_islands_faces.append(island_face_indices)
                all_loops_in_group -= current_island_loops # Remove processed loops

        # --- Function Finished ---
        func_end_time = time.perf_counter()
        total_func_time = func_end_time - func_start_time
        print(f"[INFO] Finished Pre-Grouped BMesh UV island detection. Found {len(actual_uv_islands_faces)} islands in {total_func_time:.4f}s.")
        # Optional Detailed Timing
        print("--- Pre-Grouped Island Detection Performance ---")
        print(f"  Total Time:       {total_func_time:.4f}s")
        print(f"  Build Map:        {time_build_map:.4f}s")
        print(f"  BFS Traversal:    {time_bfs_total:.4f}s")
        print("--------------------------------------------")

        return actual_uv_islands_faces
    
    
    def check_and_resize_uv_islands(mesh: bpy.types.Mesh,
                                group: set[int],
                                texture_size: int,
                                min_pixel_size: int = 8,
                                max_aspect_ratio_threshold: float = 10.0):
        """
        Check UV islands for minimum size and extreme aspect ratios, and resize if necessary.

        Uses the connected_faces_islands (based on 3D connectivity) as a proxy for UV islands.

        :param bm: BMesh object (must be in Edit Mode)
        :param group: Set of faces belonging to the current UDIM tile being processed
        :param connected_faces_islands: List of sets, where each set contains faces assumed to form an island.
        :param texture_size: The width/height of the lightmap texture (e.g., 1024).
        :param min_pixel_size: Minimum desired island bounding box dimension in pixels.
        :param max_aspect_ratio_threshold: Threshold for extreme aspect ratios (e.g., 10 means problematic if width/height > 10 or height/width > 10).
        :return: Boolean indicating if all islands met the criteria (or were successfully resized).
                Returns False if problematic islands were found and resizing was attempted.
                Returns True if no problematic islands were found initially.
        """
        
        if not mesh.uv_layers.active:
            print("[ERROR] No active UV layer found on mesh.")
            return True # Treat as OK if no layer to check
        uv_layer_name = mesh.uv_layers.active.name

        # --- Calculate target dimension in UV units ---
        # Add small epsilon to prevent division by zero if texture_size is somehow invalid
        
        epsilon = 1e-6
        target_uv_dim = min_pixel_size / (texture_size + epsilon)
        min_uv_area = target_uv_dim * target_uv_dim
        
        
        print(f"[DEBUG] Target UV Dimension: {target_uv_dim:.6f} (for {min_pixel_size} pixels on {texture_size}px texture)")

        def calculate_island_metrics(island_faces: set[int]):
            """
            Calculate UV island metrics (based on the faces provided).

            :return: (min_u, min_v, max_u, max_v, width, height, aspect_ratio)
            """
            uv_coords = []
            
            if not island_faces_indices: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            
            for face_index in island_faces_indices:
                try:
                    face = bm.faces[face_index]
                    if face.is_valid:
                        for loop in face.loops:
                            uv_coords.append(loop[uv_layer].uv.copy())
                except IndexError:
                    print(f"[WARN] Invalid face index {face_index} during metrics calculation.")


            if not uv_coords:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 # Treat as non-problematic if empty

            min_u = min(coord.x for coord in uv_coords)
            max_u = max(coord.x for coord in uv_coords)
            min_v = min(coord.y for coord in uv_coords)
            max_v = max(coord.y for coord in uv_coords)

            # Bounding box dimensions in UV space
            bbox_width = max_u - min_u
            bbox_height = max_v - min_v

            # Calculate aspect ratio safely
            if bbox_width < epsilon and bbox_height < epsilon:
                aspect_ratio = 1.0 # Degenerate point
            elif bbox_height < epsilon:
                aspect_ratio = float('inf') # Degenerate horizontal line
            elif bbox_width < epsilon:
                aspect_ratio = 0.0 # Degenerate vertical line
            else:
                aspect_ratio = bbox_width / bbox_height

            return min_u, min_v, max_u, max_v, bbox_width, bbox_height, aspect_ratio


        def resize_island(island_faces: set[int], current_metrics):
            """
            Resize a problematic UV island to meet minimum size requirements,
            using NON-UNIFORM scaling if necessary.

            :param island_faces: Set of faces forming the UV island.
            :param current_metrics: Tuple from calculate_island_metrics
            """
            min_u, min_v, max_u, max_v, current_width, current_height, _ = current_metrics

            # Handle case where metrics might indicate zero size if island was empty
            # Also handle cases where dimensions are already large enough
            if (current_width <= epsilon and current_height <= epsilon):
                print("[DEBUG] Skipping resize for zero-size island.")
                return # Cannot resize a point based on current dimensions

            # --- Calculate desired scaling factors PER AXIS ---
            scale_x = 1.0
            # Only scale up if below target AND not degenerate (avoid massive scaling from epsilon)
            if epsilon < current_width < target_uv_dim:
                scale_x = target_uv_dim / current_width
            elif current_width <= epsilon: # Is degenerate (line/point): needs some width
                # Instead of infinite scale, let's set the target size directly later?
                # For now, apply a large-ish scale, but maybe capped?
                # Let's try making it target_uv_dim wide. Scale = target_uv_dim / epsilon ~ huge.
                # This might still be problematic. Let's cap the scale factor for degenerates.
                scale_x = target_uv_dim / max(current_width, epsilon / 10) # Prevent pure zero division
                # Cap the scale factor to avoid astronomical numbers if epsilon is tiny
                scale_x = min(scale_x, 1.0 / target_uv_dim) # Heuristic cap: limits max size somewhat
                print(f"[DEBUG] Degenerate width detected. Calculated initial scale_x: {scale_x:.4f}")


            scale_y = 1.0
            if epsilon < current_height < target_uv_dim:
                scale_y = target_uv_dim / current_height
            elif current_height <= epsilon: # Is degenerate (line/point): needs some height
                scale_y = target_uv_dim / max(current_height, epsilon / 10)
                scale_y = min(scale_y, 1.0 / target_uv_dim) # Heuristic cap
                print(f"[DEBUG] Degenerate height detected. Calculated initial scale_y: {scale_y:.4f}")


            # Ensure scale factors are at least 1.0 (we only want to enlarge)
            scale_x = max(scale_x, 1.0)
            scale_y = max(scale_y, 1.0)

            if abs(scale_x - 1.0) < epsilon and abs(scale_y - 1.0) < epsilon:
                print(f"[DEBUG] Calculated scales are <= 1.0 ({scale_x:.4f}, {scale_y:.4f}). Skipping scaling.")
                return

            center_u = (min_u + max_u) / 2.0
            center_v = (min_v + max_v) / 2.0


            # Resize the island's UVs NON-UNIFORMLY
            for face_index in island_faces:
                face = bm.faces[face_index]
                if face.is_valid and face.index < len(bm.faces):
                    for loop in face.loops:
                        uv = loop[uv_layer].uv
                        # Scale from center independently for U and V
                        scaled_u = center_u + (uv.x - center_u) * scale_x
                        scaled_v = center_v + (uv.y - center_v) * scale_y
                        loop[uv_layer].uv = (scaled_u, scaled_v)
                else:
                    print(f"[WARN] Invalid face encountered during resize: index {face.index}")

        start_time = time.perf_counter()
        

        bpy.ops.object.mode_set(mode='EDIT')
        
        
        
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table() # Needed for index access
        uv_layer = bm.loops.layers.uv.get(uv_layer_name)
        if not uv_layer:
            print(f"[ERROR] Could not get BMesh UV layer '{uv_layer_name}' for checks.")
            return True # Treat as OK
        t_island_start = time.perf_counter()
        actual_uv_islands = ExportBSP.find_uv_islands_bm(bm, group, uv_layer)
        t_island_end = time.perf_counter()
        print(f"[INFO] BMesh Island Finding took: {t_island_end - t_island_start:.4f}s")
        
        
        print(f"[INFO] Finding UV Islands via select linked took {(time.perf_counter() - start_time):.2f} seconds.")
        start_time = time.perf_counter()
        
        if actual_uv_islands is None:
            print("[ERROR] Failed to identify UV islands.")
            return True # Treat failure to find islands as non-problematic for now

        if not actual_uv_islands:
            print("[INFO] No UV islands found within the group (group might be empty or fully degenerate?).")
            return True
        
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.get(uv_layer_name)
        if not uv_layer:
            print(f"[ERROR] Could not get BMesh UV layer '{uv_layer_name}' for checks.")
            bm.free()
            return True
        
        # --- Main Check and Resize Loop ---
        problematic_islands_found = False
        islands_to_resize = []

        # Check UV islands
        for i, island_faces_indices in enumerate(actual_uv_islands):
            if not island_faces_indices: continue # Skip if an empty island set was passed

            metrics = calculate_island_metrics(island_faces_indices)
            _, _, _, _, bbox_width, bbox_height, aspect_ratio = metrics

            #print(f"[DEBUG] Island {i}: Indices={island_faces_indices}")
            #print(f"[DEBUG] Island {i}: Metrics: W={bbox_width:.6f}, H={bbox_height:.6f}, AR={aspect_ratio:.2f}")

            is_problematic = False

            # Check 1: Degeneracy (very small dimensions)
            is_degenerate = bbox_width < epsilon or bbox_height < epsilon
            if is_degenerate:
                # Technically problematic, resize logic will handle aiming for target_uv_dim
                # print(f"[DEBUG] Island is degenerate (W:{bbox_width:.2E}, H:{bbox_height:.2E}). Flagged for resize.")
                is_problematic = True

            # Check 2: Minimum Dimension
            # Only check if not degenerate (avoid issues with inf aspect ratio)
            if not is_degenerate:
                if bbox_width < target_uv_dim or bbox_height < target_uv_dim:
                    # print(f"[DEBUG] Island below target UV dimension (W:{bbox_width:.6f}, H:{bbox_height:.6f} vs Target:{target_uv_dim:.6f}). Flagged.")
                    is_problematic = True
                    #print(f"[DEBUG] Island {i}: Flagged as TOO SMALL (W:{bbox_width < target_uv_dim}, H:{bbox_height < target_uv_dim}).")

                # Check 3: Aspect Ratio (only if not degenerate)
                if aspect_ratio > max_aspect_ratio_threshold or (aspect_ratio > epsilon and aspect_ratio < (1.0 / max_aspect_ratio_threshold)):
                    # print(f"[DEBUG] Island has extreme aspect ratio ({aspect_ratio:.2f}). Flagged.")
                    # For now, we only resize based on minimum dimension, not aspect ratio alone.
                    # If you want to resize purely for aspect ratio, uncomment the next line:
                    # is_problematic = True
                    pass # Decide if aspect ratio alone triggers resizing

            if is_problematic:
                #print(f"[DEBUG] Island {i}: Final verdict = PROBLEMATIC.") # Add this
                problematic_islands_found = True
                islands_to_resize.append((island_faces_indices, metrics))
            else:
                pass
                #print(f"[DEBUG] Island {i}: Final verdict = OK.") # Add this
            #print("-" * 20) # Separator


        # Resize problematic islands
        if islands_to_resize:
            #print(f"[INFO] Found {len(islands_to_resize)} problematic islands. Attempting resize...")
            for island_faces_indices, metrics in islands_to_resize:
                resize_island(island_faces_indices, metrics)
            # Important: After modifying UVs in bmesh, update the mesh
            # bmesh.update_edit_mesh(mesh) # This should be done *outside* this function call

        print(f"[INFO] Correcting UV Islands took {(time.perf_counter() - start_time):.2f} seconds.")
        start_time = time.perf_counter()
        
        # Return True if no problems were found initially
        # Return False if problems were found (resizing was attempted)
        return not problematic_islands_found
    
    
    def new_bake_lightmaps(self, objects, texture_size = 1024, marginPixels = 4, unitsPerTexel = 1):
        uv_margin = marginPixels/texture_size
        
        # To be more specific, this variable holds the data for each "object" we need to export later to the BSP as different objects.
        material_light_data = {}
        light_maps_references = []
        
        print(f"Started light map baking process. UV Margin [{uv_margin}]")
        
        bpy.ops.object.select_all(action='DESELECT')
        
        start_time: float = time.perf_counter()
        
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        
        bpy.ops.object.duplicate()
        duplicates = bpy.context.selected_objects.copy()

        bpy.context.view_layer.objects.active = duplicates[0]

        bpy.ops.object.join()

        merged_object = bpy.context.active_object
        merged_object.select_set(True)
        merged_object.name = "MERGED OBJECT FOR BSP EXPORT"
        
        elapsed_time: float = time.perf_counter() - start_time
        start_time = time.perf_counter()
        print(f"[INFO] Merging all objects for BSP export took {elapsed_time:.2f} seconds.")
        
        # Hide all other mesh objects in the scene
        original_visibility = {}
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH' and obj != merged_object:
                original_visibility[obj] = (obj.hide_viewport, obj.hide_render)
                # Step 2: Hide mesh objects except the merged one
                obj.hide_viewport = True
                obj.hide_render = True
        
        # Remember to assign this to a material later so that Cycles knows it's supposed to bake there.
        lightmap_udim = bpy.data.images.new(
            name="Lightmap_UDIM",
            width=texture_size,
            height=texture_size,
            alpha=False,
            tiled=True
        )
        
        
        # We need a new exported object for each material+light texture combination.
        bpy.ops.object.mode_set(mode='EDIT')
        
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        
        mesh: bpy.types.Mesh = merged_object.data
        bpy.ops.mesh.select_all(action="DESELECT")
        
        if "LightUVMap" not in mesh.uv_layers:
            print(f"[INFO] Creating new LightUVMap for {merged_object.name}")
            light_uv_layer = mesh.uv_layers.new(name="LightUVMap")
        else:
            light_uv_layer = mesh.uv_layers["LightUVMap"]
            print(f"[INFO] Using existing LightUVMap for {merged_object.name}")
        
        mesh.uv_layers.active = mesh.uv_layers["LightUVMap"]
        mesh.uv_layers["LightUVMap"].active = True
        
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table()
        
        unmarked_faces = {f.index for f in bm.faces}
        
        light_texture_index = 0
        
        
        
        elapsed_time: float = time.perf_counter() - start_time
        start_time = time.perf_counter()
        print(f"[INFO] Preparations for light map group processing took {elapsed_time:.2f} seconds.")
        
        while len(unmarked_faces) > 0:
            pre_face_loop_time = time.perf_counter()
            start_time = time.perf_counter()
            
            
            print(f"Processing light map group [{light_texture_index}]")
            start_face = next(iter(unmarked_faces))
            # The group here contains all the faces we are gonna use in the current light texture.
            group: set[int] = None
            connected_faces_islands: list[set[int]] = None
            group, connected_faces_islands = ExportBSP.flood_fill_with_limit_indices(start_face, unmarked_faces, texture_size*texture_size, merged_object, bm, unitsPerTexel)
            
            elapsed_time: float = time.perf_counter() - start_time
            start_time = time.perf_counter()
            print(f"[INFO] Flood fill for group took {elapsed_time:.2f} seconds.")
            
            for face_index in group:
                bm.faces[face_index].select = True
            
            bmesh.update_edit_mesh(mesh)
            
            bpy.ops.uv.select_all(action='SELECT')
            
            
            lower_angle_deg = 45.0
            lower_angle_rad = math.radians(lower_angle_deg)
            
            bpy.ops.uv.smart_project(
                angle_limit=lower_angle_rad,        #IN RADIANS
                island_margin=0,
                area_weight=0.0,
                correct_aspect=True,
                scale_to_bounds=True 
            )
            
            bmesh.update_edit_mesh(mesh)
            bm.free()
            del bm
            
            elapsed_time: float = time.perf_counter() - start_time
            start_time = time.perf_counter()
            print(f"[INFO] Smart UV Project block took {elapsed_time:.2f} seconds.")
            
            bpy.ops.uv.select_all(action='DESELECT')
            
            first_pass_result = ExportBSP.check_and_resize_uv_islands(mesh, group, texture_size)

            elapsed_time: float = time.perf_counter() - start_time
            start_time = time.perf_counter()
            print(f"[INFO] Checking and resizing uv islands took {elapsed_time:.2f} seconds.")
            
            bm = bmesh.from_edit_mesh(mesh)
            bm.faces.ensure_lookup_table()
            
            bmesh.update_edit_mesh(mesh)
            
            for face_index in group:
                bm.faces[face_index].select = True
            
            bmesh.update_edit_mesh(mesh)
            
            
            
            
            
            bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.pack_islands(margin=uv_margin, margin_method="ADD")
            
            elapsed_time: float = time.perf_counter() - start_time
            start_time = time.perf_counter()
            print(f"[INFO] Packing islands took {elapsed_time:.2f} seconds.")
            
            # The order here is important. We store the light uv data before we apply the offset. That's because R3Engine expects the light uvs to be in the
            # 0-1 range. But to use our UDIM workflow in the light bake, we need to position the uvs correctly at their UDIM offsets.
            
            group_faces = [bm.faces[face_index] for face_index in group]

            materials_in_group = set(face.material_index for face in group_faces)

            for material_index in materials_in_group:
                material_faces = [face for face in group_faces if face.material_index == material_index]
                data = ExportBSP.process_material_in_light_group(material_faces, bm, mesh)
                material_light_data[(light_texture_index, merged_object.material_slots[material_index].material)] = data
            
            u_offset = light_texture_index % 10  # Column (0-9)
            v_offset = light_texture_index // 10  # Row (0, 1, 2, ...)
            uv_offset = mathutils.Vector((float(u_offset), float(v_offset)))
            
            uv_layer = bm.loops.layers.uv.active
            for face_index in group:
                for loop in bm.faces[face_index].loops:
                    loop[uv_layer].uv += uv_offset  # Apply both U and V shifts

            bmesh.update_edit_mesh(mesh)
            bpy.ops.uv.select_all(action='DESELECT')
            bpy.ops.mesh.select_all(action="DESELECT")
            
            light_texture_index += 1
            
            elapsed_time: float = time.perf_counter() - start_time
            start_time = time.perf_counter()
            print(f"[INFO] Group finalization took {elapsed_time:.2f} seconds.")
            
            elapsed_time: float = time.perf_counter() - pre_face_loop_time
            print(f"[INFO] Overall group processing took {elapsed_time:.2f} seconds.")

        
        ExportBSP.add_udim_tiles(lightmap_udim, texture_size, 1, light_texture_index)
        
        elapsed_time: float = time.perf_counter() - start_time
        start_time = time.perf_counter()
        print(f"[INFO] Unpacking and separating uv islands took {elapsed_time:.2f} seconds.")
        
        #return
        
        """Prepare objects for baking and bake lightmaps in batches."""
        # Set up Cycles rendering settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.samples = 16
        bpy.context.scene.cycles.max_bounces = 4
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.caustics_reflective = False
        bpy.context.scene.cycles.caustics_refractive = False
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.adaptive_threshold = 0.01
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.tile_size = 1024
        bpy.context.scene.render.bake.use_pass_direct = True
        bpy.context.scene.render.bake.use_pass_indirect = True
        bpy.context.scene.render.bake.use_pass_emit = True
        bpy.context.scene.render.use_persistent_data = True
        
        bpy.context.scene.cycles.use_denoising = False
        bpy.context.scene.cycles.sample_clamp_direct = 0.2
        bpy.context.scene.cycles.sample_clamp_indirect = 0.2
        bpy.context.scene.cycles.film_exposure = 1.5
        
        bpy.context.scene.render.bake.use_selected_to_active = False
        bpy.context.scene.render.bake.use_clear = False
        
        print(f"[INFO] Cycles settings set for baking.")
        
        start_time = datetime.datetime.now()
        
        original_material = merged_object.active_material
        if not original_material:
            return
        
        if "LightUVMap" not in merged_object.data.uv_layers:
            print(f"Object {merged_object.name} is missing a LightUVMap map.")
            return
        
        merged_object.data.uv_layers["LightUVMap"].active = True
        
        original_materials = [slot.material for slot in merged_object.material_slots]
        
        # Create a neutral material for baking
        neutral_material = bpy.data.materials.new(name="NeutralMaterial")
        neutral_material.use_nodes = True
        nodes = neutral_material.node_tree.nodes
        links = neutral_material.node_tree.links
        
        for node in nodes:
            nodes.remove(node)
        
        diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
        diffuse_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White color
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        merged_object.active_material = neutral_material
        
        texture_node = nodes.new(type='ShaderNodeTexImage')
        texture_node.image = lightmap_udim
        
        # Select the object for baking
        merged_object.select_set(True)
        bpy.context.view_layer.objects.active = merged_object
        texture_node.select = True
        nodes.active = texture_node
        
        for slot in merged_object.material_slots:
            slot.material = neutral_material
        
        print(f"[INFO] Starting Cycles bake...")
        
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.bake(type='COMBINED', margin=marginPixels-1, use_clear=False)
        
        # Cleanup
        merged_object.active_material = original_material
        
        nodes.remove(texture_node)
        
        obj.select_set(False)
        
        
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        total_seconds = time_difference.total_seconds()

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        print(f"Cycle light baking took: {hours:02d}:{minutes:02d}:{seconds:06.3f}")

        # Image post-processing
        
        print()
        
        temp_dir = bpy.app.tempdir
        lightmap_udim.filepath_raw = os.path.join(temp_dir, "Lightmap_<UDIM>.png")
        lightmap_udim.file_format = 'PNG'
        lightmap_udim.save()
        print(f"Saved UDIM tiles to {temp_dir} (e.g., Lightmap_1001.png, etc.)")

        
        
        # Process each tile individually
        for i in range(light_texture_index):
            tile_number = 1001 + i
            tile_filename = os.path.join(temp_dir, f"Lightmap_{tile_number}.png")
            
            sys.stdout.write(f"\rAdjusting lightness of texture {i+1}/{light_texture_index} (tile {tile_number})...")
            sys.stdout.flush()
            
            tile_image = bpy.data.images.load(tile_filename)
            
            # Set up the compositor for denoising
            tree, image_node, denoise_node, viewer_node = ExportBSP.setup_compositor_for_denoising()
            image_node.image = tile_image  # Assign the tile image to the compositor
            
            # Apply denoising and update the tile image with the result
            ExportBSP.apply_denoising_and_save(tile_image.name, tree, image_node, denoise_node, viewer_node)
            
            # Clean up compositor nodes
            tree.nodes.remove(image_node)
            tree.nodes.remove(denoise_node)
            tree.nodes.remove(viewer_node)
            
            # Apply D3D8 adjustment to the tile image
            ExportBSP.adjust_lightmap_for_d3d8(tile_image.name)
            
            # Save the processed tile back to disk
            tile_image.filepath_raw = tile_filename
            tile_image.file_format = 'PNG'
            tile_image.save()
            light_maps_references.append(tile_image)
            
            # Remove the tile image from Blender's data to free memory
            #bpy.data.images.remove(tile_image)

        # Finish the progress output
        print()  # New line after the loop
        
        # Further cleanup
        
        for obj, (hide_viewport, hide_render) in original_visibility.items():
            obj.hide_viewport = hide_viewport
            obj.hide_render = hide_render
        
        for slot, original_material in zip(merged_object.material_slots, original_materials):
            slot.material = original_material
        
        return material_light_data, light_maps_references
        
        #Ok, what do we have currently? We have light maps 100% baked. We also have divided material data, along with their texture ids.
        
    def vector_to_shorts(vec: Vector) -> tuple[int, int, int]:
        # Convert each component to a short (16-bit integer)
        x = int(vec.x)
        y = int(vec.y)
        z = int(vec.z)
        
        # Clamp the values to the range of a short
        x = max(-32768, min(32767, x))
        y = max(-32768, min(32767, y))
        z = max(-32768, min(32767, z))
        
        return (x, y, z)

    def process_mesh_object(exporting_object: bpy.types.Object) -> dict[int, dict]:
        """
        Given a mesh object, returns data ready to be used for BSP creation and exportation. Face windings are reversed in the process.
        """
        mesh: bpy.types.Mesh = exporting_object.data
        mesh_materials = mesh.materials
        mesh_vertices = mesh.vertices
        mesh_polygons = mesh.polygons
        mesh_loops = mesh.loops
        mesh_uvs = mesh.uv_layers["UVMap"].data
        light_mesh_uvs = mesh.uv_layers["LightUVMap"].data

        # dictionary to store mesh data for each material
        material_data: dict[int, dict] = {}

        # Step 1: Get original vertices
        original_mesh_vertices = [v.co for v in mesh_vertices]

        # Step 2: Group polygons by material index
        material_polygons: dict[int, list[bpy.types.MeshPolygon]] = {}
        for poly in mesh_polygons:
            material_index = poly.material_index
            if material_index not in material_polygons:
                material_polygons[material_index] = []
            material_polygons[material_index].append(poly)

        # Process each material group
        for material_index, polygons in material_polygons.items():
            exporter_vertices: list[Vector] = []
            exporter_uvs: list[tuple[float, float]] = []
            exporter_light_uvs: list[tuple[float, float]] = []
            exporter_polygons: list[list[int]] = []

            # Step 3: Construct collection of vertex index and UVs from loops for this material
            loop_data: list[tuple[int, tuple[float, float], tuple[float, float]]] = []
            for poly in polygons:
                for loop_index in poly.loop_indices:
                    loop = mesh_loops[loop_index]
                    uv = tuple(mesh_uvs[loop.index].uv) if mesh_uvs else (0.0, 0.0)
                    light_uv = tuple(light_mesh_uvs[loop.index].uv) if light_mesh_uvs else (0.0, 0.0)
                    loop_data.append((loop.vertex_index, uv, light_uv))

            # Step 4: Create unique vertices collection for this material
            unique_vertex_indices: list[int] = []
            unique_uvs: list[tuple[float, float]] = []
            unique_light_uvs: list[tuple[float, float]] = []
            seen_combinations = set()

            for vertex_index, uv, light_uv in loop_data:
                if (vertex_index, uv, light_uv) not in seen_combinations:
                    unique_vertex_indices.append(vertex_index)
                    unique_uvs.append(uv)
                    unique_light_uvs.append(light_uv)
                    seen_combinations.add((vertex_index, uv, light_uv))

            # Step 5: Build exporter vertex collection with unique vertex+uv combinations for this material
            mesh_vertex_to_export_vertex_map: dict[tuple[int, tuple[float, float], tuple[float, float]], int] = {}
            for i, (vertex_index, uv, light_uv) in enumerate(zip(unique_vertex_indices, unique_uvs, unique_light_uvs)):
                exporter_vertices.append(original_mesh_vertices[vertex_index]/SCALE_FACTOR)
                exporter_uvs.append(uv)
                exporter_light_uvs.append(light_uv)
                mesh_vertex_to_export_vertex_map[(vertex_index, uv, light_uv)] = i

            # Step 6: Rebuild polygon indices for this material
            for poly in polygons:
                poly_indices: list[int] = []
                for loop_index in poly.loop_indices:
                    loop = mesh_loops[loop_index]
                    uv = tuple(mesh_uvs[loop.index].uv) if mesh_uvs else (0.0, 0.0)
                    light_uv = tuple(light_mesh_uvs[loop.index].uv) if light_mesh_uvs else (0.0, 0.0)
                    poly_indices.append(mesh_vertex_to_export_vertex_map[(loop.vertex_index, uv, light_uv)])

                amount_of_polys = len(poly_indices)
                if amount_of_polys == 3:
                    # Reverse the triangle
                    exporter_polygons.append([poly_indices[2], poly_indices[1], poly_indices[0]])
                elif amount_of_polys == 4:
                    # Reverse both triangles
                    exporter_polygons.append([poly_indices[2], poly_indices[1], poly_indices[0]])
                    exporter_polygons.append([poly_indices[3], poly_indices[2], poly_indices[0]])
                else:
                    # Reverse each triangle in the fan
                    v0 = poly_indices[0]
                    for i in range(1, amount_of_polys - 1):
                        exporter_polygons.append([poly_indices[i + 1], poly_indices[i], v0])

            # Store the data for this material
            material_data[material_index] = {
                "material": mesh_materials[material_index] if material_index < len(mesh_materials) else None,
                "vertices": exporter_vertices,
                "uvs": exporter_uvs,
                "light_uvs": exporter_light_uvs,
                "polygons": exporter_polygons
            }
        return material_data
    
    def process_blender_material(blender_material: bpy.types.Material, r3m_materials: dict[str, tuple[R3MMaterial, int]], texture_dictionary) -> int:
        if blender_material.name in r3m_materials:
            return r3m_materials[blender_material.name][1]

        r3m_material = R3MMaterial()
        r3m_material.flag = 1
        r3m_material.name = blender_material.name
        r3m_material.detail_surface = -1
        r3m_material.detail_scale = 0.0

        nodes = blender_material.node_tree.nodes if blender_material.node_tree else []

        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                image_name = node.image.name
                if image_name not in texture_dictionary:
                    texture_dictionary[image_name] = len(texture_dictionary) + 1

                texture_layer = TextureLayer()
                texture_layer.texture_id = texture_dictionary[image_name]
                texture_layer.argb_color[1] = 1.0
                texture_layer.argb_color[2] = 1.0
                texture_layer.argb_color[3] = 1.0

                if blender_material.blend_method != 'OPAQUE':
                    pass
                    #texture_layer.alpha_type = AlphaType.DIRECT.value
                texture_layer.alpha_type = BlendMethod.NONE.value

                if MaterialProperties.ARGB_ALPHA.value in node:
                    texture_layer.argb_color[0] = node[MaterialProperties.ARGB_ALPHA.value]

                    flag = 0

                    if MaterialProperties.ENVIROMENT_MAT.value in node:
                        flag |= LayerFlag._MAT_ENV_BUMP.value

                    if MaterialProperties.METAL_EFFECT_SIZE.value in node:
                        texture_layer.metal_effect_size = node[MaterialProperties.METAL_EFFECT_SIZE.value]
                        flag |= LayerFlag._UV_METAL.value

                    if MaterialProperties.UV_ROTATION.value in node:
                        texture_layer.uv_rotation = node[MaterialProperties.UV_ROTATION.value]
                        flag |= LayerFlag._UV_ROTATE.value

                    if MaterialProperties.STARTING_SCALE.value in node:
                        texture_layer.uv_starting_scale = node[MaterialProperties.STARTING_SCALE.value]
                        flag |= LayerFlag._UV_SCALE.value

                    if MaterialProperties.ENDING_SCALE.value in node:
                        texture_layer.uv_ending_scale = node[MaterialProperties.ENDING_SCALE.value]

                    if MaterialProperties.SCALE_SPEED.value in node:
                        texture_layer.uv_scale_speed = node[MaterialProperties.SCALE_SPEED.value]

                    if MaterialProperties.LAVA_WAVE_RATE.value in node:
                        texture_layer.lava_wave_effect_rate = node[MaterialProperties.LAVA_WAVE_RATE.value]
                        flag |= LayerFlag._UV_LAVA.value

                    if MaterialProperties.LAVA_WAVE_SPEED.value in node:
                        texture_layer.lava_wave_effect_speed = node[MaterialProperties.LAVA_WAVE_SPEED.value]

                    if MaterialProperties.SCROLL_U.value in node:
                        texture_layer.scroll_u = node[MaterialProperties.SCROLL_U.value]
                        flag |= LayerFlag._UV_SCROLL_U.value

                    if MaterialProperties.SCROLL_V.value in node:
                        texture_layer.scroll_v = node[MaterialProperties.SCROLL_V.value]
                        flag |= LayerFlag._UV_SCROLL_V.value

                    texture_layer.flags = flag

                r3m_material.texture_layers.append(texture_layer)
        r3m_material.layer_num = len(r3m_material.texture_layers)
        
        material_id = len(r3m_materials)
        r3m_materials[blender_material.name] = (r3m_material, material_id)
        return material_id
    
    def calculate_bounding_box_and_middle(vertices: list[Vector]) -> tuple['Vector3Int', 'Vector3Int', Vector]:
        if not vertices:
            raise ValueError("The vertex list is empty.")

        # Initialize min and max coordinates with the first vertex
        min_coords = Vector3Int(int(vertices[0].x), int(vertices[0].y), int(vertices[0].z))
        max_coords = Vector3Int(int(vertices[0].x), int(vertices[0].y), int(vertices[0].z))

        # Iterate through the remaining vertices to update min and max coordinates
        for vertex in vertices[1:]:
            min_coords = Vector3Int(
                min(min_coords.x, int(vertex.x)),
                min(min_coords.y, int(vertex.y)),
                min(min_coords.z, int(vertex.z))
            )
            max_coords = Vector3Int(
                max(max_coords.x, int(vertex.x)),
                max(max_coords.y, int(vertex.y)),
                max(max_coords.z, int(vertex.z))
            )

        # Calculate the middle position
        material_position = Vector((
            (min_coords.x + max_coords.x) / 2.0,
            (min_coords.y + max_coords.y) / 2.0,
            (min_coords.z + max_coords.z) / 2.0
        ))

        return min_coords, max_coords, material_position
    
    def organize_vertex_data(bsp_vertex_data: dict[Vector, tuple[int, int]]) -> dict[int, Vector]:
        vertex_id_to_position = {}
        for position, (vertex_id, _) in bsp_vertex_data.items():
            vertex_id_to_position[vertex_id] = Vector(position)
        return vertex_id_to_position
    
    @dataclass
    class GridCell:
        face_ids: Set[int] = None
        
        def __post_init__(self):
            self.face_ids = set()

    class SpatialGrid:
        def __init__(self, bounds_min: Vector, bounds_max: Vector, face_count: int):
            self.bounds_min = bounds_min
            self.bounds_max = bounds_max
            self.dimensions = bounds_max - bounds_min
            self.total_faces = face_count
            
            # Calculate adaptive grid resolution
            volume = self.dimensions.x * self.dimensions.y * self.dimensions.z
            target_cell_volume = volume / (face_count / 10)  # Aim for ~10 faces per cell
            cell_size = math.pow(target_cell_volume, 1/3)
            
            self.resolution = [
                max(1, math.ceil(self.dimensions.x / cell_size)),
                max(1, math.ceil(self.dimensions.y / cell_size)),
                max(1, math.ceil(self.dimensions.z / cell_size))
            ]
            
            self.cell_size = Vector((
                self.dimensions.x / self.resolution[0],
                self.dimensions.y / self.resolution[1],
                self.dimensions.z / self.resolution[2]
            ))
            
            # Initialize grid
            self.grid = [[[ExportBSP.GridCell() for _ in range(self.resolution[2])]
                        for _ in range(self.resolution[1])]
                        for _ in range(self.resolution[0])]
            
            self.max_faces_per_cell = 0
            self.total_occupied_cells = 0
            self.face_density_stats_valid = False
        
        def get_cell_coords(self, point: Vector) -> Tuple[int, int, int]:
            relative_pos = point - self.bounds_min
            x = int(min(self.resolution[0] - 1, max(0, relative_pos.x / self.cell_size.x)))
            y = int(min(self.resolution[1] - 1, max(0, relative_pos.y / self.cell_size.y)))
            z = int(min(self.resolution[2] - 1, max(0, relative_pos.z / self.cell_size.z)))
            return x, y, z
        
        def register_triangle(self, face_id: int, vertices: List[Vector]):
            # Get bounding box of triangle
            min_coords = Vector((
                min(v.x for v in vertices),
                min(v.y for v in vertices),
                min(v.z for v in vertices)
            ))
            max_coords = Vector((
                max(v.x for v in vertices),
                max(v.y for v in vertices),
                max(v.z for v in vertices)
            ))
            
            # Get cell ranges
            min_cell = self.get_cell_coords(min_coords)
            max_cell = self.get_cell_coords(max_coords)
            
            # Register triangle in all potentially intersecting cells
            for x in range(min_cell[0], max_cell[0] + 1):
                for y in range(min_cell[1], max_cell[1] + 1):
                    for z in range(min_cell[2], max_cell[2] + 1):
                        cell_min = self.bounds_min + Vector((
                            x * self.cell_size.x,
                            y * self.cell_size.y,
                            z * self.cell_size.z
                        ))
                        cell_max = cell_min + self.cell_size
                        
                        # Only add if triangle actually intersects cell
                        if ExportBSP.triangle_box_intersection(vertices, cell_min, cell_max):
                            cell = self.grid[x][y][z]
                            was_empty = len(cell.face_ids) == 0
                            cell.face_ids.add(face_id)
                            
                            # Update statistics
                            if len(cell.face_ids) > self.max_faces_per_cell:
                                self.max_faces_per_cell = len(cell.face_ids)
                            if was_empty and len(cell.face_ids) > 0:
                                self.total_occupied_cells += 1
            self.face_density_stats_valid = True
        
        def get_faces_in_box(self, box_min: Vector, box_max: Vector) -> Set[int]:
            min_cell = self.get_cell_coords(box_min)
            max_cell = self.get_cell_coords(box_max)
            
            face_ids = set()
            for x in range(min_cell[0], max_cell[0] + 1):
                for y in range(min_cell[1], max_cell[1] + 1):
                    for z in range(min_cell[2], max_cell[2] + 1):
                        face_ids.update(self.grid[x][y][z].face_ids)
            return face_ids
        
        def get_total_cell_count(self) -> int:
            """Returns the total number of cells in the grid."""
            return self.resolution[0] * self.resolution[1] * self.resolution[2]
        
        def get_max_faces_per_cell(self) -> int:
            """Returns the number of faces in the most populated cell."""
            return self.max_faces_per_cell
        
        def get_average_faces_per_occupied_cell(self) -> float:
            """Returns the average number of faces per occupied cell."""
            if self.total_occupied_cells == 0:
                return 0
            return self.total_faces / self.total_occupied_cells
        
        def calculate_max_faces_per_leaf(self) -> int:
            """
            Calculates a suitable maximum number of faces per leaf based on grid statistics.
            """
            if not self.face_density_stats_valid:
                raise RuntimeError("Face density statistics not valid. Register faces first.")
            
            avg_faces_per_occupied = self.get_average_faces_per_occupied_cell()
            max_faces_in_cell = self.get_max_faces_per_cell()
            
            # Base the limit on a combination of average and maximum density
            # We multiply average by 4 as a leaf might cover multiple cells
            base_limit = min(
                max(int(avg_faces_per_occupied * 4), 32),  # Don't go below 32 faces
                int(max_faces_in_cell * 1.5),  # Don't exceed 150% of densest cell
                65535  # Hard maximum due to data type constraint
            )
            
            # Scale based on total map size
            if self.total_faces < 1000:
                base_limit = min(base_limit, self.total_faces // 4)
            elif self.total_faces > 100000:
                base_limit = min(base_limit * 2, 65535)
            
            return base_limit

    def calculate_dynamic_face_limit(grid: SpatialGrid) -> int:
        """
        Wrapper function to calculate the maximum faces per leaf.
        Also validates the result and provides a reasonable fallback.
        """
        try:
            limit = grid.calculate_max_faces_per_leaf()
            # Ensure we have a reasonable minimum
            return max(32, min(limit, 65535))
        except Exception as e:
            print(f"Warning: Error calculating dynamic face limit: {e}")
            # Fallback based on total faces
            if grid.total_faces < 1000:
                return 32
            elif grid.total_faces < 10000:
                return 128
            elif grid.total_faces < 100000:
                return 512
            else:
                return 2048

    def triangle_box_intersection(triangle_vertices: List[Vector], box_min: Vector, box_max: Vector) -> bool:
        # Implementation of Separating Axis Theorem for triangle-box intersection
        
        # Get box center and half-extents
        box_center = (box_max + box_min) * 0.5
        box_half_size = (box_max - box_min) * 0.5
        
        # Get triangle edges
        edges = [
            triangle_vertices[1] - triangle_vertices[0],
            triangle_vertices[2] - triangle_vertices[1],
            triangle_vertices[0] - triangle_vertices[2]
        ]
        
        # Triangle normal
        normal = edges[0].cross(edges[1]).normalized()
        
        # Test box axes
        box_axes = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))]
        
        # Transform triangle vertices to box space
        vertices_box_space = [v - box_center for v in triangle_vertices]
        
        # Test box faces
        for i in range(3):
            axis = box_axes[i]
            p = [v.dot(axis) for v in vertices_box_space]
            r = box_half_size[i]
            if min(p) > r or max(p) < -r:
                return False
        
        # Test triangle face
        p = [v.dot(normal) for v in vertices_box_space]
        r = sum(abs(box_half_size[i] * normal[i]) for i in range(3))
        if min(p) > r or max(p) < -r:
            return False
        
        # Test edge cross products
        for edge in edges:
            for box_axis in box_axes:
                axis = edge.cross(box_axis).normalized()
                if axis.length_squared < 1e-6:  # Skip if parallel
                    continue
                
                p = [v.dot(axis) for v in vertices_box_space]
                r = sum(abs(box_half_size[i] * axis[i]) for i in range(3))
                if min(p) > r or max(p) < -r:
                    return False
        
        return True

    def create_bsp_structure(
        bsp_vertex_data: dict[Vector, tuple[int, int]],
        bsp_vertex_ids: list[int],
        bsp_face_pointers: list[ReadFaceStruct],
        max_faces_per_leaf: int = 40
    ) -> tuple[list[BSPNode], list[BSPLeaf], list[int], list[int], list[tuple[Vector, float]]]:
        
        bsp_nodes: list[BSPNode] = [BSPNode(0, 0.0, 0, 0, Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0)))]
        bsp_leaves: list[BSPLeaf] = [BSPLeaf(0, 0, 0, 0, 0, Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0)))]
        collision_face_ids: list[int] = []
        material_list_in_leaf_ids: list[int] = []
        splitting_planes: list[tuple[Vector]] = [(Vector((0.0, 0.0, 0.0)))]
        
        vertex_id_to_position = ExportBSP.organize_vertex_data(bsp_vertex_data)
        all_face_ids = list(range(len(bsp_face_pointers)))
        
        # Create spatial acceleration grid
        vertices = []
        for face_id in all_face_ids:
            face_pointer = bsp_face_pointers[face_id]
            face_vertices = [vertex_id_to_position[vertex_id] for vertex_id in 
                            bsp_vertex_ids[face_pointer.vertex_start_id:face_pointer.vertex_start_id + face_pointer.vertex_amount]]
            vertices.extend(face_vertices)
        
        global_bb_min, global_bb_max = ExportBSP.calculate_bounding_box(vertices)
        grid = ExportBSP.SpatialGrid(global_bb_min, global_bb_max, len(all_face_ids))
        
        face_amount = len(all_face_ids)
        
        # Register all faces in the grid
        for i, face_id in enumerate(all_face_ids, start=1):
            sys.stdout.write(f"\rCreating bsp generation accelerator structure {i}/{face_amount}...")
            sys.stdout.flush()
            face_pointer = bsp_face_pointers[face_id]
            face_vertices = [vertex_id_to_position[vertex_id] for vertex_id in 
                            bsp_vertex_ids[face_pointer.vertex_start_id:face_pointer.vertex_start_id + face_pointer.vertex_amount]]
            grid.register_triangle(face_id, face_vertices)
        
        max_faces_per_leaf = ExportBSP.calculate_dynamic_face_limit(grid)
        print(f"Defined [{max_faces_per_leaf}] as the goal of maximum amount of faces per leaf in this map.")
        
        def recursive_bsp(working_face_ids: list[int], previous_front_faces_len: int = 0, previous_back_faces_len: int = 0, node_bb_min: Vector = None, node_bb_max: Vector = None, depth: int = 0, stall_count: List[int] = None) -> int:
            #print(f"Processing [{len(working_face_ids)}] faces for nodes [{len(bsp_nodes)}]ND/[{len(bsp_leaves)}]LF  at depth {depth}.")
            
            if stall_count is None:
                stall_count = [0]
            
            
            vertices_within_bounds = []
            faces_within_bounds = []
            vertices_test_collection = []
            
            for face_id in working_face_ids:
                face_pointer = bsp_face_pointers[face_id]
                face_vertices = [vertex_id_to_position[vertex_id] for vertex_id in bsp_vertex_ids[face_pointer.vertex_start_id:face_pointer.vertex_start_id + face_pointer.vertex_amount]]
                should_append_face = False
                for vertex in face_vertices:
                    if (node_bb_min[0] <= vertex[0] <= node_bb_max[0] or
                        node_bb_min[1] <= vertex[1] <= node_bb_max[1] or
                        node_bb_min[2] <= vertex[2] <= node_bb_max[2]):
                        vertices_within_bounds.append(vertex)
                        should_append_face = True
                if should_append_face:
                    vertices_test_collection.append(face_vertices)
                    faces_within_bounds.append(face_id)
            
            
            
            plane_normal, plane_distance, plane_index = ExportBSP.find_optimal_splitting_plane(node_bb_min, node_bb_max, splitting_planes, vertices_test_collection)
            front_faces = []
            back_faces = []

            for face_id in faces_within_bounds:
                face_pointer = bsp_face_pointers[face_id]
                face_vertices = [vertex_id_to_position[vertex_id] for vertex_id in bsp_vertex_ids[face_pointer.vertex_start_id:face_pointer.vertex_start_id + face_pointer.vertex_amount]]

                # Compute distances of all vertices in the face
                distances = [Vector(v).dot(plane_normal) - plane_distance for v in face_vertices]
                distance_sum = sum(distances)

                if all(d >= 0 for d in distances):  # Fully in front
                    front_faces.append(face_id)
                elif all(d <= 0 for d in distances):  # Fully in back
                    back_faces.append(face_id)
                else:  # Straddles the plane
                    if distance_sum > 0:
                        front_faces.append(face_id)  # Assign to the front
                    else:
                        back_faces.append(face_id)  # Assign to the back
            
            if (len(front_faces) == previous_front_faces_len and len(back_faces) == previous_back_faces_len):
                stall_count[0] += 1
            else:
                stall_count[0] = 0
            
            #print(f"Processing node at depth {depth} || BSPNLEN BSPLLEN: [{len(bsp_nodes)}/{len(bsp_leaves)}] || Faces intersecting us:[{len(faces_within_bounds)}] \nBounding Box:[{node_bb_min}]/[{node_bb_max}] || Stall count: [{stall_count}]|| Front faces[{len(front_faces)}] || Back faces[{len(back_faces)}]")
            
            if len(faces_within_bounds) <= max_faces_per_leaf or ((not front_faces or not back_faces) and depth>100):
                intersecting_face_ids = grid.get_faces_in_box(node_bb_min, node_bb_max)
                vertices_within_bounds.clear()
                faces_within_bounds.clear()
                
                for face_id in intersecting_face_ids:
                    face_pointer = bsp_face_pointers[face_id]
                    face_vertices = [vertex_id_to_position[vertex_id] for vertex_id in 
                                bsp_vertex_ids[face_pointer.vertex_start_id:face_pointer.vertex_start_id + face_pointer.vertex_amount]]
                    
                    if ExportBSP.triangle_box_intersection(face_vertices, node_bb_min, node_bb_max):
                        faces_within_bounds.append(face_id)
                
                
                leaf_id = len(bsp_leaves)
                face_start_id = len(collision_face_ids)
                collision_face_ids.extend(faces_within_bounds)

                material_ids = set()
                for face_id in faces_within_bounds:
                    face_pointer = bsp_face_pointers[face_id]
                    material_ids.add(face_pointer.material_id)
                material_group_start_id = len(material_list_in_leaf_ids)
                material_list_in_leaf_ids.extend(material_ids)

                bsp_leaf = BSPLeaf(
                    type=0,
                    face_amount=len(faces_within_bounds),
                    face_start_id=face_start_id,
                    material_group_amount=len(material_ids),
                    material_group_start_id=material_group_start_id,
                    bounding_box_min=node_bb_min,
                    bounding_box_max=node_bb_max
                )
                bsp_leaves.append(bsp_leaf)
                return -leaf_id -1 # Negative value indicates a leaf
            
            # Create the node
            node_id = len(bsp_nodes)
            bsp_node = BSPNode(
                plane_id=plane_index,
                distance=plane_distance,
                front_id=-1,  # Temporarily invalid; updated after recursion
                back_id=-1,   # Temporarily invalid; updated after recursion
                bounding_box_min=node_bb_min,
                bounding_box_max=node_bb_max
            )
            bsp_nodes.append(bsp_node)  # Add the node immediately

            front_bb_min, front_bb_max, back_bb_min, back_bb_max = ExportBSP.split_bounding_box(node_bb_min, node_bb_max, plane_normal, plane_distance)
            
            front_id = recursive_bsp(front_faces,len(front_faces), len(back_faces), front_bb_min, front_bb_max, depth + 1)
            back_id = recursive_bsp(back_faces,len(front_faces), len(back_faces), back_bb_min, back_bb_max, depth + 1)

            bsp_node.front_id = front_id
            bsp_node.back_id = back_id
            return node_id  # Positive value indicates a node

        # Start the recursive BSP construction
        
        recursive_bsp(all_face_ids, node_bb_min=global_bb_min, node_bb_max=global_bb_max)

        return bsp_nodes, bsp_leaves, collision_face_ids, material_list_in_leaf_ids, splitting_planes
    
    def split_bounding_box(node_bb_min: Vector, node_bb_max: Vector, plane_normal: Vector, plane_distance: float):
        # Determine the dominant axis of the plane normal
        dominant_axis = max(range(3), key=lambda i: abs(plane_normal[i]))

        # Calculate the intersection point along the dominant axis
        intersection = (plane_distance - sum(plane_normal[i] * node_bb_min[i] for i in range(3) if i != dominant_axis)) / plane_normal[dominant_axis]

        # Create front and back bounding boxes
        front_bb_min = node_bb_min.copy()
        front_bb_max = node_bb_max.copy()
        back_bb_min = node_bb_min.copy()
        back_bb_max = node_bb_max.copy()

        # Adjust the bounding boxes based on the intersection point
        if plane_normal[dominant_axis] > 0:
            front_bb_min[dominant_axis] = intersection
            back_bb_max[dominant_axis] = intersection
        else:
            front_bb_max[dominant_axis] = intersection
            back_bb_min[dominant_axis] = intersection
        return front_bb_min, front_bb_max, back_bb_min, back_bb_max
    
    def calculate_bounding_box(vertices: list[Vector]) -> tuple[Vector, Vector]:
        """Calculate the bounding box for a list of vertices."""
        if not vertices:
            return Vector((0, 0, 0)), Vector((0, 0, 0))
        
        # Initialize min and max coordinates with the first vertex
        min_coords = Vector(vertices[0])
        max_coords = Vector(vertices[0])
        
        # Iterate through the remaining vertices to update min and max coordinates
        for vertex in vertices[1:]:
            min_coords = Vector((
                min(min_coords[0], vertex[0]),
                min(min_coords[1], vertex[1]),
                min(min_coords[2], vertex[2])
            ))
            max_coords = Vector((
                max(max_coords[0], vertex[0]),
                max(max_coords[1], vertex[1]),
                max(max_coords[2], vertex[2])
            ))
        
        return min_coords, max_coords
    
    def calculate_box_aspect_ratio(box_min: Vector, box_max: Vector) -> float:
        dimensions = box_max - box_min
        max_dim = max(dimensions)
        if max_dim == 0:
            return float('inf')
        min_dim = min(dim for dim in dimensions if dim != 0)
        return max_dim / min_dim
    
    def calculate_split_quality(
        parent_box_min: Vector, 
        parent_box_max: Vector,
        front_box_min: Vector, 
        front_box_max: Vector,
        back_box_min: Vector, 
        back_box_max: Vector,
        front_faces_count: int,
        back_faces_count: int,
        total_faces: int
    ) -> float:
        # Calculate volumes
        parent_volume = (parent_box_max - parent_box_min).x * (parent_box_max - parent_box_min).y * (parent_box_max - parent_box_min).z
        front_volume = (front_box_max - front_box_min).x * (front_box_max - front_box_min).y * (front_box_max - front_box_min).z
        back_volume = (back_box_max - back_box_min).x * (back_box_max - back_box_min).y * (back_box_max - back_box_min).z
        
        # Volume balance score (0-1, higher is better)
        volume_ratio = min(front_volume, back_volume) / max(front_volume, back_volume) if max(front_volume, back_volume) > 0 else 0
        
        # Volume reduction score (0-1, higher is better)
        volume_reduction = 1.0 - ((front_volume + back_volume) / (2 * parent_volume))
        
        # Aspect ratio score (0-1, higher is better)
        parent_aspect = ExportBSP.calculate_box_aspect_ratio(parent_box_min, parent_box_max)
        front_aspect = ExportBSP.calculate_box_aspect_ratio(front_box_min, front_box_max)
        back_aspect = ExportBSP.calculate_box_aspect_ratio(back_box_min, back_box_max)
        aspect_improvement = (parent_aspect - min(front_aspect, back_aspect)) / parent_aspect
        
        # Face distribution score (0-1, higher is better)
        face_balance = min(front_faces_count, back_faces_count) / max(front_faces_count, back_faces_count) if max(front_faces_count, back_faces_count) > 0 else 0
        
        aspect_weight = 0.4
        volume_weight = 0.3
        reduction_weight = 0.2
        face_weight = 0.1
        
        # Increase face weight if we have many faces
        if total_faces > 10000:
            face_factor = min(1.0, (total_faces - 10000) / 55535)  # Gradually increase importance
            aspect_weight *= (1.0 - face_factor * 0.5)
            face_weight += face_factor * 0.3
        
        return (
            aspect_weight * aspect_improvement +
            volume_weight * volume_ratio +
            reduction_weight * volume_reduction +
            face_weight * face_balance
        )
    
    def find_optimal_splitting_plane(
        box_min_coords: Vector, 
        box_max_coords: Vector,
        splitting_planes: list[tuple[Vector]],
        face_vertices: list[list[Vector]],  # List of face vertices for testing splits
        force_split: bool = False
    ) -> tuple[Vector, float, int]:
        """Enhanced splitting plane selection with spatial awareness."""
        best_score = -float('inf')
        best_normal = None
        best_distance = None
        best_index = None
        
        dimensions = box_max_coords - box_min_coords
        axes = list(range(3))
        
        # Sort axes by dimension length for better initial guesses
        axes.sort(key=lambda i: dimensions[i], reverse=True)
        
        for axis in axes:
            normal = Vector((0, 0, 0))
            normal[axis] = 1
            
            # Try multiple split positions for each axis
            split_positions = [
                box_min_coords[axis] + dimensions[axis] * ratio 
                for ratio in [0.5, 0.33, 0.66] if dimensions[axis] > 0
            ]
            
            for distance in split_positions:
                # Count faces on each side
                front_faces = []
                back_faces = []
                
                for face_verts in face_vertices:
                    distances = [Vector(v).dot(normal) - distance for v in face_verts]
                    if all(d >= 0 for d in distances):
                        front_faces.append(face_verts)
                    elif all(d <= 0 for d in distances):
                        back_faces.append(face_verts)
                    else:
                        if sum(distances) > 0:
                            front_faces.append(face_verts)
                        else:
                            back_faces.append(face_verts)
                
                if force_split and (not front_faces or not back_faces):
                    continue
                
                # Calculate split boxes
                front_min, front_max, back_min, back_max = ExportBSP.split_bounding_box(
                    box_min_coords, box_max_coords, normal, distance)
                
                score = ExportBSP.calculate_split_quality(
                    box_min_coords, box_max_coords,
                    front_min, front_max,
                    back_min, back_max,
                    len(front_faces), len(back_faces),
                    len(face_vertices)
                )
                
                if score > best_score:
                    best_score = score
                    best_normal = normal
                    best_distance = distance
                    plane_key = (normal)
                    if plane_key in splitting_planes:
                        best_index = splitting_planes.index(plane_key)
                    else:
                        splitting_planes.append(plane_key)
                        best_index = len(splitting_planes) - 1
        
        if best_normal is None and force_split:
            # Fallback for forced splits: just split in middle of longest axis
            axis = axes[0]
            best_normal = Vector((0, 0, 0))
            best_normal[axis] = 1
            best_distance = (box_min_coords[axis] + box_max_coords[axis]) / 2
            plane_key = (best_normal)
            if plane_key in splitting_planes:
                best_index = splitting_planes.index(plane_key)
            else:
                splitting_planes.append(plane_key)
                best_index = len(splitting_planes) - 1
        
        return best_normal, best_distance, best_index


    
def menu_func_import(self, context):
    self.layout.operator(ImportBSP.bl_idname, text="BSP (.BSP)")

def menu_func_export(self, context):
    pass
    #self.layout.operator(ExportBSP.bl_idname, text="BSP (.BSP)")

def register():
    bpy.utils.register_class(ImportBSP)
    bpy.utils.register_class(CBB_FH_ImportBSP)
    #bpy.utils.register_class(ExportBSP)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ImportBSP)
    bpy.utils.unregister_class(CBB_FH_ImportBSP)
    #bpy.utils.unregister_class(ExportBSP)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
