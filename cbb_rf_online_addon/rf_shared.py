from __future__ import annotations
import bpy
import struct
import traceback
from bpy.types import Operator, Context, Event
from mathutils import Vector, Quaternion, Matrix, Euler
import os
from enum import Enum
import io
from .utils import Vector3Int, Utils
import tempfile
from typing import Tuple
import numpy as np
from PIL import Image
from . import texture_utils

SCALE_FACTOR = 1

class RFShared(Operator):
    bl_idname = "cbb.rf_shared"
    bl_label = "RF Shared"
    bl_options = {'PRESET', 'UNDO'}
    
    @staticmethod
    def unlock_dds(buffer):
        password = bytes([
            0x2E, 0x80, 0x4D, 0x76, 0x2E, 0xF8, 0xD1, 0xF0, 0xBD, 0x3F, 0x86, 0x81, 0x58, 0x2C, 0x3F, 0x3F, 
            0x2E, 0x2E, 0x67, 0x6F, 0x3F, 0x40, 0x3F, 0x78, 0x3C, 0x3F, 0xF1, 0xC0, 0xA5, 0xF6, 0x3B, 0x9F, 
            0xC1, 0x20, 0x3F, 0xD7, 0xC8, 0xC1, 0xE9, 0x85, 0x86, 0xBD, 0xEF, 0x56, 0x3F, 0xA1, 0xFB, 0x2E, 
            0x87, 0x86, 0x61, 0x4C, 0x21, 0x3B, 0x4E, 0xB4, 0x78, 0x57, 0xAE, 0x97, 0x3F, 0x2E, 0x4A, 0x2E, 
            0x3F, 0x4C, 0x2E, 0x44, 0xCD, 0xC5, 0x5F, 0xE8, 0xE9, 0xEC, 0xEB, 0xBD, 0xBE, 0xBB, 0xF7, 0x6C, 
            0x2E, 0xF2, 0xE4, 0x2E, 0x3F, 0x3F, 0x97, 0x9F, 0x9D, 0xB3, 0x21, 0xB9, 0x76, 0x65, 0x54, 0x3F, 
            0xE6, 0xF6, 0xC6, 0xF0, 0x79, 0xDB, 0xE2, 0xB2, 0x4B, 0x2E, 0x2E, 0xEB, 0xD3, 0xD3, 0xCA, 0xAB, 
            0xEA, 0xC7, 0xED, 0x9C, 0xC7, 0xD9, 0xD0, 0x65, 0x48, 0xB4, 0xFA, 0x35, 0x2E, 0x2E, 0x6A, 0x9B, 
            #0xAF, 0x7E, 0xD6, 0xB7, 0x79, 
        ])
        password = struct.unpack('<32I', password)
        for i in range(32):
            buffer[i] ^= password[i]
        return buffer
    
    @staticmethod
    def get_materials_from_r3m_file(directory, file_name_stem)-> list[R3MMaterial]:
        """
        Returns a R3MMaterial collection with all the materials in the R3M file, given a directory and a file name to search for.
        """
        r3m_filepath = os.path.join(directory, file_name_stem + ".r3m")
        try:
            with open(r3m_filepath, 'rb') as r3m_file:
                return RFShared.get_materials_from_r3m_filestream(r3m_file)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"R3M file not found at: {r3m_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3M file at {r3m_filepath}") from e
        
    @staticmethod
    def get_materials_from_r3m_filestream(r3m_file: io.BufferedReader)-> list[R3MMaterial]:
        """
        Returns a R3MMaterial collection with all the materials in the R3M file from an already opened file.
        """
        try:
            reader = Utils.Serializer(r3m_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor)
            version = reader.read_float()
            if abs(version - 1.1) > 0.01:
                print(f"Warning: R3M file version [{version}] is different than the version [1.1] this addon was built in mind with.")
                
            material_amount = reader.read_uint()
            materials: list[R3MMaterial] = []
            for a in range(material_amount):
                material = R3MMaterial()
                material.layer_num = reader.read_uint()
                material.flag = reader.read_uint()
                material.detail_surface = reader.read_int()
                material.detail_scale = reader.read_float()
                material.name = reader.read_fixed_string(128, "euc-kr")
                for b in range(material.layer_num):
                    layer = reader.read_values("h i I I I h h h h h h h h h h H h h h", 46)
                    texture_layer = TextureLayer.get_texture_layer_from_unpacked_bytes(layer)
                    material.texture_layers.append(texture_layer)
                materials.append(material)
            return materials
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3M file data: {e}\n{traceback.format_exc()}")
    
    @staticmethod
    def get_color_texture_dictionary_from_r3t_file(directory, file_name_stem) -> dict[int, str]:
        r3t_filepath = os.path.join(directory, file_name_stem + ".r3t")
        try:
            with open(r3t_filepath, 'rb') as r3t_file:
                return RFShared.get_color_texture_dictionary_from_r3t_filestream(r3t_file)
    
        except FileNotFoundError:
            raise FileNotFoundError(f"R3T file not found at: {r3t_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file at {r3t_filepath}") from e
        finally:
            print()
            
    @staticmethod
    def get_color_texture_dictionary_from_r3t_filestream(r3t_file: io.BufferedReader) -> dict[int, str]:
        try:
            texture_dictionary: dict = {}
            reader = Utils.Serializer(r3t_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor)
            version = reader.read_float()
            if abs(version - 1.2) > 0.01:
                print(f"Warning: R3T file version [{version}] is different than the version [1.2] this addon was built in mind with.")
                
            texture_amount = reader.read_uint()
            
            texture_paths: list[str] = []
            for _ in range(texture_amount):
                texture_paths.append(reader.read_fixed_string(128, "euc-kr"))
            
            # TextureLayer.texture_id starts at 1 instead of 0
            for texture_id, texture_path in enumerate(texture_paths, start=1):
                print(f"\rLoading textures {texture_id}/{texture_amount}", end="")
                texture_name = os.path.basename(texture_path)
                size = reader.read_uint()
                dds_header = bytearray(r3t_file.read(128))
                if not dds_header[:4] == b'DDS ':
                    dds_header = RFShared.unlock_dds(list(struct.unpack('<32I', dds_header)))
                    dds_header = struct.pack('<32I', *dds_header)
                texture_data = r3t_file.read(size - 128)

                temp_file = tempfile.NamedTemporaryFile(suffix=".dds", delete=False)
                temp_file.write(dds_header)
                temp_file.write(texture_data)
                temp_file.flush()
                blender_image = bpy.data.images.load(temp_file.name)
                blender_image.name = texture_name
                blender_image.pack()
                temp_file.close()
                os.remove(temp_file.name)

                texture_dictionary[texture_id] = texture_name
            print()
            return texture_dictionary
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file data: {e}\n{traceback.format_exc()}")
        finally:
            print()
            
    @staticmethod
    def get_light_texture_dictionary_from_r3t_file(directory, file_name_stem) -> dict[int, str]:
        r3t_filepath = os.path.join(directory, f"{file_name_stem}Lgt.r3t")
        try:
            with open(r3t_filepath, 'rb') as r3t_file:
                return RFShared.get_light_texture_dictionary_from_r3t_filestream(r3t_file)
    
        except FileNotFoundError:
            raise FileNotFoundError(f"R3T file not found at: {r3t_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file at {r3t_filepath}") from e
        finally:
            print()
            
    @staticmethod
    def get_light_texture_dictionary_from_r3t_filestream(r3t_file: io.BufferedReader) -> dict[int, str]:
        try:
            texture_dictionary: dict = {}
            reader = Utils.Serializer(
                r3t_file,
                Utils.Serializer.Endianness.Little,
                Utils.Serializer.Quaternion_Order.XYZW,
                Utils.Serializer.Matrix_Order.ColumnMajor,
            )
            version = reader.read_float()
            texture_amount = reader.read_uint()

            for texture_id in range(texture_amount):
                print(f"\rLoading textures {texture_id + 1}/{texture_amount}", end="")
                texture_name = f"LightTexture{texture_id}.dds"
                size = reader.read_uint()
                dds_header = bytearray(r3t_file.read(128))

                if not dds_header[:4] == b'DDS ':
                    dds_header = RFShared.unlock_dds(list(struct.unpack('<32I', dds_header)))
                    dds_header = struct.pack('<32I', *dds_header)

                texture_data = r3t_file.read(size - 128)
                width, height = struct.unpack('<II', dds_header[12:20])

                # Decode RGB565
                raw_data = np.frombuffer(texture_data, dtype=np.uint16).reshape((height, width))
                r = ((raw_data & 0xF800) >> 11) << 3
                g = ((raw_data & 0x07E0) >> 5) << 2
                b = (raw_data & 0x001F) << 3
                rgb_data = np.stack((r, g, b), axis=-1).astype(np.uint8)

                # Save as a new DDS
                pil_image = Image.fromarray(rgb_data, mode="RGB")
                with tempfile.NamedTemporaryFile(suffix=".dds", delete=False) as temp_file:
                    pil_image.save(temp_file.name, format="DDS")
                    blender_image = bpy.data.images.load(temp_file.name)
                    blender_image.name = texture_name
                    blender_image.pack()
                    temp_file.close()
                    os.remove(temp_file.name)

                texture_dictionary[texture_id] = texture_name
            print()
            return texture_dictionary
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file data: {e}\n{traceback.format_exc()}")
        finally:
            print()
    
    @staticmethod
    def process_texture_layers(r3m_material: R3MMaterial, material: bpy.types.Material, 
                           nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, 
                           bsdf: bpy.types.Node,
                           texture_dictionary: dict[int, str], context: bpy.types.Context, debug_object_name = ""):
        
        material.surface_render_method = "DITHERED"
        material.use_transparency_overlap = False
        
        bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs['Emission Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs['Alpha'].default_value = 1.0
        
        specular_value_node = nodes.new(type="ShaderNodeValue")
        specular_value_node.name = "Material_Specular"
        specular_value_node.label = "Specular"
        specular_value_node.outputs[0].default_value = 0.0
        links.new(specular_value_node.outputs[0], bsdf.inputs['Specular IOR Level'])

        # --- Accumulator Sockets ---
        # These will hold the current combined result for each BSDF input stream
        # Start with 'neutral' values (black for color/emission, fully transparent for alpha if needed, or opaque)
        
        # For Base Color stream
        # Initialize with a black color node. If the first layer is opaque, it will replace this.
        # If all layers are e.g. additive, the base color will remain black.
        black_color_node = nodes.new(type="ShaderNodeRGB")
        black_color_node.name = "Initial_Base_Color"
        black_color_node.outputs[0].default_value = (0,0,0,1)
        accumulated_base_color_socket = black_color_node.outputs['Color']

        # For Alpha stream (overall material coverage/transparency)
        # Initialize with fully opaque. If any layer defines transparency, this will be updated.
        opaque_alpha_node = nodes.new(type="ShaderNodeValue")
        opaque_alpha_node.name = "Initial_Alpha"
        opaque_alpha_node.outputs[0].default_value = 1.0
        accumulated_alpha_socket = opaque_alpha_node.outputs[0]
        is_alpha_set_by_layer_zero = False
        
        # For Emission Color stream
        # Initialize with a black color node. Additive layers will add to this.
        black_emission_node = nodes.new(type="ShaderNodeRGB")
        black_emission_node.name = "Initial_Emission_Color"
        black_emission_node.outputs[0].default_value = (0,0,0,1)
        accumulated_emission_color_socket = black_emission_node.outputs['Color']

        


        texture_coordinates_node = nodes.new(type="ShaderNodeTexCoord")
        # Important for mat env bump effect: This effect is special because it seems to be used only once in the entire game, and that's for water. 
        # In all these cases, it's expected that layer 0 is a base texture layer, and layer 1 is the layer for the mat env bump. In the case that this effect is present, all other layers are ignored.
        # The effect is also special because it messes with the uvs of another layer(effect in layer 1 messes with uv of layer 0), something that does not happen with other effects.
        first_layer_uv_output = None
        first_layer_tex_image_node = None
        

        zero_strength_node = nodes.new(type="ShaderNodeValue")
        zero_strength_node.name = "Initial_Emission_Strength"
        zero_strength_node.outputs[0].default_value = 0.0 
        accumulated_emission_strength_socket = zero_strength_node.outputs[0]

        for i, texture_layer in enumerate(r3m_material.texture_layers):
            
            image = bpy.data.images[texture_dictionary[texture_layer.texture_id]]
            tex_image_node: bpy.types.TextureNodeImage = nodes.new('ShaderNodeTexImage')
            tex_image_node.name = f"Tex_Image_Layer_{i}"
            tex_image_node.image = image
            
            tex_image_node[MaterialProperties.ALPHA_TYPE.value] = texture_layer.alpha_type
            
            uvs_output = texture_coordinates_node.outputs["UV"]
            
            if texture_layer.flags & int(LayerFlag._UV_METAL.value):
                tex_image_node[MaterialProperties.METAL_EFFECT_SIZE.value] = texture_layer.metal_effect_size # Store raw short m_sUVMetal

                su_node = nodes.new(type="ShaderNodeValue")
                su_node.name = f"Metal_su_L{i}"
                su_node.label = f"Metal Divisor (su) L{i}"
                Utils.create_driver_single(
                    su_node.outputs[0], 
                    "raw_sUVMetal",
                    material, 
                    f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.METAL_EFFECT_SIZE.value}"]',
                    "max(raw_sUVMetal / 2.0, 0.0001)"
                )
                su_node.outputs[0].default_value = max(float(texture_layer.metal_effect_size) / 2.0, 0.0001)
                su_socket = su_node.outputs[0]

                geometry_node = nodes.new(type="ShaderNodeNewGeometry")
                geometry_node.name = f"Geom_Metal_L{i}"
                geometry_node.label = f"Geometry Metal L{i}"
                p_world_socket = geometry_node.outputs["Position"] 

                camera_world_pos_node = nodes.new(type="ShaderNodeVectorTransform")
                camera_world_pos_node.name = f"CamWorldPos_Metal_L{i}"
                camera_world_pos_node.label = f"Camera World Position [{image.name}]"
                camera_world_pos_node.vector_type = "POINT"
                camera_world_pos_node.convert_from = "CAMERA"
                camera_world_pos_node.convert_to = "WORLD"
                camera_world_pos_node.inputs[0].default_value = (0.0, 0.0, 0.0)
                eye_world_socket = camera_world_pos_node.outputs['Vector']

                world_face_normal_socket = geometry_node.outputs["True Normal"] 


                separate_normal_node = nodes.new(type='ShaderNodeSeparateXYZ')
                separate_normal_node.name = f"SeparateNormal_Metal_L{i}"
                links.new(world_face_normal_socket, separate_normal_node.inputs['Vector'])
                normal_z_socket = separate_normal_node.outputs['Z']
                normal_y_socket = separate_normal_node.outputs['Y']
                normal_x_socket = separate_normal_node.outputs['X']

                abs_normal_z_node = nodes.new(type='ShaderNodeMath')
                abs_normal_z_node.name = f"AbsNormalZ_Metal_L{i}"
                abs_normal_z_node.operation = 'ABSOLUTE'
                links.new(normal_z_socket, abs_normal_z_node.inputs[0])

                compare_normal_z_node = nodes.new(type='ShaderNodeMath')
                compare_normal_z_node.name = f"CompareNormalZ_Metal_L{i}"
                compare_normal_z_node.operation = 'GREATER_THAN'
                links.new(abs_normal_z_node.outputs[0], compare_normal_z_node.inputs[0]) 
                compare_normal_z_node.inputs[1].default_value = 0.98

                s_world_path1_node = nodes.new(type='ShaderNodeCombineXYZ')
                s_world_path1_node.name = f"S_World_Path1_L{i}"
                s_world_path1_node.inputs['X'].default_value = 1.0

                s_world_path2_y_neg_node = nodes.new(type='ShaderNodeMath')
                s_world_path2_y_neg_node.operation = 'MULTIPLY'
                links.new(normal_y_socket, s_world_path2_y_neg_node.inputs[0])
                s_world_path2_y_neg_node.inputs[1].default_value = -1.0

                s_world_path2_combine_node = nodes.new(type='ShaderNodeCombineXYZ')
                s_world_path2_combine_node.name = f"S_World_Path2_Combine_L{i}"
                links.new(s_world_path2_y_neg_node.outputs['Value'], s_world_path2_combine_node.inputs['X'])
                links.new(normal_x_socket, s_world_path2_combine_node.inputs['Y'])

                s_world_path2_normalize_node = nodes.new(type='ShaderNodeVectorMath')
                s_world_path2_normalize_node.name = f"S_World_Path2_Norm_L{i}"
                s_world_path2_normalize_node.operation = 'NORMALIZE'
                links.new(s_world_path2_combine_node.outputs['Vector'], s_world_path2_normalize_node.inputs[0])

                s_world_mix_node = nodes.new(type='ShaderNodeMix')
                s_world_mix_node.name = f"S_World_Mix_L{i}"
                s_world_mix_node.data_type = 'VECTOR'
                links.new(compare_normal_z_node.outputs[0], s_world_mix_node.inputs[0])
                links.new(s_world_path2_normalize_node.outputs['Vector'], s_world_mix_node.inputs[4])
                links.new(s_world_path1_node.outputs['Vector'], s_world_mix_node.inputs[5])
                s_world_socket = s_world_mix_node.outputs["Result"]

                t_world_cross_node = nodes.new(type='ShaderNodeVectorMath')
                t_world_cross_node.name = f"T_World_Cross_L{i}"
                t_world_cross_node.operation = 'CROSS_PRODUCT'
                links.new(s_world_socket, t_world_cross_node.inputs[0])
                links.new(world_face_normal_socket, t_world_cross_node.inputs[1])
                
                t_world_normalize_node = nodes.new(type='ShaderNodeVectorMath')
                t_world_normalize_node.name = f"T_World_Normalize_L{i}"
                t_world_normalize_node.operation = 'NORMALIZE'
                links.new(t_world_cross_node.outputs['Vector'], t_world_normalize_node.inputs[0])
                t_world_socket = t_world_normalize_node.outputs['Vector']


                dot_s_p_node = nodes.new(type="ShaderNodeVectorMath"); dot_s_p_node.name=f"DotSP_Metal_L{i}"; dot_s_p_node.operation = 'DOT_PRODUCT'
                links.new(s_world_socket, dot_s_p_node.inputs[0])
                links.new(p_world_socket, dot_s_p_node.inputs[1])

                dot_s_eye_node = nodes.new(type="ShaderNodeVectorMath"); dot_s_eye_node.name=f"DotSEye_Metal_L{i}"; dot_s_eye_node.operation = 'DOT_PRODUCT'
                links.new(s_world_socket, dot_s_eye_node.inputs[0])
                links.new(eye_world_socket, dot_s_eye_node.inputs[1])

                mult_s_eye_08_node = nodes.new(type="ShaderNodeMath"); mult_s_eye_08_node.name=f"MultSEye08_Metal_L{i}"; mult_s_eye_08_node.operation = 'MULTIPLY'
                links.new(dot_s_eye_node.outputs['Value'], mult_s_eye_08_node.inputs[0])
                mult_s_eye_08_node.inputs[1].default_value = 0.8

                sub_u_num_node = nodes.new(type="ShaderNodeMath"); sub_u_num_node.name=f"SubUNum_Metal_L{i}"; sub_u_num_node.operation = 'SUBTRACT'
                links.new(dot_s_p_node.outputs['Value'], sub_u_num_node.inputs[0])
                links.new(mult_s_eye_08_node.outputs['Value'], sub_u_num_node.inputs[1])

                u_coord_node = nodes.new(type="ShaderNodeMath"); u_coord_node.name=f"U_Coord_Metal_L{i}"; u_coord_node.operation = 'DIVIDE'
                links.new(sub_u_num_node.outputs['Value'], u_coord_node.inputs[0])
                links.new(su_socket, u_coord_node.inputs[1])
                u_coord_socket = u_coord_node.outputs['Value']

                dot_t_p_node = nodes.new(type="ShaderNodeVectorMath"); dot_t_p_node.name=f"DotTP_Metal_L{i}"; dot_t_p_node.operation = 'DOT_PRODUCT'
                links.new(t_world_socket, dot_t_p_node.inputs[0])
                links.new(p_world_socket, dot_t_p_node.inputs[1])

                dot_t_eye_node = nodes.new(type="ShaderNodeVectorMath"); dot_t_eye_node.name=f"DotTEye_Metal_L{i}"; dot_t_eye_node.operation = 'DOT_PRODUCT'
                links.new(t_world_socket, dot_t_eye_node.inputs[0])
                links.new(eye_world_socket, dot_t_eye_node.inputs[1])

                mult_t_eye_08_node = nodes.new(type="ShaderNodeMath"); mult_t_eye_08_node.name=f"MultTEye08_Metal_L{i}"; mult_t_eye_08_node.operation = 'MULTIPLY'
                links.new(dot_t_eye_node.outputs['Value'], mult_t_eye_08_node.inputs[0])
                mult_t_eye_08_node.inputs[1].default_value = 0.8

                sub_v_num_node = nodes.new(type="ShaderNodeMath"); sub_v_num_node.name=f"SubVNum_Metal_L{i}"; sub_v_num_node.operation = 'SUBTRACT'
                links.new(dot_t_p_node.outputs['Value'], sub_v_num_node.inputs[0])
                links.new(mult_t_eye_08_node.outputs['Value'], sub_v_num_node.inputs[1])

                v_coord_node = nodes.new(type="ShaderNodeMath"); v_coord_node.name=f"V_Coord_Metal_L{i}"; v_coord_node.operation = 'DIVIDE'
                links.new(sub_v_num_node.outputs['Value'], v_coord_node.inputs[0])
                links.new(su_socket, v_coord_node.inputs[1])
                v_coord_socket = v_coord_node.outputs['Value']

                final_metal_uv_combine_node = nodes.new(type='ShaderNodeCombineXYZ')
                final_metal_uv_combine_node.name = f"FinalMetalUV_L{i}"
                final_metal_uv_combine_node.label = f"Metal UV Output L{i}"
                links.new(u_coord_socket, final_metal_uv_combine_node.inputs['X'])
                links.new(v_coord_socket, final_metal_uv_combine_node.inputs['Y'])
                final_metal_uv_combine_node.inputs['Z'].default_value = 0.0

                uvs_output = final_metal_uv_combine_node.outputs['Vector']
                
            if texture_layer.flags & int(LayerFlag._UV_ROTATE.value):
                tex_image_node[MaterialProperties.UV_ROTATION.value] = texture_layer.uv_rotation
                uv_rotation = nodes.new(type="ShaderNodeValue")
                uv_rotation.label = f"UV Rotation [{image.name}]"
                Utils.create_driver_single(uv_rotation.outputs[0], "uv_rotation", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.UV_ROTATION.value}"]', "frame/60*(uv_rotation/255.0)")
                
                vec_rot_1 = nodes.new(type="ShaderNodeVectorRotate")
                vec_rot_1.label = f"Rotation Applier [{image.name}]"
                vec_rot_1.rotation_type = "Z_AXIS"
                links.new(uvs_output, vec_rot_1.inputs["Vector"])
                vec_rot_1.inputs["Center"].default_value = Vector((0.5, 0.5, 0.0))
                links.new(uv_rotation.outputs[0], vec_rot_1.inputs["Angle"])
                
                uvs_output = vec_rot_1.outputs[0]
            
            if texture_layer.flags & int(LayerFlag._UV_SCALE.value):
                tex_image_node[MaterialProperties.STARTING_SCALE.value] = texture_layer.uv_starting_scale
                tex_image_node[MaterialProperties.ENDING_SCALE.value] = texture_layer.uv_ending_scale
                tex_image_node[MaterialProperties.SCALE_SPEED.value] = texture_layer.uv_scale_speed
                
                starting_scale_var = f"(starting_scale/255.0)"
                ending_scale_var = f"(ending_scale/255.0)"
                scale_speed_var = f"(scale_speed/255.0)"
                
                
                scale_node = nodes.new(type="ShaderNodeValue")
                scale_node.label = f"Scale Rate [{image.name}]"
                Utils.create_driver_multiple(scale_node.outputs[0], ("starting_scale", "ending_scale", "scale_speed"), [material]*3, 
                                            (f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.STARTING_SCALE.value}"]', 
                                                f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.ENDING_SCALE.value}"]',
                                                f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.SCALE_SPEED.value}"]'), 
                                            f"({ending_scale_var}-{starting_scale_var})*(sin(((frame*40%(1.0/{scale_speed_var}*6000.0))/(1.0/{scale_speed_var}*6000.0))*2*pi)+1)*0.5+{starting_scale_var}")
                
                translation_1 = nodes.new(type="ShaderNodeVectorMath")
                translation_1.operation = 'ADD'
                translation_1.label = f"Translation 1 [{image.name}]"
                links.new(uvs_output, translation_1.inputs[0])
                translation_1.inputs[1].default_value = Vector((-0.5, -0.5, 0))
                
                mult_1 = nodes.new(type="ShaderNodeVectorMath")
                mult_1.operation = 'MULTIPLY'
                mult_1.label = f"Multiply 1 [{image.name}]"
                links.new(translation_1.outputs[0], mult_1.inputs[0])
                links.new(scale_node.outputs[0], mult_1.inputs[1])
                
                translation_2 = nodes.new(type="ShaderNodeVectorMath")
                translation_2.operation = 'ADD'
                translation_2.label = f"Translation 2 [{image.name}]"
                links.new(mult_1.outputs[0], translation_2.inputs[0])
                translation_2.inputs[1].default_value = Vector((0.5, 0.5, 0))
                
                uvs_output = translation_2.outputs[0]
            
            if texture_layer.flags & int(LayerFlag._UV_LAVA.value):
                tex_image_node[MaterialProperties.LAVA_WAVE_RATE.value] = texture_layer.lava_wave_effect_rate
                tex_image_node[MaterialProperties.LAVA_WAVE_SPEED.value] = texture_layer.lava_wave_effect_speed
                if separated_uvs_node is None:
                    separated_uvs_node = nodes.new(type="ShaderNodeSeparateXYZ")
                    separated_uvs_node.label = f"Separate UVs [{image.name}]"
                    links.new(uvs_output, separated_uvs_node.inputs[0])
                    
                combine_uvs_node = nodes.new(type="ShaderNodeCombineXYZ")
                combine_uvs_node.label = f"Combine UVs [{image.name}]"
                
                
                lava_wave_speed_node = nodes.new(type="ShaderNodeValue")
                lava_wave_speed_node.label = f"Lava Wave Speed [{image.name}]"
                Utils.create_driver_single(lava_wave_speed_node.outputs[0], "lava_wave_speed", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.LAVA_WAVE_RATE.value}"]', "frame/60*(lava_wave_speed/255.0)*3")
                lava_wave_rate_node = nodes.new(type="ShaderNodeValue")
                lava_wave_rate_node.label = f"Lava Wave Rate [{image.name}]"
                Utils.create_driver_single(lava_wave_rate_node.outputs[0], "lava_wave_rate", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.LAVA_WAVE_SPEED.value}"]', "(lava_wave_rate/255.0)/8.0")
                
                # U
                u_1 = nodes.new(type="ShaderNodeMath")
                u_1.label = "U_1"
                u_1.operation = 'MULTIPLY'
                links.new(separated_uvs_node.outputs["Y"], u_1.inputs[0])
                u_1.inputs[1].default_value = 20.0
                
                u_2 = nodes.new(type="ShaderNodeMath")
                u_2.label = "U_2"
                u_2.operation = 'ADD'
                links.new(lava_wave_speed_node.outputs[0], u_2.inputs[0])
                links.new(u_1.outputs[0], u_2.inputs[1])
                
                u_3 = nodes.new(type="ShaderNodeMath")
                u_3.label = "U_3"
                u_3.operation = 'SINE'
                links.new(u_2.outputs[0], u_3.inputs[0])
                
                u_4 = nodes.new(type="ShaderNodeMath")
                u_4.label = "U_4"
                u_4.operation = 'MULTIPLY'
                links.new(u_3.outputs[0], u_4.inputs[0])
                u_4.inputs[1].default_value = 0.15
                
                u_5 = nodes.new(type="ShaderNodeMath")
                u_5.label = "U_5"
                u_5.operation = 'MULTIPLY'
                links.new(u_4.outputs[0], u_5.inputs[0])
                links.new(lava_wave_rate_node.outputs[0], u_5.inputs[1])
                
                u_6 = nodes.new(type="ShaderNodeMath")
                u_6.label = "U_6"
                u_6.operation = 'ADD'
                links.new(u_5.outputs[0], u_6.inputs[0])
                links.new(separated_uvs_node.outputs["X"], u_6.inputs[1])
                
                links.new(u_6.outputs[0], combine_uvs_node.inputs["X"])
                
                # V
                v_1 = nodes.new(type="ShaderNodeMath")
                v_1.label = "V_1"
                v_1.operation = 'MULTIPLY'
                links.new(separated_uvs_node.outputs["X"], v_1.inputs[0])
                v_1.inputs[1].default_value = 20.0
                
                v_2 = nodes.new(type="ShaderNodeMath")
                v_2.label = "V_2"
                v_2.operation = 'ADD'
                links.new(lava_wave_speed_node.outputs[0], v_2.inputs[0])
                links.new(v_1.outputs[0], v_2.inputs[1])
                
                v_3 = nodes.new(type="ShaderNodeMath")
                v_3.label = "V_3"
                v_3.operation = 'COSINE'
                links.new(v_2.outputs[0], v_3.inputs[0])
                
                v_4 = nodes.new(type="ShaderNodeMath")
                v_4.label = "V_4"
                v_4.operation = 'MULTIPLY'
                links.new(v_3.outputs[0], v_4.inputs[0])
                v_4.inputs[1].default_value = 0.14
                
                v_5 = nodes.new(type="ShaderNodeMath")
                v_5.label = "V_5"
                v_5.operation = 'MULTIPLY'
                links.new(v_4.outputs[0], v_5.inputs[0])
                links.new(lava_wave_rate_node.outputs[0], v_5.inputs[1])
                
                v_6 = nodes.new(type="ShaderNodeMath")
                v_6.label = "V_6"
                v_6.operation = 'ADD'
                links.new(v_5.outputs[0], v_6.inputs[0])
                links.new(separated_uvs_node.outputs["Y"], v_6.inputs[1])
                
                links.new(v_6.outputs[0], combine_uvs_node.inputs["Y"])
                
                effect_adder_node = nodes.new(type="ShaderNodeVectorMath")
                effect_adder_node.label = f"Lava Wave Addition [{image.name}]"
                effect_adder_node.operation = 'ADD'
                links.new(uvs_output, effect_adder_node.inputs[0])
                links.new(combine_uvs_node.outputs['Vector'], effect_adder_node.inputs[1])
                uvs_output = effect_adder_node.outputs[0]
            
            if texture_layer.flags & int(LayerFlag._UV_SCROLL_U.value) or texture_layer.flags & int(LayerFlag._UV_SCROLL_V.value):
                combine_uvs_node = nodes.new(type="ShaderNodeCombineXYZ")
                combine_uvs_node.label = f"Combine UVs (Scroll) [{image.name}]"
                if texture_layer.flags & int(LayerFlag._UV_SCROLL_U.value):
                    tex_image_node[MaterialProperties.SCROLL_U.value] = texture_layer.scroll_u
                    scroll_u_node = nodes.new(type="ShaderNodeValue")
                    scroll_u_node.label = f"Scroll U [{image.name}]"
                    Utils.create_driver_single(scroll_u_node.outputs[0], "scroll_u", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.SCROLL_U.value}"]', "frame/60/4.0*(scroll_u/255.0)")
                    links.new(scroll_u_node.outputs[0], combine_uvs_node.inputs["X"])
                    
                if texture_layer.flags & int(LayerFlag._UV_SCROLL_V.value):
                    tex_image_node[MaterialProperties.SCROLL_V.value] = texture_layer.scroll_v
                    scroll_v_node = nodes.new(type="ShaderNodeValue")
                    scroll_v_node.label = f"Scroll V [{image.name}]"
                    Utils.create_driver_single(scroll_v_node.outputs[0], "scroll_v", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.SCROLL_V.value}"]', "frame/60/4.0*(scroll_v/255.0)")
                    links.new(scroll_v_node.outputs[0], combine_uvs_node.inputs["Y"])
                
                effect_adder_node = nodes.new(type="ShaderNodeVectorMath")
                effect_adder_node.label = f"Scroll UV Addition [{image.name}]"
                effect_adder_node.operation = 'ADD'
                links.new(uvs_output, effect_adder_node.inputs[0])
                links.new(combine_uvs_node.outputs['Vector'], effect_adder_node.inputs[1])
                uvs_output = effect_adder_node.outputs[0]
            
            if texture_layer.flags & int(LayerFlag._ANI_TEXTURE.value):
                frame_num_total = float(texture_layer.animated_texture_frame/256.0)
                tex_speed_factor = float(texture_layer.animated_texture_speed/256.0) 

                base_uv_socket = texture_coordinates_node.outputs["UV"]

                def get_div_u_py(frame_count):
                    if 4 >= frame_count: return 2.0
                    if 16 >= frame_count: return 4.0
                    if 64 >= frame_count: return 8.0
                    return 16.0

                def get_div_v_py(frame_count):
                    if 2 >= frame_count: return 1.0
                    if 8 >= frame_count: return 2.0
                    if 32 >= frame_count: return 4.0
                    return 8.0
                
                div_u_val = get_div_u_py(frame_num_total)
                div_v_val = get_div_v_py(frame_num_total)

                div_u_node = nodes.new(type="ShaderNodeValue")
                div_u_node.name = f"SpriteDivU_L{i}"
                div_u_node.label = f"Sprite Div U L{i}"
                div_u_node.outputs[0].default_value = div_u_val
                div_u_socket = div_u_node.outputs[0]

                div_v_node = nodes.new(type="ShaderNodeValue")
                div_v_node.name = f"SpriteDivV_L{i}"
                div_v_node.label = f"Sprite Div V L{i}"
                div_v_node.outputs[0].default_value = div_v_val
                div_v_socket = div_v_node.outputs[0]
                
                time_node_sprite = nodes.new(type="ShaderNodeValue")
                time_node_sprite.name = f"Time_Sprite_L{i}"
                time_node_sprite.label = f"Time Sprite L{i}"
                Utils.create_driver_single_new(
                    target_rna_item=time_node_sprite.outputs[0],
                    property_name_string="default_value", property_index=-1,
                    var_name="placeholder", source_object_id=material,
                    source_prop_datapath="name",
                    expression="frame / 60.0 * 4"
                )
                r_time_socket = time_node_sprite.outputs[0]

                su_animated_node = nodes.new(type="ShaderNodeMath")
                su_animated_node.name = f"SU_Animated_L{i}"
                su_animated_node.operation = 'MULTIPLY'
                links.new(r_time_socket, su_animated_node.inputs[0])
                su_animated_node.inputs[1].default_value = tex_speed_factor
                su_animated_socket = su_animated_node.outputs['Value']

                safe_frame_num_total_node = nodes.new(type="ShaderNodeMath")
                safe_frame_num_total_node.name = f"SafeFrameNum_L{i}"
                safe_frame_num_total_node.operation = 'MAXIMUM'
                safe_frame_num_total_node.inputs[0].default_value = frame_num_total
                safe_frame_num_total_node.inputs[1].default_value = 1.0

                current_frame_float_node = nodes.new(type="ShaderNodeMath")
                current_frame_float_node.name = f"CurrentFrameFloat_L{i}"
                current_frame_float_node.operation = 'MODULO'
                links.new(su_animated_socket, current_frame_float_node.inputs[0])
                links.new(safe_frame_num_total_node.outputs['Value'], current_frame_float_node.inputs[1])

                current_frame_int_node = nodes.new(type="ShaderNodeMath")
                current_frame_int_node.name = f"CurrentFrameInt_L{i}"
                current_frame_int_node.operation = 'FLOOR'
                links.new(current_frame_float_node.outputs['Value'], current_frame_int_node.inputs[0])
                current_frame_int_socket = current_frame_int_node.outputs['Value']

                frame_mod_div_u_node = nodes.new(type="ShaderNodeMath")
                frame_mod_div_u_node.name = f"FrameModDivU_L{i}"
                frame_mod_div_u_node.operation = 'MODULO'
                links.new(current_frame_int_socket, frame_mod_div_u_node.inputs[0])
                links.new(div_u_socket, frame_mod_div_u_node.inputs[1])

                add_u_node = nodes.new(type="ShaderNodeMath")
                add_u_node.name = f"AddU_L{i}"
                add_u_node.operation = 'DIVIDE'
                add_u_node.use_clamp = True
                links.new(frame_mod_div_u_node.outputs['Value'], add_u_node.inputs[0])
                links.new(div_u_socket, add_u_node.inputs[1])
                add_u_socket = add_u_node.outputs['Value']

                frame_div_div_u_node = nodes.new(type="ShaderNodeMath")
                frame_div_div_u_node.name = f"FrameDivDivU_L{i}"
                frame_div_div_u_node.operation = 'DIVIDE'
                links.new(current_frame_int_socket, frame_div_div_u_node.inputs[0])
                links.new(div_u_socket, frame_div_div_u_node.inputs[1])

                floor_row_index_node = nodes.new(type="ShaderNodeMath")
                floor_row_index_node.name = f"FloorRowIndex_L{i}"
                floor_row_index_node.operation = 'FLOOR'
                links.new(frame_div_div_u_node.outputs['Value'], floor_row_index_node.inputs[0])

                add_v_node = nodes.new(type="ShaderNodeMath")
                add_v_node.name = f"AddV_L{i}"
                add_v_node.operation = 'DIVIDE'
                add_v_node.use_clamp = True
                links.new(floor_row_index_node.outputs['Value'], add_v_node.inputs[0])
                links.new(div_v_socket, add_v_node.inputs[1])
                add_v_socket = add_v_node.outputs['Value']

                scale_u_node = nodes.new(type="ShaderNodeMath")
                scale_u_node.name = f"ScaleU_Val_L{i}"
                scale_u_node.operation = 'DIVIDE'
                scale_u_node.inputs[0].default_value = 1.0
                links.new(div_u_socket, scale_u_node.inputs[1])
                scale_u_val_socket = scale_u_node.outputs['Value']

                scale_v_node = nodes.new(type="ShaderNodeMath")
                scale_v_node.name = f"ScaleV_Val_L{i}"
                scale_v_node.operation = 'DIVIDE'
                scale_v_node.inputs[0].default_value = 1.0
                links.new(div_v_socket, scale_v_node.inputs[1])
                scale_v_val_socket = scale_v_node.outputs['Value']

                uv_scaler_node = nodes.new(type='ShaderNodeCombineXYZ')
                uv_scaler_node.name = f"UVScaler_L{i}"
                links.new(scale_u_val_socket, uv_scaler_node.inputs['X'])
                links.new(scale_v_val_socket, uv_scaler_node.inputs['Y'])
                uv_scaler_node.inputs['Z'].default_value = 1.0

                scaled_uv_node = nodes.new(type='ShaderNodeVectorMath')
                scaled_uv_node.name = f"ScaledUV_L{i}"
                scaled_uv_node.operation = 'MULTIPLY'
                links.new(base_uv_socket, scaled_uv_node.inputs[0])
                links.new(uv_scaler_node.outputs['Vector'], scaled_uv_node.inputs[1])

                uv_translator_node = nodes.new(type='ShaderNodeCombineXYZ')
                uv_translator_node.name = f"UVTranslator_L{i}"
                links.new(add_u_socket, uv_translator_node.inputs['X'])
                links.new(add_v_socket, uv_translator_node.inputs['Y'])
                uv_translator_node.inputs['Z'].default_value = 0.0

                final_animated_uv_node = nodes.new(type='ShaderNodeVectorMath')
                final_animated_uv_node.name = f"FinalSpriteUV_L{i}"
                final_animated_uv_node.label = f"Sprite UV L{i}"
                final_animated_uv_node.operation = 'ADD'
                links.new(scaled_uv_node.outputs['Vector'], final_animated_uv_node.inputs[0])
                links.new(uv_translator_node.outputs['Vector'], final_animated_uv_node.inputs[1])
                
                uvs_output = final_animated_uv_node.outputs['Vector']
            
            # Apply uv effects
            if uvs_output:
                links.new(uvs_output, tex_image_node.inputs['Vector'])

            # Set up ARGB tint+alpha
            tex_image_node[f"r3m_argb_r_{i}"] = texture_layer.argb_color[1]
            tex_image_node[f"r3m_argb_g_{i}"] = texture_layer.argb_color[2]
            tex_image_node[f"r3m_argb_b_{i}"] = texture_layer.argb_color[3]
            tex_image_node[f"r3m_argb_a_{i}"] = texture_layer.argb_color[0]
            
            r3m_rgb_tint_node = nodes.new(type="ShaderNodeRGB")
            r3m_rgb_tint_node.name = f"R3M_RGB_Tint_Layer_{i}"
            r3m_rgb_tint_node.label = f"R3M RGB Tint [{image.name}]"
            
            Utils.create_driver_single_new(
                target_rna_item=r3m_rgb_tint_node.outputs[0],  # The OutputSocket for Color
                property_name_string="default_value",
                property_index=0,                              # Index 0 for Red
                var_name="r_val",
                source_object_id=material,
                source_prop_datapath=f'node_tree.nodes["{tex_image_node.name}"]["r3m_argb_r_{i}"]',
                expression="r_val"
            )
            r3m_rgb_tint_node.outputs[0].default_value[0] = texture_layer.argb_color[1]

            Utils.create_driver_single_new(
                target_rna_item=r3m_rgb_tint_node.outputs[0],  # The OutputSocket for Color
                property_name_string="default_value",
                property_index=1,                              # Index 1 for Green
                var_name="g_val",
                source_object_id=material,
                source_prop_datapath=f'node_tree.nodes["{tex_image_node.name}"]["r3m_argb_g_{i}"]',
                expression="g_val"
            )
            r3m_rgb_tint_node.outputs[0].default_value[1] = texture_layer.argb_color[2]

            Utils.create_driver_single_new(
                target_rna_item=r3m_rgb_tint_node.outputs[0],  # The OutputSocket for Color
                property_name_string="default_value",
                property_index=2,                              # Index 2 for Blue
                var_name="b_val",
                source_object_id=material,
                source_prop_datapath=f'node_tree.nodes["{tex_image_node.name}"]["r3m_argb_b_{i}"]',
                expression="b_val"
            )
            r3m_rgb_tint_node.outputs[0].default_value[2] = texture_layer.argb_color[3]
            
            # Alpha component, separated.
            r3m_alpha_node = nodes.new(type="ShaderNodeValue")
            r3m_alpha_node.name = f"R3M_Alpha_Layer_{i}"
            r3m_alpha_node.label = f"R3M ARGB Alpha [{image.name}]"
            Utils.create_driver_single_new(
                target_rna_item=r3m_alpha_node.outputs[0],
                property_name_string="default_value",
                property_index=-1,
                var_name="a_val",
                source_object_id=material,
                source_prop_datapath=f'node_tree.nodes["{tex_image_node.name}"]["r3m_argb_a_{i}"]',
                expression="a_val"
            )
            r3m_alpha_node.outputs[0].default_value = texture_layer.argb_color[0]

            tinted_texture_color_node = nodes.new(type='ShaderNodeMix')
            tinted_texture_color_node.data_type = 'RGBA'
            tinted_texture_color_node.name = f"Tinted_Tex_Color_Layer_{i}"
            tinted_texture_color_node.label = f"Tinted Texture [{image.name}]"
            tinted_texture_color_node.blend_type = 'MULTIPLY'
            tinted_texture_color_node.inputs['Factor'].default_value = 1.0
            links.new(tex_image_node.outputs['Color'], tinted_texture_color_node.inputs['A'])
            links.new(r3m_rgb_tint_node.outputs['Color'], tinted_texture_color_node.inputs['B'])
            
            # At this point, tinted_texture_color_node represents the texture color already tinted by ARGB
            current_layer_tinted_color_socket = tinted_texture_color_node.outputs['Result']
            
            # Will skip other effects if present.
            if texture_layer.flags & int(LayerFlag._MAT_ENV_BUMP.value): 
                sampled_dudv_color_socket = tex_image_node.outputs['Color']
                separate_rgb_dudv_node = nodes.new(type='ShaderNodeSeparateColor')
                separate_rgb_dudv_node.mode = 'RGB'
                separate_rgb_dudv_node.name = f"Separate_DuDv_{tex_image_node.name}"
                links.new(sampled_dudv_color_socket, separate_rgb_dudv_node.inputs['Color'])

                math_subtract_r_node = nodes.new(type='ShaderNodeMath')
                math_subtract_r_node.name = f"Sub_R_DuDv_{tex_image_node.name}"
                math_subtract_r_node.operation = 'SUBTRACT'
                links.new(separate_rgb_dudv_node.outputs['Red'], math_subtract_r_node.inputs[0])
                math_subtract_r_node.inputs[1].default_value = 0.5
                du_unscaled_socket = math_subtract_r_node.outputs['Value']

                math_subtract_g_node = nodes.new(type='ShaderNodeMath')
                math_subtract_g_node.name = f"Sub_G_DuDv_{tex_image_node.name}"
                math_subtract_g_node.operation = 'SUBTRACT'
                links.new(separate_rgb_dudv_node.outputs['Green'], math_subtract_g_node.inputs[0])
                math_subtract_g_node.inputs[1].default_value = 0.5
                dv_unscaled_socket = math_subtract_g_node.outputs['Value']

                time_node_for_bumpfactor = nodes.new(type="ShaderNodeValue")
                time_node_for_bumpfactor.name = f"Time_For_BumpFactor_{tex_image_node.name}"
                time_node_for_bumpfactor.label = "Time (Anim)"
                
                Utils.create_driver_single_new(
                    target_rna_item=time_node_for_bumpfactor.outputs[0],
                    property_name_string="default_value",
                    property_index=-1,
                    var_name="frame",
                    source_object_id=material,
                    source_prop_datapath=f'name',
                    expression="frame / 60.0"
                )
                game_time_socket = time_node_for_bumpfactor.outputs[0]

                angle_node = nodes.new(type="ShaderNodeMath")
                angle_node.name = f"Angle_BumpFactor_{tex_image_node.name}"
                angle_node.operation = 'MULTIPLY'
                links.new(game_time_socket, angle_node.inputs[0])
                angle_node.inputs[1].default_value = 3.0

                cos_angle_socket = nodes.new(type="ShaderNodeMath")
                cos_angle_socket.name = f"Cos_BumpFactor_{tex_image_node.name}"
                cos_angle_socket.operation = 'COSINE'
                links.new(angle_node.outputs['Value'], cos_angle_socket.inputs[0])
                cos_angle_socket = cos_angle_socket.outputs['Value']

                sin_angle_socket = nodes.new(type="ShaderNodeMath")
                sin_angle_socket.name = f"Sin_BumpFactor_{tex_image_node.name}"
                sin_angle_socket.operation = 'SINE'
                links.new(angle_node.outputs['Value'], sin_angle_socket.inputs[0])
                sin_angle_socket = sin_angle_socket.outputs['Value']

                r_val_node = nodes.new(type="ShaderNodeValue")
                r_val_node.name = f"R_Val_BumpFactor_{tex_image_node.name}"
                r_val_node.outputs[0].default_value = 0.01
                r_socket = r_val_node.outputs[0]

                m00_node = nodes.new(type="ShaderNodeMath")
                m00_node.name = f"m00_BumpFactor_{tex_image_node.name}"
                m00_node.operation = 'MULTIPLY'
                links.new(r_socket, m00_node.inputs[0])
                links.new(cos_angle_socket, m00_node.inputs[1])
                m00_socket = m00_node.outputs['Value']

                m01_intermediate_node = nodes.new(type="ShaderNodeMath")
                m01_intermediate_node.name = f"m01_Inter_BumpFactor_{tex_image_node.name}"
                m01_intermediate_node.operation = 'MULTIPLY'
                links.new(r_socket, m01_intermediate_node.inputs[0])
                links.new(sin_angle_socket, m01_intermediate_node.inputs[1])
                
                m01_node = nodes.new(type="ShaderNodeMath")
                m01_node.name = f"m01_BumpFactor_{tex_image_node.name}"
                m01_node.operation = 'MULTIPLY'
                links.new(m01_intermediate_node.outputs['Value'], m01_node.inputs[0])
                m01_node.inputs[1].default_value = -1.0
                m01_socket = m01_node.outputs['Value']

                m10_node = nodes.new(type="ShaderNodeMath")
                m10_node.name = f"m10_BumpFactor_{tex_image_node.name}"
                m10_node.operation = 'MULTIPLY'
                links.new(r_socket, m10_node.inputs[0])
                links.new(sin_angle_socket, m10_node.inputs[1])
                m10_socket = m10_node.outputs['Value']

                m11_socket = m00_socket 

                term_du_1_node = nodes.new(type="ShaderNodeMath")
                term_du_1_node.name = f"TermDU1_Bump_{tex_image_node.name}"
                term_du_1_node.operation = 'MULTIPLY'
                links.new(du_unscaled_socket, term_du_1_node.inputs[0])
                links.new(m00_socket, term_du_1_node.inputs[1])

                term_du_2_node = nodes.new(type="ShaderNodeMath")
                term_du_2_node.name = f"TermDU2_Bump_{tex_image_node.name}"
                term_du_2_node.operation = 'MULTIPLY'
                links.new(dv_unscaled_socket, term_du_2_node.inputs[0])
                links.new(m01_socket, term_du_2_node.inputs[1])

                final_animated_du_offset_node = nodes.new(type="ShaderNodeMath")
                final_animated_du_offset_node.name = f"FinalDU_Bump_{tex_image_node.name}"
                final_animated_du_offset_node.operation = 'ADD'
                links.new(term_du_1_node.outputs['Value'], final_animated_du_offset_node.inputs[0])
                links.new(term_du_2_node.outputs['Value'], final_animated_du_offset_node.inputs[1])
                final_animated_du_offset_socket = final_animated_du_offset_node.outputs['Value']

                term_dv_1_node = nodes.new(type="ShaderNodeMath")
                term_dv_1_node.name = f"TermDV1_Bump_{tex_image_node.name}"
                term_dv_1_node.operation = 'MULTIPLY'
                links.new(du_unscaled_socket, term_dv_1_node.inputs[0])
                links.new(m10_socket, term_dv_1_node.inputs[1])

                term_dv_2_node = nodes.new(type="ShaderNodeMath")
                term_dv_2_node.name = f"TermDV2_Bump_{tex_image_node.name}"
                term_dv_2_node.operation = 'MULTIPLY'
                links.new(dv_unscaled_socket, term_dv_2_node.inputs[0])
                links.new(m11_socket, term_dv_2_node.inputs[1])

                final_animated_dv_offset_node = nodes.new(type="ShaderNodeMath")
                final_animated_dv_offset_node.name = f"FinalDV_Bump_{tex_image_node.name}"
                final_animated_dv_offset_node.operation = 'ADD'
                links.new(term_dv_1_node.outputs['Value'], final_animated_dv_offset_node.inputs[0])
                links.new(term_dv_2_node.outputs['Value'], final_animated_dv_offset_node.inputs[1])
                final_animated_dv_offset_socket = final_animated_dv_offset_node.outputs['Value']

                final_uv_perturbation_offset_node = nodes.new(type='ShaderNodeCombineXYZ')
                final_uv_perturbation_offset_node.name = f"Final_UV_Perturbation_Vec_{tex_image_node.name}"
                final_uv_perturbation_offset_node.label = "UV Perturbation Offset"
                links.new(final_animated_du_offset_socket, final_uv_perturbation_offset_node.inputs['X'])
                links.new(final_animated_dv_offset_socket, final_uv_perturbation_offset_node.inputs['Y'])
                final_uv_perturbation_offset_node.inputs['Z'].default_value = 0.0 # UVs are 2D

                original_target_input_socket = None
                link_to_modify = None

                for link_obj in list(first_layer_uv_output.links):
                    if link_obj.to_node == first_layer_tex_image_node and \
                    link_obj.to_socket == first_layer_tex_image_node.inputs['Vector']:
                        original_target_input_socket = link_obj.to_socket
                        link_to_modify = link_obj
                        break
                
                if not original_target_input_socket or not link_to_modify:
                    print(f"Error: 'first_layer_uv_output' is not currently connected to the 'Vector' input of '{first_layer_tex_image_node.name}'. Cannot insert perturbation.")
                else:
                    links.remove(link_to_modify)

                    perturbed_uv_add_node = nodes.new(type='ShaderNodeVectorMath')
                    perturbed_uv_add_node.name = f"Add_Perturbation_to_{first_layer_uv_output.node.name}_UV"
                    perturbed_uv_add_node.label = "Add ENV_BUMP UV Offset"
                    perturbed_uv_add_node.operation = 'ADD'

                    links.new(first_layer_uv_output, perturbed_uv_add_node.inputs[0])

                    links.new(final_uv_perturbation_offset_node.outputs['Vector'], perturbed_uv_add_node.inputs[1])

                    new_conjoined_uv_output_socket = perturbed_uv_add_node.outputs['Vector']

                    links.new(new_conjoined_uv_output_socket, original_target_input_socket)

                    break
            
            if texture_layer.flags & int(LayerFlag._UV_GRADIENT_ALPHA_UV.value):

                gradient_params_short = texture_layer.gradient_alpha

                # Initial gradient value
                gradient_param_node = nodes.new(type="ShaderNodeValue")
                gradient_param_node.name = f"GradientShort_L{i}"
                gradient_param_node.label = f"Gradient Param Raw L{i}"
                gradient_param_node.outputs[0].default_value = float(gradient_params_short)
                raw_short_socket = gradient_param_node.outputs[0]

                # By dividing the number by 256 and flooring it after, only the higher 8 bits of it remain
                div_by_256_for_u_node = nodes.new(type="ShaderNodeMath")
                div_by_256_for_u_node.name = f"GradDiv256_U_L{i}"
                div_by_256_for_u_node.operation = 'DIVIDE'
                links.new(raw_short_socket, div_by_256_for_u_node.inputs[0])
                div_by_256_for_u_node.inputs[1].default_value = 256.0
                
                control_u_byte_node = nodes.new(type="ShaderNodeMath")
                control_u_byte_node.name = f"ControlUByte_L{i}"
                control_u_byte_node.operation = 'FLOOR'
                links.new(div_by_256_for_u_node.outputs['Value'], control_u_byte_node.inputs[0])
                control_u_byte_socket = control_u_byte_node.outputs['Value']

                # By dividing the number by 256 using modulo, only the lower 8 bits of it remain
                control_v_byte_node = nodes.new(type="ShaderNodeMath")
                control_v_byte_node.name = f"ControlVByte_L{i}"
                control_v_byte_node.operation = 'MODULO'
                links.new(raw_short_socket, control_v_byte_node.inputs[0])
                control_v_byte_node.inputs[1].default_value = 256.0
                control_v_byte_socket = control_v_byte_node.outputs['Value']
                
                # At this point, both bytes for U and V are ready. They represent the raw gradient alpha value for each coordinate
                
                
                param_u_sub_node = nodes.new(type="ShaderNodeMath")
                param_u_sub_node.operation = 'SUBTRACT'
                links.new(control_u_byte_socket, param_u_sub_node.inputs[0])
                param_u_sub_node.inputs[1].default_value = 100.0
                param_u_div_node = nodes.new(type="ShaderNodeMath")
                param_u_div_node.operation = 'DIVIDE'
                links.new(param_u_sub_node.outputs['Value'], param_u_div_node.inputs[0])
                param_u_div_node.inputs[1].default_value = 25.0
                gradient_param_u_socket = param_u_div_node.outputs['Value']

                param_v_sub_node = nodes.new(type="ShaderNodeMath")
                param_v_sub_node.operation = 'SUBTRACT'
                links.new(control_v_byte_socket, param_v_sub_node.inputs[0])
                param_v_sub_node.inputs[1].default_value = 100.0
                param_v_div_node = nodes.new(type="ShaderNodeMath")
                param_v_div_node.operation = 'DIVIDE'
                links.new(param_v_sub_node.outputs['Value'], param_v_div_node.inputs[0])
                param_v_div_node.inputs[1].default_value = 25.0
                gradient_param_v_socket = param_v_div_node.outputs['Value']

                # At this point, v_su for U and V has been calculated and is available through the exposed sockets.
                
                static_uv_input_socket = texture_coordinates_node.outputs["UV"]
                separate_static_uv_node = nodes.new(type='ShaderNodeSeparateXYZ')
                separate_static_uv_node.name = f"SeparateStaticUV_Grad_L{i}"
                links.new(static_uv_input_socket, separate_static_uv_node.inputs['Vector'])
                static_u_coord_socket = separate_static_uv_node.outputs['X']
                original_static_v_coord_socket = separate_static_uv_node.outputs['Y']

                invert_v_node = nodes.new(type="ShaderNodeMath")
                invert_v_node.name = f"Invert_StaticV_Grad_L{i}"
                invert_v_node.operation = 'SUBTRACT'
                invert_v_node.inputs[0].default_value = 1.0
                links.new(original_static_v_coord_socket, invert_v_node.inputs[1])
                
                static_v_coord_socket = invert_v_node.outputs['Value']
                
                # At this point, individual U and V original coordinates are available too.
                
                u_alpha_component_socket = nodes.new(type="ShaderNodeValue")
                u_alpha_component_socket.outputs[0].default_value = 1.0
                u_alpha_component_socket = u_alpha_component_socket.outputs[0]
                if texture_layer.flags & int(LayerFlag._UV_GRADIENT_ALPHA_U.value):
                    
                    # Is parameter U less than 0?
                    is_param_u_neg_node = nodes.new(type="ShaderNodeMath")
                    is_param_u_neg_node.operation = 'LESS_THAN'
                    links.new(gradient_param_u_socket, is_param_u_neg_node.inputs[0]); is_param_u_neg_node.inputs[1].default_value = 0.0
                    
                    # The reason for the absolute node is that this will ever only be used when the parameter is negative. Since originally we would negate it again to turn it positive, absolute is a shortcut
                    abs_grad_param_u_node = nodes.new(type="ShaderNodeMath")
                    abs_grad_param_u_node.operation = 'ABSOLUTE'
                    links.new(gradient_param_u_socket, abs_grad_param_u_node.inputs[0])
                    safe_abs_grad_param_u_node = nodes.new(type="ShaderNodeMath")
                    safe_abs_grad_param_u_node.operation = 'MAXIMUM'; links.new(abs_grad_param_u_node.outputs['Value'], safe_abs_grad_param_u_node.inputs[0]); safe_abs_grad_param_u_node.inputs[1].default_value = 0.0001

                    # uv / v_us(U) division
                    u_raw_div_node = nodes.new(type="ShaderNodeMath")
                    u_raw_div_node.operation = 'DIVIDE'
                    links.new(static_u_coord_socket, u_raw_div_node.inputs[0])
                    links.new(gradient_param_u_socket, u_raw_div_node.inputs[1])
                    u_raw_socket = u_raw_div_node.outputs['Value']

                    # (1/ -v_su) here
                    offset_val_u_node = nodes.new(type="ShaderNodeMath")
                    offset_val_u_node.operation = 'DIVIDE'
                    offset_val_u_node.inputs[0].default_value = 1.0
                    links.new(safe_abs_grad_param_u_node.outputs['Value'], offset_val_u_node.inputs[1])

                    u_conditional_add_node = nodes.new(type="ShaderNodeMath")
                    u_conditional_add_node.operation = 'ADD'
                    links.new(u_raw_socket, u_conditional_add_node.inputs[0])
                    offset_mult_by_cond_u_node = nodes.new(type="ShaderNodeMath")
                    offset_mult_by_cond_u_node.operation = 'MULTIPLY'
                    links.new(offset_val_u_node.outputs['Value'], offset_mult_by_cond_u_node.inputs[0])
                    links.new(is_param_u_neg_node.outputs[0], offset_mult_by_cond_u_node.inputs[1])
                    # Only add the offset if parameter is negative.
                    links.new(offset_mult_by_cond_u_node.outputs['Value'], u_conditional_add_node.inputs[1])
                    
                    u_clamped_alpha_node = nodes.new(type="ShaderNodeClamp")
                    links.new(u_conditional_add_node.outputs['Value'], u_clamped_alpha_node.inputs['Value'])
                    u_clamped_alpha_node.inputs['Min'].default_value = 0.0
                    u_clamped_alpha_node.inputs['Max'].default_value = 1.0
                    u_alpha_component_socket = u_clamped_alpha_node
                    u_alpha_component_socket = u_clamped_alpha_node.outputs['Result']

                v_alpha_component_socket = nodes.new(type="ShaderNodeValue")
                v_alpha_component_socket.outputs[0].default_value = 1.0
                v_alpha_component_socket = v_alpha_component_socket.outputs[0]
                
                if texture_layer.flags & int(LayerFlag._UV_GRADIENT_ALPHA_V.value):
                    
                    # Is parameter V less than 0?
                    is_param_v_neg_node = nodes.new(type="ShaderNodeMath")
                    is_param_v_neg_node.operation = 'LESS_THAN'
                    links.new(gradient_param_v_socket, is_param_v_neg_node.inputs[0]); is_param_v_neg_node.inputs[1].default_value = 0.0
                    
                    abs_grad_param_v_node = nodes.new(type="ShaderNodeMath")
                    abs_grad_param_v_node.operation = 'ABSOLUTE'; links.new(gradient_param_v_socket, abs_grad_param_v_node.inputs[0])
                    safe_abs_grad_param_v_node = nodes.new(type="ShaderNodeMath")
                    safe_abs_grad_param_v_node.operation = 'MAXIMUM'; links.new(abs_grad_param_v_node.outputs['Value'], safe_abs_grad_param_v_node.inputs[0]); safe_abs_grad_param_v_node.inputs[1].default_value = 0.0001
                    
                    v_raw_div_node = nodes.new(type="ShaderNodeMath")
                    v_raw_div_node.operation = 'DIVIDE'; links.new(static_v_coord_socket, v_raw_div_node.inputs[0]); links.new(gradient_param_v_socket, v_raw_div_node.inputs[1])
                    v_raw_socket = v_raw_div_node.outputs['Value']
                    offset_val_v_node = nodes.new(type="ShaderNodeMath")
                    offset_val_v_node.operation = 'DIVIDE'; offset_val_v_node.inputs[0].default_value = 1.0; links.new(safe_abs_grad_param_v_node.outputs['Value'], offset_val_v_node.inputs[1])
                    offset_mult_by_cond_v_node = nodes.new(type="ShaderNodeMath")
                    offset_mult_by_cond_v_node.operation = 'MULTIPLY'; links.new(offset_val_v_node.outputs['Value'], offset_mult_by_cond_v_node.inputs[0]); links.new(is_param_v_neg_node.outputs[0], offset_mult_by_cond_v_node.inputs[1])
                    v_conditional_add_node = nodes.new(type="ShaderNodeMath")
                    v_conditional_add_node.operation = 'ADD'; links.new(v_raw_socket, v_conditional_add_node.inputs[0]); links.new(offset_mult_by_cond_v_node.outputs['Value'], v_conditional_add_node.inputs[1])
                    v_clamped_alpha_node = nodes.new(type="ShaderNodeClamp")
                    links.new(v_conditional_add_node.outputs['Value'], v_clamped_alpha_node.inputs['Value']); v_clamped_alpha_node.inputs['Min'].default_value = 0.0; v_clamped_alpha_node.inputs['Max'].default_value = 1.0
                    v_alpha_component_socket = v_clamped_alpha_node
                    v_alpha_component_socket = v_clamped_alpha_node.outputs['Result']

                

                combined_gradient_alpha_node = nodes.new(type="ShaderNodeMath")
                combined_gradient_alpha_node.name = f"CombinedGradientAlpha_L{i}"
                combined_gradient_alpha_node.label = f"Combined Gradient Alpha L{i}"
                combined_gradient_alpha_node.operation = 'MULTIPLY'
                combined_gradient_alpha_node.use_clamp = True
                links.new(u_alpha_component_socket, combined_gradient_alpha_node.inputs[0])
                links.new(v_alpha_component_socket, combined_gradient_alpha_node.inputs[1])
                
                gradient_effect_output_socket = combined_gradient_alpha_node.outputs['Value']

                conjoined_alpha_node = nodes.new(type="ShaderNodeMath")
                conjoined_alpha_node.name = f"Conjoined_R3M_Gradient_Alpha_L{i}"
                conjoined_alpha_node.label = f"R3M+Gradient Alpha L{i}"
                conjoined_alpha_node.operation = 'MULTIPLY'
                conjoined_alpha_node.use_clamp = True
                links.new(r3m_alpha_node.outputs[0], conjoined_alpha_node.inputs[0])
                links.new(gradient_effect_output_socket, conjoined_alpha_node.inputs[1])

                r3m_alpha_node = conjoined_alpha_node 
            
            alpha_for_blend_factor = tex_image_node.outputs['Alpha']
            

            animated_flicker_alpha_0_1_socket = None

            if texture_layer.flags & int(LayerFlag._ANI_ALPHA_FLICKER.value):

                flicker_speed_style_node = nodes.new(type="ShaderNodeValue")
                flicker_speed_style_node.name = f"FlickerSpeedStyle_L{i}"
                flicker_speed_style_node.label = f"Flicker Speed/Style L{i}"
                
                flicker_speed_style_node.outputs[0].default_value = float(texture_layer.alpha_flicker_rate)
                su_socket = flicker_speed_style_node.outputs[0]

                flicker_range_node = nodes.new(type="ShaderNodeValue")
                flicker_range_node.name = f"FlickerRangeRaw_L{i}"
                flicker_range_node.label = f"Flicker Range Raw L{i}"
                flicker_range_node.outputs[0].default_value = float(texture_layer.alpha_flicker_animation)
                range_raw_socket = flicker_range_node.outputs[0]
                
                divide_by_256_node = nodes.new(type="ShaderNodeMath")
                divide_by_256_node.name = f"FlickerDiv256_L{i}"
                divide_by_256_node.operation = 'DIVIDE'
                links.new(range_raw_socket, divide_by_256_node.inputs[0])
                divide_by_256_node.inputs[1].default_value = 256.0
                
                start_alpha_byte_node = nodes.new(type="ShaderNodeMath")
                start_alpha_byte_node.name = f"FlickerStartByte_L{i}"
                start_alpha_byte_node.operation = 'FLOOR'
                links.new(divide_by_256_node.outputs['Value'], start_alpha_byte_node.inputs[0])
                start_alpha_byte_socket = start_alpha_byte_node.outputs['Value']

                end_alpha_byte_node = nodes.new(type="ShaderNodeMath")
                end_alpha_byte_node.name = f"FlickerEndByte_L{i}"
                end_alpha_byte_node.operation = 'MODULO'
                links.new(range_raw_socket, end_alpha_byte_node.inputs[0])
                end_alpha_byte_node.inputs[1].default_value = 256.0
                end_alpha_byte_socket = end_alpha_byte_node.outputs['Value']
                
                se_sub_node = nodes.new(type="ShaderNodeMath")
                se_sub_node.name = f"FlickerAmp_L{i}"
                se_sub_node.operation = 'SUBTRACT'
                links.new(end_alpha_byte_socket, se_sub_node.inputs[0])
                links.new(start_alpha_byte_socket, se_sub_node.inputs[1])
                se_sub_socket = se_sub_node.outputs['Value']

                time_node_flicker = nodes.new(type="ShaderNodeValue")
                time_node_flicker.name = f"Time_Flicker_L{i}"
                time_node_flicker.label = f"Time Flicker L{i}"
                
                Utils.create_driver_single_new(
                    target_rna_item=time_node_flicker.outputs[0],
                    property_name_string="default_value", property_index=-1,
                    var_name="placeholder", source_object_id=material,
                    source_prop_datapath="name",
                    expression="frame / 60.0" 
                )
                r_time_socket = time_node_flicker.outputs[0]


                su_scaled_time_node_sin = nodes.new(type="ShaderNodeMath")
                su_scaled_time_node_sin.operation = 'MULTIPLY'
                links.new(r_time_socket, su_scaled_time_node_sin.inputs[0])
                links.new(su_socket, su_scaled_time_node_sin.inputs[1])
                
                sin_val_node = nodes.new(type="ShaderNodeMath")
                sin_val_node.operation = 'SINE'
                links.new(su_scaled_time_node_sin.outputs['Value'], sin_val_node.inputs[0])

                se_sub_half_node_sin = nodes.new(type="ShaderNodeMath")
                se_sub_half_node_sin.operation = 'DIVIDE'
                links.new(se_sub_socket, se_sub_half_node_sin.inputs[0])
                se_sub_half_node_sin.inputs[1].default_value = 2.0
                
                term1_sin_node = nodes.new(type="ShaderNodeMath")
                term1_sin_node.operation = 'MULTIPLY'
                links.new(sin_val_node.outputs['Value'], term1_sin_node.inputs[0])
                links.new(se_sub_half_node_sin.outputs['Value'], term1_sin_node.inputs[1])
                
                add1_sin_node = nodes.new(type="ShaderNodeMath")
                add1_sin_node.operation = 'ADD'
                links.new(term1_sin_node.outputs['Value'], add1_sin_node.inputs[0])
                links.new(se_sub_half_node_sin.outputs['Value'], add1_sin_node.inputs[1])
                
                result_sin_flicker_0_255_node = nodes.new(type="ShaderNodeMath")
                result_sin_flicker_0_255_node.operation = 'ADD'
                links.new(add1_sin_node.outputs['Value'], result_sin_flicker_0_255_node.inputs[0])
                links.new(start_alpha_byte_socket, result_sin_flicker_0_255_node.inputs[1])
                
                neg_su_node_lin = nodes.new(type="ShaderNodeMath")
                neg_su_node_lin.operation = 'MULTIPLY'
                links.new(su_socket, neg_su_node_lin.inputs[0])
                neg_su_node_lin.inputs[1].default_value = -1.0 
                
                term1_lin_node = nodes.new(type="ShaderNodeMath")
                term1_lin_node.operation = 'MULTIPLY'
                links.new(r_time_socket, term1_lin_node.inputs[0])
                links.new(neg_su_node_lin.outputs['Value'], term1_lin_node.inputs[1])
                
                term2_lin_node = nodes.new(type="ShaderNodeMath")
                term2_lin_node.operation = 'MULTIPLY'
                links.new(term1_lin_node.outputs['Value'], term2_lin_node.inputs[0])
                term2_lin_node.inputs[1].default_value = 200.0
                
                safe_se_sub_node_lin = nodes.new(type="ShaderNodeMath")
                safe_se_sub_node_lin.operation = 'MAXIMUM'
                links.new(se_sub_socket, safe_se_sub_node_lin.inputs[0])
                safe_se_sub_node_lin.inputs[1].default_value = 1.0 

                modulo_lin_node = nodes.new(type="ShaderNodeMath")
                modulo_lin_node.operation = 'MODULO'
                links.new(term2_lin_node.outputs['Value'], modulo_lin_node.inputs[0])
                links.new(safe_se_sub_node_lin.outputs['Value'], modulo_lin_node.inputs[1])
                
                result_lin_flicker_0_255_node = nodes.new(type="ShaderNodeMath")
                result_lin_flicker_0_255_node.operation = 'ADD'
                links.new(modulo_lin_node.outputs['Value'], result_lin_flicker_0_255_node.inputs[0])
                links.new(start_alpha_byte_socket, result_lin_flicker_0_255_node.inputs[1])

                is_su_negative_node = nodes.new(type="ShaderNodeMath")
                is_su_negative_node.name = f"IsSUNegative_L{i}"
                is_su_negative_node.operation = 'LESS_THAN'
                links.new(su_socket, is_su_negative_node.inputs[0])
                is_su_negative_node.inputs[1].default_value = 0.0
                
                select_flicker_type_node = nodes.new(type="ShaderNodeMix") 
                select_flicker_type_node.name = f"SelectFlickerPath_L{i}"
                select_flicker_type_node.data_type = 'FLOAT'
                links.new(is_su_negative_node.outputs[0], select_flicker_type_node.inputs[0])
                links.new(result_sin_flicker_0_255_node.outputs['Value'], select_flicker_type_node.inputs[2])
                links.new(result_lin_flicker_0_255_node.outputs['Value'], select_flicker_type_node.inputs[3])
                
                animated_flicker_value_0_255_socket = select_flicker_type_node.outputs[0]
                
                normalize_anim_flicker_node = nodes.new(type="ShaderNodeMath")
                normalize_anim_flicker_node.name = f"NormAnimFlicker_L{i}"
                normalize_anim_flicker_node.operation = 'DIVIDE'
                links.new(animated_flicker_value_0_255_socket, normalize_anim_flicker_node.inputs[0])
                normalize_anim_flicker_node.inputs[1].default_value = 255.0
                
                final_flicker_modulation_node = nodes.new(type="ShaderNodeMath")
                final_flicker_modulation_node.name = f"ModulateFlickerByARGB_L{i}"
                final_flicker_modulation_node.operation = 'MULTIPLY'
                final_flicker_modulation_node.use_clamp = True
                links.new(normalize_anim_flicker_node.outputs['Value'], final_flicker_modulation_node.inputs[0])
                
                links.new(r3m_alpha_node.outputs[0], final_flicker_modulation_node.inputs[1]) 
                
                animated_flicker_alpha_0_1_socket = final_flicker_modulation_node.outputs['Value']
            else:
                animated_flicker_alpha_0_1_socket = r3m_alpha_node.outputs[0]
            
            final_alpha_factor_multiply_node = nodes.new(type="ShaderNodeMath")
            final_alpha_factor_multiply_node.name = f"Final_Blend_Factor_L{i}"
            final_alpha_factor_multiply_node.label = f"Effective Blend Alpha L{i}"
            final_alpha_factor_multiply_node.operation = 'MULTIPLY'
            final_alpha_factor_multiply_node.use_clamp = True # Ensure result is 0-1
            
            links.new(alpha_for_blend_factor, final_alpha_factor_multiply_node.inputs[0])

            links.new(animated_flicker_alpha_0_1_socket, final_alpha_factor_multiply_node.inputs[1])

            alpha_for_blend_factor = final_alpha_factor_multiply_node.outputs['Value']
            
            current_blend_op_type = texture_layer.alpha_type

            if i == 0 and not is_alpha_set_by_layer_zero:
                first_layer_uv_output = uvs_output
                first_layer_tex_image_node = tex_image_node
                if current_blend_op_type == BlendMethod.OPAQUE.value:
                    accumulated_alpha_socket = alpha_for_blend_factor
                    is_alpha_set_by_layer_zero = True
                elif current_blend_op_type == BlendMethod.BRIGHT.value or current_blend_op_type == BlendMethod.DEFAULT.value:
                    # For additive, the "coverage" alpha comes from luminance
                    rgb_to_bw_node = nodes.new(type='ShaderNodeRGBToBW')
                    rgb_to_bw_node.name = f"Luminance_For_Coverage_L{i}"
                    rgb_to_bw_node.label = f"Luminance for Coverage L{i}"
                    links.new(current_layer_tinted_color_socket, rgb_to_bw_node.inputs['Color'])
                    
                    multiply_luminance_with_factor_node = nodes.new(type="ShaderNodeMath")
                    multiply_luminance_with_factor_node.name = f"Coverage_Alpha_L{i}"
                    multiply_luminance_with_factor_node.label = f"Coverage Alpha L{i}"
                    multiply_luminance_with_factor_node.operation = 'MULTIPLY'
                    multiply_luminance_with_factor_node.use_clamp = True
                    links.new(rgb_to_bw_node.outputs['Val'], multiply_luminance_with_factor_node.inputs[0])
                    links.new(alpha_for_blend_factor, multiply_luminance_with_factor_node.inputs[1])
                    
                    accumulated_alpha_socket = multiply_luminance_with_factor_node.outputs['Value']
                    is_alpha_set_by_layer_zero = True
                elif current_blend_op_type == BlendMethod.SHADOW.value or \
                    current_blend_op_type == BlendMethod.ONLY_TRANSPARENCY.value:
                    accumulated_alpha_socket = alpha_for_blend_factor
                    is_alpha_set_by_layer_zero = True

            # --- Apply Blend Operation ---
            # OPAQUE, DEFAULT, SHADOW, ONLY_TRANSPARENCY primarily affect BaseColor stream
            if current_blend_op_type == BlendMethod.NONE.value or \
            current_blend_op_type == BlendMethod.OPAQUE.value or \
            current_blend_op_type == BlendMethod.DEFAULT.value or \
            current_blend_op_type == BlendMethod.SHADOW.value or \
            current_blend_op_type == BlendMethod.ONLY_TRANSPARENCY.value:
                mix_node = nodes.new(type='ShaderNodeMix')
                mix_node.data_type = 'RGBA'
                mix_node.name = f"Blend_BaseColor_Layer_{i}"
                
                if current_blend_op_type == BlendMethod.NONE.value:
                    if i == 0:
                        accumulated_base_color_socket = current_layer_tinted_color_socket
                    else:
                        mix_node.blend_type = 'MIX'
                        links.new(accumulated_base_color_socket, mix_node.inputs['A'])
                        links.new(current_layer_tinted_color_socket, mix_node.inputs['B'])
                        links.new(alpha_for_blend_factor, mix_node.inputs['Factor'])
                        accumulated_base_color_socket = mix_node.outputs['Result']
                
                elif current_blend_op_type == BlendMethod.OPAQUE.value or \
                    current_blend_op_type == BlendMethod.ONLY_TRANSPARENCY.value:
                    mix_node.blend_type = 'MIX'
                    links.new(accumulated_base_color_socket, mix_node.inputs['A'])
                    links.new(current_layer_tinted_color_socket, mix_node.inputs['B'])
                    links.new(alpha_for_blend_factor, mix_node.inputs['Factor'])
                    accumulated_base_color_socket = mix_node.outputs['Result']

                elif current_blend_op_type == BlendMethod.DEFAULT.value or \
                    current_blend_op_type == BlendMethod.SHADOW.value:
                    term1_mix = nodes.new('ShaderNodeMix')
                    term1_mix.data_type = 'RGBA'
                    term1_mix.name = f"Term1_Default_Layer_{i}"
                    term1_mix.blend_type = 'MIX'
                    term1_mix.inputs['A'].default_value = (0,0,0,1)
                    links.new(current_layer_tinted_color_socket, term1_mix.inputs['B'])
                    links.new(alpha_for_blend_factor, term1_mix.inputs['Factor'])

                    inv_src_color_node = nodes.new('ShaderNodeVectorMath')
                    inv_src_color_node.name = f"InvSrc_Default_Layer_{i}"
                    inv_src_color_node.operation = 'SUBTRACT'
                    inv_src_color_node.inputs[0].default_value = (1,1,1)
                    links.new(current_layer_tinted_color_socket, inv_src_color_node.inputs[1])

                    term2_multiply = nodes.new('ShaderNodeMix')
                    term2_multiply.data_type = 'RGBA'
                    term2_multiply.name = f"Term2_Default_Layer_{i}"
                    term2_multiply.blend_type = 'MULTIPLY'
                    term2_multiply.inputs['Factor'].default_value = 1.0
                    links.new(accumulated_base_color_socket, term2_multiply.inputs['A'])
                    links.new(inv_src_color_node.outputs['Vector'], term2_multiply.inputs['B'])
                    
                    mix_node.blend_type = 'ADD'
                    mix_node.inputs['Factor'].default_value = 1.0
                    links.new(term1_mix.outputs['Result'], mix_node.inputs['A'])
                    links.new(term2_multiply.outputs['Result'], mix_node.inputs['B'])
                    accumulated_base_color_socket = mix_node.outputs['Result']
            
            # BRIGHT, BACK_BRIGHT primarily affect EmissionColor stream
            elif current_blend_op_type == BlendMethod.BRIGHT.value:
                add_node = nodes.new(type='ShaderNodeMix')
                add_node.data_type = 'RGBA'
                add_node.name = f"Blend_Emission_Layer_{i}"
                add_node.blend_type = 'ADD'
                # A is previous accumulated emission
                links.new(accumulated_emission_color_socket, add_node.inputs['A'])
                # B is current layer's tinted color
                links.new(current_layer_tinted_color_socket, add_node.inputs['B'])
                # Factor is current layer's R3M ARGB Alpha (intensity)
                links.new(alpha_for_blend_factor, add_node.inputs['Factor'])
                accumulated_emission_color_socket = add_node.outputs['Result']

            elif current_blend_op_type == BlendMethod.BACK_BRIGHT.value:

                term1_is_current_color = current_layer_tinted_color_socket

                inv_src_color_node = nodes.new('ShaderNodeVectorMath')
                inv_src_color_node.operation = 'SUBTRACT'
                inv_src_color_node.inputs[0].default_value = (1,1,1)
                links.new(current_layer_tinted_color_socket, inv_src_color_node.inputs[1])

                term2_multiply = nodes.new('ShaderNodeMix')
                term2_multiply.data_type = 'RGBA'
                term2_multiply.blend_type = 'MULTIPLY'
                term2_multiply.inputs['Factor'].default_value = 1.0
                links.new(accumulated_emission_color_socket, term2_multiply.inputs['A'])
                links.new(inv_src_color_node.outputs['Vector'], term2_multiply.inputs['B'])
                
                final_add_node = nodes.new('ShaderNodeMix')
                final_add_node.data_type = 'RGBA'
                final_add_node.name = f"Blend_Emission_Layer_{i}"
                final_add_node.blend_type = 'ADD'
                final_add_node.inputs['Factor'].default_value = 1.0
                links.new(term1_is_current_color, final_add_node.inputs['A'])
                links.new(term2_multiply.outputs['Result'], final_add_node.inputs['B'])
                accumulated_emission_color_socket = final_add_node.outputs['Result']
            
            
            # LIGHTMAP and INV_LIGHTMAP typically affect BaseColor
            elif current_blend_op_type == BlendMethod.LIGHTMAP.value:
                multiply_node = nodes.new(type='ShaderNodeMix')
                multiply_node.data_type = 'RGBA'
                multiply_node.name = f"Blend_BaseColor_Layer_{i}_Mul1"
                multiply_node.blend_type = 'MULTIPLY'
                multiply_node.inputs['Factor'].default_value = 1.0
                links.new(accumulated_base_color_socket, multiply_node.inputs['A'])
                links.new(current_layer_tinted_color_socket, multiply_node.inputs['B'])

                multiply_by_two_node = nodes.new(type="ShaderNodeMix")
                multiply_by_two_node.data_type = 'RGBA'
                multiply_by_two_node.name = f"Lightmap_x2_Layer_{i}"
                multiply_by_two_node.blend_type = 'MULTIPLY'
                multiply_by_two_node.inputs['A'].default_value = (2.0, 2.0, 2.0, 1.0)
                links.new(multiply_node.outputs['Result'], multiply_by_two_node.inputs['B'])
                multiply_by_two_node.inputs['Factor'].default_value = 1.0
                accumulated_base_color_socket = multiply_by_two_node.outputs['Result']

            elif current_blend_op_type == BlendMethod.INV_LIGHTMAP.value:
                # BaseColor = PrevBaseColor * (1 - TintedCurrentTexColor)
                inv_src_color_node = nodes.new('ShaderNodeVectorMath')
                inv_src_color_node.operation = 'SUBTRACT'
                inv_src_color_node.inputs[0].default_value = (1,1,1)
                links.new(current_layer_tinted_color_socket, inv_src_color_node.inputs[1])

                multiply_node = nodes.new(type='ShaderNodeMix')
                multiply_node.data_type = 'RGBA'
                multiply_node.name = f"Blend_BaseColor_Layer_{i}"
                multiply_node.blend_type = 'MULTIPLY'
                multiply_node.inputs['Factor'].default_value = 1.0
                links.new(accumulated_base_color_socket, multiply_node.inputs['A'])
                links.new(inv_src_color_node.outputs['Vector'], multiply_node.inputs['B'])
                accumulated_base_color_socket = multiply_node.outputs['Result']
            
            elif current_blend_op_type == BlendMethod.INV_BRIGHT.value:
                one_minus_alpha_node = nodes.new(type="ShaderNodeMath")
                one_minus_alpha_node.operation = 'SUBTRACT'
                one_minus_alpha_node.inputs[0].default_value = 1.0
                links.new(alpha_for_blend_factor, one_minus_alpha_node.inputs[1])

                # Contribution = TintedCurrentTexColor * (1 - R3M_ARGB_A)
                scaled_color_node = nodes.new(type='ShaderNodeMix')
                scaled_color_node.data_type = 'RGBA'
                scaled_color_node.blend_type = 'MIX'
                scaled_color_node.inputs['A'].default_value = (0,0,0,1)
                links.new(current_layer_tinted_color_socket, scaled_color_node.inputs['B'])
                links.new(one_minus_alpha_node.outputs['Value'], scaled_color_node.inputs['Factor'])
                
                # Add to accumulated emission
                add_node = nodes.new(type='ShaderNodeMix')
                add_node.data_type = 'RGBA'
                add_node.name = f"Blend_Emission_Layer_{i}"
                add_node.blend_type = 'ADD'
                add_node.inputs['Factor'].default_value = 1.0
                links.new(accumulated_emission_color_socket, add_node.inputs['A'])
                links.new(scaled_color_node.outputs['Result'], add_node.inputs['B'])
                accumulated_emission_color_socket = add_node.outputs['Result']

            # --- End of Layer Blending ---

        # --- Final Connections to BSDF ---
        if bsdf.inputs['Base Color'].is_linked: 
            links.remove(bsdf.inputs['Base Color'].links[0])
        links.new(accumulated_base_color_socket, bsdf.inputs['Base Color'])
        
        if bsdf.inputs['Emission Color'].is_linked: 
            links.remove(bsdf.inputs['Emission Color'].links[0])
        links.new(accumulated_emission_color_socket, bsdf.inputs['Emission Color'])

        if bsdf.inputs['Alpha'].is_linked: 
            links.remove(bsdf.inputs['Alpha'].links[0])
        links.new(accumulated_alpha_socket, bsdf.inputs['Alpha'])
        
        if accumulated_emission_strength_socket != None:
            bsdf.inputs["Emission Strength"].default_value = 3.0

        
        organizer = Utils.NodeOrganizer()
        organizer.arrange_nodes(context, material.node_tree, 300, 300, True)
        
    
    @staticmethod
    def convert_vector3s_to_f(vector3s_collection, scale: float, position: Vector)-> Vector:
        """
        Converts a given vector3short to a float vector.
        """
        return Vector(((vector3s_collection[0]/32767.0)*scale+position[0], (vector3s_collection[1]/32767.0)*scale+position[1], (vector3s_collection[2]/32767.0)*scale+position[2]))
    
    @staticmethod
    def convert_vector3c_to_f(vector3c_collection, scale: float, position: Vector)-> Vector:
        """
        Converts a given vector3char to a float vector.
        """
        return Vector(((vector3c_collection[0]/127.0)*scale+position[0], (vector3c_collection[1]/127.0)*scale+position[1], (vector3c_collection[2]/127.0)*scale+position[2]))
    
    @staticmethod
    def arrange_nodes(node_tree, start_x=0, start_y=0, x_offset=200, y_offset=200):
        """
        Arrange nodes in the node tree with specified offsets.
        
        :param node_tree: The node tree containing the nodes.
        :param start_x: The starting X position for the first node.
        :param start_y: The starting Y position for the first node.
        :param x_offset: The horizontal space between nodes.
        :param y_offset: The vertical space between nodes.
        """
        x = start_x
        y = start_y

        for node in node_tree.nodes:
            node.location = (x, y)
            y -= y_offset

            # If the node has an output, place the connected nodes next to it
            for output in node.outputs:
                if output.is_linked:
                    for link in output.links:
                        linked_node = link.to_node
                        linked_node.location = (x + x_offset, y)
                        y -= y_offset

    
class R3MMaterial:
    """
    k
    """
    def __init__(self):
        self.layer_num: int = 0
        self.flag: int = 0
        self.detail_surface: int = 0
        self.detail_scale: float = 0.0
        self.name: str = ""
        self.texture_layers: list[TextureLayer] = []

class MaterialGroup:
    """
    k
    """
    def __init__(
        self,
        attribute: int = 0,
        number_of_faces: int = 0,
        starting_face_id: int = 0,
        material_id: int = 0,
        light_id: int = 0,
        bounding_box_min: Vector3Int = Vector3Int(0, 0, 0),
        bounding_box_max: Vector3Int = Vector3Int(0, 0, 0),
        position: Vector = Vector((0.0, 0.0, 0.0)),
        scale: float = 1.0,
        animated_object_id: int = 0
    ):
        self.attribute: int = attribute
        self.number_of_faces: int = number_of_faces
        self.starting_face_id: int = starting_face_id
        self.material_id: int = material_id
        self.light_id: int = light_id
        self.bounding_box_min: Vector3Int = bounding_box_min
        self.bounding_box_max: Vector3Int = bounding_box_max
        self.position: Vector = position
        self.scale: float = scale
        self.animated_object_id: int = animated_object_id
    
    @staticmethod
    def bsp_material_from_unpacked_bytes(unpacked_material_group) -> "MaterialGroup":
        material_group = MaterialGroup()
        material_group.attribute = unpacked_material_group[0]
        material_group.number_of_faces = unpacked_material_group[1]
        material_group.starting_face_id = unpacked_material_group[2]
        material_group.material_id = unpacked_material_group[3]
        material_group.light_id = unpacked_material_group[4]
        material_group.bounding_box_min = Vector3Int(unpacked_material_group[5], unpacked_material_group[6], unpacked_material_group[7])
        material_group.bounding_box_max = Vector3Int(unpacked_material_group[8], unpacked_material_group[9], unpacked_material_group[10])
        material_group.position = Vector((unpacked_material_group[11], unpacked_material_group[12], unpacked_material_group[13]))
        material_group.scale = unpacked_material_group[14]
        material_group.animated_object_id = unpacked_material_group[15]
        return material_group
    
    @staticmethod
    def r3e_material_from_unpacked_bytes(unpacked_material_group) -> "MaterialGroup":
        material_group = MaterialGroup()
        material_group.number_of_faces = unpacked_material_group[0]
        material_group.starting_face_id = unpacked_material_group[1]
        material_group.material_id = unpacked_material_group[2]
        material_group.animated_object_id = unpacked_material_group[3]
        material_group.bounding_box_min = Vector((unpacked_material_group[4], unpacked_material_group[5], unpacked_material_group[6]))
        material_group.bounding_box_max = Vector((unpacked_material_group[7], unpacked_material_group[8], unpacked_material_group[9]))
        return material_group
      
        
class TextureLayer:
    """
    Class that holds data for each texture a material has.
    """
    """
    # also, each layer signifies a new texture for a certain material.
    # dwFlag for layers flags:
    
    

    dwAlphaType flags:
    blend_none: 0 opaque
    group_type_reflect: 0x4000 reflective
    """
    def __init__(self):
        self.iTileAniTexNum: int = 0
        self.texture_id: int = 0
        self.alpha_type: int = 0
        self.argb_color: Vector = Vector((0.0, 0.0, 0.0, 0.0))
        self.flags: int = 0
        self.lava_wave_effect_rate: int = 0
        self.lava_wave_effect_speed: int = 0
        self.scroll_u: int = 0
        self.scroll_v: int = 0
        self.uv_rotation: int = 0
        self.uv_starting_scale: int = 0
        self.uv_ending_scale: int = 0
        self.uv_scale_speed: int = 0
        self.metal_effect_size: int = 0
        self.alpha_flicker_rate: int = 0
        self.alpha_flicker_animation: int = 0
        self.animated_texture_frame: int = 0
        self.animated_texture_speed: int = 0
        self.gradient_alpha: int = 0

    def get_texture_layer_from_unpacked_bytes(unpacked_layer) -> "TextureLayer":
        layer = TextureLayer()
        layer.iTileAniTexNum = unpacked_layer[0]
        layer.texture_id = unpacked_layer[1]
        layer.alpha_type = unpacked_layer[2]
        argb = unpacked_layer[3]
        a = ((argb >> 24) & 0xFF)/255.0
        r = ((argb >> 16) & 0xFF)/255.0
        g = ((argb >> 8) & 0xFF)/255.0
        b = (argb & 0xFF)/255.0
        layer.argb_color = Vector((a, r, g, b))
        layer.flags = unpacked_layer[4]
        layer.lava_wave_effect_rate = unpacked_layer[5]
        layer.lava_wave_effect_speed = unpacked_layer[6]
        layer.scroll_u = unpacked_layer[7]
        # Different coordinates system
        layer.scroll_v = -unpacked_layer[8]
        layer.uv_rotation = -unpacked_layer[9]
        layer.uv_starting_scale = unpacked_layer[10]
        layer.uv_ending_scale = unpacked_layer[11]
        layer.uv_scale_speed = unpacked_layer[12]
        layer.metal_effect_size = unpacked_layer[13]
        layer.alpha_flicker_rate = unpacked_layer[14]
        layer.alpha_flicker_animation = unpacked_layer[15]
        layer.animated_texture_frame = unpacked_layer[16]
        layer.animated_texture_speed = unpacked_layer[17]
        layer.gradient_alpha = unpacked_layer[18]
        return layer

class AnimatedObject:
    """
    Class that holds data for materials that have animations.
    """
    def __init__(self):
        self.flag: int = 0
        self.parent: int = 0
        self.frames: int = 0
        self.pos_count: int = 0
        self.rot_count: int = 0
        self.scale_count: int = 0
        self.scale: Vector = Vector((1.0, 1.0, 1.0))
        self.scale_quat: Quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        self.pos: Vector = Vector((0.0, 0.0, 0.0))
        self.quat: Quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        self.pos_offset: int = 0
        self.rot_offset: int = 0
        self.scale_offset: int = 0

class ReadFaceStruct:
    def __init__(self, vertex_amount, vertex_start_id, material_id = -1):
        self.vertex_amount:int = vertex_amount
        self.vertex_start_id:int = vertex_start_id
        self.material_id:int = material_id
        
class ReadEntityStruct:
    def __init__(self, id, scale, position, rot_x, rot_y, bb_min, bb_max):
        self.id: int = id
        self.scale: float = scale
        self.position: Vector = position
        self.rot_x: float = rot_x
        self.rot_y: float = rot_y
        self.bb_min: Vector = bb_min
        self.bb_max: Vector = bb_max
        
class EntityStruct:
    def __init__(self, is_particle, is_file_exists, file_path, fade_start, fade_end, flag, shader_id, factors):
        self.is_particle: bool = True if is_particle != 0 else False
        self.is_file_exists: bool = True if is_file_exists != 0 else False
        self.file_path: str = file_path
        self.fade_start: float = fade_start
        self.fade_end: float = fade_end
        self.flag: int = flag
        self.shader_id: int = shader_id
        self.factors: Tuple[int, int] = factors
        
class EntityRPKIndices:
    def __init__(self, key_name, r3e_index = None, r3m_index = None, r3t_index = None):
        self.key_name = key_name
        self.r3e_index = r3e_index
        self.r3m_index = r3m_index
        self.r3t_index = r3t_index
        
    def is_valid(self) -> bool:
        if self.r3e_index is not None and self.r3m_index is not None and self.r3t_index is not None:
            return True
        return False

class MaterialProperties(Enum):
    ALPHA_TYPE = "alpha_type"
    ARGB_ALPHA = "argb_alpha"
    METAL_EFFECT_SIZE = "metal_effect_size"
    ENVIROMENT_MAT = "enviroment_mat"
    UV_ROTATION = "uv_rotation"
    STARTING_SCALE = "starting_scale"
    ENDING_SCALE = "ending_scale"
    SCALE_SPEED = "scale_speed"
    LAVA_WAVE_RATE = "lava_wave_rate"
    LAVA_WAVE_SPEED = "lava_wave_speed"
    SCROLL_U = "scroll_u"
    SCROLL_V = "scroll_v"

class LayerFlag(Enum):
    ALPHA_SORT_ON		= 0x80000000


    _UV_ENV					= 0x00000001	
    #//_UV_WATER				= 0x00000002
    _UV_LAVA				= 0x00000004
    _UV_METAL_FLOOR			= 0x00000002 # makes object flicker
    _UV_METAL_WALL			= 0x00000008
    _UV_METAL				= 0x0000000a#	//= 0x2 or = 0x8
    _UV_SCROLL_U			= 0x00000010
    _UV_SCROLL_V			= 0x00000020
    _UV_ROTATE				= 0x00000040
    _UV_SCALE				= 0x00000080#	//The scale itself is anime.
    # the = 0x300 below is simply a combination of U and V. If a flag has either ALPHA U or V and you use & you will see if the mat has alpha gradient
    _UV_GRADIENT_ALPHA_UV	= 0x00000300#	//Gladiating the alpha.
    _UV_GRADIENT_ALPHA_U	= 0x00000100#	//Gladiating the alpha.
    _UV_GRADIENT_ALPHA_V	= 0x00000200#	//Gladiating the alpha.

    _ANI_ALPHA_FLICKER		= 0x00000400#	//alpha flashing, literally.
    _ANI_TEXTURE			= 0x00000800#  //Texture animation...
    _ANI_TILE_TEXTURE		= 0x00001000#  //Tile texture animation...

    _MAT_ZWRITE				= 0x00002000#	//Z light the material. The default is not to do it.
    _MAT_ENV_BUMP			= 0x00008000#	//Env bump mapping.(texture is treated as bump)
    #//_MAT_VOLUME_FOG		= 0x00008000#	//The material is volume fog.
    _MAT_TEX_CLAMP			= 0x00010000#	//Texture clamp option.


    _MAT_NO_COLLISON		= 0x00020000#	//Polygons using this material do not perform any collision checks. //Use layer flag 0.
    _MAT_WATER				= 0x00040000#	//Polygons using this material have a water effect.

class BlendMethod(Enum):
    NONE	= 0
    OPAQUE = 1
    DEFAULT = 2
    BRIGHT = 3
    BACK_BRIGHT = 8
    INV_DEFAULT = 5
    INV_BRIGHT = 6
    LIGHTMAP = 10
    INV_LIGHTMAP = 11
    ONLY_TRANSPARENCY = 13
    SHADOW = 14

def register():
    bpy.utils.register_class(RFShared)

def unregister():
    bpy.utils.unregister_class(RFShared)

if __name__ == "__main__":
    register()
