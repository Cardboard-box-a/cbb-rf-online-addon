import sys
import os
import bpy
import importlib

from . import utils

from . import rf_shared
from bpy.types import Panel, Node
from rna_prop_ui import PropertyPanel

from . import bsp
from . import bn_skeleton
from . import msh
from . import ani
from . import r3e
from . import batch_glb_exporter
from . import texture_utils
texture_utils.check_imagemagick()

class ShaderNodePanel:
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_context = "node"
    bl_label = "Custom Properties"
    bl_category = "Properties"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return context.space_data.node_tree and context.active_node and len(context.selected_nodes) > 0
    
class SHADER_PROPERTIES_PANEL_PT_custom_props(ShaderNodePanel, PropertyPanel, Panel):
    _context_path = "space_data.node_tree.nodes.active"
    _property_type = bpy.types.Node
    
def register():
    bsp.register()
    utils.register()
    bn_skeleton.register()
    msh.register()
    ani.register()
    r3e.register()
    batch_glb_exporter.register()
    rf_shared.register()
    bpy.utils.register_class(SHADER_PROPERTIES_PANEL_PT_custom_props)

def unregister():
    bsp.unregister()
    utils.unregister()
    bn_skeleton.unregister()
    msh.unregister()
    ani.unregister()
    r3e.unregister()
    batch_glb_exporter.unregister()
    rf_shared.unregister()
    bpy.utils.unregister_class(SHADER_PROPERTIES_PANEL_PT_custom_props)

if __name__ == "__main__":
    register()
