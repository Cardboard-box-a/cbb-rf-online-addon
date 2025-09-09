# cbb-rf-online-addon

An addon for Blender 4.3.0 (also tested with 4.4.3) to import and export the .msh, .bn (.bbx goes together) and .ani files for RF Online. The entity (R3E) and map (BSP) formats are import only. Import operations also work with drag and drop.

There is code for exporting the BSP format inside the addon code but it is deactivated due to being incomplete. It only reaches so far as actually exporting walkable map geometry (with the BSP structure also built) and baking+exporting the light maps. Unfortunately, Blender proved to not be very suitable for the task of actually being a complete map editor for RF Online, mostly due to complexity issues with the .SPT particle format and other desirable features that would be hard to implement into it, such as mob spawn areas and portals. The R3M materials are also quite hard to simulate, since the original engine rendered the same mesh multiple times for each texture layer they had.
It is possible to reactivate the feature by manually uncommenting the three commented lines in the bsp.py's menu_func_export, register and unregister functions. Expect no support for this feature, as the more proper solution would be writing a proper dedicated software.

__If you have any technical problems with the addon, please open an issue in the repo or message me in any forum you might have found the addon.__

Expected usage order for .BN+MSH/ANI imports: Always import the .BN skeleton first, otherwise the animations and meshes won't latch onto it automatically.
Expected enviroment: The addon uses relative paths by expecting it to be used in the normal RF Online client folder structure. Deviations from this environment can cause certain automatic parts to fail, such as automatic texture extraction and assignment.

In-depth details about features:

 __.BN importer__ :
 -Along with the skeleton, the custom shapes for the bones are also imported and are linked to a collection named "Bone Shapes". You are free to either delete (this will remove the custom bone shapes from the armature too) or simply hide the collection from view (they will still work in this case).

 __.MSH importer__ :
 
 -Will try to latch onto a selected armature present in the current object selection in Blender if the "Apply to Armature in Selected" option was set. Only one armature should be selected at once for this to work. If the armature chosen is found as not compatible because it misses some of the bones used by the mesh, the armature is ignored.
 
 -If the "Apply to Armature in Selected" option is off, the importer will search for the first armature that has all bones used by the mesh and automatically latch onto it. This process does not rely on armature/mesh name and should be pretty reliable.
 
 -In the case the mesh-armature connection fails for any reason, the intended parent name will be written as custom property in each created object from the .msh file.

 -The option "Preserve Parenting" should be turned off if you pretend to export the imported .msh objects to the FBX format. The FBX format does not work very well when objects are parented to bones, so when this option is turned off, parenting of mesh vertices to bone is done through weight assignment instead. Should the option be left on, object-bone parenting will be preserved as-is, as this is the ideal thing to happen if you wish to export the objects as a .msh file again or in other formats that do not have fbx's issues.

 __.ANI importer__ :
 
 -If "Apply to Selected Objects" is on, the animation will attempt to latch onto any objects within the selection, including normal objects and armatures. The reason for that is that R3 Engine .ANI animations support animation for objects without the necessity of an armature. Although even with this option on, there should be only a single armature in the selection.
 
 -If "Apply to Selected Objects" is off, the importer will try to find any collection that has the same base name of the animation (the base name is the first word written before the first occurence of the "_" character. For example "BELMALE_MELEE_JUDGMENT_RMACE_00.ANI" has a base name of "BELMALE"). If it fails to do so, it will throw an error and cancel the import.

 -The "Ignore Not Found Objects" option shouldn't really be turned off, unless you're curious about what objects the animation is missing. Since most animations have some unused objects, the importer will probably fail most of the time.

 -If you've found an animation that has failed the automatic latch but you're sure the armature is compatible (e.g. bellato player animation for the bellato skeleton), simply select the armature and turn "Apply to Selected Objects" on when importing the animation.

__.BSP importer__ :

 -Open the Blender console to see the progress of importing a bsp map.

 -The "Import And Show Light Maps" option will import the map's lightmaps and apply the already baked lights on the map's geometry. This will allow you to see light effects and shadows on the objects.

 -The "Visualize BSP Data (Slow)" option will construct bounding boxes for the Binary Space Partitioning (BSP) present in the .bsp file. It will show normal nodes and leaf nodes, with the leaf nodes accompaning its own geometry. This is useful if you want to see how the BSP actually works, together with how each face goes along with the leaf nodes. This will make the scene rendering extremely slow, however, as there will simply be too many objects in the scene, at least for the larger maps (guild maps usually fare far better).

 -The material groups of the map will be imported to the BSP_MAP collection. Entities will get imported to the BSP_ENTITIES collection. SPT entities are being imported, but are not working properly (and will probably be displayed as mostly black textures), as no reliable way was found to simulate the R3 particle system inside Blender. There is an option for optional importation of SPT entities, which is turned off by default. The entities templates are imported to a separate BSP_ENTITIES_TEMPLATES collection, as they're used only to be copied later as they're placed around the map: Feel free to hide this collection.

 -Animated material groups are also being imported. You can see them working by pressing the play button in Blender's timeline.

__.R3E importer__ :

 -Imports object animations as well.

-----------------------------------------------------------------------------------------------------------------------

__.MSH exporter__ :

 -There are three options that define which objects will be exported as a single .msh file: You can export only the selected objects, the currently active collection (usually the collection of the last clicked object) and all present collections in the scene (each collection will be exported as a separate .msh file. this allows for multiple .msh exportations at the same time). In the first case (only selected objects will get exported), the file name will be whatever name you've given the file when Blender's operator window showed up. In the second and third cases, the name of each file will be the name of the respective collection it works from.

 -As for the format option, MESH08 is slightly lighter in drive usage, however the standard format will work with any version of the game. It's untested, but the MESH08 format might not work on older versions than 2.2.3.2.

 -The texture path that is exported along with each mesh file is the name of the first texture assigned to the BSDF Principled shader in any material.

__.BN exporter__ :

 -Will export all armatures present in the scene by default. Turn on the "Export only selected" option if you wish to export only selected armatures.

 -The name of the exported armatures will always be the name of the collection the armature is located at. Make sure to place the armatures into their appropriate collection.

__.ANI exporter__ :

 -"Action(s) to Export" option: In order, they are: Export only the currently active action (context sensitive, usually the last chosen action in Blender), export all animations that have something to do with the currently active collection, export all animations that have something to do with the objects in the current selection and the final option is to simply export all actions in the scene.

 -The exported actions will have the exact same name they were given in Blender.
