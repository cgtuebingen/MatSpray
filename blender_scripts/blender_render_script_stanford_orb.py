import argparse, sys, os
import json
import bpy
import mathutils
from mathutils import Vector
import numpy as np
import glob
import shutil
from math import radians

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from blender_aov_setup import fix_all_material_texture_colorspaces, get_texture_colorspace_from_material_input, colorspace_to_view_transform, create_aov_entries

# Set up argument parser for command line options
parser = argparse.ArgumentParser(description='Render with multiple environment maps')
parser.add_argument('--obj_file', required=True, help='Path to .obj file to import')
parser.add_argument('--env_dir', required=True, help='Directory with environment maps (.hdr/.exr)')
parser.add_argument('--results_path', required=True, help='Directory to save render results')

# Get all arguments after "--" in the Blender command
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

# Parse the arguments
args = parser.parse_args(argv)

DEBUG = False

VIEWS = 200
RESOLUTION = 512
RESULTS_PATH = args.results_path  # Use the path from command line argument
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
UPPER_VIEWS = True
CIRCLE_FIXED_START = (0,0,0)
CIRCLE_FIXED_END = (.7,0,0)

# Clear the default scene
print("Clearing default scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Delete default light if it exists
for obj in list(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)

# Create a camera immediately (needed for rendering)
print("Creating camera...")
bpy.ops.object.camera_add(location=(0, 4.0, 0.5))
cam = bpy.context.active_object
cam.name = 'Camera'
bpy.context.scene.camera = cam
print(f"Camera created and set as active: {cam.name}")

# Import the OBJ file
obj_file_path = args.obj_file
print(f"Importing OBJ file: {obj_file_path}")

if not os.path.exists(obj_file_path):
    print(f"ERROR: OBJ file not found: {obj_file_path}")
    sys.exit(1)

# Import the OBJ file (Blender 4.2+ uses wm.obj_import instead of import_scene.obj)
bpy.ops.wm.obj_import(filepath=obj_file_path)

# Get the imported objects
imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Imported {len(imported_objects)} mesh objects")

# Center and scale the object to fit in a unit cube
if imported_objects:
    # Select all imported objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objects:
        obj.select_set(True)
    
    # Set active object
    bpy.context.view_layer.objects.active = imported_objects[0]
    
    # Join all meshes into one if multiple
    if len(imported_objects) > 1:
        bpy.ops.object.join()
        print("Joined multiple meshes into one")
    
    # Get the final object
    final_obj = bpy.context.active_object
    
    # Switch to object mode and ensure we're working with mesh data
    bpy.context.view_layer.objects.active = final_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Update the view layer to ensure object transforms are current
    bpy.context.view_layer.update()
    
    # Calculate bounding box in world space
    # bound_box is a property of the object, not the mesh
    # Get bounding box corners in local space
    bbox_corners_local = [Vector(corner) for corner in final_obj.bound_box]
    
    # Transform to world space
    bbox_corners = [final_obj.matrix_world @ corner for corner in bbox_corners_local]
    
    # Calculate bounding box dimensions
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)
    
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    
    # Find the maximum dimension
    max_dimension = max(width, height, depth)
    
    # Calculate center of bounding box
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0
    
    print(f"Original bounding box: width={width:.4f}, height={height:.4f}, depth={depth:.4f}")
    print(f"Maximum dimension: {max_dimension:.4f}")
    print(f"Bounding box center: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
    
    # Calculate scale factor to fit in unit cube (with a small margin, e.g., 0.9 to leave some space)
    scale_factor = 0.9 / max_dimension if max_dimension > 0 else 1.0
    
    print(f"Scale factor: {scale_factor:.6f}")
    
    # First, move object to origin (center it)
    final_obj.location = (-center_x, -center_y, -center_z)
    
    # Apply location transform to mesh data
    bpy.context.view_layer.objects.active = final_obj
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    
    # Then scale uniformly to fit in unit cube
    final_obj.scale = (scale_factor, scale_factor, scale_factor)
    
    # Apply scale transform to mesh data
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Verify final bounding box
    bpy.context.view_layer.update()
    bbox_corners_local_final = [Vector(corner) for corner in final_obj.bound_box]
    bbox_corners_final = [final_obj.matrix_world @ corner for corner in bbox_corners_local_final]
    final_min_x = min(corner.x for corner in bbox_corners_final)
    final_max_x = max(corner.x for corner in bbox_corners_final)
    final_min_y = min(corner.y for corner in bbox_corners_final)
    final_max_y = max(corner.y for corner in bbox_corners_final)
    final_min_z = min(corner.z for corner in bbox_corners_final)
    final_max_z = max(corner.z for corner in bbox_corners_final)
    final_width = final_max_x - final_min_x
    final_height = final_max_y - final_min_y
    final_depth = final_max_z - final_min_z
    final_max_dim = max(final_width, final_height, final_depth)
    
    print(f"Final bounding box: width={final_width:.4f}, height={final_height:.4f}, depth={final_depth:.4f}")
    print(f"Final maximum dimension: {final_max_dim:.4f} (should be ~0.9)")
    print("Object centered at origin and scaled to fit in unit cube")

# Find all environment maps
env_dir = args.env_dir
env_maps = glob.glob(os.path.join(env_dir, '*.hdr')) + glob.glob(os.path.join(env_dir, '*.exr'))
if not env_maps:
    print(f"No environment maps found in {env_dir}")
    sys.exit(1)
else:
    print(f"Found {len(env_maps)} environment maps")

# Function to set environment map as world background
def set_environment_map(env_path):
    # Get the world
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    
    # Enable nodes for the world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes for the environment map
    node_background = nodes.new(type='ShaderNodeBackground')
    node_environment = nodes.new(type='ShaderNodeTexEnvironment')
    node_output = nodes.new(type='ShaderNodeOutputWorld')
    
    # Load the environment map
    try:
        # Remove existing image if it exists to avoid memory issues
        env_name = os.path.basename(env_path).split('.')[0]
        if env_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[env_name])
        node_environment.image = bpy.data.images.load(env_path)
        # Set the color space to Standard
        # node_environment.image.colorspace_settings.name = 'Linear Rec.709'
    except Exception as e:
        print(f"Error loading environment map {env_path}: {e}")
        return None
    
    # Link the nodes
    links.new(node_environment.outputs['Color'], node_background.inputs['Color'])
    links.new(node_background.outputs['Background'], node_output.inputs['Surface'])
    
    print(f"Set environment map: {os.path.basename(env_path)}")
    return os.path.basename(env_path).split('.')[0]  # Return env name without extension

def listify_matrix(matrix):
    return [list(row) for row in matrix]

# Make sure we're using Cycles renderer for proper irradiance support
if bpy.context.scene.render.engine != 'CYCLES':
    print("Switching render engine to Cycles for proper irradiance support")
    bpy.context.scene.render.engine = 'CYCLES'

# Ensure custom AOVs exist on the View Layer and are wired in materials
# This makes 'BaseColor' and 'MetallicRoughness' available in Render Layers outputs
_required_aovs = [("BaseColor", "COLOR"), ("MetallicRoughness", "COLOR"), ("NormalCamera", "COLOR"), ("NormalWorld", "COLOR")]
for _aov_name, _aov_type in _required_aovs:
    if not any(a.name == _aov_name for a in bpy.context.view_layer.aovs):
        _aov = bpy.context.view_layer.aovs.add()
        _aov.name = _aov_name
        _aov.type = _aov_type
        print(f"Added View Layer AOV: {_aov_name}")

# Wire AOV outputs for each unique material (Principled BSDF based)
_processed_mats = set()
for _obj in bpy.context.scene.objects:
    if _obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT", "HAIR", "POINTCLOUD", "VOLUME"}:
        for _slot in _obj.material_slots:
            _mat = _slot.material
            if not _mat or _mat in _processed_mats or not _mat.use_nodes or not _mat.node_tree:
                continue
            # Find Principled BSDF
            _principled = None
            for _node in _mat.node_tree.nodes:
                if _node.type == 'BSDF_PRINCIPLED':
                    _principled = _node
                    break
            if _principled is None:
                # Fallback: create AOVs for non-Principled materials (Diffuse/Emission/Defaults)
                try:
                    _nodes = _mat.node_tree.nodes
                    _links = _mat.node_tree.links
                    # Remove existing AOV outputs with these names to avoid duplicates
                    for _n in list(_nodes):
                        if getattr(_n, 'type', '') == 'OUTPUT_AOV' and getattr(_n, 'aov_name', '') in ["BaseColor", "MetallicRoughness"]:
                            _nodes.remove(_n)
                    # Create outputs
                    _aov_base = _nodes.new("ShaderNodeOutputAOV"); _aov_base.aov_name = "BaseColor"
                    _aov_mr = _nodes.new("ShaderNodeOutputAOV"); _aov_mr.aov_name = "MetallicRoughness"

                    # Determine base color source
                    _basecolor_socket = None
                    _default_base = None
                    _diffuse = next((n for n in _nodes if n.type == 'BSDF_DIFFUSE'), None)
                    if _diffuse is not None:
                        if _diffuse.inputs["Color"].links:
                            _basecolor_socket = _diffuse.inputs["Color"].links[0].from_socket
                        else:
                            _default_base = _diffuse.inputs["Color"].default_value
                    else:
                        _emission = next((n for n in _nodes if n.type == 'EMISSION'), None)
                        if _emission is not None:
                            if _emission.inputs["Color"].links:
                                _basecolor_socket = _emission.inputs["Color"].links[0].from_socket
                            else:
                                _default_base = _emission.inputs["Color"].default_value
                        else:
                            _default_base = tuple(getattr(_mat, 'diffuse_color', (0.8, 0.8, 0.8, 1.0)))

                    if _basecolor_socket is not None:
                        _links.new(_basecolor_socket, _aov_base.inputs[0])
                    else:
                        _rgb = _nodes.new("ShaderNodeRGB")
                        _rgb.outputs[0].default_value = _default_base if _default_base is not None else (0.8, 0.8, 0.8, 1.0)
                        _links.new(_rgb.outputs[0], _aov_base.inputs[0])

                    # Build MetallicRoughness: R=metallic (default 0), G=roughness (try Diffuse/Glossy default), B=0
                    _combine = _nodes.new("ShaderNodeCombineRGB")
                    # Metallic
                    _val_m = _nodes.new("ShaderNodeValue"); _val_m.outputs[0].default_value = 0.0
                    _links.new(_val_m.outputs[0], _combine.inputs[0])
                    # Roughness
                    _rough_default = 0.5
                    if _diffuse is not None:
                        _rough_default = float(_diffuse.inputs["Roughness"].default_value)
                    else:
                        _glossy = next((n for n in _nodes if n.type == 'BSDF_GLOSSY'), None)
                        if _glossy is not None:
                            _rough_default = float(_glossy.inputs["Roughness"].default_value)
                    _val_r = _nodes.new("ShaderNodeValue"); _val_r.outputs[0].default_value = _rough_default
                    _links.new(_val_r.outputs[0], _combine.inputs[1])
                    # Blue = 0
                    _val_b = _nodes.new("ShaderNodeValue"); _val_b.outputs[0].default_value = 0.0
                    _links.new(_val_b.outputs[0], _combine.inputs[2])
                    _links.new(_combine.outputs[0], _aov_mr.inputs[0])

                    _processed_mats.add(_mat)
                    print(f"Added fallback AOVs to non-Principled material '{_mat.name}'")
                except Exception as _e:
                    print(f"Failed to create fallback AOVs for material '{_mat.name}': {_e}")
                continue
            try:
                if create_aov_entries(_mat, _principled):
                    _processed_mats.add(_mat)
            except Exception as _e:
                print(f"Failed to create AOVs for material '{_mat.name}': {_e}")

# Configure GPU compute for headless rendering
def setup_gpu_compute():
    """Setup GPU compute devices for Cycles rendering"""
    import bpy
    
    # Get the preferences
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    
    # Refresh device list
    cycles_prefs.get_devices()
    
    print("Available compute devices:")
    cpu_devices = []
    gpu_devices = []
    
    for device in cycles_prefs.devices:
        print(f"  {device.name} ({device.type}) - {'ENABLED' if device.use else 'DISABLED'}")
        if device.type == 'CPU':
            cpu_devices.append(device)
        elif device.type in {'CUDA', 'OPTIX', 'OPENCL', 'HIP', 'METAL'}:
            gpu_devices.append(device)
    
    # Enable GPU devices and disable CPU if GPUs are available
    gpu_devices_found = False
    if gpu_devices:
        # Enable all GPU devices
        for device in gpu_devices:
            device.use = True
            gpu_devices_found = True
            print(f"  ✓ Enabled GPU device: {device.name} ({device.type})")
        
        # Disable CPU when using GPUs (recommended for performance)
        for device in cpu_devices:
            device.use = False
            print(f"  ✗ Disabled CPU device: {device.name} (using GPUs instead)")
    else:
        # No GPUs found, enable CPU
        for device in cpu_devices:
            device.use = True
            print(f"  ✓ Enabled CPU device: {device.name}")
    
    if gpu_devices_found:
        # Set compute device to GPU
        bpy.context.scene.cycles.device = 'GPU'
        print("✓ Set Cycles compute device to GPU")
        return True
    else:
        print("⚠️ No GPU devices found, using CPU")
        bpy.context.scene.cycles.device = 'CPU'
        return False

# Setup GPU compute
gpu_available = setup_gpu_compute()

# Fix material texture color spaces before rendering
fix_all_material_texture_colorspaces()

# Set enough bounces for indirect lighting to be visible
if hasattr(bpy.context.scene, 'cycles'):
    cycles = bpy.context.scene.cycles
    if cycles.diffuse_bounces < 3:
        print(f"Increasing diffuse bounces from {cycles.diffuse_bounces} to 3 for better indirect lighting")
        cycles.diffuse_bounces = 3
    if cycles.max_bounces < 8:
        print(f"Increasing max bounces from {cycles.max_bounces} to 8")
        cycles.max_bounces = 8

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes
# Note: use_pass_normal = False since we're using custom NormalWorld AOV instead
bpy.context.view_layer.use_pass_z = True
bpy.context.view_layer.use_pass_glossy_direct = True
bpy.context.view_layer.use_pass_emit = True
bpy.context.view_layer.use_pass_shadow = True

# IMPORTANT: Enable diffuse passes for irradiance calculation
print("Enabling diffuse lighting passes for irradiance capture")
bpy.context.view_layer.use_pass_diffuse_direct = True
bpy.context.view_layer.use_pass_diffuse_indirect = True

bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

# Clear old nodes
for node in tree.nodes:
    if node.name.endswith('Output') or node.name.startswith("Normalize"):
        tree.nodes.remove(node)

# Render layers node
render_layers = tree.nodes.new('CompositorNodeRLayers')
render_layers.label = 'Custom Outputs'
render_layers.name = 'Custom Outputs'

# Print available passes for debugging
print("=====================================================")
print("AVAILABLE RENDER PASSES:")
for output in render_layers.outputs:
    if output.enabled:
        print(f"- {output.name}")
print("=====================================================")

# Helper to pick first existing pass name among candidates
def _pick_pass_name(candidates):
    for _c in candidates:
        if _c in render_layers.outputs:
            return _c
    return None

# Detect BaseColor and MetallicRoughness pass names (handle naming variants)
_basecolor_pass = _pick_pass_name(["BaseColor", "Base Color", "Base_Color", "basecolor", "Basecolor"]) 
_metalrough_pass = _pick_pass_name([
    "MetallicRoughness", "Metallic Roughness", "Metal Roughness", "Metallic_Roughness", "MetalRoughness", "MetalRough", "metalroughness"
])
if _basecolor_pass is None:
    print("⚠️ BaseColor AOV not found on Render Layers. Check View Layer AOV names and material AOV node names match exactly.")
else:
    print(f"Detected BaseColor AOV pass: '{_basecolor_pass}'")
if _metalrough_pass is None:
    print("⚠️ MetallicRoughness AOV not found on Render Layers. Check naming/case/spaces.")
else:
    print(f"Detected MetallicRoughness AOV pass: '{_metalrough_pass}'")

# Output nodes
aov_outputs = {
    'Depth': 'depth',
    'NormalWorld': 'normal',  # Use our custom world normal AOV instead of built-in Normal
    'GlossDir': 'glossy_direct',
    'Emit': 'emission'
}
# Add detected AOVs if present
if _basecolor_pass:
    aov_outputs[_basecolor_pass] = 'basecolor'
if _metalrough_pass:
    aov_outputs[_metalrough_pass] = 'metallicroughness'

# Add irradiance outputs - check both possible naming conventions
irradiance_outputs = {}
for direct_name in ["DiffuseDirect", "DiffDir"]:
    if direct_name in render_layers.outputs:
        irradiance_outputs['direct_name'] = direct_name
        aov_outputs[direct_name] = 'irradiance_direct'
        print(f"Found direct diffuse pass: {direct_name}")
        break

for indirect_name in ["DiffuseIndirect", "DiffInd"]:
    if indirect_name in render_layers.outputs:
        irradiance_outputs['indirect_name'] = indirect_name
        aov_outputs[indirect_name] = 'irradiance_indirect'
        print(f"Found indirect diffuse pass: {indirect_name}")
        break

# Determine the appropriate view transform for MetallicRoughness by checking source textures
metallic_roughness_view_transform = "Raw"  # Default

# Check all materials to find roughness texture colorspace
for mat in bpy.data.materials:
    if mat.use_nodes and mat.node_tree:
        roughness_colorspace = get_texture_colorspace_from_material_input(mat, "Roughness")
        metallic_colorspace = get_texture_colorspace_from_material_input(mat, "Metallic")
        
        if roughness_colorspace:
            metallic_roughness_view_transform = colorspace_to_view_transform(roughness_colorspace)
            print(f"Found Roughness texture with colorspace '{roughness_colorspace}' in material '{mat.name}'")
            print(f"Using view transform: '{metallic_roughness_view_transform}' for MetallicRoughness output")
            break
        elif metallic_colorspace:
            metallic_roughness_view_transform = colorspace_to_view_transform(metallic_colorspace)
            print(f"Found Metallic texture with colorspace '{metallic_colorspace}' in material '{mat.name}'")
            print(f"Using view transform: '{metallic_roughness_view_transform}' for MetallicRoughness output")
            break

# Setup AOV output nodes
aov_file_outputs = {}
for output_name, suffix in aov_outputs.items():
    if output_name in render_layers.outputs:
        file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        file_output.label = f'{output_name} Output'
        file_output.name = f'{output_name} Output'
        
        # Set formats based on output type
        if output_name in ['Depth', 'NormalWorld', 'NormalCamera'] or 'irradiance' in suffix:
            # EXR for depth, normals, and irradiance (need high precision/HDR)
            file_output.format.file_format = 'OPEN_EXR'
            file_output.format.color_depth = '32'
            
            # FIXED: Set colorspace to Non-Color for depth and normals only
            if output_name in ['Depth', 'NormalWorld', 'NormalCamera']:
                try:
                    # Correct way to set colorspace for output file nodes
                    file_output.format.linear_colorspace_settings.name = 'Non-Color'
                    print(f"✓ Set colorspace to Non-Color for {output_name}")
                except AttributeError:
                    # Alternative approach if the above doesn't work
                    try:
                        file_output.format.color_management = 'OVERRIDE'
                        file_output.format.linear_colorspace_settings.name = 'Non-Color'
                        print(f"✓ Set colorspace override to Non-Color for {output_name}")
                    except AttributeError:
                        print(f"⚠️ Could not set Non-Color colorspace for {output_name}")
            else:
                # For irradiance passes, keep default Linear colorspace
                print(f"✓ Keeping default Linear colorspace for irradiance {output_name}")
                    
        elif output_name in ['MetallicRoughness', 'BaseColor', 'Image']:
            if output_name == 'MetallicRoughness':
                # PNG for MetallicRoughness - keep default sRGB colorspace
                file_output.format.file_format = 'PNG'
                file_output.format.color_depth = '8'
                file_output.format.color_mode = 'RGBA'
                
                # Fallback to Raw view transform
                try:
                    file_output.format.color_management = 'OVERRIDE'
                    file_output.format.view_settings.view_transform = 'Raw'
                    print(f"⚠️ Fallback: Set Raw view transform for {output_name}")
                except AttributeError:
                    print(f"⚠️ Could not set colorspace for {output_name}")
            
            elif output_name in ['BaseColor', 'Image']:
                # PNG for base color and composite with sRGB colorspace
                file_output.format.file_format = 'PNG'
                file_output.format.color_depth = '8'
                file_output.format.color_mode = 'RGBA'
                try:
                    # Keep sRGB colorspace for color images
                    file_output.format.color_settings.name = 'sRGB'
                    print(f"✓ Set colorspace to sRGB for {output_name}")
                except AttributeError:
                    # Fallback to Standard view transform
                    try:
                        file_output.format.color_management = 'OVERRIDE'
                        file_output.format.view_settings.view_transform = 'Standard'
                        print(f"⚠️ Fallback: Set Standard view transform for {output_name}")
                    except AttributeError:
                        print(f"⚠️ Could not set colorspace for {output_name}")
        else:
            file_output.format.file_format = FORMAT

        # Connect directly to the render layer outputs (may be overridden below for BaseColor)
        links.new(render_layers.outputs[output_name], file_output.inputs[0])
        
        aov_file_outputs[output_name] = {'node': file_output, 'suffix': suffix}
        print(f"Added output node for {output_name} with suffix _{suffix}")

# Special handling: BaseColor unpremultiply while preserving original alpha (no opaque forcing)
try:
    if _basecolor_pass and _basecolor_pass in aov_file_outputs and _basecolor_pass in render_layers.outputs:
        _basecolor_socket = render_layers.outputs[_basecolor_pass]
        _basecolor_out_node = aov_file_outputs[_basecolor_pass]['node']

        # Separate RGBA
        _sep = tree.nodes.new("CompositorNodeSepRGBA")
        links.new(_basecolor_socket, _sep.inputs[0])

        # Unpremultiply: divide RGB by max(alpha, eps)
        _max = tree.nodes.new("CompositorNodeMath"); _max.operation = 'MAXIMUM'; _max.inputs[1].default_value = 0.0001
        links.new(_sep.outputs[3], _max.inputs[0])

        _div_r = tree.nodes.new("CompositorNodeMath"); _div_r.operation = 'DIVIDE'
        _div_g = tree.nodes.new("CompositorNodeMath"); _div_g.operation = 'DIVIDE'
        _div_b = tree.nodes.new("CompositorNodeMath"); _div_b.operation = 'DIVIDE'
        links.new(_sep.outputs[0], _div_r.inputs[0]); links.new(_max.outputs[0], _div_r.inputs[1])
        links.new(_sep.outputs[1], _div_g.inputs[0]); links.new(_max.outputs[0], _div_g.inputs[1])
        links.new(_sep.outputs[2], _div_b.inputs[0]); links.new(_max.outputs[0], _div_b.inputs[1])

        # Combine back with original alpha
        _comb = tree.nodes.new("CompositorNodeCombRGBA")
        links.new(_div_r.outputs[0], _comb.inputs[0])
        links.new(_div_g.outputs[0], _comb.inputs[1])
        links.new(_div_b.outputs[0], _comb.inputs[2])
        links.new(_sep.outputs[3], _comb.inputs[3])

        # Connect to file output
        links.new(_comb.outputs[0], _basecolor_out_node.inputs[0])
except Exception as _e:
    print(f"BaseColor unpremultiply wiring failed: {_e}")

# MetallicRoughness: unpremultiply while preserving original alpha
try:
    if _metalrough_pass and _metalrough_pass in render_layers.outputs and _metalrough_pass in aov_file_outputs:
        _mr_socket = render_layers.outputs[_metalrough_pass]
        _mr_out_node = aov_file_outputs[_metalrough_pass]['node']

        _sep_mr = tree.nodes.new("CompositorNodeSepRGBA")
        links.new(_mr_socket, _sep_mr.inputs[0])

        _max_mr = tree.nodes.new("CompositorNodeMath"); _max_mr.operation = 'MAXIMUM'; _max_mr.inputs[1].default_value = 0.0001
        links.new(_sep_mr.outputs[3], _max_mr.inputs[0])

        _div_r_mr = tree.nodes.new("CompositorNodeMath"); _div_r_mr.operation = 'DIVIDE'
        _div_g_mr = tree.nodes.new("CompositorNodeMath"); _div_g_mr.operation = 'DIVIDE'
        _div_b_mr = tree.nodes.new("CompositorNodeMath"); _div_b_mr.operation = 'DIVIDE'
        links.new(_sep_mr.outputs[0], _div_r_mr.inputs[0]); links.new(_max_mr.outputs[0], _div_r_mr.inputs[1])
        links.new(_sep_mr.outputs[1], _div_g_mr.inputs[0]); links.new(_max_mr.outputs[0], _div_g_mr.inputs[1])
        links.new(_sep_mr.outputs[2], _div_b_mr.inputs[0]); links.new(_max_mr.outputs[0], _div_b_mr.inputs[1])

        _comb_mr = tree.nodes.new("CompositorNodeCombRGBA")
        links.new(_div_r_mr.outputs[0], _comb_mr.inputs[0])
        links.new(_div_g_mr.outputs[0], _comb_mr.inputs[1])
        links.new(_div_b_mr.outputs[0], _comb_mr.inputs[2])
        links.new(_sep_mr.outputs[3], _comb_mr.inputs[3])

        links.new(_comb_mr.outputs[0], _mr_out_node.inputs[0])
except Exception as _e:
    print(f"MetallicRoughness unpremultiply wiring failed: {_e}")

# Add total irradiance (direct + indirect) if both are available
if 'direct_name' in irradiance_outputs and 'indirect_name' in irradiance_outputs:
    direct_name = irradiance_outputs['direct_name']
    indirect_name = irradiance_outputs['indirect_name']
    
    # Create mixer node for total irradiance
    total_irradiance_add = tree.nodes.new("CompositorNodeMixRGB")
    total_irradiance_add.name = "Total Irradiance Mixer"
    total_irradiance_add.blend_type = 'ADD'
    total_irradiance_add.inputs[0].default_value = 1.0  # Factor
    
    # Connect inputs to the mixer
    links.new(render_layers.outputs[direct_name], total_irradiance_add.inputs[1])
    links.new(render_layers.outputs[indirect_name], total_irradiance_add.inputs[2])
    
    # Create output node for total irradiance
    total_output = tree.nodes.new(type="CompositorNodeOutputFile")
    total_output.name = "TotalIrradiance Output"
    total_output.format.file_format = 'OPEN_EXR'
    total_output.format.color_depth = '32'
    
    # Connect mixer to output
    links.new(total_irradiance_add.outputs[0], total_output.inputs[0])
    
    # Add to outputs dictionary
    aov_file_outputs["TotalIrradiance"] = {'node': total_output, 'suffix': 'irradiance_total'}
    print("Set up total irradiance (direct + indirect)")

# Background
bpy.context.scene.render.dither_intensity = 0.0
# Note: film_transparent will be set per-render (True for images with alpha, False for images_bg)

# Delete Empty objects
# Safely remove stray empties without selecting (selection fails if not in view layer)
for _obj in list(bpy.data.objects):
    if _obj.type == 'EMPTY' and _obj.name.startswith('Empty'):
        _name = _obj.name
        try:
            # Unlink from all collections first
            for _col in list(_obj.users_collection):
                _col.objects.unlink(_obj)
            # Remove the object datablock
            bpy.data.objects.remove(_obj, do_unlink=True)
            print(f"Removed stray empty: {_name}")
        except Exception as e:
            print(f"Could not remove {_name}: {e}")

def parent_obj_to_camera(b_camera):
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = (0, 0, 0)
    b_camera.parent = b_empty
    bpy.context.scene.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty

scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

# Get the camera (should already exist from earlier creation)
if 'Camera' in bpy.data.objects:
    cam = bpy.data.objects['Camera']
    cam.location = (0, 4.0, 0.5)
else:
    # Fallback: create camera if it doesn't exist
    bpy.ops.object.camera_add(location=(0, 4.0, 0.5))
    cam = bpy.context.active_object
    cam.name = 'Camera'

# Ensure the camera is set as the active camera for the scene
bpy.context.scene.camera = cam
print(f"Camera set: {cam.name} at location {cam.location}")

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'

stepsize = 360.0 / VIEWS
vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]

# Main function to handle rendering with each environment map
def render_with_env_maps():
    # Set color management settings
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.display_settings.display_device = 'sRGB'
    
    for env_map in env_maps:
        # Set the current environment map
        env_name = set_environment_map(env_map)
        if not env_name:
            print(f"Skipping {env_map} due to loading error")
            continue
        
        # Update base path for this environment
        # For the first env map, use it as 'base' for training
        if env_map == env_maps[0]:
            # Use base/train structure for the first environment map
            base_path = os.path.join(RESULTS_PATH, 'base')
            current_fp = os.path.join(base_path, 'train')
        else:
            # For other env maps, use envmap_name/test structure
            env_path = os.path.join(RESULTS_PATH, env_name)
            if not os.path.exists(env_path):
                os.makedirs(env_path)
            current_fp = os.path.join(env_path, 'test')
        
        # Create directories if they don't exist
        if not os.path.exists(current_fp):
            os.makedirs(current_fp)

        # Create subdirectories for each render type
        image_dir = os.path.join(current_fp, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        out_data = {
            'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
            'environment_map': env_name
        }

        # Create directories for each AOV
        for output_name, output_data in aov_file_outputs.items():
            suffix = output_data['suffix']
            dir_path = os.path.join(current_fp, f'base_{suffix}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")

        # Update output nodes to use the current environment path
        for output_name, output_data in aov_file_outputs.items():
            suffix = output_data['suffix']
            output_dir = os.path.join(current_fp, f'base_{suffix}')
            output_data['node'].base_path = output_dir

        out_data['frames'] = []

        # Set initial rotation
        b_empty.rotation_euler = CIRCLE_FIXED_START

        # --- Main rendering loop ---
        for i in range(0, VIEWS):
            if DEBUG:
                i = np.random.randint(0,VIEWS)
                b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
                b_empty.rotation_euler[2] = radians(2*stepsize*i)
            else:
                b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
                if i > 0:
                    b_empty.rotation_euler[2] += radians(stepsize)

            print(f"ENV: {env_name} - Rotation {(stepsize * i)}, {radians(stepsize * i)}")
            
            # Set the main render output path to the 'images' directory
            # Render with transparent background to get alpha channel (like nerf_synthetic)
            scene.view_settings.view_transform = 'Standard'
            scene.render.film_transparent = True
            scene.render.filepath = os.path.join(image_dir, f'r_{i}')
            
            for output_name, output_data in aov_file_outputs.items():
                suffix = output_data['suffix']
                # Set the output directory for this specific render type
                output_dir = os.path.join(current_fp, f'base_{suffix}')
                output_data['node'].base_path = output_dir
                if len(output_data['node'].file_slots) > 0:
                    output_data['node'].file_slots[0].path = f"r_{i}_{suffix}"

            if DEBUG:
                break
            else:
                bpy.ops.render.render(write_still=True)
                
                # Handle file renaming in their respective directories
                for output_name, output_data in aov_file_outputs.items():
                    suffix = output_data['suffix']
                    output_dir = os.path.join(current_fp, f'base_{suffix}')
                    pattern = os.path.join(output_dir, f"r_{i}_{suffix}*")
                    files = glob.glob(pattern)

                    for file_path in files:
                        file_name = os.path.basename(file_path)
                        base, ext = os.path.splitext(file_name)
                        if "-" in base:
                            clean_name = base.split('-')[0] + ext
                            target_path = os.path.join(output_dir, clean_name)
                            try:
                                if os.path.exists(target_path):
                                    os.remove(target_path)
                                shutil.copy2(file_path, target_path)
                                os.remove(file_path)
                                print(f"Renamed {file_name} to {clean_name} in {output_dir}")
                            except Exception as e:
                                print(f"Error renaming {file_name}: {e}")

            frame_data = {
                'file_path': os.path.join(current_fp, 'images', f'r_{i}'),
                'rotation': radians(stepsize),
                'transform_matrix': listify_matrix(cam.matrix_world)
            }
            out_data['frames'].append(frame_data)

        if not DEBUG:
            # Save transforms file in the appropriate location
            if env_map == env_maps[0]:
                # For base/train, save as transforms_train.json
                transforms_file = os.path.join(current_fp, 'transforms_train.json')
            else:
                # For test directories, save as transforms_test.json
                transforms_file = os.path.join(current_fp, '../transforms_test.json')
            with open(transforms_file, 'w') as out_file:
                json.dump(out_data, out_file, indent=4)

# Call the render function
render_with_env_maps()

print("Rendering complete!")
