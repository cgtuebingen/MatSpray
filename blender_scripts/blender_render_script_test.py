import argparse, sys, os
import json
import bpy
import mathutils
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
bpy.context.scene.render.film_transparent = False

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

cam = scene.objects['Camera']
cam.location = (0, 4.0, 0.5)
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
        env_path = os.path.join(RESULTS_PATH, env_name)
        
        # Create environment folder if it doesn't exist
        if not os.path.exists(env_path):
            os.makedirs(env_path)
            
        # Create test subfolder for all renders
        test_path = os.path.join(env_path, 'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)
            
        # Now use test_path as the base for all AOVs
        current_fp = test_path

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
            with open(os.path.join(current_fp, '../transforms_test.json'), 'w') as out_file:
                json.dump(out_data, out_file, indent=4)

# Helper: determine if a material uses Transparent BSDF or mixes with it
def _material_uses_transparency(mat):
    try:
        if not mat.use_nodes or not mat.node_tree:
            return False
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_TRANSPARENT':
                return True
            if n.type == 'MIX_SHADER':
                # Check if either shader input ultimately comes from a Transparent BSDF
                for idx in (1, 2):
                    if n.inputs[idx].links:
                        src_node = n.inputs[idx].links[0].from_node
                        if src_node.type == 'BSDF_TRANSPARENT':
                            return True
        return False
    except Exception:
        return False

# Helper: create a copy of material with transparent mixes bypassed for AOV capture
def _create_aov_mat_copy(original_mat):
    mat_copy = original_mat.copy()
    nodes = mat_copy.node_tree.nodes
    links = mat_copy.node_tree.links
    # Bypass Mix Shader nodes that mix with Transparent BSDF: replace outputs with the non-transparent input
    try:
        # Iterate a static list because we'll edit the tree
        for n in list(nodes):
            if n.type == 'MIX_SHADER':
                in1 = n.inputs[1]; in2 = n.inputs[2]
                src1 = in1.links[0].from_node if in1.links else None
                src2 = in2.links[0].from_node if in2.links else None
                replace_socket = None
                if src1 and src1.type == 'BSDF_TRANSPARENT' and in2.links:
                    replace_socket = in2.links[0].from_socket
                elif src2 and src2.type == 'BSDF_TRANSPARENT' and in1.links:
                    replace_socket = in1.links[0].from_socket
                if replace_socket is not None:
                    # Redirect all outgoing links from mix output to the kept socket
                    for out_link in list(n.outputs[0].links):
                        links.new(replace_socket, out_link.to_socket)
                        links.remove(out_link)
        # Force Principled alpha to 1.0 if present
        principled = None
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                principled = n
                break
        if principled is not None and 'Alpha' in principled.inputs:
            principled.inputs['Alpha'].default_value = 1.0
        # Optionally set material blend/shadow methods to opaque for AOV capture
        mat_copy.blend_method = 'OPAQUE'
        if hasattr(mat_copy, 'shadow_method'):
            mat_copy.shadow_method = 'OPAQUE'
        # Ensure AOV nodes exist on the copy (in case original lacked or were removed)
        if principled is not None:
            try:
                created = create_aov_entries(mat_copy, principled)
                if created:
                    print(f"Ensured AOV nodes on copy of '{original_mat.name}'")
            except Exception as e:
                print(f"create_aov_entries on copy failed for '{original_mat.name}': {e}")
        else:
            # Minimal fallback AOVs
            try:
                aov_base = nodes.new("ShaderNodeOutputAOV"); aov_base.aov_name = "BaseColor"
                rgb = nodes.new("ShaderNodeRGB"); rgb.outputs[0].default_value = tuple(getattr(original_mat, 'diffuse_color', (0.8, 0.8, 0.8, 1.0)))
                links.new(rgb.outputs[0], aov_base.inputs[0])
                aov_mr = nodes.new("ShaderNodeOutputAOV"); aov_mr.aov_name = "MetallicRoughness"
                comb = nodes.new("ShaderNodeCombineRGB")
                vm = nodes.new("ShaderNodeValue"); vm.outputs[0].default_value = 0.0
                vr = nodes.new("ShaderNodeValue"); vr.outputs[0].default_value = 0.5
                vb = nodes.new("ShaderNodeValue"); vb.outputs[0].default_value = 0.0
                links.new(vm.outputs[0], comb.inputs[0])
                links.new(vr.outputs[0], comb.inputs[1])
                links.new(vb.outputs[0], comb.inputs[2])
                links.new(comb.outputs[0], aov_mr.inputs[0])
            except Exception as e:
                print(f"Fallback AOVs on copy failed for '{original_mat.name}': {e}")
    except Exception as e:
        print(f"AOV mat copy rewrite failed for '{original_mat.name}': {e}")
    return mat_copy

# Apply AOV overrides: duplicate transparent materials, assign copies to objects and GN inputs
def _apply_aov_overrides():
    context = {
        'orig_to_copy': {},
        'obj_restore': [],  # list of (obj, slot_index, original_mat)
        'gn_restore': []    # list of (node_input, original_mat)
    }
    # Build set of materials used in scene (including GN referenced)
    used_mats = set()
    for obj in bpy.context.scene.objects:
        for slot in obj.material_slots:
            if slot.material:
                used_mats.add(slot.material)
    # Add GN referenced materials
    try:
        gn_mats = set()
        gn_objs = set(); gn_cols = set()
        for obj in bpy.context.scene.objects:
            for mod in getattr(obj, 'modifiers', []):
                if getattr(mod, 'type', '') == 'NODES' and getattr(mod, 'node_group', None) is not None:
                    _collect_from_gn_tree(mod.node_group, gn_mats, gn_objs, gn_cols)
        for o in list(gn_objs):
            for slot in getattr(o, 'material_slots', []):
                if slot.material:
                    gn_mats.add(slot.material)
        for c in list(gn_cols):
            for o in getattr(c, 'objects', []):
                for slot in getattr(o, 'material_slots', []):
                    if slot.material:
                        gn_mats.add(slot.material)
        used_mats.update(gn_mats)
    except Exception as e:
        print(f"AOV override GN scan failed: {e}")

    # Create copies for materials using transparency
    for mat in list(used_mats):
        if not mat:
            continue
        if _material_uses_transparency(mat):
            try:
                mat_copy = _create_aov_mat_copy(mat)
                context['orig_to_copy'][mat] = mat_copy
            except Exception as e:
                print(f"Failed to duplicate material '{mat.name}': {e}")

    # Reassign object material slots
    for obj in bpy.context.scene.objects:
        for idx, slot in enumerate(obj.material_slots):
            orig = slot.material
            if orig in context['orig_to_copy']:
                context['obj_restore'].append((obj, idx, orig))
                slot.material = context['orig_to_copy'][orig]

    # Reassign GN default material inputs
    try:
        for obj in bpy.context.scene.objects:
            for mod in getattr(obj, 'modifiers', []):
                if getattr(mod, 'type', '') == 'NODES' and getattr(mod, 'node_group', None) is not None:
                    ng = mod.node_group
                    for node in ng.nodes:
                        for inp in getattr(node, 'inputs', []):
                            try:
                                val = inp.default_value
                                if isinstance(val, bpy.types.Material) and val in context['orig_to_copy']:
                                    context['gn_restore'].append((inp, val))
                                    inp.default_value = context['orig_to_copy'][val]
                            except Exception:
                                pass
    except Exception as e:
        print(f"AOV override GN reassign failed: {e}")

    return context

# Restore materials and remove copies
def _restore_aov_overrides(context):
    try:
        for obj, idx, orig in context.get('obj_restore', []):
            if idx < len(obj.material_slots):
                obj.material_slots[idx].material = orig
    except Exception as e:
        print(f"Restore object slots failed: {e}")
    try:
        for inp, orig in context.get('gn_restore', []):
            try:
                inp.default_value = orig
            except Exception:
                pass
    except Exception as e:
        print(f"Restore GN inputs failed: {e}")
    # Remove copies
    try:
        for orig, copy in context.get('orig_to_copy', {}).items():
            try:
                bpy.data.materials.remove(copy)
            except Exception:
                pass
    except Exception as e:
        print(f"Cleanup copies failed: {e}")

# Call the render function
# Disable AOV transparency overrides to preserve true transparency in outputs
# _aov_ctx = _apply_aov_overrides()
# try:
#     render_with_env_maps()
# finally:
#     _restore_aov_overrides(_aov_ctx)
render_with_env_maps()

# Pack all external data and save blend file after rendering is complete
print("🗜️ Packing all external data into blend file...")
try:
    bpy.ops.file.pack_all()
    print("✅ Successfully packed all external data")
    
    # Create output path for the packed blend file
    blend_output_path = os.path.join(RESULTS_PATH, "packed_scene.blend")
    
    print(f"💾 Saving packed blend file to: {blend_output_path}")
    bpy.ops.wm.save_as_mainfile(filepath=blend_output_path)
    print("✅ Successfully saved packed blend file")
    
except Exception as e:
    print(f"❌ Error packing/saving blend file: {e}")

def fix_all_material_texture_colorspaces():
    """
    Scan all materials in the scene and fix colorspace settings for data textures.
    Call this before rendering to ensure metallic, roughness, and normal textures use Non-Color.
    """
    print("🔧 Fixing texture color spaces for all materials...")
    
    fixed_count = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            continue
            
        # Find Principled BSDF node
        principled_bsdf = None
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled_bsdf = node
                break
        
        if not principled_bsdf:
            continue
            
        print(f"  📄 Processing material: '{mat.name}'")
        
        # Check and fix data map inputs
        data_inputs = ["Metallic", "Roughness", "Normal", "Specular", "Alpha"]
        
        for input_name in data_inputs:
            if input_name in principled_bsdf.inputs and principled_bsdf.inputs[input_name].links:
                input_node = principled_bsdf.inputs[input_name].links[0].from_node
                print(f"    🔍 Checking {input_name} input:")
                traverse_and_fix_colorspace_recursive(input_node)
                
        fixed_count += 1
    
    print(f"✅ Processed {fixed_count} materials for texture color space fixes")
    return fixed_count

def fix_texture_colorspace_for_data_maps(node):
    """
    Fix color space settings for texture image nodes that contain data (non-color) maps.
    This should be called for nodes connected to Metallic, Roughness, Normal, Height, etc.
    """
    if node.type == 'TEX_IMAGE' and node.image:
        if node.image.colorspace_settings.name != 'Non-Color':
            old_colorspace = node.image.colorspace_settings.name
            node.image.colorspace_settings.name = 'Non-Color'
            print(f"      ✓ Fixed colorspace: '{node.image.name}' from '{old_colorspace}' to 'Non-Color'")
        else:
            print(f"      ✓ Colorspace already correct: '{node.image.name}' (Non-Color)")
    elif node.type == 'TEX_IMAGE' and not node.image:
        print(f"      ⚠️ Image texture node has no image loaded")

def traverse_and_fix_colorspace_recursive(node, visited_nodes=None):
    """
    Recursively traverse node tree to find and fix all image texture nodes.
    This handles cases where there might be ColorRamp, Math, or other nodes between
    the image texture and the Principled BSDF input.
    """
    if visited_nodes is None:
        visited_nodes = set()
    
    if node in visited_nodes:
        return
    visited_nodes.add(node)
    
    # Fix this node if it's an image texture
    if node.type == 'TEX_IMAGE':
        fix_texture_colorspace_for_data_maps(node)
    
    # Recursively check all input nodes
    for input_socket in node.inputs:
        if input_socket.links:
            for link in input_socket.links:
                traverse_and_fix_colorspace_recursive(link.from_node, visited_nodes)

# Convert collection instances to real objects so their materials can be edited for AOVs
try:
    _instancers = [o for o in bpy.context.scene.objects if o.type == 'EMPTY' and getattr(o, 'instance_type', 'NONE') == 'COLLECTION' and o.instance_collection is not None]
    if _instancers:
        # Deselect all
        for o in bpy.context.selected_objects:
            o.select_set(False)
        # Select each instancer and make real one by one to avoid context issues
        for _inst in _instancers:
            try:
                bpy.context.view_layer.objects.active = _inst
                _inst.select_set(True)
                bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)
                _inst.select_set(False)
                print(f"Made collection instance real: {_inst.name}")
            except Exception as _e:
                print(f"Could not make instance real for '{_inst.name}': {_e}")
except Exception as _e:
    print(f"Instance-real conversion skipped due to error: {_e}")

# Helper: collect materials, objects, collections referenced by Geometry Nodes
def _collect_from_gn_tree(_tree, _mats, _objs, _cols, _visited=None):
    if _tree is None:
        return
    if _visited is None:
        _visited = set()
    if _tree in _visited:
        return
    _visited.add(_tree)
    try:
        for _node in _tree.nodes:
            # Recurse into group nodes
            if getattr(_node, 'type', '') == 'GROUP' and getattr(_node, 'node_tree', None) is not None:
                _collect_from_gn_tree(_node.node_tree, _mats, _objs, _cols, _visited)
            # Inspect inputs for Material/Object/Collection defaults
            for _inp in getattr(_node, 'inputs', []):
                try:
                    _val = _inp.default_value
                    # Material references
                    if isinstance(_val, bpy.types.Material):
                        _mats.add(_val)
                    # Object references
                    if isinstance(_val, bpy.types.Object):
                        _objs.add(_val)
                    # Collection references
                    if isinstance(_val, bpy.types.Collection):
                        _cols.add(_val)
                except Exception:
                    pass
            # Some GN nodes expose direct properties (rare)
            for _attr in ('material', 'object', 'collection'):
                if hasattr(_node, _attr):
                    try:
                        _val = getattr(_node, _attr)
                        if isinstance(_val, bpy.types.Material):
                            _mats.add(_val)
                        if isinstance(_val, bpy.types.Object):
                            _objs.add(_val)
                        if isinstance(_val, bpy.types.Collection):
                            _cols.add(_val)
                    except Exception:
                        pass
    except Exception as _e:
        print(f"Geometry Nodes scan error: {_e}")

# Gather materials assigned inside Geometry Nodes (and from instanced objects/collections)
_gn_materials = set()
_gn_objects = set()
_gn_collections = set()
for _obj in bpy.context.scene.objects:
    for _mod in getattr(_obj, 'modifiers', []):
        if getattr(_mod, 'type', '') == 'NODES' and getattr(_mod, 'node_group', None) is not None:
            _collect_from_gn_tree(_mod.node_group, _gn_materials, _gn_objects, _gn_collections)

# Also gather materials from referenced objects/collections
for _o in list(_gn_objects):
    try:
        for _slot in _o.material_slots:
            if _slot.material:
                _gn_materials.add(_slot.material)
    except Exception:
        pass

for _c in list(_gn_collections):
    try:
        for _o in _c.objects:
            for _slot in _o.material_slots:
                if _slot.material:
                    _gn_materials.add(_slot.material)
    except Exception:
        pass

if _gn_materials:
    print(f"Geometry Nodes referenced materials detected: {[m.name for m in _gn_materials]}")

# Ensure GN-referenced materials are localized if linked and wired with AOVs
_localized_gn_map = {}
for _mat in list(_gn_materials):
    if not _mat:
        continue
    # Localize if linked
    if getattr(_mat, 'library', None) is not None:
        _key = _mat.name_full if hasattr(_mat, 'name_full') else _mat.name
        if _key not in _localized_gn_map:
            _local = _mat.copy()
            _localized_gn_map[_key] = _local
            print(f"Localized GN material '{_key}' → '{_local.name}'")
        else:
            _local = _localized_gn_map[_key]
        # Reassign in objects and GN references where possible (object slots handled; GN inputs will use default_value references which will still point to old; that's okay for render but we cannot mutate node defaults safely here)
        for _obj in bpy.context.scene.objects:
            for _slot in _obj.material_slots:
                if _slot.material == _mat:
                    _slot.material = _local
        _mat = _local
    # Add AOVs using principled if present; else fallback
    try:
        if not _mat.use_nodes or not _mat.node_tree:
            continue
        _nodes = _mat.node_tree.nodes
        _links = _mat.node_tree.links
        _principled = None
        for _node in _nodes:
            if _node.type == 'BSDF_PRINCIPLED':
                _principled = _node
                break
        if _principled is not None:
            created = False
            try:
                created = create_aov_entries(_mat, _principled)
            except Exception as _e:
                print(f"create_aov_entries failed for GN mat '{_mat.name}': {_e}")
            if created:
                print(f"Wired AOVs for GN Principled material '{_mat.name}'")
                continue
        # Fallback minimal AOVs
        # Remove old duplicates
        for _n in list(_nodes):
            if getattr(_n, 'type', '') == 'OUTPUT_AOV' and getattr(_n, 'aov_name', '') in ["BaseColor", "MetallicRoughness"]:
                _nodes.remove(_n)
        _aov_base = _nodes.new("ShaderNodeOutputAOV"); _aov_base.aov_name = "BaseColor"
        _rgb = _nodes.new("ShaderNodeRGB"); _rgb.outputs[0].default_value = tuple(getattr(_mat, 'diffuse_color', (0.8, 0.8, 0.8, 1.0)))
        _links.new(_rgb.outputs[0], _aov_base.inputs[0])
        _aov_mr = _nodes.new("ShaderNodeOutputAOV"); _aov_mr.aov_name = "MetallicRoughness"
        _comb = _nodes.new("ShaderNodeCombineRGB")
        _vm = _nodes.new("ShaderNodeValue"); _vm.outputs[0].default_value = 0.0
        _vr = _nodes.new("ShaderNodeValue"); _vr.outputs[0].default_value = 0.5
        _vb = _nodes.new("ShaderNodeValue"); _vb.outputs[0].default_value = 0.0
        _links.new(_vm.outputs[0], _comb.inputs[0])
        _links.new(_vr.outputs[0], _comb.inputs[1])
        _links.new(_vb.outputs[0], _comb.inputs[2])
        _links.new(_comb.outputs[0], _aov_mr.inputs[0])
        print(f"Added fallback AOVs to GN material '{_mat.name}'")
    except Exception as _e:
        print(f"Failed to wire AOVs for GN material '{_mat.name}': {_e}")

# Collect materials from evaluated depsgraph (instances/geometry nodes at render time)
def _ensure_aovs_from_evaluated_instances():
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        eval_mats = set()
        for inst in dg.object_instances:
            try:
                obj_eval = inst.object.evaluated_get(dg) if inst.object else None
                if obj_eval and obj_eval.type == 'MESH':
                    # Prefer evaluated mesh
                    me = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
                    # Collect from mesh materials if present
                    if me is not None:
                        for m in me.materials:
                            if m is not None:
                                eval_mats.add(m)
                        obj_eval.to_mesh_clear()
                    else:
                        # Fallback to slots on evaluated object
                        for slot in obj_eval.material_slots:
                            if slot.material:
                                eval_mats.add(slot.material)
            except Exception as e:
                print(f"Depsgraph instance scan error: {e}")
        if eval_mats:
            print(f"Evaluated materials detected: {[m.name for m in eval_mats]}")
        # Ensure AOVs for all evaluated materials
        for mat in list(eval_mats):
            try:
                if not mat or not mat.use_nodes or not mat.node_tree:
                    continue
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                principled = None
                for n in nodes:
                    if n.type == 'BSDF_PRINCIPLED':
                        principled = n
                        break
                if principled is not None:
                    try:
                        created = create_aov_entries(mat, principled)
                        if created:
                            print(f"Wired AOVs for evaluated Principled material '{mat.name}'")
                            continue
                    except Exception as e:
                        print(f"create_aov_entries failed for evaluated mat '{mat.name}': {e}")
                # Fallback minimal AOVs
                for n in list(nodes):
                    if getattr(n, 'type', '') == 'OUTPUT_AOV' and getattr(n, 'aov_name', '') in ["BaseColor", "MetallicRoughness"]:
                        nodes.remove(n)
                aov_base = nodes.new("ShaderNodeOutputAOV"); aov_base.aov_name = "BaseColor"
                rgb = nodes.new("ShaderNodeRGB"); rgb.outputs[0].default_value = tuple(getattr(mat, 'diffuse_color', (0.8, 0.8, 0.8, 1.0)))
                links.new(rgb.outputs[0], aov_base.inputs[0])
                aov_mr = nodes.new("ShaderNodeOutputAOV"); aov_mr.aov_name = "MetallicRoughness"
                comb = nodes.new("ShaderNodeCombineRGB")
                vm = nodes.new("ShaderNodeValue"); vm.outputs[0].default_value = 0.0
                vr = nodes.new("ShaderNodeValue"); vr.outputs[0].default_value = 0.5
                vb = nodes.new("ShaderNodeValue"); vb.outputs[0].default_value = 0.0
                links.new(vm.outputs[0], comb.inputs[0])
                links.new(vr.outputs[0], comb.inputs[1])
                links.new(vb.outputs[0], comb.inputs[2])
                links.new(comb.outputs[0], aov_mr.inputs[0])
                print(f"Added fallback AOVs to evaluated material '{mat.name}'")
            except Exception as e:
                print(f"Failed to wire AOVs for evaluated material '{mat.name}': {e}")
    except Exception as e:
        print(f"Evaluated depsgraph scan failed: {e}")

# Run evaluated scan to capture GN/instanced foliage materials
_ensure_aovs_from_evaluated_instances()
