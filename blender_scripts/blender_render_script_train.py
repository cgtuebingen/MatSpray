import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
import glob
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from blender_aov_setup import fix_all_material_texture_colorspaces, get_texture_colorspace_from_material_input, colorspace_to_view_transform
         
DEBUG = False
            
VIEWS = 100
RESOLUTION = 512
RESOLUTION_X = 1280
RESOLUTION_Y = 704
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = True
UPPER_VIEWS = True
ENV_MAP_PATH = './env_maps/base.exr'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Render images from Blender')
parser.add_argument('--object_name', type=str, required=True, 
                    help='Name of the object being rendered')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Base directory for output files')
parser.add_argument('--views', type=int, default=VIEWS,
                    help=f'Number of views to render (default: {VIEWS})')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode (no actual rendering)')

# Get arguments from command line
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
args = parser.parse_args(argv)

OBJECT_NAME = args.object_name
BASE_OUTPUT_DIR = args.output_dir
VIEWS = args.views
DEBUG = args.debug

# Construct the base path
BASE_PATH = os.path.join(BASE_OUTPUT_DIR, OBJECT_NAME, 'base')

# Set environment map
def set_environment_map(env_map_path):
    # Get the world
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    # Enable nodes for the world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links
    
    # Clear existing nodes
    world_nodes.clear()
    
    # Create new nodes
    node_background = world_nodes.new(type='ShaderNodeBackground')
    node_environment = world_nodes.new(type='ShaderNodeTexEnvironment')
    node_output = world_nodes.new(type='ShaderNodeOutputWorld')
    
    # Load the environment texture and set color space to Linear Rec.709
    node_environment.image = bpy.data.images.load(env_map_path)
    node_environment.image.colorspace_settings.name = 'Linear Rec.709'
    
    # Set up links
    world_links.new(node_environment.outputs['Color'], node_background.inputs['Color'])
    world_links.new(node_background.outputs['Background'], node_output.inputs['Surface'])
    
    print(f"Environment map set to: {env_map_path}")

# Function to render a dataset (train or val)
def render_dataset(dataset_type):
    # Set color management settings
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.display_settings.display_device = 'sRGB'
    
    fp = os.path.join(BASE_PATH, dataset_type)
    
    # Create base directory if it doesn't exist
    if not os.path.exists(fp):
        os.makedirs(fp)
    
    # Create subdirectories for each render type
    image_dir = os.path.join(fp, 'images')
    image_bg_dir = os.path.join(fp, 'images_bg')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(image_bg_dir):
        os.makedirs(image_bg_dir)
        
    # Data to store in JSON file
    out_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }
    
    # Create directories for each AOV
    for output_name, suffix in aov_outputs.items():
        if output_name == 'MetallicRoughness':
            # Create separate directories for metallic and roughness
            metallic_dir = os.path.join(fp, 'base_metallic')
            roughness_dir = os.path.join(fp, 'base_roughness')
            if not os.path.exists(metallic_dir):
                os.makedirs(metallic_dir)
                print(f"Created directory: {metallic_dir}")
            if not os.path.exists(roughness_dir):
                os.makedirs(roughness_dir)
                print(f"Created directory: {roughness_dir}")
        else:
            folder_name = folder_name_mapping.get(suffix, suffix)  # Use mapping for folder name
            dir_path = os.path.join(fp, f'base_{folder_name}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
    
    # Also create directory for total irradiance
    irradiance_total_dir = os.path.join(fp, 'base_irradiance')  # Changed from 'base_irradiance' to 'irradiance'
    if not os.path.exists(irradiance_total_dir):
        os.makedirs(irradiance_total_dir)
        print(f"Created directory: {irradiance_total_dir}")
    
    # Clear old nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # Create input render layer node
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'
    
    # Print available render passes for debugging
    print("=====================================================")
    print("AVAILABLE RENDER PASSES:")
    for output in render_layers.outputs:
        if output.enabled:
            print(f"- {output.name}")
    print("=====================================================")
    
    # Setup the total irradiance mixer node (adds direct and indirect)
    irradiance_mixer = tree.nodes.new('CompositorNodeMixRGB')
    irradiance_mixer.name = "Total Irradiance Mixer"
    irradiance_mixer.blend_type = 'ADD'
    irradiance_mixer.inputs[0].default_value = 1.0  # Factor = 100%
    irradiance_mixer.location = (300, -600)
    
    # Create dedicated output node for total irradiance
    total_irradiance_output = tree.nodes.new(type="CompositorNodeOutputFile")
    total_irradiance_output.name = "Total Irradiance Output"
    total_irradiance_output.location = (500, -600)
    total_irradiance_output.format.file_format = 'OPEN_EXR'
    total_irradiance_output.format.color_depth = '32'
    total_irradiance_output.format.exr_codec = 'PIZ'
    
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
    
    # Setup file output nodes for each AOV
    aov_file_outputs = {}
    for output_name, suffix in aov_outputs.items():
        if output_name in render_layers.outputs:
            if output_name == 'MetallicRoughness':
                # Create separate output nodes for metallic and roughness
                metallic_output = tree.nodes.new(type="CompositorNodeOutputFile")
                metallic_output.label = 'Metallic Output'
                metallic_output.name = 'Metallic Output'
                metallic_output.format.file_format = 'PNG'
                metallic_output.format.color_depth = '8'
                metallic_output.format.color_management = 'OVERRIDE'
                metallic_output.format.view_settings.view_transform = 'Raw'
                
                roughness_output = tree.nodes.new(type="CompositorNodeOutputFile")
                roughness_output.label = 'Roughness Output'
                roughness_output.name = 'Roughness Output'
                roughness_output.format.file_format = 'PNG'
                roughness_output.format.color_depth = '8'
                roughness_output.format.color_management = 'OVERRIDE'
                roughness_output.format.view_settings.view_transform = 'Raw'
                
                # Connect both to the same MetallicRoughness output
                links.new(render_layers.outputs[output_name], metallic_output.inputs[0])
                links.new(render_layers.outputs[output_name], roughness_output.inputs[0])
                
                aov_file_outputs['Metallic'] = {'node': metallic_output, 'suffix': 'metallic'}
                aov_file_outputs['Roughness'] = {'node': roughness_output, 'suffix': 'roughness'}
                print(f"Added separate output nodes for Metallic and Roughness")
            else:
                file_output = tree.nodes.new(type="CompositorNodeOutputFile")
                file_output.label = f'{output_name} Output'
                file_output.name = f'{output_name} Output'
                
                # Set formats based on output type
                if output_name in ['Depth', 'NormalWorld', 'NormalCamera'] or 'Diffuse' in output_name or 'Diff' in output_name:
                    # EXR for depth, normals, and irradiance (need high precision/HDR)
                    file_output.format.file_format = 'OPEN_EXR'
                    file_output.format.color_depth = '32'
                    file_output.format.exr_codec = 'PIZ'
                    
                    # For raw data passes: Set color management override with Raw view transform
                    if output_name in ['Depth', 'NormalWorld', 'NormalCamera']:
                        try:
                            file_output.format.color_management = 'OVERRIDE'
                            file_output.format.view_settings.view_transform = 'Raw'
                            print(f"✓ Set Color Management Override + Raw view for {output_name}")
                        except AttributeError:
                            print(f"⚠️ Could not set color management override for {output_name}")
                elif output_name in ['BaseColor']:
                    # PNG for base color
                    file_output.format.file_format = 'PNG'
                    file_output.format.color_depth = '8'
                    # Force Standard view transform for BaseColor to keep consistent outputs
                    try:
                        file_output.format.color_management = 'OVERRIDE'
                        file_output.format.view_settings.view_transform = 'Standard'
                        print(f"✓ Set BaseColor to Standard view transform")
                    except AttributeError:
                        print(f"⚠️ Could not set Standard view transform override for BaseColor")
                else:
                    file_output.format.file_format = FORMAT
        
                # Connect directly to the render layer outputs
                links.new(render_layers.outputs[output_name], file_output.inputs[0])
                
                aov_file_outputs[output_name] = {'node': file_output, 'suffix': suffix}
                print(f"Added output node for {output_name} with suffix _{suffix}")
    
    # Find the direct and indirect diffuse passes - try both naming conventions
    diffuse_direct = None
    diffuse_indirect = None
    
    # Try to find direct diffuse pass
    for pass_name in ['DiffuseDirect', 'DiffDir']:
        if pass_name in render_layers.outputs:
            diffuse_direct = render_layers.outputs[pass_name]
            print(f"✅ Found direct diffuse pass: {pass_name}")
            break
    
    # Try to find indirect diffuse pass
    for pass_name in ['DiffuseIndirect', 'DiffInd']:
        if pass_name in render_layers.outputs:
            diffuse_indirect = render_layers.outputs[pass_name]
            print(f"✅ Found indirect diffuse pass: {pass_name}")
            break
    
    # Connect the mixer node if both passes were found
    if diffuse_direct and diffuse_indirect:
        links.new(diffuse_direct, irradiance_mixer.inputs[1])
        links.new(diffuse_indirect, irradiance_mixer.inputs[2])
        links.new(irradiance_mixer.outputs[0], total_irradiance_output.inputs[0])
        print("✅ Connected total irradiance mixer successfully")
    else:
        if not diffuse_direct:
            print("❌ ERROR: Direct diffuse pass not found!")
        if not diffuse_indirect:
            print("❌ ERROR: Indirect diffuse pass not found!")
        print("⚠️ WARNING: Could not set up total irradiance - missing diffuse passes")
    
    # Frames data
    out_data['frames'] = []
    
    # Reset camera orientation
    b_empty.rotation_euler = (0, 0, 0)
    
    for i in range(0, VIEWS):
        if RANDOM_VIEWS:
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
    
        # Set paths for AOV outputs
        for output_name, output_data in aov_file_outputs.items():
            suffix = output_data['suffix']
            folder_name = folder_name_mapping.get(suffix, suffix)  # Use mapping for folder name
            # Set the output directory for this specific render type
            output_dir = os.path.join(fp, f'base_{folder_name}')
            output_data['node'].base_path = output_dir
            if len(output_data['node'].file_slots) > 0:
                output_data['node'].file_slots[0].path = f"r_{i}_{suffix}"  # Keep original suffix for file naming
        
        # Set path for total irradiance output
        total_irradiance_output.base_path = irradiance_total_dir
        if len(total_irradiance_output.file_slots) > 0:
            total_irradiance_output.file_slots[0].path = f"r_{i}_irradiance_total"
    
        if DEBUG:
            print(f"Debug mode: skipping render for {dataset_type} view {i}")
        else:
            # First render: transparent background (original)
            # Ensure Standard tonemapping for images
            scene.view_settings.view_transform = 'Standard'
            scene.render.film_transparent = True
            scene.render.filepath = os.path.join(image_dir, f'r_{i}')
            bpy.ops.render.render(write_still=True)
            
            # Second render: with environment map background
            # Use AgX tonemapping for images_bg
            scene.view_settings.view_transform = 'AgX'
            scene.render.film_transparent = False
            scene.render.filepath = os.path.join(image_bg_dir, f'r_{i}')
            bpy.ops.render.render(write_still=True)
            
            # Handle file renaming in their respective directories
            for output_name, output_data in aov_file_outputs.items():
                suffix = output_data['suffix']
                folder_name = folder_name_mapping.get(suffix, suffix)  # Use mapping for folder name
                output_dir = os.path.join(fp, f'base_{folder_name}')
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
            
            # Handle renaming for images directory
            pattern = os.path.join(image_dir, f"r_{i}*")
            files = glob.glob(pattern)
            for file_path in files:
                file_name = os.path.basename(file_path)
                base, ext = os.path.splitext(file_name)
                if "-" in base:
                    clean_name = base.split('-')[0] + ext
                    target_path = os.path.join(image_dir, clean_name)
                    try:
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        shutil.copy2(file_path, target_path)
                        os.remove(file_path)
                        print(f"Renamed {file_name} to {clean_name} in {image_dir}")
                    except Exception as e:
                        print(f"Error renaming {file_name}: {e}")
            
            # Handle renaming for images_bg directory
            pattern = os.path.join(image_bg_dir, f"r_{i}*")
            files = glob.glob(pattern)
            for file_path in files:
                file_name = os.path.basename(file_path)
                base, ext = os.path.splitext(file_name)
                if "-" in base:
                    clean_name = base.split('-')[0] + ext
                    target_path = os.path.join(image_bg_dir, clean_name)
                    try:
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        shutil.copy2(file_path, target_path)
                        os.remove(file_path)
                        print(f"Renamed {file_name} to {clean_name} in {image_bg_dir}")
                    except Exception as e:
                        print(f"Error renaming {file_name}: {e}")
            
            # Also handle renaming for total irradiance
            pattern = os.path.join(irradiance_total_dir, f"r_{i}_irradiance_total*")
            files = glob.glob(pattern)
            for file_path in files:
                file_name = os.path.basename(file_path)
                base, ext = os.path.splitext(file_name)
                if "-" in base:
                    clean_name = base.split('-')[0] + ext
                    target_path = os.path.join(irradiance_total_dir, clean_name)
                    try:
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        shutil.copy2(file_path, target_path)
                        os.remove(file_path)
                        print(f"Renamed {file_name} to {clean_name} in {irradiance_total_dir}")
                    except Exception as e:
                        print(f"Error renaming {file_name}: {e}")
    
        frame_data = {
            'file_path': os.path.join(fp, 'images', f'r_{i}'), 
            'rotation': radians(stepsize),
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)
    
        if RANDOM_VIEWS:
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            b_empty.rotation_euler[2] += radians(stepsize)
    
    if not DEBUG:
        json_filename = f"transforms_{dataset_type}.json"
        with open(os.path.join(BASE_PATH, json_filename), 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
        print(f"Saved {json_filename} to {BASE_PATH}")
    
    return out_data

# Rest of your setup code
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

# Set environment map
set_environment_map(ENV_MAP_PATH)

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Ensure Cycles is set as the render engine
bpy.context.scene.render.engine = 'CYCLES'

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

# ADD THIS: Fix material texture color spaces before rendering
print("🔧 Fixing texture color spaces for metallic/roughness maps...")
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

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes
# Note: use_pass_normal = False since we're using custom NormalWorld AOV instead
bpy.context.view_layer.use_pass_z = True
bpy.context.view_layer.use_pass_glossy_direct = True
bpy.context.view_layer.use_pass_emit = True
bpy.context.view_layer.use_pass_shadow = True

# IMPORTANT: Enable diffuse passes for irradiance capture
print("Enabling diffuse lighting passes for irradiance capture")
bpy.context.view_layer.use_pass_diffuse_direct = True
bpy.context.view_layer.use_pass_diffuse_indirect = True

bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

# Output nodes - keep original file naming
aov_outputs = {
    'Depth': 'depth',
    'NormalWorld': 'normal',  # Use our custom world normal AOV instead of built-in Normal
    'GlossDir': 'glossy_direct',
    'Emit': 'emission',
    'BaseColor': 'basecolor',  # Keep original for file naming
    'MetallicRoughness': 'metallicroughness',  # Keep original for file naming
    'NormalCamera': 'normalcamera',
    # Add the irradiance passes - use both possible naming conventions
    'DiffuseDirect': 'irradiance_direct',
    'DiffuseIndirect': 'irradiance_indirect',
    'DiffDir': 'irradiance_direct',
    'DiffInd': 'irradiance_indirect'
}

# Separate mapping for folder names only
folder_name_mapping = {
    'basecolor': 'baseColor',
    'metallic': 'metallic',
    'roughness': 'roughness',
    'irradiance_direct': 'irradiance_direct',
    'irradiance_indirect': 'irradiance_indirect',
    'depth': 'depth',
    'normal': 'normal',
    'glossy_direct': 'glossy_direct',
    'emission': 'emission',
    'normalcamera': 'normalcamera'
}

# Background
bpy.context.scene.render.dither_intensity = 0.0

# Delete Empty objects
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.select_all(action='DESELECT')
for obj in objs:
    obj.select_set(True)
bpy.ops.object.delete()

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
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

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians
stepsize = 360.0 / VIEWS
rotation_mode = 'XYZ'

# Render train and val datasets
print(f"Rendering for object: {OBJECT_NAME}")
print(f"Output directory: {BASE_PATH}")
print("Rendering training set...")
train_data = render_dataset('train')

print("Rendering validation set...")
val_data = render_dataset('val')

print("Rendering complete!")