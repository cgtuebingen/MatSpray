import bpy
from mathutils import Vector

def setup_aovs_for_scene(view_transform="Standard"):
    """Create and setup AOVs for the current scene"""
    
    # Ensure correct render engine
    if bpy.context.scene.render.engine != 'CYCLES':
        print("Switching render engine to Cycles for proper irradiance support")
        bpy.context.scene.render.engine = 'CYCLES'
    
    # Set enough bounces for indirect lighting to be visible
    if hasattr(bpy.context.scene, 'cycles'):
        cycles = bpy.context.scene.cycles
        if cycles.diffuse_bounces < 3:
            print(f"Increasing diffuse bounces from {cycles.diffuse_bounces} to 3 for better indirect lighting")
            cycles.diffuse_bounces = 3
        if cycles.max_bounces < 8:
            print(f"Increasing max bounces from {cycles.max_bounces} to 8")
            cycles.max_bounces = 8
    
    # Create custom AOVs for material properties only (not irradiance)
    aov_list = [("BaseColor", "COLOR"),
                ("MetallicRoughness", "COLOR"),
                ("NormalCamera", "COLOR"),
                ("NormalWorld", "COLOR")]
    
    for (aov_name, aov_type) in aov_list:
        if not any(a.name == aov_name for a in bpy.context.view_layer.aovs):
            aov = bpy.context.view_layer.aovs.add()
            aov.name = aov_name
            aov.type = aov_type
    
    # Enable necessary passes
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_diffuse_color = True
    bpy.context.view_layer.use_pass_object_index = True
    bpy.context.view_layer.use_pass_z = True  # Enable built-in depth pass
    
    # EXPLICITLY Enable diffuse lighting passes for irradiance
    print("Enabling diffuse lighting passes for irradiance capture")
    bpy.context.view_layer.use_pass_diffuse_direct = True
    bpy.context.view_layer.use_pass_diffuse_indirect = True

    # Convert collection instances to real objects so their materials can be edited
    try:
        instancers = [o for o in bpy.context.scene.objects if o.type == 'EMPTY' and getattr(o, 'instance_type', 'NONE') == 'COLLECTION' and o.instance_collection is not None]
        if instancers:
            for o in bpy.context.selected_objects:
                o.select_set(False)
            for inst in instancers:
                try:
                    bpy.context.view_layer.objects.active = inst
                    inst.select_set(True)
                    bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)
                    inst.select_set(False)
                    print(f"Made collection instance real: {inst.name}")
                except Exception as e:
                    print(f"Could not make instance real for '{inst.name}': {e}")
    except Exception as e:
        print(f"Instance-real conversion skipped due to error: {e}")

    # Localize linked materials used in the scene so we can add AOV nodes
    localized_map = {}
    for obj in bpy.context.scene.objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT", "HAIR", "POINTCLOUD", "VOLUME"}:
            for slot in obj.material_slots:
                mat = slot.material
                if not mat:
                    continue
                if getattr(mat, "library", None) is not None:
                    key = mat.name_full if hasattr(mat, "name_full") else mat.name
                    if key not in localized_map:
                        local_mat = mat.copy()
                        localized_map[key] = local_mat
                        print(f"Localized linked material '{key}' → '{local_mat.name}' for AOV wiring")
                    else:
                        local_mat = localized_map[key]
                    slot.material = local_mat

    # Configure rendering settings
    scene = bpy.context.scene
    if hasattr(scene.view_settings, "view_transform"):
        # Set global color management settings
        scene.view_settings.view_transform = view_transform
        scene.view_settings.look = "None"
        scene.display_settings.display_device = "sRGB"
    
    # Process all objects in the scene for material-specific AOVs
    handled_materials = set()
    processed_count = 0
    
    for obj in bpy.context.scene.objects:
        # Check if the object can have materials
        if obj.type in {
            "MESH", "CURVE", "SURFACE", "META", 
            "FONT", "HAIR", "POINTCLOUD", "VOLUME",
        }:
            for slot in obj.material_slots:
                mat = slot.material
                if not mat:  # Skip empty slots
                    continue
                
                if mat in handled_materials:
                    continue
                
                handled_materials.add(mat)
                
                if not mat.use_nodes or not mat.node_tree:
                    print(f"Skipping material '{mat.name}' - no node tree")
                    continue
                
                # Find Principled BSDF node (top-level only)
                principled_bsdf = None
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principled_bsdf = node
                        break
                
                if not principled_bsdf:
                    # Non-Principled fallback: create basic AOVs from defaults/diffuse
                    try:
                        nodes = mat.node_tree.nodes
                        links = mat.node_tree.links
                        # Remove existing AOV outputs to avoid duplicates
                        for n in list(nodes):
                            if getattr(n, 'type', '') == 'OUTPUT_AOV' and getattr(n, 'aov_name', '') in ["BaseColor", "MetallicRoughness"]:
                                nodes.remove(n)
                        # BaseColor
                        aov_base = nodes.new("ShaderNodeOutputAOV"); aov_base.aov_name = "BaseColor"
                        rgb = nodes.new("ShaderNodeRGB")
                        rgb.outputs[0].default_value = tuple(getattr(mat, 'diffuse_color', (0.8, 0.8, 0.8, 1.0)))
                        links.new(rgb.outputs[0], aov_base.inputs[0])
                        # MetallicRoughness
                        aov_mr = nodes.new("ShaderNodeOutputAOV"); aov_mr.aov_name = "MetallicRoughness"
                        comb = nodes.new("ShaderNodeCombineRGB")
                        val_m = nodes.new("ShaderNodeValue"); val_m.outputs[0].default_value = 0.0
                        val_r = nodes.new("ShaderNodeValue"); val_r.outputs[0].default_value = 0.5
                        val_b = nodes.new("ShaderNodeValue"); val_b.outputs[0].default_value = 0.0
                        links.new(val_m.outputs[0], comb.inputs[0])
                        links.new(val_r.outputs[0], comb.inputs[1])
                        links.new(val_b.outputs[0], comb.inputs[2])
                        links.new(comb.outputs[0], aov_mr.inputs[0])
                        processed_count += 1
                        print(f"Added fallback AOVs to non-Principled material '{mat.name}'")
                    except Exception as e:
                        print(f"Failed to add fallback AOVs to '{mat.name}': {e}")
                    continue
                
                # Setup AOV nodes for this Principled material
                if create_aov_entries(mat, principled_bsdf):
                    processed_count += 1
    
    print(f"Successfully processed {processed_count} materials")
    
    # Setup the compositor for all passes
    print("Setting up compositor for standard and irradiance passes")
    setup_compositor_for_output("/tmp")  # Default path
    
    return processed_count > 0

def process_world_normal(links, nodes, aov_name, input_socket=None):
    """Process normal vectors for world-space normal AOV - RAW format"""
    aov_normal_world = nodes.new("ShaderNodeOutputAOV")
    aov_normal_world.aov_name = aov_name

    # Connect directly without any transformations - output raw world space normals
    if input_socket:
        links.new(input_socket, aov_normal_world.inputs["Color"])
    else:
        geometry_node = nodes.new("ShaderNodeNewGeometry")
        links.new(geometry_node.outputs["Normal"], aov_normal_world.inputs["Color"])

    return geometry_node.outputs["Normal"] if not input_socket else input_socket


def process_normal(links, nodes, aov_name, input_socket=None):
    """Process normal vectors for camera-space normal AOV - RAW format to match built-in Normal pass"""
    aov_normal = nodes.new("ShaderNodeOutputAOV")
    aov_normal.aov_name = aov_name

    vector_transform = nodes.new("ShaderNodeVectorTransform")
    vector_transform.convert_from = "WORLD"
    vector_transform.convert_to = "CAMERA"

    vector_scale = nodes.new("ShaderNodeVectorMath")
    vector_scale.operation = "MULTIPLY_ADD"
    vector_scale.inputs[1].default_value = Vector((0.5, 0.5, -0.5))
    vector_scale.inputs[2].default_value = Vector((0.5, 0.5, 0.5))

    # Connect the input to the vector transform
    if input_socket:
        links.new(input_socket, vector_transform.inputs["Vector"])
    else:
        geometry_node = nodes.new("ShaderNodeNewGeometry")
        links.new(geometry_node.outputs["Normal"], vector_transform.inputs["Vector"])

    # Connect vector transform to vector scale (MULTIPLY_ADD to convert [-1,1] to [0,1])
    links.new(vector_transform.outputs["Vector"], vector_scale.inputs["Vector"])
    
    # Connect the scaled output to the AOV
    links.new(vector_scale.outputs["Vector"], aov_normal.inputs["Color"])

    return vector_transform.outputs["Vector"]

def fix_texture_colorspace_for_data_maps(node):
    """
    Fix color space settings for texture image nodes that contain data (non-color) maps.
    This should be called for nodes connected to Metallic, Roughness, Normal, Height, etc.
    """
    if node.type == 'TEX_IMAGE' and node.image:
        if node.image.colorspace_settings.name != 'Non-Color':
            old_colorspace = node.image.colorspace_settings.name
            node.image.colorspace_settings.name = 'Non-Color'
            print(f"  ✓ Fixed colorspace: '{node.image.name}' from '{old_colorspace}' to 'Non-Color'")
        else:
            print(f"  ✓ Colorspace already correct: '{node.image.name}' (Non-Color)")
    elif node.type == 'TEX_IMAGE' and not node.image:
        print(f"  ⚠️ Image texture node has no image loaded")

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

def create_aov_entries(mat, principled_bsdf):
    """Create AOV nodes for a material with Principled BSDF"""
    try:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clean up any existing AOV nodes (to avoid duplicates)
        for node in nodes:
            if node.type == 'OUTPUT_AOV':
                if node.aov_name in ["BaseColor", "MetallicRoughness", "NormalCamera", "NormalWorld"]:
                    nodes.remove(node)

        # Create AOV output nodes
        aov_basecolor = nodes.new("ShaderNodeOutputAOV")
        aov_basecolor.aov_name = "BaseColor"

        aov_metallic_roughness = nodes.new("ShaderNodeOutputAOV")
        aov_metallic_roughness.aov_name = "MetallicRoughness"
        
        # Create a node to combine metallic and roughness values
        combine_rgb = nodes.new("ShaderNodeCombineRGB")

        # Connect Base Color
        if principled_bsdf.inputs["Base Color"].links:
            basecolor_socket = principled_bsdf.inputs["Base Color"].links[0].from_socket
            links.new(basecolor_socket, aov_basecolor.inputs[0])
        else:
            # Create a constant color with the default value
            rgb_node = nodes.new("ShaderNodeRGB")
            rgb_node.outputs[0].default_value = principled_bsdf.inputs["Base Color"].default_value
            links.new(rgb_node.outputs[0], aov_basecolor.inputs[0])

        # METALLIC - GOING TO R CHANNEL (changed from G to R)
        if principled_bsdf.inputs["Metallic"].links:
            # Connected value - get the socket directly
            metallic_socket = principled_bsdf.inputs["Metallic"].links[0].from_socket
            metallic_node = principled_bsdf.inputs["Metallic"].links[0].from_node
            
            # Fix color space for metallic textures
            print(f"  Checking Metallic input for material '{mat.name}':")
            traverse_and_fix_colorspace_recursive(metallic_node)
            
            links.new(metallic_socket, combine_rgb.inputs[0])  # Metallic → R
        else:
            # Default value - create a Value node with exact value
            value_node = nodes.new("ShaderNodeValue")
            value_node.outputs[0].default_value = principled_bsdf.inputs["Metallic"].default_value
            links.new(value_node.outputs[0], combine_rgb.inputs[0])  # Metallic → R
            
        # ROUGHNESS - GOING TO G CHANNEL (unchanged)
        if principled_bsdf.inputs["Roughness"].links:
            # Connected value - get the socket directly
            roughness_socket = principled_bsdf.inputs["Roughness"].links[0].from_socket
            roughness_node = principled_bsdf.inputs["Roughness"].links[0].from_node
            
            # Fix color space for roughness textures
            print(f"  Checking Roughness input for material '{mat.name}':")
            traverse_and_fix_colorspace_recursive(roughness_node)
            
            links.new(roughness_socket, combine_rgb.inputs[1])  # Roughness → G
        else:
            # Default value - create a Value node with exact value
            value_node = nodes.new("ShaderNodeValue")
            value_node.outputs[0].default_value = principled_bsdf.inputs["Roughness"].default_value
            links.new(value_node.outputs[0], combine_rgb.inputs[1])  # Roughness → G
            
        # Set B channel to 0
        value_zero = nodes.new("ShaderNodeValue")
        value_zero.outputs[0].default_value = 0.0
        links.new(value_zero.outputs[0], combine_rgb.inputs[2])  # 0 → B

        # Connect the combined values to the AOV
        links.new(combine_rgb.outputs[0], aov_metallic_roughness.inputs[0])

        # Also fix normal map color spaces (Normal maps should also be Non-Color)
        if principled_bsdf.inputs["Normal"].links:
            normal_node = principled_bsdf.inputs["Normal"].links[0].from_node
            print(f"  Checking Normal input for material '{mat.name}':")
            traverse_and_fix_colorspace_recursive(normal_node)

        # Process normals
        process_normal(links, nodes, "NormalCamera")  # Raw camera space normals
        process_world_normal(links, nodes, "NormalWorld")  # Raw world space normals
        
        # Add a custom property to mark this material as processed
        mat["__aov_processed__"] = True
        return True
        
    except Exception as e:
        print(f"Error processing material '{mat.name}': {str(e)}")
        return False

def get_texture_colorspace_from_material_input(material, input_name):
    """
    Get the color space setting from a texture connected to a specific input of a Principled BSDF.
    Returns the colorspace name or None if no texture is found.
    """
    if not material.use_nodes or not material.node_tree:
        return None
    
    # Find Principled BSDF node
    principled_bsdf = None
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_bsdf = node
            break
    
    if not principled_bsdf or input_name not in principled_bsdf.inputs:
        return None
    
    if not principled_bsdf.inputs[input_name].links:
        return None
    
    # Traverse the node tree to find the first image texture node
    def find_image_texture_recursive(node, visited=None):
        if visited is None:
            visited = set()
        
        if node in visited:
            return None
        visited.add(node)
        
        if node.type == 'TEX_IMAGE' and node.image:
            return node.image.colorspace_settings.name
        
        # Check all input nodes
        for input_socket in node.inputs:
            if input_socket.links:
                for link in input_socket.links:
                    result = find_image_texture_recursive(link.from_node, visited)
                    if result:
                        return result
        
        return None
    
    # Start from the connected node
    connected_node = principled_bsdf.inputs[input_name].links[0].from_node
    return find_image_texture_recursive(connected_node)

def colorspace_to_view_transform(colorspace_name, file_format="OPEN_EXR"):
    """
    Map texture colorspace settings to appropriate compositor settings.
    PNG uses 'view' settings, OpenEXR uses 'color space' settings.
    """
    if file_format == "PNG":
        # For PNG: use view transform
        mapping = {
            "Non-Color": "Raw",
            "sRGB": "Standard", 
            "Linear Rec.709": "Raw",
            "Raw": "Raw",
            "Linear": "Raw"
        }
        return mapping.get(colorspace_name, "Raw")
    else:  # OpenEXR
        # For OpenEXR: use color space
        mapping = {
            "Non-Color": "Non-Color",
            "sRGB": "sRGB", 
            "Linear Rec.709": "Linear Rec.709",
            "Raw": "Non-Color",
            "Linear": "Linear Rec.709"
        }
        return mapping.get(colorspace_name, "Non-Color")

def setup_compositor_for_output(output_dir):
    """Set up compositor nodes for rendering with proper outputs including irradiance"""
    
    # Clear and setup basic nodes
    scene = bpy.context.scene
    scene.use_nodes = True
    node_tree = scene.node_tree
    
    # Clear existing nodes
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)
    
    # Create render layers node
    render_layers = node_tree.nodes.new("CompositorNodeRLayers")
    render_layers.location = (0, 0)
    
    # Print all available passes to debug
    print("=====================================================")
    print("AVAILABLE RENDER PASSES:")
    for output in render_layers.outputs:
        if output.enabled:
            print(f"- {output.name}")
    print("=====================================================")
    
    # Determine the appropriate view transform for MetallicRoughness by checking source textures
    metallic_roughness_view_transform = "Raw"  # Default
    
    # Check all materials to find roughness texture colorspace
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            roughness_colorspace = get_texture_colorspace_from_material_input(mat, "Roughness")
            metallic_colorspace = get_texture_colorspace_from_material_input(mat, "Metallic")
            
            if roughness_colorspace:
                metallic_roughness_view_transform = colorspace_to_view_transform(roughness_colorspace, "OPEN_EXR")
                print(f"Found Roughness texture with colorspace '{roughness_colorspace}' in material '{mat.name}'")
                print(f"Using view transform: '{metallic_roughness_view_transform}' for MetallicRoughness output")
                break
            elif metallic_colorspace:
                metallic_roughness_view_transform = colorspace_to_view_transform(metallic_colorspace, "OPEN_EXR")
                print(f"Found Metallic texture with colorspace '{metallic_colorspace}' in material '{mat.name}'")
                print(f"Using view transform: '{metallic_roughness_view_transform}' for MetallicRoughness output")
                break
    
    # Create file output node for MetallicRoughness with detected view transform
    metalrough_output = node_tree.nodes.new("CompositorNodeOutputFile")
    metalrough_output.name = "MetallicRoughness Output"
    metalrough_output.location = (300, 0)
    metalrough_output.base_path = f"{output_dir}/metallicroughmask"
    metalrough_output.file_slots[0].path = "####"
    
    # Configure format for raw output - Use detected color management
    fmt = metalrough_output.format
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "16"
    fmt.exr_codec = "PIZ"
    fmt.color_management = "OVERRIDE"  # Override scene color management
    
    # Connect MetallicRoughness AOV to file output
    if "MetallicRoughness" in render_layers.outputs:
        node_tree.links.new(render_layers.outputs["MetallicRoughness"], metalrough_output.inputs[0])
    else:
        print("Warning: MetallicRoughness AOV not found in render layer outputs")
    
    # Add BaseColor output with Standard view transform
    basecolor_output = node_tree.nodes.new("CompositorNodeOutputFile")
    basecolor_output.name = "BaseColor Output"
    basecolor_output.location = (300, 200)
    basecolor_output.base_path = f"{output_dir}/basecolor"
    basecolor_output.file_slots[0].path = "####"
    
    # Configure format for BaseColor
    fmt = basecolor_output.format
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "16"
    fmt.exr_codec = "PIZ"
    
    if "BaseColor" in render_layers.outputs:
        node_tree.links.new(render_layers.outputs["BaseColor"], basecolor_output.inputs[0])
    else:
        print("Warning: BaseColor AOV not found in render layer outputs")
    
    # Normal output (using EXR for precision) - Also use Raw for normals
    normal_output = node_tree.nodes.new("CompositorNodeOutputFile")
    normal_output.name = "Normal Output"
    normal_output.location = (300, -200)
    normal_output.base_path = f"{output_dir}/normal"
    normal_output.file_slots[0].path = "####"
    
    # Configure format for Normal - Use Raw for normals too
    fmt = normal_output.format
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "16"
    fmt.exr_codec = "PIZ"
    fmt.color_management = "OVERRIDE"
    fmt.view_settings.view_transform = "Raw"  # Raw for normals
    
    if "NormalCamera" in render_layers.outputs:
        node_tree.links.new(render_layers.outputs["NormalCamera"], normal_output.inputs[0])
        
    # Depth output (using EXR for precision) - Raw for depth
    depth_output = node_tree.nodes.new("CompositorNodeOutputFile")
    depth_output.name = "Depth Output"
    depth_output.location = (300, -400)
    depth_output.base_path = f"{output_dir}/depth"
    depth_output.file_slots[0].path = "####"
    
    # Configure format for Depth - Use Raw for depth values
    fmt = depth_output.format
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "BW"
    fmt.color_depth = "16"
    fmt.exr_codec = "PIZ"
    fmt.color_management = "OVERRIDE"
    fmt.view_settings.view_transform = "Raw"  # Raw for depth
    
    # Link built-in depth pass to depth output
    if "Depth" in render_layers.outputs:
        node_tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    else:
        print("Warning: Depth pass not found in render layer outputs")
    
    # IMPROVED IRRADIANCE SETUP
    
    # Direct Irradiance
    direct_irradiance_output = node_tree.nodes.new("CompositorNodeOutputFile")
    direct_irradiance_output.name = "Direct Irradiance Output"
    direct_irradiance_output.location = (300, -600)
    direct_irradiance_output.base_path = f"{output_dir}/irradiance_direct"
    direct_irradiance_output.file_slots[0].path = "####"
    
    # Configure format for direct irradiance (HDR)
    fmt = direct_irradiance_output.format
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "32"
    fmt.exr_codec = "PIZ"
    
    # Indirect Irradiance
    indirect_irradiance_output = node_tree.nodes.new("CompositorNodeOutputFile")
    indirect_irradiance_output.name = "Indirect Irradiance Output"
    indirect_irradiance_output.location = (300, -800)
    indirect_irradiance_output.base_path = f"{output_dir}/irradiance_indirect"
    indirect_irradiance_output.file_slots[0].path = "####"
    
    # Configure format
    fmt = indirect_irradiance_output.format
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "32"
    fmt.exr_codec = "PIZ"
    
    # Total Irradiance output
    total_irradiance_output = node_tree.nodes.new("CompositorNodeOutputFile")
    total_irradiance_output.name = "Total Irradiance Output"
    total_irradiance_output.location = (400, -1000)
    total_irradiance_output.base_path = f"{output_dir}/irradiance_total"
    total_irradiance_output.file_slots[0].path = "####"
    
    # Configure format
    fmt = total_irradiance_output.format
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "32"
    fmt.exr_codec = "PIZ"
    
    # Create a Debug Viewer node to visualize irradiance during rendering
    viewer_node = node_tree.nodes.new("CompositorNodeViewer")
    viewer_node.name = "Irradiance Viewer"
    viewer_node.location = (600, -1000)
    
    # Find diffuse direct and indirect passes (check multiple possible pass names)
    diffuse_direct = None
    diffuse_indirect = None
    
    # Check ALL possible diffuse pass naming variations
    for pass_name in ["DiffuseDirect", "DiffDir", "Diffuse Direct", "Diff Dir"]:
        if pass_name in render_layers.outputs:
            diffuse_direct = render_layers.outputs[pass_name]
            print(f"Found direct diffuse pass: {pass_name}")
            break
    
    for pass_name in ["DiffuseIndirect", "DiffInd", "Diffuse Indirect", "Diff Ind"]:
        if pass_name in render_layers.outputs:
            diffuse_indirect = render_layers.outputs[pass_name]
            print(f"Found indirect diffuse pass: {pass_name}")
            break
    
    # Create mixer and connect only if BOTH passes are found
    if diffuse_direct and diffuse_indirect:
        # Connect direct irradiance
        node_tree.links.new(diffuse_direct, direct_irradiance_output.inputs[0])
        
        # Connect indirect irradiance
        node_tree.links.new(diffuse_indirect, indirect_irradiance_output.inputs[0])
        
        # Set up mixer for total irradiance
        total_irradiance_add = node_tree.nodes.new("CompositorNodeMixRGB")
        total_irradiance_add.name = "Total Irradiance Mixer"
        total_irradiance_add.blend_type = 'ADD'
        total_irradiance_add.inputs[0].default_value = 1.0  # Factor
        total_irradiance_add.location = (200, -1000)
        
        # Connect both passes to the mixer
        node_tree.links.new(diffuse_direct, total_irradiance_add.inputs[1])
        node_tree.links.new(diffuse_indirect, total_irradiance_add.inputs[2])
        
        # Connect mixer to output and viewer
        node_tree.links.new(total_irradiance_add.outputs[0], total_irradiance_output.inputs[0])
        node_tree.links.new(total_irradiance_add.outputs[0], viewer_node.inputs[0])
        
        print("✓ SUCCESSFULLY set up all irradiance compositing nodes")
    else:
        if not diffuse_direct:
            print("❌ ERROR: Direct diffuse pass not found!")
        if not diffuse_indirect:
            print("❌ ERROR: Indirect diffuse pass not found!")
        print("⚠️ WARNING: Could not set up irradiance - missing diffuse passes")
        print("Make sure you're using Cycles renderer and have enabled the necessary passes")

def normalize_scene_geometry():
    """
    Normalize the scene geometry to fit in a -1 to 1 cube while preserving proportions.
    Will center the scene at origin (0,0,0) and scale it to fit within a -1 to 1 bounding box.
    This function preserves object relationships and is idempotent.
    """
    # Get all visible mesh objects in the scene
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and not obj.hide_viewport]
    
    if not mesh_objects:
        print("No mesh objects found in the scene to normalize")
        return False
    
    # Get current min and max bounds for all objects
    min_bound = Vector((float('inf'), float('inf'), float('inf')))
    max_bound = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in mesh_objects:
        # Check all vertices in global space
        for vertex in obj.bound_box:
            # Convert local coordinates to global
            global_coord = obj.matrix_world @ Vector(vertex)
            
            # Update min and max bounds
            min_bound.x = min(min_bound.x, global_coord.x)
            min_bound.y = min(min_bound.y, global_coord.y)
            min_bound.z = min(min_bound.z, global_coord.z)
            
            max_bound.x = max(max_bound.x, global_coord.x)
            max_bound.y = max(max_bound.y, global_coord.y)
            max_bound.z = max(max_bound.z, global_coord.z)
    
    # Calculate the center and size of the bounding box
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    
    # Calculate the scaling factor to fit in a 2x2x2 cube (-1 to 1)
    # We need to preserve proportions, so we use the largest dimension
    max_dimension = max(size.x, size.y, size.z)
    scale_factor = 2.0 / max_dimension if max_dimension > 0 else 1.0
    
    print(f"Original bounding box: min={min_bound}, max={max_bound}")
    print(f"Center: {center}, Size: {size}")
    print(f"Scale factor: {scale_factor}")
    
    # Apply transformations directly to all objects
    for obj in bpy.context.scene.objects:
        if obj.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'HAIR', 'POINTCLOUD', 'VOLUME'}:
            # First translate to center the scene at origin
            obj.location = obj.location - center
            
            # Then apply uniform scale to fit within -1 to 1 cube
            # Multiply existing scale to preserve any existing scaling
            obj.scale = obj.scale * scale_factor
    
    # Verify the result by checking bounds again
    min_bound_new = Vector((float('inf'), float('inf'), float('inf')))
    max_bound_new = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in mesh_objects:
        for vertex in obj.bound_box:
            global_coord = obj.matrix_world @ Vector(vertex)
            min_bound_new.x = min(min_bound_new.x, global_coord.x)
            min_bound_new.y = min(min_bound_new.y, global_coord.y)
            min_bound_new.z = min(min_bound_new.z, global_coord.z)
            max_bound_new.x = max(max_bound_new.x, global_coord.x)
            max_bound_new.y = max(max_bound_new.y, global_coord.y)
            max_bound_new.z = max(max_bound_new.z, global_coord.z)
    
    center_new = (min_bound_new + max_bound_new) / 2
    size_new = max_bound_new - min_bound_new
    
    print(f"New bounding box: min={min_bound_new}, max={max_bound_new}")
    print(f"New center: {center_new}, New size: {size_new}")
    print(f"Scene normalized: Centered at origin, scaled by factor {scale_factor}")
    print("Object relationships preserved - no individual origins were modified")
    
    return True

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

# Run the script
if __name__ == "__main__":
    # Example usage
    setup_aovs_for_scene(view_transform="Standard")
    # Note: Geometry normalization is intentionally not executed in this variant. 