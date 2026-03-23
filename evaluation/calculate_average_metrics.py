#!/usr/bin/env python3
"""
Script to calculate average PSNR, LPIPS, and SSIM metrics across multiple objects
for different supervision methods and environment maps.
"""

import os
from collections import defaultdict

# Base directory
BASE_DIR = os.environ.get("RESULTS_DIR", "/path/to/results/aria")

# Objects to process
OBJECTS = [
    "airplane_whitegold",
    "apartment",
    "birdhouse_BoatHouse",
    "birdhouse_RedBlackRoof",
    "birdhouse_StoneHouse",
    "teapot_jade"
]

# Method folders (in order)
METHOD_FOLDERS = [
    "supervision_BaseColor_Roughness_Metallic_diffusionrenderer",
    "projected_Averaged_BaseColor_Roughness_Metallic_diffusionrenderer",
    "mlp_softmax_BaseColor_Roughness_Metallic_median_deferred_moving_geom_diffusionrenderer"
]

# Fourth method folder (check for pure_supervision first, then neilf)
FOURTH_METHOD_OPTIONS = [
    "pure_supervision_BaseColor_Roughness_Metallic_diffusionrenderer",
    "neilf_BaseColor_Roughness_Metallic_diffusionrenderer"
]

# Environment map folders
ENVMAP_FOLDERS = ["envmap3", "envmap6", "envmap12", "envmap24"]


def load_metrics(metrics_path):
    """
    Load metrics from metrics.txt file.
    Returns (psnr, ssim, lpips) or None if file doesn't exist or can't be parsed.
    Handles both formats:
    - "psnr: value" or "Average PSNR: value"
    - "ssim: value" or "Average SSIM: value"
    - "lpips: value" or "Average LPIPS: value"
    """
    if not os.path.exists(metrics_path):
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            lines = f.readlines()
        
        psnr = None
        ssim = None
        lpips_value = None
        
        for line in lines:
            line = line.strip().lower()
            # Handle both "psnr:" and "average psnr:" formats
            if 'psnr' in line and ':' in line:
                psnr = float(line.split(':')[1].strip())
            # Handle both "ssim:" and "average ssim:" formats
            elif 'ssim' in line and ':' in line:
                ssim = float(line.split(':')[1].strip())
            # Handle both "lpips:" and "average lpips:" formats
            elif 'lpips' in line and ':' in line:
                lpips_value = float(line.split(':')[1].strip())
        
        # Ensure all values were found
        if psnr is None or ssim is None or lpips_value is None:
            return None
        
        return psnr, ssim, lpips_value
    
    except Exception as e:
        print(f"Error loading metric from {metrics_path}: {e}")
        return None


def get_fourth_method_folder(object_path):
    """
    Get the fourth method folder name for a given object.
    Checks for pure_supervision first, then neilf.
    """
    for method_option in FOURTH_METHOD_OPTIONS:
        method_path = os.path.join(object_path, method_option)
        if os.path.exists(method_path):
            return method_option
    # If neither exists, return the first option as default (will show error later)
    return FOURTH_METHOD_OPTIONS[0]


def collect_all_metrics():
    """
    Collect metrics from all objects, methods, and envmaps.
    Returns a dictionary: method_name -> list of (psnr, ssim, lpips) tuples
    Note: Both pure_supervision and neilf are treated as the same "fourth method" category
    """
    method_metrics = defaultdict(list)
    
    # Track which objects have which methods
    object_method_status = defaultdict(lambda: defaultdict(bool))
    
    # Standardized method name for the fourth method (combines pure_supervision and neilf)
    FOURTH_METHOD_STANDARD_NAME = "fourth_method_BaseColor_Roughness_Metallic_diffusionrenderer"
    
    for obj_name in OBJECTS:
        object_path = os.path.join(BASE_DIR, obj_name)
        
        if not os.path.exists(object_path):
            print(f"Warning: Object folder not found: {object_path}")
            continue
        
        # Get fourth method folder for this object
        fourth_method = get_fourth_method_folder(object_path)
        all_methods = METHOD_FOLDERS + [fourth_method]
        
        for method_folder in all_methods:
            method_path = os.path.join(object_path, method_folder)
            
            if not os.path.exists(method_path):
                print(f"Warning: Method folder not found: {method_path}")
                # Track status using standardized name for fourth method
                if method_folder in FOURTH_METHOD_OPTIONS:
                    object_method_status[obj_name][FOURTH_METHOD_STANDARD_NAME] = False
                else:
                    object_method_status[obj_name][method_folder] = False
                continue
            
            # Track status using standardized name for fourth method
            if method_folder in FOURTH_METHOD_OPTIONS:
                object_method_status[obj_name][FOURTH_METHOD_STANDARD_NAME] = True
            else:
                object_method_status[obj_name][method_folder] = True
            
            # Collect metrics from all envmaps
            for envmap_folder in ENVMAP_FOLDERS:
                envmap_path = os.path.join(method_path, envmap_folder)
                metrics_path = os.path.join(envmap_path, "metrics.txt")
                
                metrics = load_metrics(metrics_path)
                if metrics is not None:
                    # Use standardized name for fourth method to combine pure_supervision and neilf
                    if method_folder in FOURTH_METHOD_OPTIONS:
                        method_metrics[FOURTH_METHOD_STANDARD_NAME].append(metrics)
                    else:
                        method_metrics[method_folder].append(metrics)
                else:
                    print(f"Warning: Could not load metrics from {metrics_path}")
    
    return method_metrics, object_method_status


def calculate_averages(method_metrics):
    """
    Calculate average PSNR, SSIM, and LPIPS for each method.
    """
    results = {}
    
    for method_name, metrics_list in method_metrics.items():
        if not metrics_list:
            print(f"Warning: No metrics found for {method_name}")
            results[method_name] = {
                'psnr': 0.0,
                'ssim': 0.0,
                'lpips': 0.0,
                'count': 0
            }
            continue
        
        # Calculate averages
        psnr_values = [m[0] for m in metrics_list]
        ssim_values = [m[1] for m in metrics_list]
        lpips_values = [m[2] for m in metrics_list]
        
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        avg_lpips = sum(lpips_values) / len(lpips_values)
        
        results[method_name] = {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
            'count': len(metrics_list)
        }
    
    return results


def print_results(results, object_method_status):
    """
    Print the results in a formatted way.
    """
    FOURTH_METHOD_STANDARD_NAME = "fourth_method_BaseColor_Roughness_Metallic_diffusionrenderer"
    FOURTH_METHOD_DISPLAY_NAME = "pure_supervision/neilf_BaseColor_Roughness_Metallic_diffusionrenderer"
    
    print("\n" + "="*80)
    print("AVERAGE METRICS ACROSS ALL OBJECTS")
    print("="*80)
    print("\n")
    
    # Print header
    print(f"{'Method':<70} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'Count':>8}")
    print("-" * 110)
    
    # Print results for each method
    for method_name in METHOD_FOLDERS:
        if method_name in results:
            r = results[method_name]
            print(f"{method_name:<70} {r['psnr']:>10.3f} {r['ssim']:>10.4f} {r['lpips']:>10.4f} {r['count']:>8}")
        else:
            print(f"{method_name:<70} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'0':>8}")
    
    # Print fourth method (combined pure_supervision and neilf)
    if FOURTH_METHOD_STANDARD_NAME in results:
        r = results[FOURTH_METHOD_STANDARD_NAME]
        print(f"{FOURTH_METHOD_DISPLAY_NAME:<70} {r['psnr']:>10.3f} {r['ssim']:>10.4f} {r['lpips']:>10.4f} {r['count']:>8}")
    else:
        print(f"{FOURTH_METHOD_DISPLAY_NAME:<70} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'0':>8}")
    
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY OBJECT")
    print("="*80)
    print("\n")
    
    FOURTH_METHOD_STANDARD_NAME = "fourth_method_BaseColor_Roughness_Metallic_diffusionrenderer"
    
    # Print per-object status
    for obj_name in OBJECTS:
        print(f"\n{obj_name}:")
        for method_name in METHOD_FOLDERS:
            status = "✓" if object_method_status[obj_name][method_name] else "✗"
            print(f"  {status} {method_name}")
        # Check fourth method (could be pure_supervision or neilf)
        status = "✓" if object_method_status[obj_name][FOURTH_METHOD_STANDARD_NAME] else "✗"
        fourth_actual = get_fourth_method_folder(os.path.join(BASE_DIR, obj_name))
        print(f"  {status} {fourth_actual} (fourth method)")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\n")
    
    # Calculate and print summary
    FOURTH_METHOD_STANDARD_NAME = "fourth_method_BaseColor_Roughness_Metallic_diffusionrenderer"
    FOURTH_METHOD_DISPLAY_NAME = "pure_supervision/neilf_BaseColor_Roughness_Metallic_diffusionrenderer"
    
    for method_name in METHOD_FOLDERS:
        if method_name in results and results[method_name]['count'] > 0:
            r = results[method_name]
            print(f"{method_name}:")
            print(f"  Average PSNR:  {r['psnr']:.4f}")
            print(f"  Average SSIM:  {r['ssim']:.4f}")
            print(f"  Average LPIPS: {r['lpips']:.4f}")
            print(f"  Total samples: {r['count']} (expected: {len(OBJECTS) * len(ENVMAP_FOLDERS)} = {len(OBJECTS) * len(ENVMAP_FOLDERS)})")
            print()
    
    # Print fourth method summary
    if FOURTH_METHOD_STANDARD_NAME in results and results[FOURTH_METHOD_STANDARD_NAME]['count'] > 0:
        r = results[FOURTH_METHOD_STANDARD_NAME]
        print(f"{FOURTH_METHOD_DISPLAY_NAME}:")
        print(f"  Average PSNR:  {r['psnr']:.4f}")
        print(f"  Average SSIM:  {r['ssim']:.4f}")
        print(f"  Average LPIPS: {r['lpips']:.4f}")
        print(f"  Total samples: {r['count']} (expected: {len(OBJECTS) * len(ENVMAP_FOLDERS)} = {len(OBJECTS) * len(ENVMAP_FOLDERS)})")
        print()


def main():
    import sys
    
    # Check if debug mode is enabled
    debug = '--debug' in sys.argv
    
    print("Collecting metrics from all objects...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Objects: {', '.join(OBJECTS)}")
    print(f"Methods: {', '.join(METHOD_FOLDERS + [FOURTH_METHOD_OPTIONS[0]])}")
    print(f"Environment maps: {', '.join(ENVMAP_FOLDERS)}")
    print()
    
    if debug:
        print("DEBUG MODE: Checking folder structure...")
        for obj_name in OBJECTS[:2]:  # Check first 2 objects
            object_path = os.path.join(BASE_DIR, obj_name)
            print(f"\nChecking {obj_name}:")
            if os.path.exists(object_path):
                print(f"  Object folder exists: {object_path}")
                print(f"  Contents: {os.listdir(object_path)}")
            else:
                print(f"  Object folder NOT found: {object_path}")
        print()
    
    method_metrics, object_method_status = collect_all_metrics()
    
    print(f"\nCollected metrics from {sum(len(m) for m in method_metrics.values())} files")
    
    results = calculate_averages(method_metrics)
    
    print_results(results, object_method_status)
    
    # Also save to a text file
    FOURTH_METHOD_STANDARD_NAME = "fourth_method_BaseColor_Roughness_Metallic_diffusionrenderer"
    FOURTH_METHOD_DISPLAY_NAME = "pure_supervision/neilf_BaseColor_Roughness_Metallic_diffusionrenderer"
    
    output_file = "average_metrics_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AVERAGE METRICS ACROSS ALL OBJECTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Method':<70} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'Count':>8}\n")
        f.write("-" * 110 + "\n")
        
        for method_name in METHOD_FOLDERS:
            if method_name in results:
                r = results[method_name]
                f.write(f"{method_name:<70} {r['psnr']:>10.3f} {r['ssim']:>10.4f} {r['lpips']:>10.4f} {r['count']:>8}\n")
            else:
                f.write(f"{method_name:<70} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'0':>8}\n")
        
        # Write fourth method
        if FOURTH_METHOD_STANDARD_NAME in results:
            r = results[FOURTH_METHOD_STANDARD_NAME]
            f.write(f"{FOURTH_METHOD_DISPLAY_NAME:<70} {r['psnr']:>10.3f} {r['ssim']:>10.4f} {r['lpips']:>10.4f} {r['count']:>8}\n")
        else:
            f.write(f"{FOURTH_METHOD_DISPLAY_NAME:<70} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'0':>8}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED BREAKDOWN BY OBJECT\n")
        f.write("="*80 + "\n\n")
        
        for obj_name in OBJECTS:
            f.write(f"\n{obj_name}:\n")
            for method_name in METHOD_FOLDERS:
                status = "✓" if object_method_status[obj_name][method_name] else "✗"
                f.write(f"  {status} {method_name}\n")
            # Check fourth method
            status = "✓" if object_method_status[obj_name][FOURTH_METHOD_STANDARD_NAME] else "✗"
            fourth_actual = get_fourth_method_folder(os.path.join(BASE_DIR, obj_name))
            f.write(f"  {status} {fourth_actual} (fourth method)\n")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

