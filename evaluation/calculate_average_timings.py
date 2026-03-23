#!/usr/bin/env python3
"""
Script to calculate average timings across multiple folders.
Reads timings.txt files from each folder and averages specific timing entries.
"""

import os
import re
from collections import defaultdict

# Base directory
BASE_DIR = os.environ.get("RESULTS_DIR", "/path/to/results/navi")

# Timing entries to extract (in order)
TIMING_ENTRIES = [
    "diffusion_renderer",
    "gaussian_splatting",
    "train_3dgs_preprocessed",
    "train_base_neilf",
    "train_mlp_BaseColor_Roughness_Metallic_Deferred_diffusionrenderer"
]


def extract_timing_value(line):
    """
    Extract timing value from a line like:
    [timestamp] step_name: 123.456s
    Returns the float value or None if not found.
    """
    # Match pattern: ...: number followed by 's'
    match = re.search(r':\s*([\d.]+)s', line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def count_images(folder_path):
    """
    Count the number of images in a folder.
    Looks in: {folder}/mlp_BaseColor_Roughness_Metallic_Deferred_diffusionrenderer/envmap3/{any_subfolder}/
    Returns the number of files in the first subdirectory found, or None if not found.
    """
    mlp_path = os.path.join(folder_path, "mlp_BaseColor_Roughness_Metallic_Deferred_diffusionrenderer")
    envmap_path = os.path.join(mlp_path, "envmap3")
    
    if not os.path.exists(envmap_path):
        return None
    
    try:
        # Get first subdirectory in envmap3
        subdirs = [d for d in os.listdir(envmap_path) 
                   if os.path.isdir(os.path.join(envmap_path, d))]
        
        if not subdirs:
            return None
        
        # Count files in the first subdirectory
        first_subdir = os.path.join(envmap_path, subdirs[0])
        files = [f for f in os.listdir(first_subdir) 
                 if os.path.isfile(os.path.join(first_subdir, f))]
        
        return len(files)
    
    except Exception as e:
        print(f"Error counting images in {folder_path}: {e}")
        return None


def load_timings(timings_path):
    """
    Load timings from timings.txt file.
    Returns a dictionary: timing_entry -> timing_value (first occurrence)
    """
    if not os.path.exists(timings_path):
        return None
    
    try:
        with open(timings_path, 'r') as f:
            lines = f.readlines()
        
        timings = {}
        
        # Track which entries we've found (we only want the first occurrence)
        found_entries = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check each timing entry
            for entry in TIMING_ENTRIES:
                if entry not in found_entries and entry in line:
                    # Extract the timing value
                    timing_value = extract_timing_value(line)
                    if timing_value is not None:
                        timings[entry] = timing_value
                        found_entries.add(entry)
                        break  # Found this entry, move to next line
        
        return timings
    
    except Exception as e:
        print(f"Error loading timings from {timings_path}: {e}")
        return None


def collect_all_timings():
    """
    Collect timings from all folders in the base directory.
    Returns a tuple: (timing_values, image_counts, per_object_timings)
    - timing_values: dictionary: timing_entry -> list of timing values
    - image_counts: dictionary: folder_name -> image_count for diffusion_renderer
    - per_object_timings: dictionary: folder_name -> dictionary of timing_entry -> timing_value
    """
    timing_values = defaultdict(list)
    image_counts = {}  # Dictionary: folder_name -> image_count
    per_object_timings = {}  # Dictionary: folder_name -> {timing_entry: timing_value}
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory not found: {BASE_DIR}")
        return timing_values, image_counts, {}, per_object_timings
    
    # Get all folders in the base directory
    folders = [f for f in os.listdir(BASE_DIR) 
               if os.path.isdir(os.path.join(BASE_DIR, f))]
    
    print(f"Found {len(folders)} folders in {BASE_DIR}")
    
    # Store folder names with timings for diffusion_renderer
    diffusion_renderer_timings = {}  # folder_name -> timing_value
    
    for folder in sorted(folders):
        folder_path = os.path.join(BASE_DIR, folder)
        timings_path = os.path.join(folder_path, "timings", "timings.txt")
        
        if not os.path.exists(timings_path):
            print(f"Warning: timings.txt not found in {folder_path}")
            continue
        
        timings = load_timings(timings_path)
        if timings is None:
            print(f"Warning: Could not load timings from {timings_path}")
            continue
        
        # Store per-object timings
        per_object_timings[folder] = timings.copy()
        
        # Add timings to the collection
        for entry, value in timings.items():
            timing_values[entry].append(value)
            # Store diffusion_renderer timing with folder name
            if entry == "diffusion_renderer":
                diffusion_renderer_timings[folder] = value
        
        # Count images for diffusion_renderer timing calculation
        if "diffusion_renderer" in timings:
            image_count = count_images(folder_path)
            if image_count is not None:
                image_counts[folder] = image_count
            else:
                print(f"Warning: Could not count images for {folder}")
        
        # Report which entries were found for this folder
        found_entries = list(timings.keys())
        missing_entries = [e for e in TIMING_ENTRIES if e not in timings]
        if missing_entries:
            print(f"  {folder}: Found {len(found_entries)}/{len(TIMING_ENTRIES)} entries "
                  f"(missing: {', '.join(missing_entries)})")
    
    return timing_values, image_counts, diffusion_renderer_timings, per_object_timings


def calculate_averages(timing_values, image_counts, diffusion_renderer_timings):
    """
    Calculate average timings for each entry.
    For diffusion_renderer, also calculates time per image.
    """
    results = {}
    
    for entry in TIMING_ENTRIES:
        if entry not in timing_values or not timing_values[entry]:
            print(f"Warning: No timings found for {entry}")
            results[entry] = {
                'average': 0.0,
                'count': 0,
                'min': 0.0,
                'max': 0.0
            }
            continue
        
        values = timing_values[entry]
        avg = sum(values) / len(values)
        results[entry] = {
            'average': avg,
            'count': len(values),
            'min': min(values),
            'max': max(values)
        }
        
        # Special handling for diffusion_renderer: calculate time per image
        if entry == "diffusion_renderer" and image_counts and diffusion_renderer_timings:
            # Match timings with image counts by folder name
            time_per_image_values = []
            image_count_list = []
            
            # Find folders that have both timing and image count
            for folder_name in sorted(image_counts.keys()):
                if folder_name in diffusion_renderer_timings:
                    timing_value = diffusion_renderer_timings[folder_name]
                    image_count = image_counts[folder_name]
                    time_per_image = timing_value / image_count
                    time_per_image_values.append(time_per_image)
                    image_count_list.append(image_count)
            
            if time_per_image_values:
                results[entry]['time_per_image'] = {
                    'average': sum(time_per_image_values) / len(time_per_image_values),
                    'count': len(time_per_image_values),
                    'min': min(time_per_image_values),
                    'max': max(time_per_image_values),
                    'avg_images_per_folder': sum(image_count_list) / len(image_count_list) if image_count_list else 0
                }
    
    return results


def print_per_object_timings(per_object_timings, image_counts):
    """
    Print timings for each object (folder) individually.
    """
    print("\n" + "="*100)
    print("PER-OBJECT TIMINGS")
    print("="*100)
    print("\n")
    
    # Sort folders alphabetically
    sorted_folders = sorted(per_object_timings.keys())
    
    # Print header
    header = f"{'Object/Folder':<40}"
    for entry in TIMING_ENTRIES:
        header += f" {entry[:20]:>20}"
    if image_counts:
        header += f" {'Images':>10} {'Time/img (s)':>15}"
    print(header)
    print("-" * (40 + len(TIMING_ENTRIES) * 21 + (30 if image_counts else 0)))
    
    # Print each object's timings
    for folder in sorted_folders:
        timings = per_object_timings[folder]
        row = f"{folder:<40}"
        
        for entry in TIMING_ENTRIES:
            if entry in timings:
                row += f" {timings[entry]:>20.3f}"
            else:
                row += f" {'N/A':>20}"
        
        # Add image count and time per image for diffusion_renderer
        if image_counts and folder in image_counts:
            image_count = image_counts[folder]
            row += f" {image_count:>10}"
            if "diffusion_renderer" in timings:
                time_per_image = timings["diffusion_renderer"] / image_count
                row += f" {time_per_image:>15.3f}"
            else:
                row += f" {'N/A':>15}"
        
        print(row)
    
    print()


def print_results(results):
    """
    Print the results in a formatted way.
    """
    print("\n" + "="*100)
    print("AVERAGE TIMINGS ACROSS ALL FOLDERS")
    print("="*100)
    print("\n")
    
    # Print header
    print(f"{'Timing Entry':<70} {'Average (s)':>15} {'Count':>8} {'Min (s)':>15} {'Max (s)':>15}")
    print("-" * 123)
    
    # Print results for each timing entry
    for entry in TIMING_ENTRIES:
        if entry in results:
            r = results[entry]
            if r['count'] > 0:
                print(f"{entry:<70} {r['average']:>15.3f} {r['count']:>8} "
                      f"{r['min']:>15.3f} {r['max']:>15.3f}")
            else:
                print(f"{entry:<70} {'N/A':>15} {'0':>8} {'N/A':>15} {'N/A':>15}")
        else:
            print(f"{entry:<70} {'N/A':>15} {'0':>8} {'N/A':>15} {'N/A':>15}")
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print("\n")
    
    for entry in TIMING_ENTRIES:
        if entry in results and results[entry]['count'] > 0:
            r = results[entry]
            print(f"{entry}:")
            print(f"  Average: {r['average']:.3f} seconds")
            print(f"  Count:   {r['count']} folders")
            print(f"  Min:     {r['min']:.3f} seconds")
            print(f"  Max:     {r['max']:.3f} seconds")
            
            # Special output for diffusion_renderer with time per image
            if entry == "diffusion_renderer" and 'time_per_image' in r:
                tpi = r['time_per_image']
                print(f"  Time per image:")
                print(f"    Average: {tpi['average']:.3f} seconds/image")
                print(f"    Count:   {tpi['count']} folders")
                print(f"    Min:     {tpi['min']:.3f} seconds/image")
                print(f"    Max:     {tpi['max']:.3f} seconds/image")
                print(f"    Average images per folder: {tpi['avg_images_per_folder']:.1f}")
            print()


def main():
    import sys
    
    # Check if debug mode is enabled
    debug = '--debug' in sys.argv
    
    print("Collecting timings from all folders...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Timing entries to extract: {', '.join(TIMING_ENTRIES)}")
    print()
    
    timing_values, image_counts, diffusion_renderer_timings, per_object_timings = collect_all_timings()
    
    total_timings = sum(len(v) for v in timing_values.values())
    print(f"\nCollected {total_timings} timing values across {len(TIMING_ENTRIES)} entry types")
    if image_counts:
        print(f"Collected image counts for {len(image_counts)} folders")
    
    # Print per-object timings first
    print_per_object_timings(per_object_timings, image_counts)
    
    results = calculate_averages(timing_values, image_counts, diffusion_renderer_timings)
    
    print_results(results)
    
    # Also save to a text file
    output_file = "average_timings_results.txt"
    with open(output_file, 'w') as f:
        # Write per-object timings
        f.write("="*100 + "\n")
        f.write("PER-OBJECT TIMINGS\n")
        f.write("="*100 + "\n\n")
        
        sorted_folders = sorted(per_object_timings.keys())
        
        # Write header
        header = f"{'Object/Folder':<40}"
        for entry in TIMING_ENTRIES:
            header += f" {entry[:20]:>20}"
        if image_counts:
            header += f" {'Images':>10} {'Time/img (s)':>15}"
        f.write(header + "\n")
        f.write("-" * (40 + len(TIMING_ENTRIES) * 21 + (30 if image_counts else 0)) + "\n")
        
        # Write each object's timings
        for folder in sorted_folders:
            timings = per_object_timings[folder]
            row = f"{folder:<40}"
            
            for entry in TIMING_ENTRIES:
                if entry in timings:
                    row += f" {timings[entry]:>20.3f}"
                else:
                    row += f" {'N/A':>20}"
            
            # Add image count and time per image for diffusion_renderer
            if image_counts and folder in image_counts:
                image_count = image_counts[folder]
                row += f" {image_count:>10}"
                if "diffusion_renderer" in timings:
                    time_per_image = timings["diffusion_renderer"] / image_count
                    row += f" {time_per_image:>15.3f}"
                else:
                    row += f" {'N/A':>15}"
            
            f.write(row + "\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("AVERAGE TIMINGS ACROSS ALL FOLDERS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Timing Entry':<70} {'Average (s)':>15} {'Count':>8} {'Min (s)':>15} {'Max (s)':>15}\n")
        f.write("-" * 123 + "\n")
        
        for entry in TIMING_ENTRIES:
            if entry in results:
                r = results[entry]
                if r['count'] > 0:
                    f.write(f"{entry:<70} {r['average']:>15.3f} {r['count']:>8} "
                           f"{r['min']:>15.3f} {r['max']:>15.3f}\n")
                else:
                    f.write(f"{entry:<70} {'N/A':>15} {'0':>8} {'N/A':>15} {'N/A':>15}\n")
            else:
                f.write(f"{entry:<70} {'N/A':>15} {'0':>8} {'N/A':>15} {'N/A':>15}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        for entry in TIMING_ENTRIES:
            if entry in results and results[entry]['count'] > 0:
                r = results[entry]
                f.write(f"{entry}:\n")
                f.write(f"  Average: {r['average']:.3f} seconds\n")
                f.write(f"  Count:   {r['count']} folders\n")
                f.write(f"  Min:     {r['min']:.3f} seconds\n")
                f.write(f"  Max:     {r['max']:.3f} seconds\n")
                
                # Special output for diffusion_renderer with time per image
                if entry == "diffusion_renderer" and 'time_per_image' in r:
                    tpi = r['time_per_image']
                    f.write(f"  Time per image:\n")
                    f.write(f"    Average: {tpi['average']:.3f} seconds/image\n")
                    f.write(f"    Count:   {tpi['count']} folders\n")
                    f.write(f"    Min:     {tpi['min']:.3f} seconds/image\n")
                    f.write(f"    Max:     {tpi['max']:.3f} seconds/image\n")
                    f.write(f"    Average images per folder: {tpi['avg_images_per_folder']:.1f}\n")
                
                f.write("\n")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

