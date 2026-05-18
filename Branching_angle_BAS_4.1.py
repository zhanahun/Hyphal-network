#Keep one output per branch point

import cv2
import numpy as np
from skimage.morphology import skeletonize
import pandas as pd
from pathlib import Path
from math import degrees, atan2, pi
import os

# --- Configuration ---
# Yellow range
YELLOW_HUE_LOWER = 5
YELLOW_HUE_UPPER = 65

# Teal/Cyan range
TEAL_HUE_LOWER = 80
TEAL_HUE_UPPER = 100

CLOSING_KERNEL_SIZE = 7

# Set minimum required branch length (in pixels) for measurement validity
MIN_BRANCH_LENGTH = 10
# Distance along the branch to sample the direction vector
SAMPLE_DISTANCE = 60

# Junction merging distance - junctions closer than this will be merged
JUNCTION_MERGE_DISTANCE = 10

def find_color_mask(image, hue_lower, hue_upper, color_name="color"):
    """Convert image to HSV and create binary mask for specified hue range."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([hue_lower, 50, 50])
    upper_bound = np.array([hue_upper, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Morphological closing
    kernel = np.ones((CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def skeletonize_mask(mask):
    """Convert binary mask to 1-pixel skeleton."""
    binary = mask > 0
    skeleton = skeletonize(binary)
    return skeleton.astype(np.uint8) * 255

def get_neighbors(skeleton, y, x):
    """Get 8-connected neighbors of a pixel."""
    h, w = skeleton.shape
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                neighbors.append((ny, nx))
    return neighbors

def find_junctions(skeleton):
    """Find pixels with 3 or more connections (branch points)."""
    junctions = []
    h, w = skeleton.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] > 0:
                n_neighbors = len(get_neighbors(skeleton, y, x))
                if n_neighbors >= 3:
                    junctions.append((y, x))
    return junctions

def merge_nearby_junctions(junctions, merge_distance=JUNCTION_MERGE_DISTANCE):
    """
    Merge junctions that are within merge_distance of each other.
    Returns a list of merged junction coordinates (centroids of clusters).
    """
    if not junctions:
        return []
    
    # Convert to numpy array for easier computation
    junctions = np.array(junctions)
    n = len(junctions)
    
    # Track which junctions have been merged
    merged = [False] * n
    merged_junctions = []
    
    for i in range(n):
        if merged[i]:
            continue
        
        # Start a new cluster with this junction
        cluster = [junctions[i]]
        merged[i] = True
        
        # Find all nearby junctions to merge
        for j in range(i + 1, n):
            if merged[j]:
                continue
            
            # Calculate distance
            dist = np.hypot(junctions[i][0] - junctions[j][0], 
                           junctions[i][1] - junctions[j][1])
            
            if dist <= merge_distance:
                cluster.append(junctions[j])
                merged[j] = True
        
        # Calculate centroid of the cluster
        cluster = np.array(cluster)
        centroid = tuple(np.mean(cluster, axis=0).astype(int))
        merged_junctions.append(centroid)
    
    return merged_junctions

def extract_branches_with_color(skeleton, junction, color_mask, branch_pixels_limit=50, all_junctions=None):
    """
    Extracts branches and determines their color (yellow or teal).
    Returns list of (branch_pixels, color_type) where color_type is 'yellow' or 'teal'.
    """
    y0, x0 = junction
    branches = []
    
    # Create a set of all junction positions for quick lookup
    junction_set = set(all_junctions) if all_junctions else set()
    
    initial_neighbors = get_neighbors(skeleton, y0, x0)
    visited_pixels = set([(y0, x0)])
    
    for start_y, start_x in initial_neighbors:
        if (start_y, start_x) in visited_pixels:
            continue
            
        current_branch = [(start_y, start_x)]
        visited_pixels.add((start_y, start_x))
        queue = [(start_y, start_x)]
        
        while queue:
            cy, cx = queue.pop(0)
            potential_next_pixels = get_neighbors(skeleton, cy, cx)
            
            for ny, nx in potential_next_pixels:
                if (ny, nx) not in visited_pixels and skeleton[ny, nx] > 0:
                    neighbor_count = len(get_neighbors(skeleton, ny, nx))
                    
                    # Stop if we hit another merged junction (not the current one)
                    if (ny, nx) in junction_set and (ny, nx) != junction:
                        continue
                    
                    if (neighbor_count < 3 or (ny, nx) == junction):
                        current_branch.append((ny, nx))
                        visited_pixels.add((ny, nx))
                        queue.append((ny, nx))
                        
                        if len(current_branch) >= branch_pixels_limit:
                            break
                    
            if len(current_branch) >= branch_pixels_limit:
                 break
        
        # Determine branch color by checking pixels against color mask
        if len(current_branch) >= MIN_BRANCH_LENGTH:
            # Count how many pixels are in the color mask
            color_pixel_count = sum(1 for y, x in current_branch if color_mask[y, x] > 0)
            # If more than 50% of pixels are in the color mask, consider it that color
            color_type = 'colored' if color_pixel_count > len(current_branch) * 0.5 else 'other'
            branches.append((current_branch, color_type))

    return branches

def calculate_branch_angle(junction, branch):
    """
    Calculates the angle of a branch relative to the horizontal axis (0-360 degrees).
    """
    y0, x0 = junction
    
    distances = [np.hypot(y - y0, x - x0) for y, x in branch]
    
    if not distances:
        return None
    
    distance_diff = np.abs(np.array(distances) - SAMPLE_DISTANCE)
    sample_index = np.argmin(distance_diff)
    
    if distances[sample_index] < MIN_BRANCH_LENGTH:
        sample_index = np.argmax(distances)
    
    xf, yf = branch[sample_index][1], branch[sample_index][0]
    
    dx = xf - x0
    dy = y0 - yf  # y-axis is inverted in images
    
    angle = degrees(atan2(dy, dx))
    angle = angle % 360
    if angle < 0:
        angle += 360
        
    return angle

def calculate_angle_between(angle1, angle2):
    """
    Calculate the acute angle between two directions (0-360 degrees).
    Returns the smaller of the two possible angles.
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff

def get_text_position_offset(junction_id, base_x, base_y):
    """
    Calculate text position with offset based on junction ID to avoid overlap.
    """
    # Create a spiral pattern for text positioning
    offsets = [
        (10, -10),   # Default position
        (10, -30),   # Above
        (10, 10),    # Below
        (-80, -10),  # Left
        (30, -10),   # Right
        (10, -50),   # Far above
        (10, 30),    # Far below
    ]
    
    offset_index = junction_id % len(offsets)
    dx, dy = offsets[offset_index]
    
    return base_x + dx, base_y + dy

def process_image(img_path, output_dir):
    """Main processing pipeline for a single image."""
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not load image {img_path.name}")
        return []

    # Create masks for both colors
    yellow_mask = find_color_mask(image, YELLOW_HUE_LOWER, YELLOW_HUE_UPPER, "yellow")
    teal_mask = find_color_mask(image, TEAL_HUE_LOWER, TEAL_HUE_UPPER, "teal")
    
    # Combine masks for skeleton
    combined_mask = cv2.bitwise_or(yellow_mask, teal_mask)
    
    if np.sum(combined_mask) == 0:
        print(f"Info: No yellow or teal hyphae found in {img_path.name} after masking.")
        return []

    skeleton = skeletonize_mask(combined_mask)
    
    # Find junctions and merge nearby ones
    raw_junctions = find_junctions(skeleton)
    junctions = merge_nearby_junctions(raw_junctions, JUNCTION_MERGE_DISTANCE)
    
    print(f"  Found {len(raw_junctions)} raw junctions, merged to {len(junctions)} unique junctions")

    results = []
    annotated_image = image.copy()
    valid_junctions_found = 0
    
    # Create global visited set to track which junctions have been processed
    processed_junctions = set()

    for i, junction in enumerate(junctions):
        y, x = junction
        
        # Skip if this junction is too close to an already processed one
        skip_junction = False
        for py, px in processed_junctions:
            dist = np.hypot(y - py, x - px)
            if dist < JUNCTION_MERGE_DISTANCE:
                skip_junction = True
                break
        
        if skip_junction:
            continue
        
        # Mark this junction as processed
        processed_junctions.add((y, x))
        
        # Draw junction point for visualization
        cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
        
        # Extract branches and determine their colors
        yellow_branches = []
        teal_branches = []
        
        branches_yellow = extract_branches_with_color(skeleton, junction, yellow_mask, all_junctions=junctions)
        branches_teal = extract_branches_with_color(skeleton, junction, teal_mask, all_junctions=junctions)
        
        # Classify branches
        all_branches = []
        for branch, _ in branches_yellow:
            yellow_count = sum(1 for py, px in branch if yellow_mask[py, px] > 0)
            teal_count = sum(1 for py, px in branch if teal_mask[py, px] > 0)
            
            if yellow_count > teal_count:
                yellow_branches.append(branch)
                all_branches.append(('yellow', branch))
            elif teal_count > yellow_count:
                teal_branches.append(branch)
                all_branches.append(('teal', branch))
        
        # Check if we have exactly 2 yellow and 1 teal branches
        if len(yellow_branches) == 2 and len(teal_branches) == 1:
            # Calculate angles
            yellow_angle_1 = calculate_branch_angle(junction, yellow_branches[0])
            yellow_angle_2 = calculate_branch_angle(junction, yellow_branches[1])
            teal_angle = calculate_branch_angle(junction, teal_branches[0])
            
            if yellow_angle_1 is not None and yellow_angle_2 is not None and teal_angle is not None:
                # Calculate angles between each yellow branch and the teal branch
                angle_y1_teal = calculate_angle_between(yellow_angle_1, teal_angle)
                angle_y2_teal = calculate_angle_between(yellow_angle_2, teal_angle)
                
                # Output the smaller of the two angles
                smaller_angle = min(angle_y1_teal, angle_y2_teal)
                
                valid_junctions_found += 1
                
                results.append({
                    'filename': img_path.name,
                    'junction_id': i + 1,
                    'y_coord': y,
                    'x_coord': x,
                    'yellow_branch_1_direction': yellow_angle_1,
                    'yellow_branch_2_direction': yellow_angle_2,
                    'teal_branch_direction': teal_angle,
                    'angle_yellow1_teal': angle_y1_teal,
                    'angle_yellow2_teal': angle_y2_teal,
                    'smaller_angle': smaller_angle,
                    'status': 'Success'
                })
                
                # Draw the branches with different colors
                for idx, (branch_type, branch) in enumerate(all_branches):
                    angle = calculate_branch_angle(junction, branch)
                    if angle is not None:
                        angle_rad = np.deg2rad(angle)
                        xf_vis = int(x + SAMPLE_DISTANCE * np.cos(angle_rad))
                        yf_vis = int(y - SAMPLE_DISTANCE * np.sin(angle_rad))
                        
                        if branch_type == 'yellow':
                            color = (0, 255, 255)  # Yellow in BGR
                        else:
                            color = (255, 255, 0)  # Cyan/Teal in BGR
                        
                        cv2.line(annotated_image, (x, y), (xf_vis, yf_vis), color, 2)
                
                # Add angle text - single label per junction
                text = f"{smaller_angle:.1f} deg"
                text_x, text_y = x + 10, y - 10
                
                cv2.putText(annotated_image, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                results.append({
                    'filename': img_path.name, 'junction_id': i + 1, 'y_coord': y, 'x_coord': x,
                    'yellow_branch_1_direction': np.nan, 'yellow_branch_2_direction': np.nan,
                    'teal_branch_direction': np.nan, 'angle_yellow1_teal': np.nan,
                    'angle_yellow2_teal': np.nan, 'smaller_angle': np.nan,
                    'status': 'Failed: Could not calculate angles'
                })
        else:
            results.append({
                'filename': img_path.name, 'junction_id': i + 1, 'y_coord': y, 'x_coord': x,
                'yellow_branch_1_direction': np.nan, 'yellow_branch_2_direction': np.nan,
                'teal_branch_direction': np.nan, 'angle_yellow1_teal': np.nan,
                'angle_yellow2_teal': np.nan, 'smaller_angle': np.nan,
                'status': f'Failed: Not 2 yellow + 1 teal (found {len(yellow_branches)} yellow, {len(teal_branches)} teal)'
            })

    # Save the annotated image
    output_path = output_dir / f"annotated_{img_path.name}"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Annotation saved to {output_path.name}. Total junctions: {len(junctions)}. Valid 2Y+1T junctions: {valid_junctions_found}")

    return results

def main():
    """Main entry point."""
    # --- DIRECTORIES ---
    input_dir = Path(r"C:\Work\Exp\E1\pre\HP326v61\cleanup")
    output_dir = Path(r"C:\Work\Exp\E1\pre\HP326v61\cleanup\BASangle")
    # -------------------
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG, JPG, and JPEG images
    image_files = list(input_dir.glob("*.png"))
    image_files.extend(list(input_dir.glob("*.jpg")))
    image_files.extend(list(input_dir.glob("*.jpeg")))
    
    if not image_files:
        print(f"No image files (*.png, *.jpg, *.jpeg) found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Process all images
    all_results = []
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        results = process_image(img_path, output_dir)
        all_results.extend(results)
    
    # Save results to CSV and Excel
    if all_results:
        df = pd.DataFrame(all_results)
        
        csv_path = output_dir / "yellow_teal_junction_angles.csv"
        df.to_csv(csv_path, index=False)
        
        excel_path = output_dir / "yellow_teal_junction_angles.xlsx"
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Total images processed: {len(image_files)}")
        
        success_df = df[df['status'] == 'Success']
        total_valid_junctions = len(success_df)

        print(f"  Total successful 2-yellow + 1-teal junctions: {total_valid_junctions}")
        
        if not success_df.empty:
            print(f"  Average angle (Yellow1-Teal): {success_df['angle_yellow1_teal'].mean():.2f}°")
            print(f"  Average angle (Yellow2-Teal): {success_df['angle_yellow2_teal'].mean():.2f}°")
            print(f"  Average smaller angle: {success_df['smaller_angle'].mean():.2f}°")
            print(f"  Min smaller angle: {success_df['smaller_angle'].min():.2f}°")
            print(f"  Max smaller angle: {success_df['smaller_angle'].max():.2f}°")
        else:
            print("  No successful measurements to report statistics.")
            
        print(f"Results saved to:\n  CSV: {csv_path}\n  Excel: {excel_path}")
        print(f"{'='*60}")
    else:
        print("\nAnalysis finished. No valid junctions found in any image.")

if __name__ == "__main__":
    main()