import cv2
import numpy as np
from skimage.morphology import skeletonize
import pandas as pd
from pathlib import Path
from math import degrees, atan2, pi
import os

# --- Configuration ---
# Yellow range
HUE_TOLERANCE_LOWER = 5
HUE_TOLERANCE_UPPER = 65
CLOSING_KERNEL_SIZE = 7

# Set minimum required branch length (in pixels) for measurement validity
MIN_BRANCH_LENGTH = 10
# Distance along the branch to sample the direction vector
SAMPLE_DISTANCE = 60

# Junction merging distance - junctions closer than this will be merged
JUNCTION_MERGE_DISTANCE = 10

def find_yellow_mask(image):
    """Convert image to HSV and create binary mask of yellow hyphae."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Yellow threshold
    # H: 5-65 (captures orange-yellow to greenish-yellow)
    # S: 50-255 (allowing for slightly lower saturation/paler yellows)
    # V: 50-255 (allowing for slightly darker yellows)
    lower_yellow = np.array([HUE_TOLERANCE_LOWER, 50, 50])
    upper_yellow = np.array([HUE_TOLERANCE_UPPER, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
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

def extract_branches(skeleton, junction, all_junctions=None, branch_pixels_limit=50):
    """
    Extracts the pixels belonging to the distinct branches radiating from the junction.
    
    Only count a branch if its segment length is >= MIN_BRANCH_LENGTH.
    """
    y0, x0 = junction
    branches = []
    
    # Create a set of all junction positions for quick lookup
    junction_set = set(all_junctions) if all_junctions else set()
    
    # Start by getting all neighbors of the junction
    initial_neighbors = get_neighbors(skeleton, y0, x0)
    
    # Keep track of pixels already assigned to a branch
    visited_pixels = set([(y0, x0)])
    
    # Use the neighbors as starting points for new branches
    for start_y, start_x in initial_neighbors:
        if (start_y, start_x) in visited_pixels:
            continue
            
        current_branch = [(start_y, start_x)]
        visited_pixels.add((start_y, start_x))
        queue = [(start_y, start_x)]
        
        # Traverse outward from the starting neighbor
        while queue:
            cy, cx = queue.pop(0)
            
            # Look at neighbors in the original skeleton
            potential_next_pixels = get_neighbors(skeleton, cy, cx)
            
            for ny, nx in potential_next_pixels:
                if (ny, nx) not in visited_pixels and skeleton[ny, nx] > 0:
                    neighbor_count = len(get_neighbors(skeleton, ny, nx))
                    
                    # Stop if we hit another merged junction (not the current one)
                    if (ny, nx) in junction_set and (ny, nx) != junction:
                        continue
                    
                    # Only continue if the neighbor is not another junction (n_neighbors < 3)
                    if (neighbor_count < 3 or (ny, nx) == junction):
                        # Continue the branch
                        current_branch.append((ny, nx))
                        visited_pixels.add((ny, nx))
                        queue.append((ny, nx))
                        
                        # Stop if the branch gets too long to speed things up
                        if len(current_branch) >= branch_pixels_limit:
                            break
                    
            if len(current_branch) >= branch_pixels_limit:
                break
        
        # A valid branch must have a segment length of at least MIN_BRANCH_LENGTH
        if len(current_branch) >= MIN_BRANCH_LENGTH:
            branches.append(current_branch)

    return branches

def calculate_branch_angle(junction, branch):
    """
    Calculates the angle of a branch relative to the horizontal axis (0-360 degrees).
    """
    y0, x0 = junction
    
    distances = [np.hypot(y - y0, x - x0) for y, x in branch]
    
    if not distances:
        return None
    
    # Find the pixel closest to SAMPLE_DISTANCE or the furthest point if too short
    distance_diff = np.abs(np.array(distances) - SAMPLE_DISTANCE)
    sample_index = np.argmin(distance_diff)
    
    # If the closest point is still too close, use the furthest point
    if distances[sample_index] < MIN_BRANCH_LENGTH:
        sample_index = np.argmax(distances)
    
    xf, yf = branch[sample_index][1], branch[sample_index][0]  # x, y
    
    # Calculate vector from junction to the sample point
    dx = xf - x0
    dy = y0 - yf  # y-axis is inverted in images
    
    # Calculate angle in degrees (0 to 360)
    angle = degrees(atan2(dy, dx))
    
    # Normalize to 0-360 degrees
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

def calculate_smallest_angle_between_branches(angles):
    """
    Given 3 branch angles, calculate all pairwise angles and return the smallest.
    """
    if len(angles) != 3:
        return None
    
    angle_01 = calculate_angle_between(angles[0], angles[1])
    angle_02 = calculate_angle_between(angles[0], angles[2])
    angle_12 = calculate_angle_between(angles[1], angles[2])
    
    return min(angle_01, angle_02, angle_12)

def process_image(img_path, output_dir):
    """Main processing pipeline for a single image."""
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not load image {img_path.name}")
        return []

    mask = find_yellow_mask(image)
    if np.sum(mask) == 0:
        print(f"Info: No yellow hyphae found in {img_path.name} after masking.")
        return []

    skeleton = skeletonize_mask(mask)
    
    # Find junctions and merge nearby ones
    raw_junctions = find_junctions(skeleton)
    junctions = merge_nearby_junctions(raw_junctions, JUNCTION_MERGE_DISTANCE)
    
    print(f"  Found {len(raw_junctions)} raw junctions, merged to {len(junctions)} unique junctions")

    results = []
    annotated_image = image.copy()
    valid_y_branches_found = 0
    
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
        cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)  # Red circle
        
        # Extract branches from the junction (only long enough ones will be returned)
        branches = extract_branches(skeleton, junction, all_junctions=junctions)
        
        # CRITICAL FILTER: ONLY proceed if exactly 3 distinct, sufficiently long branches
        if len(branches) == 3:
            # Calculate angles for all 3 branches
            angles = []
            for branch in branches:
                angle = calculate_branch_angle(junction, branch)
                if angle is not None:
                    angles.append(angle)
            
            if len(angles) == 3:
                # Calculate the smallest angle between any two branches
                smallest_angle = calculate_smallest_angle_between_branches(angles)
                
                if smallest_angle is not None:
                    valid_y_branches_found += 1
                    
                    results.append({
                        'filename': img_path.name,
                        'junction_id': i + 1,
                        'y_coord': y,
                        'x_coord': x,
                        'branch_1_direction': angles[0],
                        'branch_2_direction': angles[1],
                        'branch_3_direction': angles[2],
                        'smallest_angle': smallest_angle,
                        'status': 'Success'
                    })

                    # Draw the branch directions
                    for angle_idx, angle_val in enumerate(angles):
                        # Convert angle back to standard math angle for vector calculation
                        angle_rad = np.deg2rad(angle_val)
                        # Extend vector SAMPLE_DISTANCE pixels for visualization
                        xf_vis = int(x + SAMPLE_DISTANCE * np.cos(angle_rad))
                        yf_vis = int(y - SAMPLE_DISTANCE * np.sin(angle_rad))  # Image y-axis is inverted
                        
                        # Colors for the 3 branches (Green, Blue, Cyan)
                        color = [(0, 255, 0), (255, 0, 0), (0, 255, 255)][angle_idx]
                        cv2.line(annotated_image, (x, y), (xf_vis, yf_vis), color, 2)
                    
                    # Add smallest angle text to the image
                    text = f"{smallest_angle:.1f} deg"
                    cv2.putText(annotated_image, text, (x + 10, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    results.append({
                        'filename': img_path.name, 'junction_id': i + 1, 'y_coord': y, 'x_coord': x,
                        'branch_1_direction': np.nan, 'branch_2_direction': np.nan, 'branch_3_direction': np.nan,
                        'smallest_angle': np.nan, 'status': 'Failed: Could not calculate smallest angle'
                    })
            else:
                results.append({
                    'filename': img_path.name, 'junction_id': i + 1, 'y_coord': y, 'x_coord': x,
                    'branch_1_direction': np.nan, 'branch_2_direction': np.nan, 'branch_3_direction': np.nan,
                    'smallest_angle': np.nan, 'status': 'Failed: Could not calculate all branch angles'
                })
        else:
            branch_count = len(branches)
            results.append({
                'filename': img_path.name, 'junction_id': i + 1, 'y_coord': y, 'x_coord': x,
                'branch_1_direction': np.nan, 'branch_2_direction': np.nan, 'branch_3_direction': np.nan,
                'smallest_angle': np.nan, 'status': f'Failed: Not exactly 3 branches (found {branch_count})'
            })

    # Save the annotated image
    output_path = output_dir / f"annotated_{img_path.name}"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Annotation saved to {output_path.name}. Total junctions: {len(junctions)}. Valid 3-branch junctions: {valid_y_branches_found}")

    return results

def main():
    """Main entry point."""
    # --- DIRECTORIES ---
    input_dir = Path(r"C:\Work\Exp\E1\pre\HP326v61\cleanup")
    output_dir = Path(r"C:\Work\Exp\E1\pre\HP326v61\cleanup\RHangleRH")
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
        
        csv_path = output_dir / "three_yellow_branch_angles.csv"
        df.to_csv(csv_path, index=False)
        
        excel_path = output_dir / "three_yellow_branch_angles.xlsx"
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Total images processed: {len(image_files)}")
        
        success_df = df[df['status'] == 'Success']
        total_valid_junctions = len(success_df)

        print(f"  Total successful 3-yellow-branch junctions: {total_valid_junctions}")
        
        if not success_df.empty:
            print(f"  Average smallest angle: {success_df['smallest_angle'].mean():.2f} deg")
            print(f"  Min smallest angle: {success_df['smallest_angle'].min():.2f} deg")
            print(f"  Max smallest angle: {success_df['smallest_angle'].max():.2f} deg")
        else:
            print("  No successful measurements to report statistics.")
            
        print(f"Results saved to:\n  CSV: {csv_path}\n  Excel: {excel_path}")
        print(f"{'='*60}")
    else:
        print("\nAnalysis finished. No junctions or valid angles found in any image.")

if __name__ == "__main__":
    main()