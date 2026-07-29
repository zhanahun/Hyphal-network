##BAS/RH from masks
##Measure BAS and RH with Ratio
#Major#

import cv2
import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops
import pandas as pd
import os
from pathlib import Path

def extract_color_mask(image, color_name):
    """Extract mask for specific color (teal or yellow)"""
    # Convert BGR to RGB for processing
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    if color_name == 'teal':
        # Teal color range in RGB
        lower_teal = np.array([0, 100, 100])
        upper_teal = np.array([50, 255, 255])
        mask = cv2.inRange(image_rgb, lower_teal, upper_teal)
    elif color_name == 'yellow':
        # Yellow color range in RGB
        lower_yellow = np.array([200, 200, 0])
        upper_yellow = np.array([255, 255, 100])
        mask = cv2.inRange(image_rgb, lower_yellow, upper_yellow)
    else:
        raise ValueError("Color must be 'teal' or 'yellow'")
    
    return mask

def skeletonize_image(mask):
    """Skeletonize binary mask"""
    # Convert to binary (0 and 1)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Skeletonize
    skeleton = morphology.skeletonize(binary_mask)
    
    return skeleton.astype(np.uint8) * 255

def remove_small_objects(skeleton, min_size=3):
    """Remove connected components smaller than min_size pixels"""
    # Label connected components
    labeled = label(skeleton > 0)
    
    # Get region properties
    regions = regionprops(labeled)
    
    # Create mask for objects to keep
    clean_skeleton = np.zeros_like(skeleton)
    
    for region in regions:
        if region.area >= min_size:
            # Keep this region
            coords = region.coords
            clean_skeleton[coords[:, 0], coords[:, 1]] = 255
    
    return clean_skeleton

def process_single_image(image_path):
    """Process a single image and return teal and yellow pixel counts"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Process both colors
    teal_mask = extract_color_mask(image, 'teal')
    teal_skeleton = skeletonize_image(teal_mask)
    teal_clean = remove_small_objects(teal_skeleton, min_size=3)
    teal_pixels = np.sum(teal_clean > 0)
    
    yellow_mask = extract_color_mask(image, 'yellow')
    yellow_skeleton = skeletonize_image(yellow_mask)
    yellow_clean = remove_small_objects(yellow_skeleton, min_size=3)
    yellow_pixels = np.sum(yellow_clean > 0)
    
    return {
        'teal_pixels': teal_pixels,
        'yellow_pixels': yellow_pixels
    }

def process_folder(folder_path, output_csv_path):
    """Process all images in a folder and save results to CSV"""
    folder_path = Path(folder_path)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        #image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    results = []
    
    for image_path in image_files:
        print(f"Processing: {image_path.name}")
        
        result = process_single_image(image_path)
        
        if result is not None:
            teal_px = result['teal_pixels']
            yellow_px = result['yellow_pixels']
            total_px = teal_px + yellow_px
            
            # Calculate BAS/RH ratio (teal/yellow)
            if yellow_px > 0:
                bas_rh_ratio = teal_px / yellow_px
            else:
                # Handle division by zero - you can choose how to handle this
                bas_rh_ratio = float('inf') if teal_px > 0 else 0
            
            # Calculate new ratio: teal / (teal + yellow)
            if total_px > 0:
                teal_to_total_ratio = teal_px / total_px
            else:
                teal_to_total_ratio = 0  # No pixels of either color
            
            # Calculate yellow / (teal + yellow) for completeness
            if total_px > 0:
                yellow_to_total_ratio = yellow_px / total_px
            else:
                yellow_to_total_ratio = 0
            
            results.append({
                'filename': image_path.name,
                'teal_pixels': teal_px,
                'yellow_pixels': yellow_px,
                'total_pixels': total_px,
                'bas_rh_ratio': bas_rh_ratio,
                'teal_to_total_ratio': teal_to_total_ratio,
                'yellow_to_total_ratio': yellow_to_total_ratio
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nResults saved to: {output_csv_path}")
    print(f"Total images processed: {len(results)}")
    
    # Display summary statistics
    if len(results) > 0:
        print("\nSummary Statistics:")
        print(f"Total teal pixels across all images: {df['teal_pixels'].sum()}")
        print(f"Total yellow pixels across all images: {df['yellow_pixels'].sum()}")
        print(f"Average teal pixels per image: {df['teal_pixels'].mean():.2f}")
        print(f"Average yellow pixels per image: {df['yellow_pixels'].mean():.2f}")
        
        # Handle infinite values for ratio statistics
        finite_ratios = df['bas_rh_ratio'][df['bas_rh_ratio'] != float('inf')]
        if len(finite_ratios) > 0:
            print(f"Average BAS/RH ratio: {finite_ratios.mean():.3f}")
            print(f"Median BAS/RH ratio: {finite_ratios.median():.3f}")
            print(f"Min BAS/RH ratio: {finite_ratios.min():.3f}")
            print(f"Max BAS/RH ratio: {finite_ratios.max():.3f}")
        
        # Statistics for new ratios
        print(f"\nTeal-to-Total Ratio Statistics:")
        print(f"Average teal/(teal+yellow) ratio: {df['teal_to_total_ratio'].mean():.3f}")
        print(f"Median teal/(teal+yellow) ratio: {df['teal_to_total_ratio'].median():.3f}")
        print(f"Min teal/(teal+yellow) ratio: {df['teal_to_total_ratio'].min():.3f}")
        print(f"Max teal/(teal+yellow) ratio: {df['teal_to_total_ratio'].max():.3f}")
        
        print(f"\nYellow-to-Total Ratio Statistics:")
        print(f"Average yellow/(teal+yellow) ratio: {df['yellow_to_total_ratio'].mean():.3f}")
        print(f"Median yellow/(teal+yellow) ratio: {df['yellow_to_total_ratio'].median():.3f}")
        print(f"Min yellow/(teal+yellow) ratio: {df['yellow_to_total_ratio'].min():.3f}")
        print(f"Max yellow/(teal+yellow) ratio: {df['yellow_to_total_ratio'].max():.3f}")
        
        # Count images with infinite ratios (yellow_pixels = 0)
        inf_count = sum(df['bas_rh_ratio'] == float('inf'))
        if inf_count > 0:
            print(f"\nImages with no yellow pixels (infinite BAS/RH ratio): {inf_count}")
        
        # Count images with no pixels of either color
        no_pixels_count = sum(df['total_pixels'] == 0)
        if no_pixels_count > 0:
            print(f"Images with no teal or yellow pixels: {no_pixels_count}")
    
    return df

def visualize_processing_steps(image_path, output_folder):
    """Visualize the processing steps for debugging"""
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Read original image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image {image_path}")
        return
    
    image_name = Path(image_path).stem
    
    # Save original
    cv2.imwrite(str(output_folder / f"{image_name}_01_original.png"), image)
    
    for color in ['teal', 'yellow']:
        # Extract color mask
        color_mask = extract_color_mask(image, color)
        cv2.imwrite(str(output_folder / f"{image_name}_02_{color}_mask.png"), color_mask)
        
        # Skeletonize
        skeleton = skeletonize_image(color_mask)
        cv2.imwrite(str(output_folder / f"{image_name}_03_{color}_skeleton.png"), skeleton)
        
        # Remove small objects
        clean_skeleton = remove_small_objects(skeleton, min_size=3)
        cv2.imwrite(str(output_folder / f"{image_name}_04_{color}_clean.png"), clean_skeleton)
        
        pixel_count = np.sum(clean_skeleton > 0)
        print(f"{color.capitalize()} pixels: {pixel_count}")


# Example usage
if __name__ == "__main__":
    # Process all images in a folder
    folder_path = "C:\Files\OSU\Projects\Experiment\Exp2\Analysis\Process\Inpatch\Post_temporal"  # Change this to your image folder
    output_csv = "C:\Files\OSU\Projects\Experiment\Exp2\Analysis\Process\Inpatch\Post_temporal/bas-ratio_v326.csv"
    
    # Process folder and save results
    df = process_folder(folder_path, output_csv)
    
    # Optional: Visualize processing steps for one image
    # sample_image = "path/to/sample/image.png"
    # visualize_processing_steps(sample_image, "debug_output")