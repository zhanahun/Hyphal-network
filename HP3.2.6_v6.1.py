#revised from v5.2
#more aggressive on RH
#based on Gemini

## Prediction - v5.4 (Advanced Stitching + TTA + Reconnection + Bridge Correction)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import os
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage # Required for bridge correction

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from patchify import patchify

# --- Model Definitions (Matching HP3.2.6_v5.2.py) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1, max_pooling=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        self.max_pooling = max_pooling
        self.pool = nn.MaxPool2d(2, 2) if max_pooling else None
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        conv = self.relu(self.bn1(self.conv1(x)))
        conv = self.relu(self.bn2(self.conv2(conv)))
        if self.dropout: conv = self.dropout(conv)
        next_layer = self.pool(conv) if self.max_pooling else conv
        return next_layer, conv

class DenseBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels, dropout_prob=0.1):
        super(DenseBlock, self).__init__()
        total_in_channels = sum(in_channels_list)
        self.conv1 = nn.Conv2d(total_in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, inputs):
        x = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
        conv = self.relu(self.bn1(self.conv1(x)))
        conv = self.relu(self.bn2(self.conv2(conv)))
        if self.dropout: conv = self.dropout(conv)
        return conv

class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels=3, n_filters=32, n_classes=3, deep_supervision=False, dropout_prob=0.1):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]
        self.cblock0_0 = ConvBlock(input_channels, filters[0]); self.cblock1_0 = ConvBlock(filters[0], filters[1])
        self.cblock2_0 = ConvBlock(filters[1], filters[2]); self.cblock3_0 = ConvBlock(filters[2], filters[3])
        self.cblock4_0 = ConvBlock(filters[3], filters[4], max_pooling=False)
        self.up4_3 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=1, output_padding=1)
        self.up3_2 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=1, output_padding=1)
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1, output_padding=1)
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[0], 3, stride=2, padding=1, output_padding=1)
        self.dense0_1 = DenseBlock([filters[0], filters[0]], filters[0]); self.dense0_2 = DenseBlock([filters[0], filters[0], filters[0]], filters[0])
        self.dense0_3 = DenseBlock([filters[0], filters[0], filters[0], filters[0]], filters[0]); self.dense0_4 = DenseBlock([filters[0], filters[0], filters[0], filters[0], filters[0]], filters[0])
        self.dense1_1 = DenseBlock([filters[1], filters[1]], filters[1]); self.dense1_2 = DenseBlock([filters[1], filters[1], filters[1]], filters[1])
        self.dense1_3 = DenseBlock([filters[1], filters[1], filters[1], filters[1]], filters[1]); self.dense2_1 = DenseBlock([filters[2], filters[2]], filters[2])
        self.dense2_2 = DenseBlock([filters[2], filters[2], filters[2]], filters[2]); self.dense3_1 = DenseBlock([filters[3], filters[3]], filters[3])
        self.conv_final = nn.Conv2d(filters[0], filters[0], 3, padding=1); self.bn_final = nn.BatchNorm2d(filters[0])
        self.out_final = nn.Conv2d(filters[0], n_classes, 1); self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0_0_p, x0_0 = self.cblock0_0(x); x1_0_p, x1_0 = self.cblock1_0(x0_0_p)
        x2_0_p, x2_0 = self.cblock2_0(x1_0_p); x3_0_p, x3_0 = self.cblock3_0(x2_0_p)
        _, x4_0 = self.cblock4_0(x3_0_p)
        x3_1 = self.dense3_1([x3_0, self.up4_3(x4_0)]); x2_1 = self.dense2_1([x2_0, self.up3_2(x3_0)])
        x1_1 = self.dense1_1([x1_0, self.up2_1(x2_0)]); x0_1 = self.dense0_1([x0_0, self.up1_0(x1_0)])
        x2_2 = self.dense2_2([x2_0, x2_1, self.up3_2(x3_1)]); x1_2 = self.dense1_2([x1_0, x1_1, self.up2_1(x2_1)])
        x0_2 = self.dense0_2([x0_0, x0_1, self.up1_0(x1_1)]); x1_3 = self.dense1_3([x1_0, x1_1, x1_2, self.up2_1(x2_2)])
        x0_3 = self.dense0_3([x0_0, x0_1, x0_2, self.up1_0(x1_2)]); x0_4 = self.dense0_4([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)])
        return self.out_final(self.relu(self.bn_final(self.conv_final(x0_4))))

# --- 1. TTA Prediction Helper ---
def predict_patch_with_tta(model, patch, device):
    base_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
    predictions = []
    # Original
    out = model(base_tensor)
    if isinstance(out, list): out = out[-1]
    predictions.append(F.softmax(out, dim=1))
    # Horizontal Flip
    out_hf = model(torch.flip(base_tensor, [3]))
    if isinstance(out_hf, list): out_hf = out_hf[-1]
    predictions.append(torch.flip(F.softmax(out_hf, dim=1), [3]))
    # Vertical Flip
    out_vf = model(torch.flip(base_tensor, [2]))
    if isinstance(out_vf, list): out_vf = out_vf[-1]
    predictions.append(torch.flip(F.softmax(out_vf, dim=1), [2]))
    # Rotate 90
    if patch.shape[0] == patch.shape[1]:
        out_r90 = model(torch.rot90(base_tensor, 1, [2, 3]))
        if isinstance(out_r90, list): out_r90 = out_r90[-1]
        predictions.append(torch.rot90(F.softmax(out_r90, dim=1), -1, [2, 3]))
    avg_probs = torch.stack(predictions).mean(dim=0)
    return avg_probs[0].cpu().numpy()

# --- 2. Advanced Stitcher ---
def advanced_stitch_predictions(model, image, patch_size=512, overlap=0.50, device='cuda'):
    image_np = np.array(image.convert('RGB'))
    h_orig, w_orig = image_np.shape[:2]
    h_win, w_win = np.hanning(patch_size), np.hanning(patch_size)
    window_2d = np.outer(h_win, w_win)
    step = int(patch_size * (1 - overlap))
    pad_h = (step - h_orig % step) % step
    if (h_orig + pad_h) < patch_size: pad_h = patch_size - h_orig
    pad_w = (step - w_orig % step) % step
    if (w_orig + pad_w) < patch_size: pad_w = patch_size - w_orig
    padded_image = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    h_pad, w_pad = padded_image.shape[:2]
    prob_map = np.zeros((3, h_pad, w_pad), dtype=np.float32)
    weight_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    for y in range(0, h_pad - patch_size + 1, step):
        for x in range(0, w_pad - patch_size + 1, step):
            patch = padded_image[y:y+patch_size, x:x+patch_size, :]
            patch_prob = predict_patch_with_tta(model, patch, device)
            prob_map[:, y:y+patch_size, x:x+patch_size] += patch_prob * window_2d
            weight_map[y:y+patch_size, x:x+patch_size] += window_2d
    weight_map[weight_map == 0] = 1
    final_prob = (prob_map / weight_map)[:, :h_orig, :w_orig]
    return np.argmax(final_prob, axis=0).astype(np.uint8)

# --- 3. Post-Processing (Reconnection & Denoising) ---
def reconnect_hyphae(mask, kernel_size=3):
    mask_rh = (mask == 1).astype(np.uint8)
    mask_bas = (mask == 2).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_rh_closed = cv.morphologyEx(mask_rh, cv.MORPH_CLOSE, kernel)
    mask_bas_closed = cv.morphologyEx(mask_bas, cv.MORPH_CLOSE, kernel)
    final_mask = np.zeros_like(mask)
    final_mask[mask_rh_closed == 1] = 1
    final_mask[mask_bas_closed == 1] = 2
    return final_mask

def remove_small_objects_by_area(mask, min_area=10):
    cleaned = np.zeros_like(mask)
    for cid in [1, 2]:
        num, labels, stats, _ = cv.connectedComponentsWithStats((mask == cid).astype(np.uint8), connectivity=8)
        for i in range(1, num):
            if stats[i, cv.CC_STAT_AREA] >= min_area: cleaned[labels == i] = cid
    return cleaned

def remove_poorly_connected_particles(mask, particle_max_area=300, min_contact_area=500):
    main_struct = np.isin(mask, [1, 2]); final = np.zeros_like(mask); kernel = np.ones((3,3), np.uint8)
    for cid in [1, 2]:
        num, labels, stats, _ = cv.connectedComponentsWithStats((mask == cid).astype(np.uint8), connectivity=8)
        for i in range(1, num):
            if stats[i, cv.CC_STAT_AREA] >= particle_max_area:
                final[labels == i] = cid
                continue
            dilated = cv.dilate((labels == i).astype(np.uint8), kernel, iterations=1)
            contact = np.sum(np.logical_and(dilated, np.logical_and(main_struct, labels != i)))
            if contact >= min_contact_area: final[labels == i] = cid
    return final

# --- 4. Bridge Correction ---
def correct_misclassified_bridges(mask, max_bridge_size_rh=1000, max_bridge_size_bas=2000, min_neighbor_size=1000):
    from scipy.ndimage import binary_dilation as scipy_dilation
    corrected_mask = mask.copy()
    processing_order = [2, 1]
    for current_class in processing_order:
        opposite_class = 2 if current_class == 1 else 1
        max_bridge_size = max_bridge_size_rh if current_class == 1 else max_bridge_size_bas
        labeled_current, num_current = ndimage.label(corrected_mask == current_class)
        opposite_binary = (corrected_mask == opposite_class).astype(np.uint8)
        labeled_opposite, num_opposite = ndimage.label(opposite_binary)
        opposite_sizes = ndimage.sum(opposite_binary, labeled_opposite, range(num_opposite + 1))
        for comp_id in range(1, num_current + 1):
            comp_mask = (labeled_current == comp_id)
            if np.sum(comp_mask) >= max_bridge_size: continue
            dilated = scipy_dilation(comp_mask, iterations=3)
            touching_labels = np.unique(labeled_opposite[dilated & (labeled_opposite > 0)])
            if len([l for l in touching_labels if opposite_sizes[l] >= min_neighbor_size]) >= 2:
                corrected_mask[comp_mask] = opposite_class
    return corrected_mask

# --- Final Save & Predict ---
def save_colored_mask(mask, filepath):
    h, w = mask.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {0: (0, 0, 0), 1: (128, 128, 0), 2: (0, 255, 255)}
    for class_id, color in colors.items():
        colored_mask[mask == class_id] = color
    cv.imwrite(str(filepath), colored_mask)

def show_and_save_predictions_v54(model, file_paths, num=1, save_path=None, patch_size=512, overlap=0.50):
    if save_path: Path(save_path).mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for idx, file_path in enumerate(file_paths[:num]):
            full_img = Image.open(file_path).convert('RGB')
            # 1. Prediction & Stitching
            stitched = advanced_stitch_predictions(model, full_img, patch_size, overlap, device)
            # 2. Cleanup & Processing
            mask = remove_small_objects_by_area(stitched, min_area=20)
            #mask = remove_poorly_connected_particles(mask, particle_max_area=300, min_contact_area=500)
            mask = reconnect_hyphae(mask, kernel_size=3)
            # 3. Bridge Correction
            final_mask = correct_misclassified_bridges(mask)
            if save_path:
                save_colored_mask(final_mask, Path(save_path) / f"{Path(file_path).stem}_v54.png")

# --- Run ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('HD_v3-2-7_1.pth', map_location=device, weights_only=False)
input_folder = r'C:\Files\OSU\Projects\Experiment\Exp1\Analysis\test'
output_path = r'C:\Files\OSU\Projects\Experiment\Exp1\Analysis\test\post326v61'
files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg'))]
show_and_save_predictions_v54(model, files, num=500, save_path=output_path, overlap=0.50)

print("Prediction completed!")