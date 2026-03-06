##HD early stopping test - PyTorch Version with U-Net++ (Improved Foreground IoU)

##Predict BAS and RH separately with filtered patches and overlap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import amp
import torchvision.transforms as transforms

import os
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Set paths
path = os.getcwd()
image_path = os.path.join(path, 'data/ori/')
mask_path = os.path.join(path, 'data/mask/')
image_list_orig = os.listdir(image_path)
mask_list_orig = os.listdir(mask_path)

image_list = [os.path.join(image_path, i) for i in image_list_orig]
mask_list = [os.path.join(mask_path, i) for i in mask_list_orig]

##Custom Dataset Class with Improved Patch Selection##
class PatchSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_size=512, overlap_ratio=0.25, 
                 min_foreground_ratio=0.05, max_background_ratio=0.95):
        self.patches = []
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.min_foreground_ratio = min_foreground_ratio
        self.max_background_ratio = max_background_ratio
        
        # Calculate step size for overlapping patches
        self.step_size = int(patch_size * (1 - overlap_ratio))
        
        self._extract_patches(image_paths, mask_paths)
        
    def _extract_patches(self, image_paths, mask_paths):
        """Extract patches with filtering and overlap"""
        total_patches = 0
        filtered_patches = 0
        foreground_patches = 0
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            # Load image
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
            mask = np.array(mask)

            # Process mask: reduce to single channel with 3 classes
            mask = np.max(mask, axis=-1)
            mask = np.where(mask > 100, 1, np.where(mask < 90, 0, 2))

            h, w = mask.shape
            
            # Extract overlapping patches
            for i in range(0, h - self.patch_size + 1, self.step_size):
                for j in range(0, w - self.patch_size + 1, self.step_size):
                    img_patch = image[i:i+self.patch_size, j:j+self.patch_size, :]
                    mask_patch = mask[i:i+self.patch_size, j:j+self.patch_size]
                    
                    total_patches += 1
                    
                    # Calculate class ratios
                    total_pixels = mask_patch.size
                    background_pixels = np.sum(mask_patch == 0)
                    foreground_pixels = total_pixels - background_pixels
                    
                    background_ratio = background_pixels / total_pixels
                    foreground_ratio = foreground_pixels / total_pixels
                    
                    # Filter patches with too little foreground content
                    if (foreground_ratio < self.min_foreground_ratio or 
                        background_ratio > self.max_background_ratio):
                        filtered_patches += 1
                        continue
                    
                    foreground_patches += 1
                    
                    # Convert to tensors
                    img_patch = torch.from_numpy(img_patch).permute(2, 0, 1)  # HWC → CHW
                    mask_patch = torch.from_numpy(mask_patch).long()

                    self.patches.append((img_patch, mask_patch))
        
        print(f"Total patches extracted: {total_patches}")
        print(f"Patches filtered out: {filtered_patches}")
        print(f"Patches with sufficient foreground: {foreground_patches}")
        print(f"Final dataset size: {len(self.patches)}")
        
        # Analyze class distribution in final dataset
        self._analyze_class_distribution()
        
    def _analyze_class_distribution(self):
        """Analyze class distribution in the filtered dataset"""
        if not self.patches:
            return
            
        total_pixels = 0
        class_counts = [0, 0, 0]  # Background, RH, BAS
        
        for _, mask_patch in self.patches:
            mask_np = mask_patch.numpy()
            total_pixels += mask_np.size
            
            for class_id in range(3):
                class_counts[class_id] += np.sum(mask_np == class_id)
        
        print(f"\nClass distribution in filtered dataset:")
        class_names = ['Background', 'RH', 'BAS']
        for i, (name, count) in enumerate(zip(class_names, class_counts)):
            ratio = count / total_pixels
            print(f"  {name}: {ratio:.4f} ({count:,} pixels)")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch, mask_patch = self.patches[idx]
        return img_patch, mask_patch


##Split Data##
train_ratio = 0.80
validation_ratio = 0.15
test_ratio = 0.05

x_train, x_test, y_train, y_test = train_test_split(image_list, mask_list, test_size=1 - train_ratio, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)

print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

# Create datasets with improved patch extraction
print("\n=== Creating Training Dataset ===")
train_dataset = PatchSegmentationDataset(x_train, y_train, patch_size=512, 
                                        overlap_ratio=0.25, min_foreground_ratio=0.05)

print("\n=== Creating Validation Dataset ===")
val_dataset = PatchSegmentationDataset(x_val, y_val, patch_size=512, 
                                      overlap_ratio=0.25, min_foreground_ratio=0.05)

print("\n=== Creating Test Dataset ===")
test_dataset = PatchSegmentationDataset(x_test, y_test, patch_size=512, 
                                       overlap_ratio=0.25, min_foreground_ratio=0.05)


##IoU Metric Function##
def calculate_iou(pred_mask, true_mask, num_classes=3, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for each class
    """
    # Use .reshape() to handle potential non-contiguous tensors
    pred_mask = pred_mask.reshape(-1)
    true_mask = true_mask.reshape(-1)
    
    iou_per_class = []
    
    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id).float()
        true_class = (true_mask == class_id).float()
        
        intersection = (pred_class * true_class).sum()
        union = pred_class.sum() + true_class.sum() - intersection
        
        # Handle edge case where union is 0
        # Ensure iou is a tensor for consistency
        if union == 0:
            iou = torch.tensor(1.0, device=pred_mask.device) if intersection == 0 else torch.tensor(0.0, device=pred_mask.device)
        else:
            iou = (intersection + smooth) / (union + smooth)
        
        # Convert the tensor to a standard Python float
        iou_per_class.append(iou.item())
    
    mean_iou = np.mean(iou_per_class)
    return iou_per_class, mean_iou

##IoU Monitor Class##
class IoUMonitor:
    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.total_iou_per_class = [0.0] * self.num_classes
        self.total_mean_iou = 0.0
        self.count = 0
    
    def update(self, pred_mask, true_mask):
        iou_per_class, mean_iou = calculate_iou(pred_mask, true_mask, self.num_classes)
        
        for i in range(self.num_classes):
            self.total_iou_per_class[i] += iou_per_class[i]
        
        self.total_mean_iou += mean_iou
        self.count += 1
    
    def get_average_iou(self):
        if self.count == 0:
            return [0.0] * self.num_classes, 0.0
        
        avg_iou_per_class = [iou / self.count for iou in self.total_iou_per_class]
        avg_mean_iou = self.total_mean_iou / self.count
        
        return avg_iou_per_class, avg_mean_iou
    
    def print_iou_summary(self, prefix=""):
        avg_iou_per_class, avg_mean_iou = self.get_average_iou()
        
        print(f"{prefix}IoU Summary:")
        for i, (class_name, iou) in enumerate(zip(self.class_names, avg_iou_per_class)):
            print(f"  {class_name}: {iou:.4f}")
        print(f"  Mean IoU: {avg_mean_iou:.4f}")


##Conv Block with Improved Regularization##
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1, max_pooling=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Added batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Added batch normalization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        self.max_pooling = max_pooling
        self.pool = nn.MaxPool2d(2, 2) if max_pooling else None
        
        # He normal initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        conv = self.relu(self.bn1(self.conv1(x)))
        conv = self.relu(self.bn2(self.conv2(conv)))
        
        if self.dropout:
            conv = self.dropout(conv)
        
        if self.max_pooling:
            next_layer = self.pool(conv)
        else:
            next_layer = conv
            
        return next_layer, conv  # next_layer, skip_connection

##Dense Block for U-Net++ with BatchNorm##
class DenseBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels, dropout_prob=0.1):
        super(DenseBlock, self).__init__()
        total_in_channels = sum(in_channels_list)
        self.conv1 = nn.Conv2d(total_in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Added batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Added batch normalization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        
        # He normal initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, inputs):
        # Concatenate all inputs
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[0]
        
        conv = self.relu(self.bn1(self.conv1(x)))
        conv = self.relu(self.bn2(self.conv2(conv)))
        
        if self.dropout:
            conv = self.dropout(conv)
        
        return conv

##U-Net++ Model with Regularization##
class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels=3, n_filters=32, n_classes=3, deep_supervision=False, dropout_prob=0.1):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        
        filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]
        
        # Encoder (contracting path) with increased dropout for deeper layers
        self.cblock0_0 = ConvBlock(input_channels, filters[0], dropout_prob=dropout_prob*0.5)
        self.cblock1_0 = ConvBlock(filters[0], filters[1], dropout_prob=dropout_prob)
        self.cblock2_0 = ConvBlock(filters[1], filters[2], dropout_prob=dropout_prob*1.5)
        self.cblock3_0 = ConvBlock(filters[2], filters[3], dropout_prob=dropout_prob*2)
        self.cblock4_0 = ConvBlock(filters[3], filters[4], dropout_prob=dropout_prob*2, max_pooling=False)
        
        # Upsampling layers
        self.up4_3 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=1, output_padding=1)
        self.up3_2 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=1, output_padding=1)
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1, output_padding=1)
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[0], 3, stride=2, padding=1, output_padding=1)
        
        # Dense blocks for nested skip connections with dropout
        # Level 0
        self.dense0_1 = DenseBlock([filters[0], filters[0]], filters[0], dropout_prob=dropout_prob*0.5)
        self.dense0_2 = DenseBlock([filters[0], filters[0], filters[0]], filters[0], dropout_prob=dropout_prob*0.5)
        self.dense0_3 = DenseBlock([filters[0], filters[0], filters[0], filters[0]], filters[0], dropout_prob=dropout_prob*0.5)
        self.dense0_4 = DenseBlock([filters[0], filters[0], filters[0], filters[0], filters[0]], filters[0], dropout_prob=dropout_prob*0.5)
        
        # Level 1
        self.dense1_1 = DenseBlock([filters[1], filters[1]], filters[1], dropout_prob=dropout_prob)
        self.dense1_2 = DenseBlock([filters[1], filters[1], filters[1]], filters[1], dropout_prob=dropout_prob)
        self.dense1_3 = DenseBlock([filters[1], filters[1], filters[1], filters[1]], filters[1], dropout_prob=dropout_prob)
        
        # Level 2
        self.dense2_1 = DenseBlock([filters[2], filters[2]], filters[2], dropout_prob=dropout_prob*1.5)
        self.dense2_2 = DenseBlock([filters[2], filters[2], filters[2]], filters[2], dropout_prob=dropout_prob*1.5)
        
        # Level 3
        self.dense3_1 = DenseBlock([filters[3], filters[3]], filters[3], dropout_prob=dropout_prob*2)
        
        # Output layers
        if self.deep_supervision:
            # Multiple output heads for deep supervision
            self.out1 = nn.Conv2d(filters[0], n_classes, 1)
            self.out2 = nn.Conv2d(filters[0], n_classes, 1)
            self.out3 = nn.Conv2d(filters[0], n_classes, 1)
            self.out4 = nn.Conv2d(filters[0], n_classes, 1)
            
            # Initialize output layers
            nn.init.kaiming_normal_(self.out1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.out2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.out3.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.out4.weight, mode='fan_out', nonlinearity='relu')
        else:
            # Single output (final layer)
            self.conv_final = nn.Conv2d(filters[0], filters[0], 3, padding=1)
            self.bn_final = nn.BatchNorm2d(filters[0])  # Added batch normalization
            self.out_final = nn.Conv2d(filters[0], n_classes, 1)
            self.relu = nn.ReLU(inplace=True)
            
            # Initialize final layers
            nn.init.kaiming_normal_(self.conv_final.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.out_final.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize upsampling layers
        nn.init.kaiming_normal_(self.up4_3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up3_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up2_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up1_0.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # Encoder path
        x0_0_pool, x0_0 = self.cblock0_0(x)
        x1_0_pool, x1_0 = self.cblock1_0(x0_0_pool)
        x2_0_pool, x2_0 = self.cblock2_0(x1_0_pool)
        x3_0_pool, x3_0 = self.cblock3_0(x2_0_pool)
        x4_0_pool, x4_0 = self.cblock4_0(x3_0_pool)
        
        # Nested skip connections
        # Column 1
        x3_1 = self.dense3_1([x3_0, self.up4_3(x4_0)])
        x2_1 = self.dense2_1([x2_0, self.up3_2(x3_0)])
        x1_1 = self.dense1_1([x1_0, self.up2_1(x2_0)])
        x0_1 = self.dense0_1([x0_0, self.up1_0(x1_0)])
        
        # Column 2
        x2_2 = self.dense2_2([x2_0, x2_1, self.up3_2(x3_1)])
        x1_2 = self.dense1_2([x1_0, x1_1, self.up2_1(x2_1)])
        x0_2 = self.dense0_2([x0_0, x0_1, self.up1_0(x1_1)])
        
        # Column 3
        x1_3 = self.dense1_3([x1_0, x1_1, x1_2, self.up2_1(x2_2)])
        x0_3 = self.dense0_3([x0_0, x0_1, x0_2, self.up1_0(x1_2)])
        
        # Column 4
        x0_4 = self.dense0_4([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)])
        
        if self.deep_supervision:
            # Return multiple outputs for deep supervision
            out1 = self.out1(x0_1)
            out2 = self.out2(x0_2)
            out3 = self.out3(x0_3)
            out4 = self.out4(x0_4)
            return [out1, out2, out3, out4]
        else:
            # Return single output
            conv_final = self.relu(self.bn_final(self.conv_final(x0_4)))
            output = self.out_final(conv_final)
            return output

##Enhanced Loss Function with Adaptive Weights##
class AdaptiveHyphalFocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, dice_weight=0.7, connectivity_weight=0.1, 
                 class_weights=None):
        super(AdaptiveHyphalFocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.connectivity_weight = connectivity_weight
        
        # Default class weights favor foreground classes
        if class_weights is None:
            self.class_weights = torch.tensor([1.0, 8.0, 20.0])  # Background, RH, BAS
        else:
            self.class_weights = torch.tensor(class_weights)
        
    def focal_loss(self, inputs, targets):
        """Focal loss with class weights to handle class imbalance"""
        self.class_weights = self.class_weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets, smooth=1):
        """Weighted Dice loss for segmentation"""
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        class_weights_norm = self.class_weights / self.class_weights.sum()
        
        for i in range(inputs.shape[1]):
            input_flat = inputs[:, i].contiguous().view(-1)
            target_flat = targets_one_hot[:, i].contiguous().view(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
            
            # Weight by class importance
            weighted_dice = dice * class_weights_norm[i].to(dice.device)
            dice_scores.append(weighted_dice)
        
        return 1 - torch.stack(dice_scores).sum()
    
    def connectivity_loss(self, inputs, targets):
        """Preserve connectivity of hyphal structures"""
        pred = F.softmax(inputs, dim=1)[:, 1]  # Get hyphal class probability
        
        # Simple gradient-based connectivity loss
        grad_x = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        grad_y = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        
        # Penalize strong gradients (disconnections)
        connectivity_loss = grad_x.mean() + grad_y.mean()
        return connectivity_loss
    
    def forward(self, inputs, targets):
        # Handle deep supervision (multiple outputs)
        if isinstance(inputs, list):
            total_loss = 0
            for i, output in enumerate(inputs):
                focal = self.focal_loss(output, targets)
                dice = self.dice_loss(output, targets)
                connectivity = self.connectivity_loss(output, targets)
                
                # Weight the losses (later outputs get higher weight)
                weight = (i + 1) / len(inputs)
                loss = weight * ((1 - self.dice_weight) * focal + 
                               self.dice_weight * dice + 
                               self.connectivity_weight * connectivity)
                total_loss += loss
            return total_loss
        else:
            # Handle single output
            focal = self.focal_loss(inputs, targets)
            dice = self.dice_loss(inputs, targets)
            connectivity = self.connectivity_loss(inputs, targets)
            
            total_loss = (1 - self.dice_weight) * focal + \
                         self.dice_weight * dice + \
                         self.connectivity_weight * connectivity
            
            return total_loss

##Initialize Model - U-Net++ with Regularization##
model = UNetPlusPlus(input_channels=3, n_filters=32, n_classes=3, deep_supervision=False, dropout_prob=0.15).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Define loss and optimizer with enhanced class weighting
criterion = AdaptiveHyphalFocalDiceLoss(dice_weight=0.8, class_weights=[1.0, 8.0, 20.0])
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, min_lr=1e-7)

##Data Loaders##
BATCH_SIZE = 16 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

##Early Stopping Class## 
class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best weights")
            return True
        return False

##Training Function with IoU Monitoring##
def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize IoU monitor
    iou_monitor = IoUMonitor(num_classes=3, class_names=['Background', 'RH', 'BAS'])
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with amp.autocast("cuda"):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Handle both single output and deep supervision
        if isinstance(output, list):
            pred = output[-1].argmax(dim=1)  # Use final output for metrics
        else:
            pred = output.argmax(dim=1)
            
        total += target.numel()
        correct += pred.eq(target).sum().item()
        
        # Update IoU monitor
        iou_monitor.update(pred, target)
        
        if batch_idx % 10 == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)} '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    # Get average IoU for the epoch
    avg_iou_per_class, avg_mean_iou = iou_monitor.get_average_iou()
    
    return running_loss / len(train_loader), 100. * correct / total, avg_iou_per_class, avg_mean_iou

##Validation Function with IoU Monitoring##
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize IoU monitor
    iou_monitor = IoUMonitor(num_classes=3, class_names=['Background', 'RH', 'BAS'])
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with amp.autocast("cuda"):
                output = model(data)
                loss = criterion(output, target)
            
            running_loss += loss.item()
            
            # Handle both single output and deep supervision
            if isinstance(output, list):
                pred = output[-1].argmax(dim=1)  # Use final output for metrics
            else:
                pred = output.argmax(dim=1)
                
            total += target.numel()
            correct += pred.eq(target).sum().item()
            
            # Update IoU monitor
            iou_monitor.update(pred, target)
    
    # Get average IoU for the epoch
    avg_iou_per_class, avg_mean_iou = iou_monitor.get_average_iou()
    
    return running_loss / len(val_loader), 100. * correct / total, avg_iou_per_class, avg_mean_iou

##Training Loop with IoU Monitoring##
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100):
    early_stopping = EarlyStopping(patience=8, min_delta=0.001, verbose=True)
    scaler = amp.GradScaler("cuda")
    
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_iou': [], 'val_iou': [], 'train_mean_iou': [], 'val_mean_iou': []
    }
    best_val_loss = float('inf')
    best_mean_iou = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc, train_iou_per_class, train_mean_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        
        # Validation
        val_loss, val_acc, val_iou_per_class, val_mean_iou = validate_epoch(
            model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_iou'].append(train_iou_per_class)
        history['val_iou'].append(val_iou_per_class)
        history['train_mean_iou'].append(train_mean_iou)
        history['val_mean_iou'].append(val_mean_iou)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'Train Mean IoU: {train_mean_iou:.4f}, Val Mean IoU: {val_mean_iou:.4f}')
        print(f'Val IoU per class - Background: {val_iou_per_class[0]:.4f}, '
              f'RH: {val_iou_per_class[1]:.4f}, BAS: {val_iou_per_class[2]:.4f}')
        
        # Save best model based on mean IoU (better metric for segmentation)
        if val_mean_iou > best_mean_iou:
            best_mean_iou = val_mean_iou
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_plus_plus_model_filtered.pth')
            print(f'New best model saved with validation mean IoU: {val_mean_iou:.4f}')
        
        # Early stopping based on validation loss
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return history

##Display Function##
def display_predictions(model, dataset, device, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Initialize IoU monitor for display
    iou_monitor = IoUMonitor(num_classes=3, class_names=['Background', 'RH', 'BAS'])
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            image_batch = image.unsqueeze(0).to(device)
            
            output = model(image_batch)
            
            # Handle both single output and deep supervision
            if isinstance(output, list):
                pred_mask = output[-1].argmax(dim=1).squeeze().cpu().numpy()
            else:
                pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
            
            # Calculate IoU for this sample
            pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0)
            mask_tensor = mask.unsqueeze(0)
            iou_monitor.update(pred_tensor, mask_tensor)
            
            # Convert tensors to numpy for display
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()
            
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='tab10')
            axes[i, 1].set_title('True Mask')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='tab10')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print IoU summary for displayed samples
    iou_monitor.print_iou_summary("Display Samples ")

##Plot Training History with IoU##
def plot_training_history(history):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    
    # Plot accuracy
    ax1.plot(history['train_acc'], label='Training Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['train_loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Mean IoU
    ax3.plot(history['train_mean_iou'], label='Training Mean IoU')
    ax3.plot(history['val_mean_iou'], label='Validation Mean IoU')
    ax3.set_title('Mean IoU')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean IoU')
    ax3.legend()
    ax3.grid(True)
    
    # Plot IoU per class (validation only for clarity)
    val_iou_bg = [iou[0] for iou in history['val_iou']]
    val_iou_rh = [iou[1] for iou in history['val_iou']]
    val_iou_bas = [iou[2] for iou in history['val_iou']]
    
    ax4.plot(val_iou_bg, label='Background IoU')
    ax4.plot(val_iou_rh, label='RH IoU')
    ax4.plot(val_iou_bas, label='BAS IoU')
    ax4.set_title('Validation IoU per Class')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IoU')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print training summary
    print(f"Training completed after {len(history['train_loss'])} epochs")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}%")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation mean IoU: {max(history['val_mean_iou']):.4f}")
    
    # Print final IoU per class
    final_val_iou = history['val_iou'][-1]
    print(f"Final validation IoU per class:")
    print(f"  Background: {final_val_iou[0]:.4f}")
    print(f"  RH: {final_val_iou[1]:.4f}")
    print(f"  BAS: {final_val_iou[2]:.4f}")

##Test Function with IoU Evaluation##
def test_model(model, test_loader, device):
    """Evaluate model on test set with detailed IoU metrics"""
    model.eval()
    
    # Initialize IoU monitor
    iou_monitor = IoUMonitor(num_classes=3, class_names=['Background', 'RH', 'BAS'])
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with amp.autocast("cuda"):
                output = model(data)
            
            # Handle both single output and deep supervision
            if isinstance(output, list):
                pred = output[-1].argmax(dim=1)  # Use final output for metrics
            else:
                pred = output.argmax(dim=1)
                
            total += target.numel()
            correct += pred.eq(target).sum().item()
            
            # Update IoU monitor
            iou_monitor.update(pred, target)
    
    test_acc = 100. * correct / total
    avg_iou_per_class, avg_mean_iou = iou_monitor.get_average_iou()
    
    print(f"\n=== Test Results ===")
    print(f"Test Accuracy: {test_acc:.4f}%")
    print(f"Test Mean IoU: {avg_mean_iou:.4f}")
    print(f"Test IoU per class:")
    print(f"  Background: {avg_iou_per_class[0]:.4f}")
    print(f"  RH: {avg_iou_per_class[1]:.4f}")
    print(f"  BAS: {avg_iou_per_class[2]:.4f}")
    
    return test_acc, avg_mean_iou, avg_iou_per_class

##Additional Dataset Analysis Function##
def analyze_dataset_balance():
    """Analyze and compare class distribution across datasets"""
    datasets = [
        ("Training", train_dataset),
        ("Validation", val_dataset), 
        ("Test", test_dataset)
    ]
    
    print("\n=== Dataset Balance Analysis ===")
    for name, dataset in datasets:
        if len(dataset) == 0:
            continue
            
        total_pixels = 0
        class_counts = [0, 0, 0]
        
        for _, mask_patch in dataset.patches:
            mask_np = mask_patch.numpy()
            total_pixels += mask_np.size
            
            for class_id in range(3):
                class_counts[class_id] += np.sum(mask_np == class_id)
        
        print(f"\n{name} Dataset ({len(dataset)} patches):")
        class_names = ['Background', 'RH', 'BAS']
        for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
            ratio = count / total_pixels
            print(f"  {class_name}: {ratio:.4f} ({count:,} pixels)")

##Main Training##
if __name__ == "__main__":
    # Analyze dataset balance
    analyze_dataset_balance()
    
    # Display sample
    print("\nSample from filtered dataset:")
    display_predictions(model, val_dataset, device, num_samples=1)
    
    # Train model
    print("Starting training with filtered patches and enhanced class weighting...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100)
    
    # Plot results
    plot_training_history(history)
    
    # Test the model
    test_model(model, test_loader, device)
    
    # Save final model
    torch.save(model, 'HD_v3-2-6_filtered_b16.pth')
    print("Final filtered U-Net++ model saved!")
    
    # Display final predictions
    print("Final predictions on filtered dataset:")
    display_predictions(model, val_dataset, device, num_samples=6)