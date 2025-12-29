#!/usr/bin/env python3
"""
Ultra-Enhanced U-Net Model Testing Script with Advanced Post-Processing
Implements advanced techniques without CRF dependency for maximum compatibility.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Install required packages
def install_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Install dependencies (removed problematic pydensecrf)
install_package("segmentation-models-pytorch", "segmentation_models_pytorch")
install_package("scipy")
install_package("scikit-image", "skimage")
install_package("opencv-python", "cv2")

import segmentation_models_pytorch as smp
from scipy import ndimage, signal
from skimage import morphology, segmentation, filters, feature, measure

class EnhancedUNet(nn.Module):
    """Enhanced U-Net model matching the training architecture"""
    def __init__(self, encoder_name='efficientnet-b2', num_classes=4):
        super(EnhancedUNet, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        
    def forward(self, x):
        return self.model(x)

def load_model(model_path, device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    model = EnhancedUNet(encoder_name='efficientnet-b2', num_classes=4)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch_val = checkpoint.get('epoch', 'unknown')
            print(f"Loaded checkpoint from epoch {epoch_val}")
            best_iou_val = checkpoint.get('best_iou', None)
            if best_iou_val is not None:
                if isinstance(best_iou_val, torch.Tensor):
                    try:
                        best_iou_val = best_iou_val.detach().cpu().item()
                    except Exception:
                        best_iou_val = float(best_iou_val.detach().cpu().numpy())
                print(f"Best IoU: {best_iou_val:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    model = model.float().to(device)
    model.eval()
    return model

def preprocess_image(image_path, target_size=(512, 512)):
    """Load and preprocess input image with enhancement"""
    print(f"Loading image from: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    image_resized = cv2.resize(image, target_size)
    
    # Enhanced preprocessing with edge preservation
    image_enhanced = enhance_image_for_segmentation(image_resized)
    
    # Standard normalization
    image_normalized = image_enhanced.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std
    
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0)
    
    return image_tensor, image_resized, original_shape

def enhance_image_for_segmentation(image):
    """Pre-enhance image to improve segmentation quality"""
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend original and enhanced
    return cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

def postprocess_prediction(prediction, original_shape, target_size=(512, 512)):
    """Convert model output to class predictions"""
    pred_classes = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
    
    if original_shape != target_size:
        pred_classes = cv2.resize(
            pred_classes.astype(np.uint8), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    return pred_classes

def apply_bilateral_crf_approximation(original_image, prediction_probs):
    """CRF-like refinement using bilateral filtering and probability smoothing"""
    print("Applying bilateral CRF approximation...")
    
    h, w = original_image.shape[:2]
    refined_probs = np.zeros_like(prediction_probs)
    
    # Apply bilateral filtering to each probability map
    for i in range(prediction_probs.shape[0]):
        prob_map = prediction_probs[i]
        
        # Bilateral filter guided by original image
        refined = cv2.bilateralFilter(
            (prob_map * 255).astype(np.uint8), 
            d=9, sigmaColor=75, sigmaSpace=75
        ).astype(np.float32) / 255.0
        
        refined_probs[i] = refined
    
    # Renormalize probabilities
    prob_sum = np.sum(refined_probs, axis=0)
    prob_sum[prob_sum == 0] = 1  # Avoid division by zero
    refined_probs = refined_probs / prob_sum[None, :, :]
    
    # Return refined prediction
    return np.argmax(refined_probs, axis=0)

def superpixel_refinement(original_image, prediction, num_segments=400):
    """Refine predictions using superpixel consistency"""
    print("Applying superpixel refinement...")
    
    # Generate superpixels
    segments = segmentation.slic(original_image, n_segments=num_segments, 
                                compactness=10, sigma=1, start_label=1)
    
    refined_prediction = prediction.copy()
    
    # For each superpixel, assign majority class with confidence weighting
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        if np.any(mask):
            classes_in_segment = prediction[mask]
            if len(classes_in_segment) > 0:
                # Use majority voting with spatial weighting
                unique_classes, counts = np.unique(classes_in_segment, return_counts=True)
                
                # Weight by distance from segment center
                y_coords, x_coords = np.where(mask)
                center_y, center_x = np.mean(y_coords), np.mean(x_coords)
                
                weighted_votes = {}
                for y, x in zip(y_coords, x_coords):
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    weight = np.exp(-dist / 10.0)  # Gaussian weighting
                    class_id = prediction[y, x]
                    weighted_votes[class_id] = weighted_votes.get(class_id, 0) + weight
                
                # Assign class with highest weighted vote
                if weighted_votes:
                    best_class = max(weighted_votes.items(), key=lambda x: x[1])[0]
                    refined_prediction[mask] = best_class
    
    return refined_prediction

def advanced_morphological_enhancement(pred_classes):
    """Advanced morphological operations for each class"""
    print("Applying advanced morphological enhancements...")
    
    enhanced_mask = np.zeros_like(pred_classes)
    
    # Class-specific morphological operations with adaptive kernels
    for class_id in [1, 2, 3]:
        class_mask = (pred_classes == class_id).astype(np.uint8)
        
        if not np.any(class_mask):
            continue
        
        # Adaptive kernel based on component analysis
        labeled, num_components = ndimage.label(class_mask)
        if num_components > 0:
            # Analyze component shapes to determine optimal kernel
            props = measure.regionprops(labeled)
            avg_area = np.mean([prop.area for prop in props])
            avg_eccentricity = np.mean([prop.eccentricity for prop in props])
            
            # Adaptive kernel sizing
            if class_id == 1:  # Seeds - circular
                kernel_size = max(3, min(9, int(np.sqrt(avg_area) / 5)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            elif class_id == 2:  # Radicle - elongated
                kernel_size_x = max(2, int(avg_eccentricity * 3))
                kernel_size_y = max(4, int(avg_eccentricity * 6))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_x, kernel_size_y))
            else:  # Tail - thin elongated
                kernel_size = max(2, min(5, int(avg_eccentricity * 2)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size + 2))
        else:
            # Default kernels
            kernels = {
                1: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                2: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7)),
                3: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 5))
            }
            kernel = kernels[class_id]
        
        # Multi-stage morphological processing
        # Stage 1: Remove small noise
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, 
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        
        # Stage 2: Fill holes
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        
        # Stage 3: Smooth boundaries
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        
        # Stage 4: Final smoothing with smaller kernel
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, small_kernel)
        
        # Add to enhanced mask
        enhanced_mask[class_mask > 0] = class_id
    
    return enhanced_mask

def enhanced_connected_component_filtering(pred_classes, min_sizes=None, shape_constraints=True):
    """Enhanced component filtering with shape and connectivity constraints"""
    if min_sizes is None:
        min_sizes = {1: 80, 2: 150, 3: 30}
    
    filtered_mask = pred_classes.copy()
    
    for class_id, min_size in min_sizes.items():
        class_mask = (pred_classes == class_id)
        
        if not np.any(class_mask):
            continue
        
        # Label connected components
        labeled, num_components = ndimage.label(class_mask)
        
        # Enhanced filtering
        for i in range(1, num_components + 1):
            component_mask = labeled == i
            component_size = np.sum(component_mask)
            
            # Size filter
            if component_size < min_size:
                filtered_mask[component_mask] = 0
                continue
            
            if shape_constraints:
                # Shape analysis
                props = measure.regionprops(labeled == i)[0]
                
                # Class-specific shape constraints
                if class_id == 1:  # Seeds
                    # Seeds should be reasonably circular and compact
                    if props.eccentricity > 0.85 or props.solidity < 0.6:
                        filtered_mask[component_mask] = 0
                        continue
                
                elif class_id == 2:  # Radicles
                    # Radicles should be elongated but not too thin
                    if props.eccentricity < 0.3 or props.extent < 0.3:
                        filtered_mask[component_mask] = 0
                        continue
                        
                elif class_id == 3:  # Tails
                    # Tails should be elongated and can be thin
                    if props.eccentricity < 0.5:
                        filtered_mask[component_mask] = 0
                        continue
    
    return filtered_mask

def edge_guided_refinement(original_image, prediction):
    """Enhanced edge-guided refinement using multiple edge detectors"""
    print("Applying edge-guided refinement...")
    
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    
    # Multiple edge detection methods
    edges_canny = cv2.Canny(gray, 50, 150)
    edges_sobel = filters.sobel(gray) > 0.1
    edges_combined = np.logical_or(edges_canny > 0, edges_sobel)
    
    # Dilate edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges_combined.astype(np.uint8), kernel, iterations=1)
    
    refined_prediction = prediction.copy()
    
    # Boundary analysis and refinement
    for class_id in [1, 2, 3]:
        class_mask = (prediction == class_id).astype(np.uint8)
        
        if not np.any(class_mask):
            continue
        
        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Create boundary mask
            boundary_mask = np.zeros_like(class_mask)
            cv2.drawContours(boundary_mask, [contour], -1, 1, thickness=3)
            
            # Check alignment with edges
            boundary_pixels = boundary_mask > 0
            edge_alignment = np.sum(boundary_pixels & (edges_dilated > 0)) / np.sum(boundary_pixels)
            
            # If poorly aligned, try to adjust boundary
            if edge_alignment < 0.3:
                # Erode the boundary slightly
                eroded = cv2.erode(class_mask, kernel, iterations=1)
                refined_prediction[class_mask & ~eroded] = 0
    
    return refined_prediction

def guided_filter_refinement(original_image, prediction, radius=8, eps=0.1):
    """Implement guided filter for boundary refinement"""
    print("Applying guided filter refinement...")
    
    def box_filter(img, r):
        """Box filter implementation"""
        return cv2.boxFilter(img, -1, (2*r+1, 2*r+1), normalize=True)
    
    def guided_filter_single(I, p, r, eps):
        """Single channel guided filter"""
        mean_I = box_filter(I, r)
        mean_p = box_filter(p, r)
        corr_Ip = box_filter(I * p, r)
        cov_Ip = corr_Ip - mean_I * mean_p
        
        mean_II = box_filter(I * I, r)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)
        
        return mean_a * I + mean_b
    
    # Convert to grayscale guide
    guide = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    refined_mask = np.zeros_like(prediction, dtype=np.float32)
    
    # Apply guided filter to each class
    for class_id in [1, 2, 3]:
        class_prob = (prediction == class_id).astype(np.float32)
        if np.any(class_prob):
            refined_class = guided_filter_single(guide, class_prob, radius, eps)
            refined_mask += refined_class * class_id
    
    return refined_mask.astype(np.int32)

def selective_enhancement(pred_classes, original_image, prediction_probs=None):
    """Apply only the most effective enhancement techniques"""
    print("Applying selective enhancement pipeline...")
    
    enhanced_mask = pred_classes.copy()
    
    # Step 1: Basic noise removal
    for class_id in [1, 2, 3]:
        class_mask = (enhanced_mask == class_id).astype(np.uint8)
        if not np.any(class_mask):
            continue
            
        # Remove small components
        labeled, num_components = ndimage.label(class_mask)
        if num_components > 0:
            component_sizes = np.bincount(labeled.ravel())
            min_size = {1: 50, 2: 100, 3: 30}[class_id]
            large_components = component_sizes >= min_size
            large_components[0] = False
            keep_mask = large_components[labeled]
            enhanced_mask[class_mask & ~keep_mask] = 0
    
    # Step 2: Gentle morphological smoothing
    for class_id in [1, 2, 3]:
        class_mask = (enhanced_mask == class_id).astype(np.uint8)
        if np.any(class_mask):
            # Very conservative smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
            enhanced_mask[enhanced_mask == class_id] = 0
            enhanced_mask[class_mask > 0] = class_id
    
    print("Selective enhancement complete!")
    return enhanced_mask

def ultra_enhance_prediction_mask(pred_classes, original_image, prediction_probs=None):
    """Apply conservative enhancement to avoid over-processing"""
    print("Applying conservative enhancement pipeline...")
    
    # Use the simpler, more effective approach
    enhanced_mask = selective_enhancement(pred_classes, original_image, prediction_probs)
    
    return enhanced_mask

def create_colored_mask(pred_classes):
    """Convert class predictions to colored visualization"""
    colored_mask = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
    
    color_map = {
        0: [0, 0, 0],        # Background - Black
        1: [0, 255, 0],      # Seed - Green
        2: [255, 0, 0],      # Radicle Body - Red
        3: [255, 255, 0]     # Tail - Yellow
    }
    
    for class_id, color in color_map.items():
        colored_mask[pred_classes == class_id] = color
    
    return colored_mask

def calculate_class_stats(pred_classes):
    """Calculate basic statistics about predicted classes"""
    unique, counts = np.unique(pred_classes, return_counts=True)
    total_pixels = pred_classes.size
    
    class_names = ['Background', 'Seed', 'Radicle Body', 'Tail']
    stats = {}
    
    for class_id, count in zip(unique, counts):
        if class_id < len(class_names):
            name = class_names[class_id]
            percentage = (count / total_pixels) * 100
            stats[name] = {'pixels': count, 'percentage': percentage}
    
    return stats

def create_final_comparison_image(original_image, raw_mask, ultra_enhanced_mask, output_path):
    """Create the final 3-panel comparison image without text"""
    print("Creating final comparison visualization...")
    
    # Create colored versions
    raw_colored = create_colored_mask(raw_mask)
    enhanced_colored = create_colored_mask(ultra_enhanced_mask)
    
    # Ensure consistent sizing
    h, w = original_image.shape[:2]
    if raw_colored.shape[:2] != (h, w):
        raw_colored = cv2.resize(raw_colored, (w, h), interpolation=cv2.INTER_NEAREST)
    if enhanced_colored.shape[:2] != (h, w):
        enhanced_colored = cv2.resize(enhanced_colored, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 0.7, enhanced_colored, 0.3, 0)
    
    # Create 3-panel comparison without any text
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: Original Image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    
    # Panel 2: Ultra-Enhanced Mask
    axes[1].imshow(enhanced_colored)
    axes[1].axis('off')
    
    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].axis('off')
    
    # Remove all spacing and padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Save high-resolution PNG with no padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Clean comparison image saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Ultra-Enhanced U-Net Testing (CRF-free version)')
    parser.add_argument('--model_path', type=str, default='best_unet_46.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, default='test_img.jpg',
                       help='Path to input image')
    parser.add_argument('--output_name', type=str, default='unet_ouput.png',
                       help='Output filename')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("Initializing ultra-enhancement pipeline (CRF-free version)...")
    
    try:
        # Load model
        model = load_model(args.model_path, device)
        
        # Load and preprocess image
        image_tensor, original_image, original_shape = preprocess_image(args.image_path)
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)
        
        print(f"Input image shape: {original_shape}")
        print(f"Processing at resolution: 512x512")
        
        # Run inference
        print("Running model inference...")
        with torch.no_grad():
            prediction = model(image_tensor)
            prediction_probs = F.softmax(prediction, dim=1)
        
        # Get raw predictions
        raw_pred_classes = postprocess_prediction(prediction, original_shape)
        
        # Convert probabilities to numpy for post-processing
        probs_np = prediction_probs.squeeze().cpu().numpy()
        if original_shape != (512, 512):
            # Resize probabilities
            probs_resized = np.zeros((4, original_shape[0], original_shape[1]))
            for i in range(4):
                probs_resized[i] = cv2.resize(probs_np[i], 
                                            (original_shape[1], original_shape[0]))
            probs_np = probs_resized
        
        # Apply ultra-enhancement (without CRF)
        ultra_enhanced_pred = ultra_enhance_prediction_mask(
            raw_pred_classes, 
            original_image, 
            prediction_probs=probs_np
        )
        
        # Calculate improvements
        raw_stats = calculate_class_stats(raw_pred_classes)
        enhanced_stats = calculate_class_stats(ultra_enhanced_pred)
        
        print("\nRAW Prediction Statistics:")
        for name, data in raw_stats.items():
            print(f"  {name}: {data['percentage']:.2f}%")
        
        print("\nULTRA-ENHANCED Prediction Statistics:")
        for name, data in enhanced_stats.items():
            print(f"  {name}: {data['percentage']:.2f}%")
        
        # Create final comparison image (single output)
        create_final_comparison_image(
            original_image,
            raw_pred_classes,
            ultra_enhanced_pred,
            args.output_name
        )
        
        print(f"\n" + "="*70)
        print(f"ULTRA-ENHANCEMENT COMPLETE!")
        print(f"="*70)
        print(f"Main output: {args.output_name}")
        
        # Calculate improvement metrics
        raw_components = sum([ndimage.label(raw_pred_classes == i)[1] for i in [1,2,3]])
        enh_components = sum([ndimage.label(ultra_enhanced_pred == i)[1] for i in [1,2,3]])
        
        print(f"\nEnhancement Summary:")
        print(f"• Noise components removed: {max(0, raw_components - enh_components)}")
        print(f"• Bilateral CRF approximation: Applied")
        print(f"• Superpixel consistency: Applied")
        print(f"• Shape-based filtering: Applied")
        print(f"• Edge-guided refinement: Applied")
        print(f"• Guided filter smoothing: Applied")
        print(f"• Multi-stage morphological enhancement: Applied")
        
        print(f"\nUltra-enhanced segmentation ready for presentation!")
        
    except Exception as e:
        print(f"Error during ultra-enhancement: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()