#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# DarkPose Heatmap Visualization Demo
# Generates pose estimation heatmaps and verifies alignment with body joints.
# Works standalone without pretrained weights or dataset downloads.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add lib to path
this_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(this_dir, '..', 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from config import cfg
from core.inference import get_max_preds, get_final_preds
import models.pose_resnet


# COCO keypoint names and skeleton connections
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

# Joint colors by body part group
JOINT_COLORS = [
    (255, 0, 0),    # 0  nose - red
    (255, 85, 0),   # 1  left_eye
    (255, 170, 0),  # 2  right_eye
    (255, 255, 0),  # 3  left_ear
    (170, 255, 0),  # 4  right_ear
    (85, 255, 0),   # 5  left_shoulder - green spectrum
    (0, 255, 0),    # 6  right_shoulder
    (0, 255, 85),   # 7  left_elbow
    (0, 255, 170),  # 8  right_elbow
    (0, 255, 255),  # 9  left_wrist - cyan
    (0, 170, 255),  # 10 right_wrist
    (0, 85, 255),   # 11 left_hip - blue spectrum
    (0, 0, 255),    # 12 right_hip
    (85, 0, 255),   # 13 left_knee
    (170, 0, 255),  # 14 right_knee
    (255, 0, 255),  # 15 left_ankle - magenta
    (255, 0, 170),  # 16 right_ankle
]


def create_synthetic_person_image(width=192, height=256):
    """Create a synthetic image with a simple stick figure person drawn on it.

    Returns:
        image: numpy array (height, width, 3) BGR
        keypoints: numpy array (17, 2) with approximate joint locations
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray bg

    # Define keypoints for a standing person (x, y) in image coords
    cx = width // 2
    keypoints = np.zeros((17, 2), dtype=np.float32)

    # Head
    keypoints[0] = [cx, 35]          # nose
    keypoints[1] = [cx - 8, 28]      # left_eye
    keypoints[2] = [cx + 8, 28]      # right_eye
    keypoints[3] = [cx - 16, 30]     # left_ear
    keypoints[4] = [cx + 16, 30]     # right_ear

    # Upper body
    keypoints[5] = [cx - 30, 70]     # left_shoulder
    keypoints[6] = [cx + 30, 70]     # right_shoulder
    keypoints[7] = [cx - 45, 110]    # left_elbow
    keypoints[8] = [cx + 45, 110]    # right_elbow
    keypoints[9] = [cx - 50, 150]    # left_wrist
    keypoints[10] = [cx + 50, 150]   # right_wrist

    # Lower body
    keypoints[11] = [cx - 20, 150]   # left_hip
    keypoints[12] = [cx + 20, 150]   # right_hip
    keypoints[13] = [cx - 25, 195]   # left_knee
    keypoints[14] = [cx + 25, 195]   # right_knee
    keypoints[15] = [cx - 28, 235]   # left_ankle
    keypoints[16] = [cx + 28, 235]   # right_ankle

    # Draw head (circle)
    cv2.circle(image, (cx, 30), 18, (100, 80, 60), -1)

    # Draw torso
    cv2.line(image, (cx, 48), (cx, 150), (80, 60, 40), 8)

    # Draw skeleton connections with colored lines
    for (i, j) in COCO_SKELETON:
        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
        color = (
            (JOINT_COLORS[i][0] + JOINT_COLORS[j][0]) // 2,
            (JOINT_COLORS[i][1] + JOINT_COLORS[j][1]) // 2,
            (JOINT_COLORS[i][2] + JOINT_COLORS[j][2]) // 2,
        )
        cv2.line(image, pt1, pt2, color, 4)

    # Draw joint circles
    for idx, (x, y) in enumerate(keypoints):
        cv2.circle(image, (int(x), int(y)), 4, JOINT_COLORS[idx], -1)
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 0), 1)

    return image, keypoints


def generate_ground_truth_heatmaps(keypoints, heatmap_width, heatmap_height,
                                   image_width, image_height, sigma=2):
    """Generate ground truth Gaussian heatmaps from keypoint coordinates.

    Args:
        keypoints: (num_joints, 2) array of (x, y) coordinates in image space
        heatmap_width: width of heatmap
        heatmap_height: height of heatmap
        image_width: width of input image
        image_height: height of input image
        sigma: Gaussian standard deviation

    Returns:
        heatmaps: (num_joints, heatmap_height, heatmap_width) numpy array
    """
    num_joints = keypoints.shape[0]
    heatmaps = np.zeros((num_joints, heatmap_height, heatmap_width),
                        dtype=np.float32)

    scale_x = heatmap_width / image_width
    scale_y = heatmap_height / image_height

    for j in range(num_joints):
        mu_x = keypoints[j, 0] * scale_x
        mu_y = keypoints[j, 1] * scale_y

        # Generate 2D Gaussian
        x = np.arange(0, heatmap_width, 1, np.float32)
        y = np.arange(0, heatmap_height, 1, np.float32)
        y = y[:, np.newaxis]

        heatmaps[j] = np.exp(
            -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2)
        )

    return heatmaps


def build_model(cfg):
    """Build a pose_resnet model from config (randomly initialized)."""
    model = models.pose_resnet.get_pose_net(cfg, is_train=False)
    model.eval()
    return model


def prepare_input(image):
    """Prepare image tensor for model input.

    Args:
        image: BGR numpy array (H, W, 3)

    Returns:
        input_tensor: (1, 3, H, W) float tensor, normalized
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1] then apply ImageNet normalization
    image_float = image_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_float - mean) / std
    # HWC -> CHW, add batch dimension
    tensor = torch.from_numpy(
        image_normalized.transpose(2, 0, 1)
    ).unsqueeze(0).float()
    return tensor


def visualize_individual_heatmaps(image, heatmaps, preds, maxvals,
                                  gt_keypoints, output_dir, prefix):
    """Visualize each joint's heatmap overlaid on the image.

    Creates a grid showing: original image + each joint's heatmap overlay.

    Args:
        image: BGR numpy array (H, W, 3)
        heatmaps: (num_joints, hm_h, hm_w) numpy array
        preds: (num_joints, 2) predicted coordinates in heatmap space
        maxvals: (num_joints, 1) peak confidence values
        gt_keypoints: (num_joints, 2) ground truth keypoints in image space
        output_dir: directory to save visualizations
        prefix: filename prefix
    """
    num_joints = heatmaps.shape[0]
    hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]
    img_h, img_w = image.shape[0], image.shape[1]

    # Resize image to heatmap size for overlay
    image_resized = cv2.resize(image, (hm_w, hm_h))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Scale ground truth keypoints to heatmap space
    scale_x = hm_w / img_w
    scale_y = hm_h / img_h
    gt_hm = gt_keypoints.copy()
    gt_hm[:, 0] *= scale_x
    gt_hm[:, 1] *= scale_y

    # Create figure: 3 rows x 6 cols = 18 slots (1 for original + 17 joints)
    ncols = 6
    nrows = 3
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.25)

    # First panel: original image with all GT keypoints and skeleton
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image_rgb)
    for idx in range(num_joints):
        gx, gy = gt_hm[idx]
        ax0.plot(gx, gy, 'o', color=np.array(JOINT_COLORS[idx]) / 255.0,
                 markersize=4, markeredgecolor='black', markeredgewidth=0.5)
    for (i, j) in COCO_SKELETON:
        ax0.plot([gt_hm[i, 0], gt_hm[j, 0]],
                 [gt_hm[i, 1], gt_hm[j, 1]],
                 '-', color='white', linewidth=1, alpha=0.7)
    ax0.set_title('Input + GT', fontsize=9, fontweight='bold')
    ax0.axis('off')

    # Plot each joint heatmap
    for j in range(num_joints):
        row = (j + 1) // ncols
        col = (j + 1) % ncols
        ax = fig.add_subplot(gs[row, col])

        # Overlay heatmap on image
        hm = heatmaps[j]
        hm_normalized = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm_colored = cv2.applyColorMap(
            (hm_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        hm_colored_rgb = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        overlay = (hm_colored_rgb * 0.6 + image_rgb * 0.4).astype(np.uint8)

        ax.imshow(overlay)

        # Mark predicted peak (red x)
        px, py = preds[j]
        ax.plot(px, py, 'rx', markersize=8, markeredgewidth=2,
                label='Pred')

        # Mark ground truth (green circle)
        gx, gy = gt_hm[j]
        ax.plot(gx, gy, 'go', markersize=6, markeredgewidth=1.5,
                fillstyle='none', label='GT')

        conf = maxvals[j, 0]
        dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        ax.set_title(
            f'{COCO_KEYPOINT_NAMES[j]}\nconf={conf:.3f} dist={dist:.1f}px',
            fontsize=7
        )
        ax.axis('off')

    fig.suptitle(
        'DarkPose Heatmap Visualization - Per-Joint Analysis\n'
        'Red X = Predicted Peak | Green O = Ground Truth',
        fontsize=13, fontweight='bold'
    )

    save_path = os.path.join(output_dir, f'{prefix}_individual_heatmaps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved individual heatmaps: {save_path}')
    return save_path


def visualize_combined_heatmap(image, heatmaps, preds, maxvals,
                               gt_keypoints, output_dir, prefix):
    """Visualize combined (max across joints) heatmap with skeleton overlay.

    Args:
        image, heatmaps, preds, maxvals, gt_keypoints: same as above
        output_dir: directory to save
        prefix: filename prefix
    """
    num_joints = heatmaps.shape[0]
    hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]
    img_h, img_w = image.shape[0], image.shape[1]

    image_resized = cv2.resize(image, (hm_w, hm_h))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    scale_x = hm_w / img_w
    scale_y = hm_h / img_h
    gt_hm = gt_keypoints.copy()
    gt_hm[:, 0] *= scale_x
    gt_hm[:, 1] *= scale_y

    # Compute combined heatmap (max over joints)
    combined_hm = np.max(heatmaps, axis=0)
    combined_normalized = (combined_hm - combined_hm.min()) / (
        combined_hm.max() - combined_hm.min() + 1e-8
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: Original image with GT skeleton
    axes[0].imshow(image_rgb)
    for idx in range(num_joints):
        gx, gy = gt_hm[idx]
        axes[0].plot(gx, gy, 'o',
                     color=np.array(JOINT_COLORS[idx]) / 255.0,
                     markersize=6, markeredgecolor='black',
                     markeredgewidth=0.8)
    for (i, j) in COCO_SKELETON:
        axes[0].plot([gt_hm[i, 0], gt_hm[j, 0]],
                     [gt_hm[i, 1], gt_hm[j, 1]],
                     '-', color='lime', linewidth=1.5, alpha=0.8)
    axes[0].set_title('Input Image + GT Skeleton', fontsize=11,
                      fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Combined heatmap overlay with predicted skeleton
    hm_colored = cv2.applyColorMap(
        (combined_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    hm_colored_rgb = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    overlay = (hm_colored_rgb * 0.6 + image_rgb * 0.4).astype(np.uint8)
    axes[1].imshow(overlay)
    for idx in range(num_joints):
        px, py = preds[idx]
        axes[1].plot(px, py, 'o',
                     color=np.array(JOINT_COLORS[idx]) / 255.0,
                     markersize=6, markeredgecolor='white',
                     markeredgewidth=0.8)
    for (i, j) in COCO_SKELETON:
        axes[1].plot([preds[i, 0], preds[j, 0]],
                     [preds[i, 1], preds[j, 1]],
                     '-', color='white', linewidth=1.5, alpha=0.8)
    axes[1].set_title('Combined Heatmap + Predicted Skeleton', fontsize=11,
                      fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Side-by-side GT vs Pred comparison
    axes[2].imshow(image_rgb)
    for idx in range(num_joints):
        gx, gy = gt_hm[idx]
        px, py = preds[idx]
        # Draw arrow from GT to Pred
        axes[2].annotate(
            '', xy=(px, py), xytext=(gx, gy),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
        axes[2].plot(gx, gy, 'go', markersize=5, markeredgewidth=1)
        axes[2].plot(px, py, 'rx', markersize=7, markeredgewidth=2)
    axes[2].set_title('GT (green) vs Pred (red) Displacement', fontsize=11,
                      fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(
        'DarkPose Combined Heatmap Visualization',
        fontsize=14, fontweight='bold'
    )

    save_path = os.path.join(output_dir, f'{prefix}_combined_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved combined heatmap: {save_path}')
    return save_path


def visualize_gt_vs_pred_heatmaps(gt_heatmaps, pred_heatmaps, preds,
                                  gt_keypoints_hm, output_dir, prefix):
    """Compare ground truth Gaussian heatmaps with model predicted heatmaps.

    Args:
        gt_heatmaps: (num_joints, hm_h, hm_w) ground truth heatmaps
        pred_heatmaps: (num_joints, hm_h, hm_w) predicted heatmaps
        preds: (num_joints, 2) predicted peak coordinates
        gt_keypoints_hm: (num_joints, 2) GT keypoints in heatmap space
        output_dir: directory to save
        prefix: filename prefix
    """
    num_joints = gt_heatmaps.shape[0]

    # Select 6 representative joints for detailed comparison
    selected = [0, 5, 6, 9, 13, 15]  # nose, shoulders, wrist, knee, ankle
    n_selected = len(selected)

    fig, axes = plt.subplots(n_selected, 3, figsize=(15, n_selected * 3))

    for row, j in enumerate(selected):
        gt_hm = gt_heatmaps[j]
        pred_hm = pred_heatmaps[j]

        # Normalize for visualization
        gt_norm = (gt_hm - gt_hm.min()) / (gt_hm.max() - gt_hm.min() + 1e-8)
        pred_norm = (pred_hm - pred_hm.min()) / (
            pred_hm.max() - pred_hm.min() + 1e-8
        )

        # GT heatmap
        axes[row, 0].imshow(gt_norm, cmap='jet', vmin=0, vmax=1)
        gx, gy = gt_keypoints_hm[j]
        axes[row, 0].plot(gx, gy, 'w+', markersize=12, markeredgewidth=2)
        axes[row, 0].set_title(
            f'{COCO_KEYPOINT_NAMES[j]} - GT Heatmap\n'
            f'Peak: ({gx:.1f}, {gy:.1f})',
            fontsize=9
        )
        axes[row, 0].axis('off')

        # Predicted heatmap
        axes[row, 1].imshow(pred_norm, cmap='jet', vmin=0, vmax=1)
        px, py = preds[j]
        axes[row, 1].plot(px, py, 'w+', markersize=12, markeredgewidth=2)
        axes[row, 1].set_title(
            f'{COCO_KEYPOINT_NAMES[j]} - Predicted Heatmap\n'
            f'Peak: ({px:.1f}, {py:.1f}) val={pred_hm.max():.4f}',
            fontsize=9
        )
        axes[row, 1].axis('off')

        # Difference map
        diff = np.abs(gt_norm - pred_norm)
        axes[row, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
        dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        axes[row, 2].set_title(
            f'|GT - Pred| Difference\n'
            f'Peak dist: {dist:.2f}px',
            fontsize=9
        )
        axes[row, 2].axis('off')

    fig.suptitle(
        'Ground Truth vs Predicted Heatmap Comparison\n'
        '(Left: GT Gaussian | Center: Model Output | Right: Absolute Diff)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'{prefix}_gt_vs_pred_heatmaps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved GT vs Pred comparison: {save_path}')
    return save_path


def visualize_peak_analysis(heatmaps, preds, maxvals, gt_keypoints_hm,
                            output_dir, prefix):
    """Analyze peak values and alignment statistics.

    Creates a bar chart of confidence values and a scatter plot of
    GT vs predicted positions.

    Args:
        heatmaps: (num_joints, hm_h, hm_w)
        preds: (num_joints, 2) predicted peaks
        maxvals: (num_joints, 1) peak confidence
        gt_keypoints_hm: (num_joints, 2) GT in heatmap space
        output_dir: directory to save
        prefix: filename prefix
    """
    num_joints = heatmaps.shape[0]

    distances = np.sqrt(
        np.sum((preds - gt_keypoints_hm) ** 2, axis=1)
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Confidence bar chart
    colors = [np.array(c) / 255.0 for c in JOINT_COLORS]
    bars = axes[0, 0].bar(
        range(num_joints), maxvals.flatten(), color=colors, edgecolor='black',
        linewidth=0.5
    )
    axes[0, 0].set_xticks(range(num_joints))
    axes[0, 0].set_xticklabels(
        [n.replace('_', '\n') for n in COCO_KEYPOINT_NAMES],
        fontsize=6, rotation=45, ha='right'
    )
    axes[0, 0].set_ylabel('Peak Confidence Value')
    axes[0, 0].set_title('Heatmap Peak Confidence per Joint', fontweight='bold')
    axes[0, 0].axhline(y=np.mean(maxvals), color='red', linestyle='--',
                        label=f'Mean={np.mean(maxvals):.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Panel 2: Distance bar chart
    bars2 = axes[0, 1].bar(
        range(num_joints), distances, color=colors, edgecolor='black',
        linewidth=0.5
    )
    axes[0, 1].set_xticks(range(num_joints))
    axes[0, 1].set_xticklabels(
        [n.replace('_', '\n') for n in COCO_KEYPOINT_NAMES],
        fontsize=6, rotation=45, ha='right'
    )
    axes[0, 1].set_ylabel('Distance (heatmap pixels)')
    axes[0, 1].set_title('GT-to-Pred Peak Distance per Joint',
                         fontweight='bold')
    axes[0, 1].axhline(y=np.mean(distances), color='red', linestyle='--',
                        label=f'Mean={np.mean(distances):.2f}px')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Panel 3: GT vs Pred scatter (X coords)
    axes[1, 0].scatter(gt_keypoints_hm[:, 0], preds[:, 0], c=colors, s=80,
                       edgecolors='black', linewidths=0.5, zorder=3)
    lim = max(gt_keypoints_hm[:, 0].max(), preds[:, 0].max()) + 2
    axes[1, 0].plot([0, lim], [0, lim], 'r--', alpha=0.5, label='Perfect')
    axes[1, 0].set_xlabel('GT X coordinate')
    axes[1, 0].set_ylabel('Predicted X coordinate')
    axes[1, 0].set_title('X-coordinate: GT vs Predicted', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # Panel 4: GT vs Pred scatter (Y coords)
    axes[1, 1].scatter(gt_keypoints_hm[:, 1], preds[:, 1], c=colors, s=80,
                       edgecolors='black', linewidths=0.5, zorder=3)
    lim = max(gt_keypoints_hm[:, 1].max(), preds[:, 1].max()) + 2
    axes[1, 1].plot([0, lim], [0, lim], 'r--', alpha=0.5, label='Perfect')
    axes[1, 1].set_xlabel('GT Y coordinate')
    axes[1, 1].set_ylabel('Predicted Y coordinate')
    axes[1, 1].set_title('Y-coordinate: GT vs Predicted', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_aspect('equal')

    fig.suptitle(
        'DarkPose Heatmap Peak Analysis\n'
        'Alignment & Confidence Statistics',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'{prefix}_peak_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved peak analysis: {save_path}')
    return save_path


def visualize_heatmap_grid(image, heatmaps, output_dir, prefix):
    """Create the classic DarkPose-style heatmap grid visualization
    (matches save_batch_heatmaps in lib/utils/vis.py).

    Args:
        image: BGR numpy array (H, W, 3)
        heatmaps: (num_joints, hm_h, hm_w) numpy array
        output_dir: directory to save
        prefix: filename prefix
    """
    num_joints = heatmaps.shape[0]
    hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

    preds_np = heatmaps.reshape(num_joints, -1)
    idx = np.argmax(preds_np, axis=1)
    pred_x = idx % hm_w
    pred_y = idx // hm_w

    resized = cv2.resize(image, (hm_w, hm_h))

    grid = np.zeros((hm_h, (num_joints + 1) * hm_w, 3), dtype=np.uint8)

    # First column: resized input image
    grid[0:hm_h, 0:hm_w, :] = resized

    for j in range(num_joints):
        hm = heatmaps[j]
        hm_uint8 = ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8) * 255
                     ).astype(np.uint8)
        colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        overlay = (colored * 0.7 + resized * 0.3).astype(np.uint8)

        # Mark peak
        cv2.circle(overlay, (int(pred_x[j]), int(pred_y[j])), 1,
                   (0, 0, 255), 1)

        w_begin = hm_w * (j + 1)
        w_end = hm_w * (j + 2)
        grid[0:hm_h, w_begin:w_end, :] = overlay

    save_path = os.path.join(output_dir, f'{prefix}_heatmap_grid.png')
    cv2.imwrite(save_path, grid)
    print(f'Saved heatmap grid: {save_path}')
    return save_path


def print_alignment_report(preds, maxvals, gt_keypoints_hm):
    """Print a detailed text report on heatmap alignment and peak analysis."""
    num_joints = preds.shape[0]
    distances = np.sqrt(np.sum((preds - gt_keypoints_hm) ** 2, axis=1))

    print('\n' + '=' * 70)
    print('DARKPOSE HEATMAP ALIGNMENT REPORT')
    print('=' * 70)
    print(f'{"Joint":<18} {"GT (x,y)":<16} {"Pred (x,y)":<16} '
          f'{"Dist (px)":<10} {"Conf":<8}')
    print('-' * 70)

    for j in range(num_joints):
        gt_str = f'({gt_keypoints_hm[j, 0]:.1f}, {gt_keypoints_hm[j, 1]:.1f})'
        pred_str = f'({preds[j, 0]:.1f}, {preds[j, 1]:.1f})'
        print(f'{COCO_KEYPOINT_NAMES[j]:<18} {gt_str:<16} {pred_str:<16} '
              f'{distances[j]:<10.2f} {maxvals[j, 0]:<8.4f}')

    print('-' * 70)
    print(f'Mean distance:  {np.mean(distances):.2f} px')
    print(f'Max distance:   {np.max(distances):.2f} px '
          f'({COCO_KEYPOINT_NAMES[np.argmax(distances)]})')
    print(f'Min distance:   {np.min(distances):.2f} px '
          f'({COCO_KEYPOINT_NAMES[np.argmin(distances)]})')
    print(f'Mean confidence: {np.mean(maxvals):.4f}')
    print(f'Max confidence:  {np.max(maxvals):.4f} '
          f'({COCO_KEYPOINT_NAMES[np.argmax(maxvals.flatten())]})')
    print(f'Min confidence:  {np.min(maxvals):.4f} '
          f'({COCO_KEYPOINT_NAMES[np.argmin(maxvals.flatten())]})')

    # Alignment check
    threshold = 5.0  # heatmap pixels
    aligned = np.sum(distances < threshold)
    print(f'\nAlignment check (threshold={threshold:.1f} hm-px):')
    print(f'  Aligned joints:    {aligned}/{num_joints}')
    print(f'  Misaligned joints: {num_joints - aligned}/{num_joints}')

    if aligned == num_joints:
        print('  Status: ALL JOINTS CORRECTLY ALIGNED')
    else:
        misaligned_names = [
            COCO_KEYPOINT_NAMES[j]
            for j in range(num_joints) if distances[j] >= threshold
        ]
        print(f'  Misaligned: {", ".join(misaligned_names)}')

    # Peak value correspondence check
    print(f'\nPeak value correspondence check:')
    for j in range(num_joints):
        hm_peak_x, hm_peak_y = preds[j]
        gt_x, gt_y = gt_keypoints_hm[j]
        is_close = distances[j] < threshold
        status = 'OK' if is_close else 'MISMATCH'
        print(f'  {COCO_KEYPOINT_NAMES[j]:<18}: peak@({hm_peak_x:.1f},'
              f'{hm_peak_y:.1f}) gt@({gt_x:.1f},{gt_y:.1f}) -> [{status}]')

    print('=' * 70)

    note = (
        '\nNOTE: This demo uses randomly initialized weights (no pretrained '
        'model).\nWith a trained model, heatmap peaks would closely match GT '
        'joint locations\nand confidence values would be much higher. The '
        'visualization pipeline\nitself is correct and ready for use with '
        'pretrained DarkPose models.'
    )
    print(note)
    print('=' * 70 + '\n')

    return distances, aligned


def parse_args():
    parser = argparse.ArgumentParser(
        description='DarkPose Heatmap Visualization Demo'
    )
    parser.add_argument(
        '--output-dir', type=str, default='demo_output',
        help='Directory to save visualization outputs'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to input image (uses synthetic if not provided)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {os.path.abspath(output_dir)}')

    # Image and heatmap dimensions (matching ResNet-50 config)
    img_w, img_h = 192, 256
    hm_w, hm_h = 48, 64

    # --- Step 1: Prepare input image ---
    print('\n[Step 1] Preparing input image...')
    if args.image is not None and os.path.isfile(args.image):
        image = cv2.imread(args.image)
        image = cv2.resize(image, (img_w, img_h))
        # Use center of image as rough keypoint estimate
        gt_keypoints = np.zeros((17, 2), dtype=np.float32)
        gt_keypoints[:, 0] = img_w / 2
        gt_keypoints[:, 1] = np.linspace(20, img_h - 20, 17)
        print(f'Loaded image: {args.image}')
    else:
        image, gt_keypoints = create_synthetic_person_image(img_w, img_h)
        print('Created synthetic person image with known joint locations')

    # Save input image
    input_path = os.path.join(output_dir, 'input_image.png')
    cv2.imwrite(input_path, image)
    print(f'Saved input image: {input_path}')

    # --- Step 2: Generate ground truth heatmaps ---
    print('\n[Step 2] Generating ground truth Gaussian heatmaps...')
    gt_heatmaps = generate_ground_truth_heatmaps(
        gt_keypoints, hm_w, hm_h, img_w, img_h, sigma=2
    )
    print(f'GT heatmaps shape: {gt_heatmaps.shape}')

    # Verify GT heatmap peaks
    gt_hm_4d = gt_heatmaps[np.newaxis, ...]  # (1, 17, 64, 48)
    gt_preds, gt_maxvals = get_max_preds(gt_hm_4d)
    gt_preds = gt_preds[0]       # (17, 2)
    gt_maxvals = gt_maxvals[0]   # (17, 1)

    scale_x = hm_w / img_w
    scale_y = hm_h / img_h
    gt_keypoints_hm = gt_keypoints.copy()
    gt_keypoints_hm[:, 0] *= scale_x
    gt_keypoints_hm[:, 1] *= scale_y

    print('GT heatmap peak verification:')
    for j in range(min(5, 17)):
        print(f'  {COCO_KEYPOINT_NAMES[j]}: '
              f'expected=({gt_keypoints_hm[j, 0]:.1f}, '
              f'{gt_keypoints_hm[j, 1]:.1f}), '
              f'peak=({gt_preds[j, 0]:.1f}, {gt_preds[j, 1]:.1f}), '
              f'val={gt_maxvals[j, 0]:.4f}')

    # --- Step 3: Build model and run inference ---
    print('\n[Step 3] Building pose_resnet model (random init)...')
    cfg.defrost()
    cfg.MODEL.NAME = 'pose_resnet'
    cfg.MODEL.NUM_JOINTS = 17
    cfg.MODEL.IMAGE_SIZE = [img_w, img_h]
    cfg.MODEL.HEATMAP_SIZE = [hm_w, hm_h]
    cfg.MODEL.INIT_WEIGHTS = False
    cfg.MODEL.PRETRAINED = ''
    cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
    cfg.MODEL.EXTRA.DECONV_WITH_BIAS = False
    cfg.MODEL.EXTRA.NUM_DECONV_LAYERS = 3
    cfg.MODEL.EXTRA.NUM_DECONV_FILTERS = [256, 256, 256]
    cfg.MODEL.EXTRA.NUM_DECONV_KERNELS = [4, 4, 4]
    cfg.MODEL.EXTRA.NUM_LAYERS = 50
    cfg.TEST.BLUR_KERNEL = 11
    cfg.freeze()

    model = build_model(cfg)
    print(f'Model created: {cfg.MODEL.NAME} (ResNet-50)')

    input_tensor = prepare_input(image)
    print(f'Input tensor shape: {input_tensor.shape}')

    print('Running forward pass...')
    with torch.no_grad():
        output = model(input_tensor)
    print(f'Output heatmaps shape: {output.shape}')

    # Extract heatmaps
    pred_heatmaps = output[0].cpu().numpy()  # (17, 64, 48)

    # Get predictions using DarkPose get_max_preds
    pred_hm_4d = output.cpu().numpy()  # (1, 17, 64, 48)
    preds, maxvals = get_max_preds(pred_hm_4d)
    preds = preds[0]       # (17, 2)
    maxvals = maxvals[0]   # (17, 1)

    # --- Step 4: Generate all visualizations ---
    print('\n[Step 4] Generating visualizations...')

    # 4a: Individual joint heatmaps
    visualize_individual_heatmaps(
        image, pred_heatmaps, preds, maxvals,
        gt_keypoints, output_dir, 'demo'
    )

    # 4b: Combined heatmap overlay
    visualize_combined_heatmap(
        image, pred_heatmaps, preds, maxvals,
        gt_keypoints, output_dir, 'demo'
    )

    # 4c: GT vs Predicted heatmap comparison
    visualize_gt_vs_pred_heatmaps(
        gt_heatmaps, pred_heatmaps, preds,
        gt_keypoints_hm, output_dir, 'demo'
    )

    # 4d: Peak analysis charts
    visualize_peak_analysis(
        pred_heatmaps, preds, maxvals,
        gt_keypoints_hm, output_dir, 'demo'
    )

    # 4e: Classic heatmap grid
    visualize_heatmap_grid(
        image, pred_heatmaps, output_dir, 'demo'
    )

    # --- Step 5: Print alignment report ---
    print('\n[Step 5] Alignment and peak value analysis...')
    distances, aligned = print_alignment_report(
        preds, maxvals, gt_keypoints_hm
    )

    print(f'\nAll visualizations saved to: {os.path.abspath(output_dir)}')
    print('Demo complete!')


if __name__ == '__main__':
    main()
