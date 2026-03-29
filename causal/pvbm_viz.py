"""
pvbm_viz.py — Zone and vessel overlay visualization utilities.

Usage example:
    from pvbm_viz import plot_zones
    plot_zones(fundus_img, disc_mask, artery_mask, vein_mask)
    # cx, cy, radius are optional — derived from disc_mask if not provided
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


def _disc_geometry_from_mask(
    disc_mask: np.ndarray,
) -> tuple[float, float, float]:
    """
    Derive disc centre (cx, cy) and equivalent-circle radius from a binary
    disc segmentation mask, using the largest connected component.

    Returns (cx, cy, radius) in pixel coordinates of the mask.
    """
    labeled = label(disc_mask > 0)
    props = regionprops(labeled)
    if not props:
        raise ValueError("No optic disc region found in disc mask.")
    largest = max(props, key=lambda p: p.area)
    cy, cx = largest.centroid       # regionprops: (row, col) = (y, x)
    radius = np.sqrt(largest.area / np.pi)
    return float(cx), float(cy), float(radius)


def plot_zones(
    fundus: np.ndarray | Image.Image,
    disc_mask: np.ndarray,
    artery_mask: np.ndarray,
    vein_mask: np.ndarray,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    radius: Optional[float] = None,
    figsize: tuple[int, int] = (14, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel figure:
      Left  — fundus with zone A / B / C annuli overlaid as coloured rings.
      Right — artery/vein masks with the zone-B annulus highlighted.

    Parameters
    ----------
    fundus       : RGB fundus image (H×W×3 ndarray or PIL Image).
    disc_mask    : Binary disc segmentation — any resolution, resized internally.
    artery_mask  : Binary artery segmentation — any resolution, resized internally.
    vein_mask    : Binary vein segmentation — any resolution, resized internally.
    cx, cy       : Optional disc centre override (fundus pixel coords).
                   If None, derived from disc_mask after resizing.
    radius       : Optional disc radius override (fundus pixels).
                   If None, derived from disc_mask after resizing.
                   Passing pre-computed PVBM values is NOT recommended because
                   they are in mask-space and may not match fundus resolution.
    figsize      : Matplotlib figure size.
    save_path    : If given, save figure to this path.
    """
    if isinstance(fundus, Image.Image):
        fundus = np.array(fundus.convert("RGB"))

    H, W = fundus.shape[:2]

    # --- Resize all masks to fundus resolution ---
    def _resize_mask(mask: np.ndarray) -> np.ndarray:
        if mask.shape[:2] == (H, W):
            return mask
        return np.array(
            Image.fromarray((mask > 0).astype(np.uint8) * 255)
            .resize((W, H), Image.NEAREST)
        ) // 255   # back to binary 0/1

    disc_mask_r   = _resize_mask(disc_mask)
    artery_mask_r = _resize_mask(artery_mask)
    vein_mask_r   = _resize_mask(vein_mask)

    # --- Derive disc geometry from the RESIZED mask ---
    # This guarantees cx, cy, r are all in fundus pixel space.
    # Any cx/cy/radius passed in are ignored — they are typically in
    # mask-space (VascX preprocessed resolution) and will be wrong here.
    try:
        cx_m, cy_m, r_m = _disc_geometry_from_mask(disc_mask_r)
    except ValueError:
        # Fallback to passed-in values (scaled if they differ from mask-space)
        if cx is None or cy is None or radius is None:
            raise
        orig_h = disc_mask.shape[0]
        scale  = H / orig_h
        cx_m, cy_m, r_m = cx * scale, cy * scale, radius * scale

    cx_draw, cy_draw, r = cx_m, cy_m, r_m

    # --- Build zone masks ---
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - cx_draw) ** 2 + (yy - cy_draw) ** 2)

    zone_a = dist <= r
    zone_b = (dist > r) & (dist <= 2 * r)
    zone_c = (dist > 2 * r) & (dist <= 3 * r)

    # --- Left panel: fundus + zone overlays ---
    overlay = fundus.copy().astype(np.float32)

    def _tint(mask, rgb, alpha=0.35):
        for c, val in enumerate(rgb):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + val * alpha,
                overlay[:, :, c],
            )

    _tint(zone_a, (255, 80,  80),  alpha=0.40)   # red   — zone A (disc)
    _tint(zone_b, (60,  200, 60),  alpha=0.30)   # green — zone B (analysis)
    _tint(zone_c, (80,  140, 255), alpha=0.25)   # blue  — zone C (outer)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # --- Right panel: vessel map with zone-B boundary rings ---
    vessel_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vessel_rgb[artery_mask_r > 0] = (220, 60, 60)
    vessel_rgb[vein_mask_r > 0]   = (60, 100, 220)

    for c, val in enumerate((30, 80, 30)):
        vessel_rgb[:, :, c] = np.where(
            zone_b & (artery_mask_r == 0) & (vein_mask_r == 0),
            np.clip(vessel_rgb[:, :, c].astype(int) + val, 0, 255),
            vessel_rgb[:, :, c],
        )

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.set_xticks([])
        ax.set_yticks([])

    ring_styles = [
        (r,      "#FF5050", "--", "Zone A boundary (r)"),
        (2 * r,  "#50C850", "-",  "Zone B boundary (2r)"),
        (3 * r,  "#5090FF", "--", "Zone C boundary (3r)"),
    ]

    # Left panel
    axes[0].imshow(overlay)
    axes[0].set_title("Fundus — Zone overlay", color="white", fontsize=12, pad=8)
    for ring_r, color, ls, _ in ring_styles:
        axes[0].add_patch(plt.Circle(
            (cx_draw, cy_draw), ring_r,
            color=color, fill=False, linewidth=1.5, linestyle=ls,
        ))
    axes[0].plot(cx_draw, cy_draw, "+", color="white", markersize=10, markeredgewidth=1.5)
    axes[0].legend(
        handles=[
            mpatches.Patch(facecolor="#FF5050", alpha=0.7, label=f"Zone A — disc (r={r:.0f}px)"),
            mpatches.Patch(facecolor="#50C850", alpha=0.7, label="Zone B — analysis (r–2r)"),
            mpatches.Patch(facecolor="#5090FF", alpha=0.7, label="Zone C — outer (2r–3r)"),
        ],
        loc="lower left", fontsize=8,
        facecolor="#222222", labelcolor="white", framealpha=0.8,
    )

    # Right panel
    axes[1].imshow(vessel_rgb)
    axes[1].set_title("A/V segmentation — Zone B highlighted", color="white", fontsize=12, pad=8)
    for ring_r, color, ls, label_txt in ring_styles:
        axes[1].add_patch(plt.Circle(
            (cx_draw, cy_draw), ring_r,
            color=color, fill=False, linewidth=1.5, linestyle=ls,
        ))
        angle_rad = math.radians(45)
        tx = cx_draw + ring_r * math.cos(angle_rad)
        ty = cy_draw - ring_r * math.sin(angle_rad)
        if 0 < tx < W and 0 < ty < H:
            axes[1].annotate(
                label_txt, xy=(tx, ty), fontsize=7, color=color,
                xytext=(6, -6), textcoords="offset points",
            )
    axes[1].plot(cx_draw, cy_draw, "+", color="white", markersize=10, markeredgewidth=1.5)
    axes[1].legend(
        handles=[
            mpatches.Patch(facecolor="#DC3C3C", label="Arteries"),
            mpatches.Patch(facecolor="#3C64DC", label="Veins"),
            mpatches.Patch(facecolor="#1E501E", label="Zone B (background)"),
        ],
        loc="lower left", fontsize=8,
        facecolor="#222222", labelcolor="white", framealpha=0.8,
    )

    plt.tight_layout(pad=1.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")

    return fig
