"""
disc_morphology.py — Structural glaucoma features derived from existing masks.

All features are computed from outputs already produced by the CDR pipeline
(SegFormer segmentation masks) and VascX (fovea coordinates, disc mask).
No additional models are required.

Features
--------
Vertical CDR (vcdr)
    Clinical graders measure CDR as the vertical cup height divided by the
    vertical disc height, rather than the area-derived linear_cdr.  For
    round cups/discs both converge, but glaucomatous cupping preferentially
    enlarges the inferior pole, making vertical CDR more sensitive early on.

    vcdr = cup_height_px / disc_height_px
    where height = number of rows containing at least one mask pixel (bounding
    box height), consistent with clinical practice.

Horizontal CDR (hcdr)
    Analogous ratio along the horizontal axis.  Less clinically used than vcdr
    but useful as a cross-check and for asymmetry analysis.

Rim area and rim-to-disc ratio (RDR)
    The neuroretinal rim is the tissue between the cup and disc boundaries.
    rim_area_px  = disc_area_px - cup_area_px
    rdr          = rim_area_px / disc_area_px   (0–1; lower = more cupping)

    A healthy rim follows the ISNT rule (inferior ≥ superior ≥ nasal ≥
    temporal thickness).  The RDR scalar captures global rim loss; sectoral
    breakdown requires RNFL data (available in GRAPE clinical fields).

Disc area normalised by image size (disc_area_norm)
    disc_area_px / (image_height * image_width)
    Removes sensor/FOV dependence so disc size is comparable across images.
    A large disc with a large cup is physiologically normal; this field lets
    the causal model condition CDR on disc size.

Disc-fovea distance (df_distance_px, df_distance_norm)
    Euclidean distance between disc centre and fovea in pixels, and
    normalised by image diagonal.  Clinically stable (~4–5 disc diameters),
    but shifts with certain pathologies and helps verify disc localisation.

Disc-fovea angle (df_angle_deg)
    Angle of the fovea relative to the disc centre, measured from the
    positive-x axis (clockwise positive, consistent with fundus image coords
    where y increases downward).  Normally ~0°–(−10°) for right eyes (fovea
    slightly inferior to disc) with the sign flipping for left eyes.
    A shift in this angle may indicate disc displacement.

Cup centre offset (cup_offset_x, cup_offset_y, cup_offset_norm)
    Displacement of the cup centroid from the disc centroid, in pixels and
    normalised by disc radius.  Infero-nasal cup displacement relative to the
    disc is a known early glaucoma sign.

Usage
-----
    from disc_morphology import compute_disc_morphology, DiscMorphology

    morph = compute_disc_morphology(
        cdr_result=cdr_result,          # CDRResult with loaded masks
        image_shape=(H, W),             # tuple from np.array(Image.open(...)).shape[:2]
        fovea_xy=vascx_result.fovea,    # pd.Series with 'x', 'y' — or None
    )

    print(morph.vcdr)               # vertical cup-disc ratio
    print(morph.df_distance_norm)   # disc-fovea distance / image diagonal
    morph_dict = morph.to_flat_dict()  # for CSV / DataFrame row
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from cdr_models import CDRResult


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DiscMorphology:
    """
    Structural optic disc features derivable without additional models.

    All pixel measurements are in the coordinate space of the SegFormer
    output mask (same resolution as the saved disc_mask / cup_mask PNGs).
    Normalised fields remove this resolution dependence.
    """

    # -- Vertical / horizontal CDR -------------------------------------------
    vcdr: Optional[float] = None
    """Vertical CDR: cup bounding-box height / disc bounding-box height."""

    hcdr: Optional[float] = None
    """Horizontal CDR: cup bounding-box width / disc bounding-box width."""

    disc_height_px: Optional[int] = None
    """Vertical extent of the optic disc mask (bounding-box height, pixels)."""

    disc_width_px: Optional[int] = None
    """Horizontal extent of the optic disc mask (bounding-box width, pixels)."""

    cup_height_px: Optional[int] = None
    cup_width_px: Optional[int] = None

    # -- Rim -----------------------------------------------------------------
    rim_area_px: Optional[int] = None
    """disc_area_px − cup_area_px (pixels)."""

    rdr: Optional[float] = None
    """Rim-to-disc ratio: rim_area_px / disc_area_px. Higher = healthier rim."""

    # -- Disc size normalised by image area ----------------------------------
    disc_area_norm: Optional[float] = None
    """disc_area_px / (image_H * image_W). Removes FOV / sensor dependence."""

    disc_radius_norm: Optional[float] = None
    """disc_radius_px / image_diagonal. Comparable across resolutions."""

    # -- Disc-fovea geometry -------------------------------------------------
    df_distance_px: Optional[float] = None
    """Euclidean disc-centre to fovea distance in pixels."""

    df_distance_norm: Optional[float] = None
    """df_distance_px / image_diagonal (0–1 scale)."""

    df_angle_deg: Optional[float] = None
    """
    Angle from disc centre to fovea, degrees.
    0° = fovea directly right of disc, positive = clockwise (y-down coords).
    Typically ~0° to −10° (fovea slightly inferior) for right eyes.
    """

    # -- Cup offset from disc centre -----------------------------------------
    cup_offset_x: Optional[float] = None
    """Horizontal displacement of cup centroid relative to disc centroid (px)."""

    cup_offset_y: Optional[float] = None
    """Vertical displacement of cup centroid relative to disc centroid (px)."""

    cup_offset_norm: Optional[float] = None
    """Euclidean cup offset / disc_radius. >0.1 may indicate asymmetric cupping."""

    # -- Quality flags -------------------------------------------------------
    disc_fovea_available: bool = False
    """True if fovea coordinates were present and disc-fovea metrics computed."""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_flat_dict(self) -> dict:
        return {
            "vcdr":                 self.vcdr,
            "hcdr":                 self.hcdr,
            "disc_height_px":       self.disc_height_px,
            "disc_width_px":        self.disc_width_px,
            "cup_height_px":        self.cup_height_px,
            "cup_width_px":         self.cup_width_px,
            "rim_area_px":          self.rim_area_px,
            "rdr":                  self.rdr,
            "disc_area_norm":       self.disc_area_norm,
            "disc_radius_norm":     self.disc_radius_norm,
            "df_distance_px":       self.df_distance_px,
            "df_distance_norm":     self.df_distance_norm,
            "df_angle_deg":         self.df_angle_deg,
            "cup_offset_x":         self.cup_offset_x,
            "cup_offset_y":         self.cup_offset_y,
            "cup_offset_norm":      self.cup_offset_norm,
            "disc_fovea_available": self.disc_fovea_available,
        }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _bounding_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Return (row_min, row_max, col_min, col_max) for a binary mask, or None
    if the mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def compute_disc_morphology(
    cdr_result: "CDRResult",
    image_shape: tuple[int, int],
    fovea_xy: Optional["pd.Series"] = None,
) -> DiscMorphology:
    """
    Compute structural disc features from an existing CDRResult.

    Parameters
    ----------
    cdr_result   : CDRResult with populated disc / cup geometry and saved masks.
    image_shape  : (H, W) of the *original* fundus image (not the mask).
                   Used only for normalisation — mask-space measurements are
                   scaled to image-space before normalising.
    fovea_xy     : pd.Series with 'x' and 'y' keys (VascX fovea output),
                   in the coordinate space of the *original* fundus image.
                   Pass None to skip disc-fovea metrics.

    Returns
    -------
    DiscMorphology dataclass with all fields populated where possible.
    """
    m = DiscMorphology()

    img_H, img_W = image_shape
    img_diagonal = math.sqrt(img_H ** 2 + img_W ** 2)

    # -- Load masks ----------------------------------------------------------
    try:
        disc_mask = cdr_result.load_disc_mask()   # (H_mask, W_mask) uint8 0/1
        cup_mask  = cdr_result.load_cup_mask()
    except Exception:
        return m   # masks not available — return empty

    mask_H, mask_W = disc_mask.shape

    # Scale factor from mask pixels to image pixels (for normalisation)
    scale_h = img_H / mask_H
    scale_w = img_W / mask_W

    # -- Vertical / horizontal CDR -------------------------------------------
    disc_bb = _bounding_box(disc_mask)
    cup_bb  = _bounding_box(cup_mask)

    if disc_bb is not None and cup_bb is not None:
        d_rmin, d_rmax, d_cmin, d_cmax = disc_bb
        c_rmin, c_rmax, c_cmin, c_cmax = cup_bb

        disc_h = d_rmax - d_rmin + 1
        disc_w = d_cmax - d_cmin + 1
        cup_h  = c_rmax - c_rmin + 1
        cup_w  = c_cmax - c_cmin + 1

        m.disc_height_px = disc_h
        m.disc_width_px  = disc_w
        m.cup_height_px  = cup_h
        m.cup_width_px   = cup_w

        m.vcdr = cup_h / disc_h if disc_h > 0 else None
        m.hcdr = cup_w / disc_w if disc_w > 0 else None

    # -- Rim -----------------------------------------------------------------
    if (cdr_result.disc.area_px is not None
            and cdr_result.cup.area_px is not None):
        rim = cdr_result.disc.area_px - cdr_result.cup.area_px
        m.rim_area_px = max(0, rim)
        if cdr_result.disc.area_px > 0:
            m.rdr = m.rim_area_px / cdr_result.disc.area_px

    # -- Disc size normalised by image -----------------------------------------
    if cdr_result.disc.area_px is not None:
        # Convert mask-space area to image-space area
        area_img = cdr_result.disc.area_px * scale_h * scale_w
        m.disc_area_norm = area_img / (img_H * img_W)

    if cdr_result.disc.radius is not None:
        # Radius in mask space → image space (use geometric mean of scale factors)
        radius_img = cdr_result.disc.radius * math.sqrt(scale_h * scale_w)
        m.disc_radius_norm = radius_img / img_diagonal

    # -- Cup centroid offset from disc centroid --------------------------------
    if (cdr_result.disc.center_x is not None
            and cdr_result.cup.center_x is not None):
        m.cup_offset_x = cdr_result.cup.center_x - cdr_result.disc.center_x
        m.cup_offset_y = cdr_result.cup.center_y - cdr_result.disc.center_y
        offset_px = math.sqrt(m.cup_offset_x ** 2 + m.cup_offset_y ** 2)
        if cdr_result.disc.radius and cdr_result.disc.radius > 0:
            m.cup_offset_norm = offset_px / cdr_result.disc.radius

    # -- Disc-fovea geometry --------------------------------------------------
    if fovea_xy is not None and cdr_result.disc.center_x is not None:
        fov_x, fov_y = _extract_fovea_coords(fovea_xy)

        if fov_x is not None and fov_y is not None:
            # Fovea is in original image space; disc centre is in mask space.
            # Scale disc centre to image space for a consistent comparison.
            disc_cx_img = cdr_result.disc.center_x * scale_w
            disc_cy_img = cdr_result.disc.center_y * scale_h

            dx = fov_x - disc_cx_img
            dy = fov_y - disc_cy_img   # positive = fovea below disc (y-down)

            m.df_distance_px   = math.sqrt(dx ** 2 + dy ** 2)
            m.df_distance_norm = m.df_distance_px / img_diagonal
            # atan2 with y-down: angle measured clockwise from positive-x axis
            m.df_angle_deg     = math.degrees(math.atan2(dy, dx))
            m.disc_fovea_available = True

    return m


def _extract_fovea_coords(
    fovea_xy: "pd.Series",
) -> tuple[Optional[float], Optional[float]]:
    """
    Extract (x, y) from a VascX fovea Series robustly.

    VascX HeatmapRegressionEnsemble may return columns named 'x'/'y',
    'col'/'row', or integer positions 0/1 depending on the model version.
    Try each convention in order and return the first that works.
    """
    idx = fovea_xy.index.tolist()

    # Named: 'x' / 'y'
    if "x" in idx and "y" in idx:
        return float(fovea_xy["x"]), float(fovea_xy["y"])

    # Named: 'col' / 'row'
    if "col" in idx and "row" in idx:
        return float(fovea_xy["col"]), float(fovea_xy["row"])

    # Named: 'fovea_x' / 'fovea_y'
    if "fovea_x" in idx and "fovea_y" in idx:
        return float(fovea_xy["fovea_x"]), float(fovea_xy["fovea_y"])

    # Positional fallback: first two values are x, y
    vals = fovea_xy.values
    if len(vals) >= 2:
        try:
            return float(vals[0]), float(vals[1])
        except (TypeError, ValueError):
            pass

    return None, None
