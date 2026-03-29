"""
cdr_models.py — Data models for cup-disc ratio (CDR) pipeline output.

Model: pamixsun/segformer_for_optic_disc_cup_segmentation
Label map: 0=background, 1=disc annulus (ring only), 2=optic cup (inner)
full_disc = pred >= 1  (disc annulus + cup combined)

CDR is computed two ways:
  - area_cdr   : cup_area / full_disc_area  (pixel count ratio)
  - linear_cdr : cup_diameter / disc_diameter  (equivalent-circle diameters)
               = sqrt(cup_area / disc_area)  — matches clinical convention

Extended structural features (DiscMorphology) are attached after calling
compute_disc_morphology() and stored in CDRResult.morph.  They include:
  - vertical / horizontal CDR (bounding-box method)
  - rim area and rim-to-disc ratio
  - disc size normalised by image area
  - disc-fovea distance and angle (requires VascX fovea output)
  - cup centroid offset from disc centroid
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image

if TYPE_CHECKING:
    from disc_morphology import DiscMorphology


@dataclass
class DiscGeometry:
    """
    Geometric properties of the optic disc derived from the full_disc mask
    (pred >= 1).  Centre and radius are used downstream by PVBM zone-B logic.
    """
    center_x: Optional[float] = None   # col (x) in image pixel coords
    center_y: Optional[float] = None   # row (y) in image pixel coords
    radius: Optional[float] = None     # equivalent-circle radius = sqrt(area/π)
    area_px: Optional[int] = None      # pixel count of full disc mask


@dataclass
class CupGeometry:
    """
    Geometric properties of the optic cup (pred == 2).
    """
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    radius: Optional[float] = None     # equivalent-circle radius
    area_px: Optional[int] = None


@dataclass
class CDRResult:
    """
    Top-level result for one fundus image.

    Attributes
    ----------
    image_id     : stem of the input filename (no extension)
    output_dir   : directory where masks were saved
    disc          : DiscGeometry — full disc (annulus + cup)
    cup           : CupGeometry  — cup only
    area_cdr      : cup_area / disc_area  (pixel area ratio)
    linear_cdr    : sqrt(cup_area / disc_area) = cup_r / disc_r
                    (equivalent to vertical CDR used clinically)
    segmap_path   : path to saved uint8 segmentation map (0/1/2)
    disc_mask_path: path to saved binary full-disc mask (uint8, 0 or 255)
    cup_mask_path : path to saved binary cup mask (uint8, 0 or 255)
    """
    image_id: str
    output_dir: Path

    disc: DiscGeometry = None
    cup: CupGeometry = None

    area_cdr: Optional[float] = None
    linear_cdr: Optional[float] = None

    segmap_path: Optional[Path] = None
    disc_mask_path: Optional[Path] = None
    cup_mask_path: Optional[Path] = None

    # Populated after calling compute_disc_morphology(cdr_result, image_shape, fovea)
    morph: Optional[Any] = None   # type: DiscMorphology | None at runtime

    def __post_init__(self):
        if self.disc is None:
            self.disc = DiscGeometry()
        if self.cup is None:
            self.cup = CupGeometry()
        self.output_dir = Path(self.output_dir)

    # ------------------------------------------------------------------
    # Convenience loaders
    # ------------------------------------------------------------------

    def load_segmap(self) -> np.ndarray:
        """Return the raw uint8 segmentation map (0=bg, 1=disc ring, 2=cup)."""
        return np.array(Image.open(self.segmap_path))

    def load_disc_mask(self) -> np.ndarray:
        """Return binary full-disc mask (H×W, values 0 or 1)."""
        return (np.array(Image.open(self.disc_mask_path)) > 0).astype(np.uint8)

    def load_cup_mask(self) -> np.ndarray:
        """Return binary cup mask (H×W, values 0 or 1)."""
        return (np.array(Image.open(self.cup_mask_path)) > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_flat_dict(self) -> dict:
        d = {
            "image_id":        self.image_id,
            "disc_cx":         self.disc.center_x,
            "disc_cy":         self.disc.center_y,
            "disc_radius":     self.disc.radius,
            "disc_area_px":    self.disc.area_px,
            "cup_cx":          self.cup.center_x,
            "cup_cy":          self.cup.center_y,
            "cup_radius":      self.cup.radius,
            "cup_area_px":     self.cup.area_px,
            "area_cdr":        self.area_cdr,
            "linear_cdr":      self.linear_cdr,
        }
        if self.morph is not None:
            for k, v in self.morph.to_flat_dict().items():
                d[f"morph_{k}"] = v
        return d

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_flat_dict())
