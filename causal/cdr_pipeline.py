"""
cdr_pipeline.py — Cup-disc ratio segmentation pipeline.

Model : pamixsun/segformer_for_optic_disc_cup_segmentation (HuggingFace)
Labels: 0=background, 1=disc annulus (ring only), 2=optic cup (inner)

Outputs written to  <output_root>/<image_id>/
  segmap.png        — uint8 label map (0/1/2)
  disc_mask.png     — binary full-disc mask (pred >= 1), 0 or 255
  cup_mask.png      — binary cup mask  (pred == 2), 0 or 255

Usage
-----
    from cdr_pipeline import CDRPipeline

    cdr = CDRPipeline(output_root="outputs", device="cuda")
    result = cdr.run("datasets/grape/CFPs/100_OD_1.jpg")

    print(result.area_cdr)      # pixel-area CDR
    print(result.linear_cdr)    # sqrt(cup/disc area) — clinical convention
    print(result.disc.radius)   # equivalent-circle disc radius (px)
    print(result.to_series())   # flat pd.Series for DataFrame rows
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from cdr_models import CDRResult, CupGeometry, DiscGeometry

# Lazy-imported to avoid mandatory HF dependency at module load time
_transformers_imported = False
_AutoImageProcessor = None
_SegformerForSemanticSegmentation = None

MODEL_ID = "pamixsun/segformer_for_optic_disc_cup_segmentation"


def _import_transformers():
    global _transformers_imported, _AutoImageProcessor, _SegformerForSemanticSegmentation
    if not _transformers_imported:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        _AutoImageProcessor = AutoImageProcessor
        _SegformerForSemanticSegmentation = SegformerForSemanticSegmentation
        _transformers_imported = True


def _regionprops_simple(binary: np.ndarray) -> tuple[float, float, float, int]:
    """
    Lightweight region properties without skimage dependency.
    Returns (center_x, center_y, equiv_radius, area_px).
    center_x = col (x), center_y = row (y).
    """
    area = int(binary.sum())
    if area == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    rows, cols = np.where(binary)
    cy = float(rows.mean())
    cx = float(cols.mean())
    radius = math.sqrt(area / math.pi)
    return cx, cy, radius, area


class CDRPipeline:
    """
    Runs SegFormer cup-disc segmentation on one or more fundus images.

    Parameters
    ----------
    output_root : str | Path
        Root directory; per-image subdirs are created automatically.
    device      : str | torch.device
        "cuda", "cpu", or a torch.device object.
    """

    def __init__(
        self,
        output_root: Union[str, Path] = "outputs",
        device: Union[str, torch.device] = "cuda",
    ):
        self.output_root = Path(output_root)
        self.device = torch.device(device) if isinstance(device, str) else device

        _import_transformers()
        print(f"[cdr] Loading {MODEL_ID} …")
        self._processor = _AutoImageProcessor.from_pretrained(MODEL_ID)
        self._model = (
            _SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
            .to(self.device)
            .eval()
        )
        print("[cdr] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image_path: Union[str, Path]) -> CDRResult:
        """
        Segment a single fundus image and return a CDRResult.

        Parameters
        ----------
        image_path : path to a fundus image (any PIL-readable format).
        """
        image_path = Path(image_path)
        image_id = image_path.stem
        out_dir = self.output_root / image_id
        out_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        w, h = image.size  # PIL: (width, height)

        # --- Inference ---
        inputs = self._processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, 3, H/4, W/4)

        upsampled = nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        # pred: 0=background, 1=disc annulus, 2=optic cup

        # --- Derive masks ---
        full_disc_mask = (pred >= 1).astype(np.uint8)   # disc + cup
        cup_mask       = (pred == 2).astype(np.uint8)   # cup only

        # --- Geometry ---
        disc_cx, disc_cy, disc_r, disc_area = _regionprops_simple(full_disc_mask)
        cup_cx,  cup_cy,  cup_r,  cup_area  = _regionprops_simple(cup_mask)

        # --- CDR ---
        if disc_area > 0:
            area_cdr   = cup_area / disc_area
            linear_cdr = math.sqrt(area_cdr)   # cup_r / disc_r
        else:
            area_cdr = linear_cdr = float("nan")

        # --- Save outputs ---
        segmap_path     = out_dir / "segmap.png"
        disc_mask_path  = out_dir / "disc_mask.png"
        cup_mask_path   = out_dir / "cup_mask.png"

        Image.fromarray(pred).save(segmap_path)
        Image.fromarray(full_disc_mask * 255).save(disc_mask_path)
        Image.fromarray(cup_mask * 255).save(cup_mask_path)

        result = CDRResult(
            image_id=image_id,
            output_dir=out_dir,
            disc=DiscGeometry(
                center_x=disc_cx,
                center_y=disc_cy,
                radius=disc_r,
                area_px=disc_area,
            ),
            cup=CupGeometry(
                center_x=cup_cx,
                center_y=cup_cy,
                radius=cup_r,
                area_px=cup_area,
            ),
            area_cdr=area_cdr,
            linear_cdr=linear_cdr,
            segmap_path=segmap_path,
            disc_mask_path=disc_mask_path,
            cup_mask_path=cup_mask_path,
        )

        print(
            f"[cdr] {image_id}: "
            f"area_CDR={area_cdr:.3f}  linear_CDR={linear_cdr:.3f}  "
            f"disc_r={disc_r:.1f}px"
        )
        return result

    def run_batch(self, image_paths: list[Union[str, Path]]) -> list[CDRResult]:
        """Run on a list of images and return a list of CDRResult objects."""
        return [self.run(p) for p in image_paths]
