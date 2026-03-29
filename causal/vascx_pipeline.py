import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_inference import (
    ClassificationEnsemble,
    HeatmapRegressionEnsemble,
    SegmentationEnsemble,
)
from rtnls_fundusprep.preprocessor import parallel_preprocess

from vascx_models import VascXResult

QUALITY_LABELS = {0: "quality_good", 1: "quality_usable", 2: "quality_reject"}

class VascXPipeline:
    """
    Runs VascX models for a single image, writing intermediate preprocessed
    files (RGB + CE) to a temporary directory that is cleaned up automatically,
    and saving only the final segmentation outputs to per-image output folders.

    Usage:
        pipeline = VascXPipeline(output_root="outputs", device="cuda")
        result = pipeline.run("path/to/image.jpg")
        artery = result.load_artery_mask()
    """

    def __init__(
        self,
        output_root: str | Path = "outputs",
        device: str = "cuda",
        run_vessels: bool = True,
        run_av: bool = True,
        run_disc: bool = True,
        run_fovea: bool = True,
        run_quality: bool = True,
        num_workers: int = 2,
        preprocess_jobs: int = 4,
    ):
        self.output_root = Path(output_root)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.run_vessels = run_vessels
        self.run_av = run_av
        self.run_disc = run_disc
        self.run_fovea = run_fovea
        self.run_quality = run_quality
        self.num_workers = num_workers
        self.preprocess_jobs = preprocess_jobs

        self._models: dict = {}

    def _load_models(self):
        """Lazy-load models once on first use."""
        if self._models:
            return
        d = self.device
        if self.run_av:
            self._models["av"] = SegmentationEnsemble.from_huggingface(
                "Eyened/vascx:artery_vein/av_july24.pt"
            ).to(d)
        if self.run_disc:
            self._models["disc"] = SegmentationEnsemble.from_huggingface(
                "Eyened/vascx:disc/disc_july24.pt"
            ).to(d)
        if self.run_fovea:
            self._models["fovea"] = HeatmapRegressionEnsemble.from_huggingface(
                "Eyened/vascx:fovea/fovea_july24.pt"
            ).to(d)
        if self.run_quality:
            from huggingface_hub import list_repo_files
            quality_files = [
                f for f in list_repo_files("Eyened/vascx")
                if "quality" in f and f.endswith(".pt")
            ]
            if quality_files:
                self._models["quality"] = ClassificationEnsemble.from_huggingface(
                    f"Eyened/vascx:{quality_files[0]}"
                ).to(d)

    def _preprocess(
        self,
        image_path: Path,
        tmp_rgb_dir: Path,
        tmp_ce_dir: Path,
    ) -> tuple[Path, Path]:
        bounds = parallel_preprocess(
            [image_path],
            rgb_path=tmp_rgb_dir,
            ce_path=tmp_ce_dir,
            n_jobs=1,
        )
        if not bounds or not bounds[0].get("success", True):
            raise RuntimeError(f"Preprocessing failed for {image_path}")

        stem = image_path.stem
        return tmp_rgb_dir / f"{stem}.png", tmp_ce_dir / f"{stem}.png"

    def _preprocess_batch(
        self,
        image_paths: list[Path],
        tmp_rgb_dir: Path,
        tmp_ce_dir: Path,
    ) -> list[tuple[Path, Path]]:
        bounds = parallel_preprocess(
            image_paths,
            rgb_path=tmp_rgb_dir,
            ce_path=tmp_ce_dir,
            n_jobs=self.preprocess_jobs,
        )
        failed = [b["id"] for b in bounds if not b.get("success", True)]
        if failed:
            raise RuntimeError(f"Preprocessing failed for: {failed}")

        return [
            (tmp_rgb_dir / f"{p.stem}.png", tmp_ce_dir / f"{p.stem}.png")
            for p in image_paths
        ]

    def _run_av(
        self,
        paired_paths: list,
        image_stem: str,
        tmp_av_dir: Path,
        out_dir: Path,
    ):
        self._models["av"].predict_preprocessed(
            paired_paths,
            dest_path=str(tmp_av_dir),
            num_workers=self.num_workers,
        )
        tmp_file = tmp_av_dir / f"{image_stem}.png"
        if not tmp_file.exists():
            return
        shutil.copy(tmp_file, out_dir / "artery_vein.png")
        if self.run_vessels:
            av = np.array(Image.open(tmp_file))
            vessel_mask = ((av >= 1) * 255).astype("uint8")
            Image.fromarray(vessel_mask).save(out_dir / "vessels.png")

    def _run_disc(
        self,
        paired_paths: list,
        image_stem: str,
        tmp_disc_dir: Path,
        out_dir: Path,
    ):
        self._models["disc"].predict_preprocessed(
            paired_paths,
            dest_path=str(tmp_disc_dir),
            num_workers=self.num_workers,
        )
        tmp_file = tmp_disc_dir / f"{image_stem}.png"
        if tmp_file.exists():
            shutil.copy(tmp_file, out_dir / "disc.png")

    def _run_fovea(
        self,
        paired_paths: list,
        out_dir: Path,
    ) -> Optional[pd.Series]:
        df = self._models["fovea"].predict_preprocessed(
            paired_paths,
            num_workers=self.num_workers,
        )
        if df is not None:
            df.to_csv(out_dir / "fovea.csv", index=False)
            return df.iloc[0]
        return None


    def _run_quality(
            self,
            rgb_paths: list[str],   # RGB only — quality model is 3-channel ResNet
            out_dir: Path,
    ) -> Optional[pd.Series]:
        df = self._models["quality"].predict_preprocessed(
            rgb_paths,
            num_workers=self.num_workers,
        )
        if df is None:
            return None
 
        # Rename integer columns → readable names
        df = df.rename(columns=QUALITY_LABELS)
 
        # Add "quality" column: label of the highest-probability class
        prob_cols = list(QUALITY_LABELS.values())          # ordered good→usable→reject
        df["quality"] = df[prob_cols].idxmax(axis=1).str.replace("quality_", "")
 
        df.to_csv(out_dir / "quality.csv", index=False)
        return df.iloc[0]    

    
    def run(self, image_path: str | Path) -> VascXResult:
        """
        Process a single fundus image. Intermediates go to a tempdir;
        outputs go to output_root / image_id /.
        """
        image_path = Path(image_path)
        image_id = image_path.stem
        out_dir = self.output_root / image_id
        out_dir.mkdir(parents=True, exist_ok=True)

        self._load_models()

        with tempfile.TemporaryDirectory(prefix="vascx_tmp_") as tmp_root:
            tmp_root = Path(tmp_root)
            tmp_rgb = tmp_root / "preprocessed_rgb"
            tmp_ce = tmp_root / "preprocessed_ce"
            tmp_av = tmp_root / "artery_vein"
            tmp_disc = tmp_root / "disc"
            for d in (tmp_rgb, tmp_ce, tmp_av, tmp_disc):
                d.mkdir()

            rgb_path, ce_path = self._preprocess(image_path, tmp_rgb, tmp_ce)
            paired_paths = [(str(rgb_path), str(ce_path))]

            result = VascXResult(image_id=image_id, output_dir=out_dir)

            if self.run_av and "av" in self._models:
                self._run_av(paired_paths, image_path.stem, tmp_av, out_dir)
            if self.run_disc and "disc" in self._models:
                self._run_disc(paired_paths, image_path.stem, tmp_disc, out_dir)
            if self.run_fovea and "fovea" in self._models:
                result.fovea = self._run_fovea(paired_paths, out_dir)
            if self.run_quality and "quality" in self._models:
                result.quality = self._run_quality([str(rgb_path)], out_dir)

        return result

    def run_batch(self, image_paths: list[str | Path]) -> list[VascXResult]:
        """
        Process multiple images. Preprocessing is parallelised across the
        whole batch (CPU-heavy); GPU inference then runs per image.
        """
        image_paths = [Path(p) for p in image_paths]
        self._load_models()

        with tempfile.TemporaryDirectory(prefix="vascx_tmp_") as tmp_root:
            tmp_root = Path(tmp_root)
            tmp_rgb = tmp_root / "preprocessed_rgb"
            tmp_ce = tmp_root / "preprocessed_ce"
            tmp_av = tmp_root / "artery_vein"
            tmp_disc = tmp_root / "disc"
            for d in (tmp_rgb, tmp_ce, tmp_av, tmp_disc):
                d.mkdir()

            preprocessed = self._preprocess_batch(image_paths, tmp_rgb, tmp_ce)

            results = []
            for image_path, (rgb_path, ce_path) in zip(image_paths, preprocessed):
                image_id = image_path.stem
                out_dir = self.output_root / image_id
                out_dir.mkdir(parents=True, exist_ok=True)

                paired_paths = [(str(rgb_path), str(ce_path))]
                result = VascXResult(image_id=image_id, output_dir=out_dir)

                if self.run_av and "av" in self._models:
                    self._run_av(paired_paths, image_path.stem, tmp_av, out_dir)
                if self.run_disc and "disc" in self._models:
                    self._run_disc(paired_paths, image_path.stem, tmp_disc, out_dir)
                if self.run_fovea and "fovea" in self._models:
                    result.fovea = self._run_fovea(paired_paths, out_dir)
                if self.run_quality and "quality" in self._models:
                    result.quality = self._run_quality([str(rgb_path)], out_dir)                    

                results.append(result)

        return results
