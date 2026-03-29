from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class VascXResult:
    image_id: str
    output_dir: Path
    quality: Optional[pd.Series] = None
    fovea: Optional[pd.Series] = None

    @property
    def artery_vein_path(self) -> Path:
        return self.output_dir / "artery_vein.png"

    @property
    def vessels_path(self) -> Path:
        return self.output_dir / "vessels.png"

    @property
    def disc_path(self) -> Path:
        return self.output_dir / "disc.png"

    def load_av_mask(self) -> np.ndarray:
        """Returns array with: 0=background, 1=artery, 2=vein, 3=crossing."""
        return np.array(Image.open(self.artery_vein_path))

    def load_artery_mask(self, include_crossings: bool = False) -> np.ndarray:
        """
        Returns binary artery mask.
        If include_crossings=True, junction pixels (value 3) are counted as artery.
        """
        av = self.load_av_mask()
        mask = av == 1
        if include_crossings:
            mask = mask | (av == 3)
        return mask.astype("uint8")

    def load_vein_mask(self, include_crossings: bool = False) -> np.ndarray:
        """
        Returns binary vein mask.
        If include_crossings=True, junction pixels (value 3) are counted as vein.
        """
        av = self.load_av_mask()
        mask = av == 2
        if include_crossings:
            mask = mask | (av == 3)
        return mask.astype("uint8")

    def load_vessel_mask(self) -> np.ndarray:
        """All vessels (arteries + veins + crossings)."""
        return (self.load_av_mask() >= 1).astype("uint8")

    def load_disc_mask(self) -> np.ndarray:
        return np.array(Image.open(self.disc_path))

    def save_disc_mask(
            self,
    ) -> Path:
        mask = self.load_disc_mask()
        path = self.output_dir / "optic_disc.png"
        Image.fromarray(mask * 255).save(path)
        return path
        

    
    def save_artery_mask(
        self,
        crossings: Literal["exclude", "artery", "vein"] = "exclude",
    ) -> Path:
        """
        Saves a binary artery-only PNG to the output directory.
        crossings controls how junction pixels (value 3) are handled:
          - "exclude": junction pixels are background
          - "artery":  junction pixels are counted as artery
          - "vein":    junction pixels are counted as artery (excluded from this mask)
        Returns the path to the saved file.
        """
        include = crossings == "artery"
        mask = self.load_artery_mask(include_crossings=include)
        path = self.output_dir / f"artery_crossings_{crossings}.png"
        Image.fromarray(mask * 255).save(path)
        return path

    def save_vein_mask(
        self,
        crossings: Literal["exclude", "artery", "vein"] = "exclude",
    ) -> Path:
        """
        Saves a binary vein-only PNG to the output directory.
        crossings controls how junction pixels (value 3) are handled:
          - "exclude": junction pixels are background
          - "vein":    junction pixels are counted as vein
          - "artery":  junction pixels are excluded from this mask
        Returns the path to the saved file.
        """
        include = crossings == "vein"
        mask = self.load_vein_mask(include_crossings=include)
        path = self.output_dir / f"vein_crossings_{crossings}.png"
        Image.fromarray(mask * 255).save(path)
        return path
