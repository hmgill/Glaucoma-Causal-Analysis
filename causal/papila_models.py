"""
papila_models.py — PAPILA dataset-specific data model.

Extends base_models with PAPILA-specific clinical fields that are not
present in GRAPE:
  - Pachymetry        (corneal thickness µm) — confounds raw IOP measurement
  - Axial length      (mm)                  — confounds disc appearance in myopia
  - Refractive error  (3 components)        — related to axial length
  - Phakic status     (lens present/absent) — affects IOP measurement
  - IOP_Perkins       (second tonometer)    — alternate IOP measurement
  - IOP_corrected     (Pachymetry-adjusted) — deconfounded IOP estimate

Causal DAG role
---------------
PAPILA is cross-sectional (single visit per patient), so there is no
longitudinal progression label.  The DAG outcome node is glaucoma_label
("Healthy" / "Suspect" / "Glaucoma").

Key causal additions over GRAPE:
  Pachymetry → IOP_corrected → (true IOP) → optic nerve damage
  AxialLength → disc_appearance_bias → (confounds CDR from images)
  Phakic → IOP_measurement_error → (confounds raw IOP)

Diagnosis coding (PAPILA raw)
------------------------------
  0 → "Healthy"   → GlaucomaLabel "Healthy"
  1 → "Glaucoma"  → GlaucomaLabel "Glaucoma"
  2 → "Suspect"   → GlaucomaLabel "Suspect"

Gender coding (PAPILA raw)
--------------------------
  0 → "M"
  1 → "F"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from base_models import (
    BaseComputedMetrics,
    BaseDemographics,
    BaseDiagnosis,
    BaseEyeData,
    BaseFundusImage,
    BaseParticipant,
    GlaucomaLabel,
    Laterality,
)

# ── Reference values ─────────────────────────────────────────────────────────

PACHYMETRY_REFERENCE_UM  = 545.0   # µm — population mean used in IOP correction
IOP_CORRECTION_PER_10UM  = 0.7     # mmHg per 10 µm deviation (Doughty & Zaman)

# PAPILA diagnosis code → harmonised GlaucomaLabel
PAPILA_DIAGNOSIS_MAP: dict[int, GlaucomaLabel] = {
    0: "Healthy",
    1: "Glaucoma",
    2: "Suspect",
}

PAPILA_GENDER_MAP: dict[int, str] = {0: "M", 1: "F"}


# ── PAPILA-specific leaf models ───────────────────────────────────────────────

@dataclass
class RefractiveError:
    """
    Refractive error decomposed into three components (dioptres).
    All three are needed to fully characterise the optical defect.

    dioptre_1    : sphere component (positive = hyperopia, negative = myopia)
    dioptre_2    : cylinder component (magnitude of astigmatism)
    astigmatism  : axis of astigmatism in degrees (0–180)

    Spherical equivalent = dioptre_1 + (dioptre_2 / 2); negative SE is myopia.
    """
    dioptre_1:   Optional[float] = None   # sphere (D)
    dioptre_2:   Optional[float] = None   # cylinder (D)
    astigmatism: Optional[float] = None   # axis (degrees)

    @property
    def spherical_equivalent(self) -> Optional[float]:
        """SE = sphere + cylinder/2.  Negative = myopic."""
        if self.dioptre_1 is not None and self.dioptre_2 is not None:
            return self.dioptre_1 + self.dioptre_2 / 2.0
        return None

    def to_flat_dict(self, prefix: str = "refraction") -> dict:
        return {
            f"{prefix}_d1":  self.dioptre_1,
            f"{prefix}_d2":  self.dioptre_2,
            f"{prefix}_axis": self.astigmatism,
            f"{prefix}_se":  self.spherical_equivalent,
        }


@dataclass
class PAPILAOcularData:
    """
    PAPILA-specific per-eye measurements absent from GRAPE.

    pachymetry_um    : central corneal thickness (µm).  Thin corneas
                       underestimate true IOP; used to compute iop_corrected.
    axial_length_mm  : eye axial length (mm).  Longer eyes (myopia) have
                       tilted discs that mimic glaucomatous CDR — a confounder
                       on image-derived features.
    refractive_error : sphere, cylinder, axis.
    phakic           : True = natural lens in place; False = pseudophakic
                       (IOL implant, e.g. post-cataract).  Affects IOP.
    iop_perkins      : IOP measured by Perkins applanation tonometer (mmHg).
                       PAPILA has both Pneumatic and Perkins; use whichever
                       is non-null, preferring Pneumatic.
    """
    pachymetry_um:   Optional[float] = None
    axial_length_mm: Optional[float] = None
    refractive_error: RefractiveError = field(default_factory=RefractiveError)
    phakic:          Optional[bool]  = None   # True = phakic
    iop_perkins:     Optional[float] = None   # mmHg (Perkins tonometer)

    def iop_corrected(self, iop_raw: Optional[float]) -> Optional[float]:
        """
        Pachymetry-adjusted IOP estimate.

        Applies the Doughty & Zaman (2000) correction:
          IOP_corrected = IOP_raw − 0.7 × (pachymetry − 545) / 10

        Returns None if either iop_raw or pachymetry_um is missing.
        This value is a causal DAG node that deconfounds the IOP → damage path.
        """
        if iop_raw is None or self.pachymetry_um is None:
            return None
        correction = IOP_CORRECTION_PER_10UM * (
            (self.pachymetry_um - PACHYMETRY_REFERENCE_UM) / 10.0
        )
        return round(iop_raw - correction, 2)

    def to_flat_dict(self, prefix: str = "") -> dict:
        pre = f"{prefix}_" if prefix else ""
        d: dict = {
            f"{pre}pachymetry_um":   self.pachymetry_um,
            f"{pre}axial_length_mm": self.axial_length_mm,
            f"{pre}phakic":          self.phakic,
            f"{pre}iop_perkins":     self.iop_perkins,
        }
        d.update(self.refractive_error.to_flat_dict(prefix=f"{pre}refraction"))
        return d


# ── PAPILA eye data ───────────────────────────────────────────────────────────

@dataclass
class PAPILAEyeData(BaseEyeData):
    """
    One eye's data for a PAPILA participant.

    Extends BaseEyeData with PAPILA-specific clinical measurements.
    The inherited fields (iop, vf_md, fundus, computed) are the shared
    causal DAG nodes.  The ocular field contains PAPILA-only additions.

    iop is set to IOP_Pneumatic when available, IOP_Perkins otherwise.
    iop_corrected is the Pachymetry-adjusted estimate stored for DAG use.
    """
    ocular:        PAPILAOcularData  = field(default_factory=PAPILAOcularData)
    iop_corrected: Optional[float]   = None   # Pachymetry-adjusted IOP (mmHg)

    def __post_init__(self):
        # Populate iop_corrected from ocular data whenever iop is set
        if self.iop is not None and self.ocular.pachymetry_um is not None:
            self.iop_corrected = self.ocular.iop_corrected(self.iop)

    def causal_node_dict(self, prefix: str = "") -> dict:
        """Override to add iop_corrected to the shared causal node set."""
        d = super().causal_node_dict(prefix=prefix)
        pre = f"{prefix}_" if prefix else ""
        d[f"{pre}iop_corrected"] = self.iop_corrected
        return d

    def to_flat_dict(self, prefix: str = "") -> dict:
        """Full serialisation including PAPILA-specific fields."""
        d = self.causal_node_dict(prefix=prefix)
        pre = f"{prefix}_" if prefix else ""
        d.update(self.ocular.to_flat_dict(prefix=pre.rstrip("_")))
        return d


# ── PAPILA participant ────────────────────────────────────────────────────────

@dataclass
class PAPILAParticipant(BaseParticipant):
    """
    Top-level data model for a PAPILA participant.

    PAPILA is cross-sectional: one visit, up to two eyes (OD / OS loaded
    from separate files).  Eyes are PAPILAEyeData instances.

    Dataset-level context
    ---------------------
    dataset     : always "PAPILA"
    subject_id  : string ID from the dataset (e.g. "#002")
    eyes        : dict keyed "OD" / "OS" — PAPILAEyeData instances

    Asymmetry features (iop_asymmetry, vf_md_asymmetry, cdr_asymmetry)
    are inherited from BaseParticipant and computed by calling
    compute_asymmetry() after both eyes are loaded.

    PAPILA-specific asymmetry additions
    ------------------------------------
    pachymetry_asymmetry : abs(OD - OS) corneal thickness (µm)
    axial_length_asymmetry : abs(OD - OS) axial length (mm)
    Both are clinically meaningful — unilateral thin cornea or long eye
    can drive unilateral glaucoma risk.
    """
    dataset: str = "PAPILA"

    # PAPILA-specific cross-eye asymmetry
    pachymetry_asymmetry:    Optional[float] = None   # µm
    axial_length_asymmetry:  Optional[float] = None   # mm
    iop_corrected_asymmetry: Optional[float] = None   # mmHg

    def compute_asymmetry(self) -> None:
        """Extend base asymmetry with PAPILA-specific fields."""
        super().compute_asymmetry()

        od: Optional[PAPILAEyeData] = self.eyes.get("OD")
        os: Optional[PAPILAEyeData] = self.eyes.get("OS")

        if od is not None and os is not None:
            od_pach = od.ocular.pachymetry_um
            os_pach = os.ocular.pachymetry_um
            if od_pach is not None and os_pach is not None:
                self.pachymetry_asymmetry = abs(od_pach - os_pach)

            od_al = od.ocular.axial_length_mm
            os_al = os.ocular.axial_length_mm
            if od_al is not None and os_al is not None:
                self.axial_length_asymmetry = abs(od_al - os_al)

            if od.iop_corrected is not None and os.iop_corrected is not None:
                self.iop_corrected_asymmetry = abs(
                    od.iop_corrected - os.iop_corrected
                )

    def to_flat_dict(self) -> dict:
        """Full serialisation for CSV / DataFrame row."""
        d = self.causal_node_dict()   # shared nodes
        d.update({
            # PAPILA-specific asymmetry
            "pachymetry_asymmetry":    self.pachymetry_asymmetry,
            "axial_length_asymmetry":  self.axial_length_asymmetry,
            "iop_corrected_asymmetry": self.iop_corrected_asymmetry,
            # Full per-eye data including PAPILA-specific fields
        })
        for lat, eye in self.eyes.items():
            d.update(eye.to_flat_dict(prefix=lat.lower()))
        return d

    def __repr__(self) -> str:
        eye_str = "/".join(self.eyes.keys())
        return (
            f"PAPILAParticipant(id={self.subject_id}, "
            f"age={self.demographics.age}, "
            f"label={self.diagnosis.glaucoma_label}, "
            f"eyes={eye_str})"
        )
