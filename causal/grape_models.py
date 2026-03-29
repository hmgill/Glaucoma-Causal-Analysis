"""
grape_models.py — GRAPE dataset-specific data model.

Extends base_models with GRAPE-specific clinical structure:
  - Full 61-point visual field map (vs PAPILA's scalar VF_MD only)
  - RNFL thickness (OCT, quadrant-level)
  - CCT (central corneal thickness — equivalent to PAPILA Pachymetry)
  - Longitudinal exam record (multiple visits per participant)
  - Progression labels (PLR2, PLR3, MD binary flags)
  - Detailed fundus image metadata (camera model, resolution)

Inheritance
-----------
GRAPEParticipant  ← BaseParticipant
GRAPEDiagnosis    ← BaseDiagnosis        (adds ProgressionStatus)
GRAPEEyeExam      ← BaseEyeData          (adds VisualField, FundusImageMetadata)

Shared causal nodes (via base classes)
---------------------------------------
  age, gender, glaucoma_label, iop, vf_md (derived from full map),
  fundus filename, computed pipeline metrics (CDR, AVR, fractal, ...),
  cross-eye asymmetry features

GRAPE-specific nodes
---------------------
  vf_map (61 points), vf_n_depressed, vf_n_missing
  rnfl_mean/superior/nasal/inferior/temporal
  cct (µm)  — analogous to PAPILA Pachymetry
  progression (plr2, plr3, md flags)
  visit_number, interval_years

Harmonised diagnosis mapping (GRAPE → GlaucomaLabel)
-----------------------------------------------------
  OAG, ACG, NTG → "Glaucoma"
  OHT           → "Suspect"
  (no "Healthy" category in GRAPE; all participants have ocular hypertension
   or confirmed glaucoma)

Data source alignment
---------------------
  Baseline sheet   → Demographics, Diagnosis, BaselineOcularData, visit-1 EyeExam
  Follow-Up sheet  → All EyeExam objects (visit 1 redundantly included)

Sentinel values
---------------
  VF value of -1 → blind spot / outside stimulus grid → stored as None
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import pandas as pd

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

if TYPE_CHECKING:
    from pvbm_models import PVBMResult
    from cdr_models import CDRResult

# ── Type aliases ──────────────────────────────────────────────────────────────

GlaucomaCategory = Literal["OAG", "ACG", "NTG", "OHT"]

# GRAPE glaucoma category → harmonised GlaucomaLabel for the DAG
GRAPE_DIAGNOSIS_MAP: dict[str, GlaucomaLabel] = {
    "OAG": "Glaucoma",
    "ACG": "Glaucoma",
    "NTG": "Glaucoma",
    "OHT": "Suspect",
}

VF_N_POINTS = 61  # Humphrey 24-2 equivalent


# ── GRAPE-specific leaf models ────────────────────────────────────────────────

@dataclass
class ProgressionStatus:
    """
    Pointwise linear regression (PLR) progression flags.
    plr2 / plr3 : binary (0=stable, 1=progressing) under two criteria
    md          : binary progression flag from mean deviation slope
    """
    plr2: Optional[int] = None
    plr3: Optional[int] = None
    md:   Optional[int] = None

    @property
    def any_progression(self) -> Optional[bool]:
        flags = [x for x in (self.plr2, self.plr3, self.md) if x is not None]
        return bool(any(flags)) if flags else None


@dataclass
class GRAPEDiagnosis(BaseDiagnosis):
    """
    Extends BaseDiagnosis with GRAPE-specific glaucoma category and
    longitudinal progression labels.

    glaucoma_category is the original GRAPE string ("OAG", "ACG", etc.).
    glaucoma_label (inherited) is the harmonised cross-dataset value.
    """
    glaucoma_category: Optional[GlaucomaCategory] = None
    progression: ProgressionStatus = field(default_factory=ProgressionStatus)

    @classmethod
    def from_category(cls, category: str) -> "GRAPEDiagnosis":
        """Build from a raw GRAPE category string, populating both fields."""
        label = GRAPE_DIAGNOSIS_MAP.get(category)
        return cls(
            glaucoma_label    = label,
            source_category   = category,
            glaucoma_category = category if category in GRAPE_DIAGNOSIS_MAP else None,
        )

    @property
    def has_glaucoma(self) -> Optional[bool]:
        if self.glaucoma_category in ("OAG", "ACG", "NTG"):
            return True
        if self.glaucoma_category == "OHT":
            return False
        return None

    @property
    def is_progressor(self) -> Optional[bool]:
        return self.progression.any_progression


@dataclass
class RNFLMeasurement:
    """
    Retinal nerve fibre layer (RNFL) thickness from OCT, in µm.
    Quadrant convention: S=superior, N=nasal, I=inferior, T=temporal.
    GRAPE-specific — PAPILA does not provide OCT RNFL.
    """
    mean:     Optional[float] = None
    superior: Optional[float] = None
    nasal:    Optional[float] = None
    inferior: Optional[float] = None
    temporal: Optional[float] = None

    def to_flat_dict(self, prefix: str = "rnfl") -> dict:
        return {
            f"{prefix}_mean":     self.mean,
            f"{prefix}_superior": self.superior,
            f"{prefix}_nasal":    self.nasal,
            f"{prefix}_inferior": self.inferior,
            f"{prefix}_temporal": self.temporal,
        }


@dataclass
class BaselineOcularData:
    """
    Per-eye measurements taken at baseline only (static across visits).
    GRAPE-specific structure.

    cct  : central corneal thickness (µm) — analogous to PAPILA Pachymetry.
           Note: GRAPE does not provide IOP correction from CCT; for the
           shared DAG node use PAPILA's iop_corrected where available.
    rnfl : OCT RNFL thickness by quadrant.
    """
    laterality: Optional[Laterality] = None
    cct:  Optional[float] = None
    rnfl: RNFLMeasurement = field(default_factory=RNFLMeasurement)

    def to_flat_dict(self, prefix: str = "") -> dict:
        pre = f"{prefix}_" if prefix else ""
        d: dict = {f"{pre}cct": self.cct}
        d.update(self.rnfl.to_flat_dict(prefix=f"{pre}rnfl"))
        return d


@dataclass
class GRAPEFundusImage(BaseFundusImage):
    """
    Extended fundus image metadata for GRAPE (camera model, resolution).
    Extends BaseFundusImage which holds filename and laterality.
    """
    camera:         Optional[str] = None
    resolution_str: Optional[str] = None
    width:          Optional[int] = None
    height:         Optional[int] = None
    fundus_type:    Optional[str] = "CFP"

    @classmethod
    def from_raw(
        cls, filename: str, camera: str,
        resolution_str: str, laterality: Optional[Laterality] = None,
    ) -> "GRAPEFundusImage":
        w, h = None, None
        if resolution_str and "×" in resolution_str:
            try:
                parts = resolution_str.split("×")
                w, h = int(parts[0].strip()), int(parts[1].strip())
            except (ValueError, IndexError):
                pass
        return cls(
            filename=filename, laterality=laterality,
            camera=camera, resolution_str=resolution_str,
            width=w, height=h, fundus_type="CFP",
        )

    @property
    def megapixels(self) -> Optional[float]:
        if self.width and self.height:
            return round(self.width * self.height / 1_000_000, 2)
        return None


@dataclass
class VisualField:
    """
    61-point visual field sensitivity values (dB) — GRAPE-specific.
    Index layout matches Humphrey 24-2 convention used in GRAPE.

    Sentinel -1 in raw data → stored as None (blind spot / out-of-grid).

    vf_md (the shared causal DAG node) is computed as mean_sensitivity
    and surfaced on GRAPEEyeExam so it maps cleanly to PAPILA's scalar.
    """
    values: list[Optional[float]] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw_values: list) -> "VisualField":
        cleaned = [
            float(v) if (v is not None and v != -1) else None
            for v in raw_values
        ]
        return cls(values=cleaned)

    @property
    def valid_values(self) -> list[float]:
        return [v for v in self.values if v is not None]

    @property
    def mean_sensitivity(self) -> Optional[float]:
        vals = self.valid_values
        return float(np.mean(vals)) if vals else None

    @property
    def n_depressed_points(self) -> int:
        return sum(1 for v in self.valid_values if v < 20)

    @property
    def n_missing(self) -> int:
        return sum(1 for v in self.values if v is None)

    def to_flat_dict(self, prefix: str = "vf") -> dict:
        return {f"{prefix}_{i}": v for i, v in enumerate(self.values)}


# ── GRAPE eye exam (per-visit, per-eye) ───────────────────────────────────────

@dataclass
class GRAPEEyeExam(BaseEyeData):
    """
    All data for one eye at one GRAPE visit.

    Extends BaseEyeData:
      iop        : intraocular pressure (mmHg) — shared DAG node
      vf_md      : mean visual field deviation — shared DAG node
                   populated from visual_field.mean_sensitivity on load
      fundus     : GRAPEFundusImage (extended metadata)
      computed   : BaseComputedMetrics (CDR, PVBM — populated post-hoc)

    GRAPE-specific additions:
      visual_field : full 61-point sensitivity map
    """
    visual_field: VisualField        = field(default_factory=VisualField)
    fundus:       GRAPEFundusImage   = field(default_factory=GRAPEFundusImage)
    computed:     BaseComputedMetrics = field(default_factory=BaseComputedMetrics)

    def __post_init__(self):
        # Keep vf_md (shared DAG node) in sync with the full map
        if self.vf_md is None and self.visual_field.values:
            self.vf_md = self.visual_field.mean_sensitivity

    def to_flat_dict(self, prefix: str = "") -> dict:
        """Full serialisation including GRAPE-specific VF map."""
        pre = f"{prefix}_" if prefix else ""
        d: dict = {
            f"{pre}laterality":      self.laterality,
            f"{pre}iop":             self.iop,
            f"{pre}vf_md":           self.vf_md,
            f"{pre}vf_n_depressed":  self.visual_field.n_depressed_points,
            f"{pre}vf_n_missing":    self.visual_field.n_missing,
            f"{pre}cfp_file":        self.fundus.filename,
            f"{pre}camera":          self.fundus.camera,
            f"{pre}resolution":      self.fundus.resolution_str,
        }
        # Full 61-point map
        d.update(self.visual_field.to_flat_dict(prefix=f"{pre}vf"))
        # Pipeline metrics
        d.update(self.computed.to_flat_dict(prefix=pre.rstrip("_")))
        return d


# ── Exam (one visit, both eyes) ───────────────────────────────────────────────

@dataclass
class Exam:
    """A single clinical visit. May contain one or both eyes."""
    visit_number:   int   = 1
    is_baseline:    bool  = True
    interval_years: float = 0.0
    eyes: dict[str, GRAPEEyeExam] = field(default_factory=dict)

    @property
    def od(self) -> Optional[GRAPEEyeExam]:
        return self.eyes.get("OD")

    @property
    def os(self) -> Optional[GRAPEEyeExam]:
        return self.eyes.get("OS")

    def to_flat_dict(self) -> dict:
        d = {
            "visit_number":   self.visit_number,
            "is_baseline":    self.is_baseline,
            "interval_years": self.interval_years,
        }
        for lat, eye in self.eyes.items():
            d.update(eye.to_flat_dict(prefix=lat.lower()))
        return d


@dataclass
class ExamRecord:
    """Ordered longitudinal visit container for one participant."""
    total_exams: int = 0
    exams: list[Exam] = field(default_factory=list)

    def __post_init__(self):
        if self.total_exams == 0 and self.exams:
            self.total_exams = len(self.exams)

    @property
    def baseline(self) -> Optional[Exam]:
        for e in self.exams:
            if e.is_baseline:
                return e
        return None

    @property
    def follow_ups(self) -> list[Exam]:
        return [e for e in self.exams if not e.is_baseline]

    def get_exam(self, visit_number: int) -> Optional[Exam]:
        for e in self.exams:
            if e.visit_number == visit_number:
                return e
        return None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_flat_dict() for e in self.exams])


# ── GRAPE participant ─────────────────────────────────────────────────────────

@dataclass
class GRAPEParticipant(BaseParticipant):
    """
    Top-level data model for a GRAPE study participant.

    Extends BaseParticipant with:
      - GRAPEDiagnosis      (adds glaucoma_category + progression flags)
      - BaselineOcularData  (per-eye CCT and RNFL — static)
      - ExamRecord          (longitudinal visit structure)
      - total_visits        (as reported in the dataset)

    The eyes dict (inherited) is not used for GRAPE — longitudinal eye data
    lives in exam_record.  Cross-eye asymmetry is computed from the baseline
    exam after loading.

    Shared causal node access
    -------------------------
    causal_node_dict() returns the cross-dataset comparable feature set
    at the baseline visit.  For longitudinal analysis iterate exam_record.
    """
    dataset:     str = "GRAPE"
    diagnosis:   GRAPEDiagnosis   = field(default_factory=GRAPEDiagnosis)
    exam_record: ExamRecord       = field(default_factory=ExamRecord)
    baseline_ocular: dict[str, BaselineOcularData] = field(default_factory=dict)
    total_visits: Optional[int] = None

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def od_baseline(self) -> Optional[BaselineOcularData]:
        return self.baseline_ocular.get("OD")

    @property
    def os_baseline(self) -> Optional[BaselineOcularData]:
        return self.baseline_ocular.get("OS")

    @property
    def n_exams(self) -> int:
        return self.exam_record.total_exams

    @property
    def is_progressor(self) -> Optional[bool]:
        return self.diagnosis.progression.any_progression

    def compute_asymmetry(self) -> None:
        """
        Compute cross-eye asymmetry from the baseline exam.
        Populates inherited iop_asymmetry, vf_md_asymmetry, cdr_asymmetry.
        """
        baseline = self.exam_record.baseline
        if baseline is None:
            return
        od = baseline.eyes.get("OD")
        os = baseline.eyes.get("OS")
        if od is None or os is None:
            return
        if od.iop is not None and os.iop is not None:
            self.iop_asymmetry = abs(od.iop - os.iop)
        if od.vf_md is not None and os.vf_md is not None:
            self.vf_md_asymmetry = abs(od.vf_md - os.vf_md)
        od_cdr = od.computed.linear_cdr
        os_cdr = os.computed.linear_cdr
        if od_cdr is not None and os_cdr is not None:
            self.cdr_asymmetry = abs(od_cdr - os_cdr)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def causal_node_dict(self) -> dict:
        """
        Shared causal DAG nodes at the baseline visit.
        Safe to concatenate with PAPILA causal_node_dict() output.
        """
        baseline = self.exam_record.baseline
        d: dict = {
            "subject_id":      self.subject_id,
            "dataset":         self.dataset,
            "age":             self.demographics.age,
            "gender":          self.demographics.gender,
            "glaucoma_label":  self.diagnosis.glaucoma_label,
            "has_glaucoma":    self.diagnosis.has_glaucoma,
            "source_category": self.diagnosis.source_category,
            "iop_asymmetry":   self.iop_asymmetry,
            "vf_md_asymmetry": self.vf_md_asymmetry,
            "cdr_asymmetry":   self.cdr_asymmetry,
        }
        if baseline is not None:
            for lat, eye in baseline.eyes.items():
                d.update(eye.causal_node_dict(prefix=lat.lower()))
        return d

    def to_flat_dict(self) -> dict:
        """
        Full participant summary (one row, baseline visit only).
        Use exam_record.to_dataframe() for longitudinal rows.
        """
        d: dict = {
            "subject_id":        self.subject_id,
            "dataset":           self.dataset,
            "age":               self.demographics.age,
            "gender":            self.demographics.gender,
            "glaucoma_label":    self.diagnosis.glaucoma_label,
            "glaucoma_category": self.diagnosis.glaucoma_category,
            "has_glaucoma":      self.diagnosis.has_glaucoma,
            "source_category":   self.diagnosis.source_category,
            "progression_plr2":  self.diagnosis.progression.plr2,
            "progression_plr3":  self.diagnosis.progression.plr3,
            "progression_md":    self.diagnosis.progression.md,
            "any_progression":   self.is_progressor,
            "total_visits":      self.total_visits,
            "n_exams_loaded":    self.n_exams,
            "iop_asymmetry":     self.iop_asymmetry,
            "vf_md_asymmetry":   self.vf_md_asymmetry,
            "cdr_asymmetry":     self.cdr_asymmetry,
        }
        for lat, bod in self.baseline_ocular.items():
            d.update(bod.to_flat_dict(prefix=lat.lower()))
        return d

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_flat_dict())

    def __repr__(self) -> str:
        return (
            f"GRAPEParticipant(subject_id={self.subject_id}, "
            f"gender={self.demographics.gender}, "
            f"age={self.demographics.age}, "
            f"dx={self.diagnosis.glaucoma_category}, "
            f"label={self.diagnosis.glaucoma_label}, "
            f"progressor={self.is_progressor}, "
            f"visits={self.n_exams})"
        )
