"""
base_models.py — Shared causal node schema for glaucoma datasets.

This module defines the dataset-agnostic core that both GRAPE and PAPILA
(and any future dataset) populate.  Dataset-specific models import these
base classes and extend them with their own fields.

Causal node alignment
---------------------
The fields defined here correspond directly to nodes in the causal DAG that
are estimable from both datasets:

  Demographics ─── Age, Gender
  Diagnosis    ─── GlaucomaLabel (harmonised across datasets)
  BaseEyeData  ─── IOP, VF_MD, FundusImage, ComputedMetrics (CDR, AVR, ...)
  BaseParticipant ─ cross-eye asymmetry features (IOP, VF_MD, CDR)

Dataset-specific nodes (Pachymetry, AxialLength, FullVFMap, RNFL, ...)
are defined in the dataset submodels (grape_models, papila_models) and
attached to the causal DAG as conditionally-observed nodes.

Harmonised diagnosis labels
---------------------------
Both datasets use different coding schemes:
  GRAPE  : "OAG", "ACG", "NTG", "OHT"
  PAPILA : 0=Healthy, 1=Glaucoma, 2=Suspect

BaseDiagnosis stores a harmonised GlaucomaLabel ("Healthy", "Suspect",
"Glaucoma") that is comparable across datasets, alongside the original
dataset-specific category string for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from pvbm_models import PVBMResult
    from cdr_models import CDRResult

# ── Type aliases ──────────────────────────────────────────────────────────────

Laterality = Literal["OD", "OS"]
Dataset    = Literal["GRAPE", "PAPILA"]

# Harmonised 3-way label usable across datasets for the DAG outcome node.
# GRAPE OAG/ACG/NTG → "Glaucoma"
# GRAPE OHT         → "Suspect"   (elevated IOP without damage)
# PAPILA 0          → "Healthy"
# PAPILA 1          → "Glaucoma"
# PAPILA 2          → "Suspect"
GlaucomaLabel = Literal["Healthy", "Suspect", "Glaucoma"]


# ── Shared leaf models ────────────────────────────────────────────────────────

@dataclass
class BaseDemographics:
    """
    Subject-level demographic information.
    Both GRAPE and PAPILA provide age and binary gender.
    """
    age:    Optional[int] = None   # years at baseline / enrolment
    gender: Optional[str] = None   # "M" or "F"  (0→"M", 1→"F" in PAPILA)


@dataclass
class BaseDiagnosis:
    """
    Harmonised diagnostic classification comparable across datasets.

    glaucoma_label    : harmonised label used in the causal DAG outcome node
    source_category   : original dataset-specific value for traceability
                        (e.g. "OAG" for GRAPE, "1" for PAPILA)
    """
    glaucoma_label:  Optional[GlaucomaLabel] = None
    source_category: Optional[str] = None     # raw value from dataset

    @property
    def has_glaucoma(self) -> Optional[bool]:
        """True = confirmed glaucoma, False = healthy/suspect, None = unknown."""
        if self.glaucoma_label == "Glaucoma":
            return True
        if self.glaucoma_label in ("Healthy", "Suspect"):
            return False
        return None

    @property
    def is_suspect(self) -> Optional[bool]:
        return self.glaucoma_label == "Suspect" if self.glaucoma_label else None


@dataclass
class BaseFundusImage:
    """
    Minimal fundus image reference shared across datasets.
    Dataset-specific metadata (camera model, resolution) extends this in
    the submodels.
    """
    filename: Optional[str] = None   # image filename on disk
    laterality: Optional[Laterality] = None


@dataclass
class BaseComputedMetrics:
    """
    Image-derived pipeline outputs (VascX → PVBM + CDR + DiscMorphology).
    Identical structure for both datasets since the same pipeline runs on
    both.  Fields are Optional — populated post-hoc by the pipeline.

    The scalars exposed as properties are the causal DAG node values:
      avr, crae, crve      — vascular calibre / AVR
      area_cdr, linear_cdr — cup-disc ratio (area and diameter conventions)
      vcdr, rdr            — vertical CDR and rim-to-disc ratio
      fractal_d0           — vessel network fractal dimension
      vessel_density_norm  — normalised vessel density in zone-B
      df_angle_deg         — disc-fovea angle (structural geometry)
    """
    pvbm: Optional[Any] = None   # PVBMResult | None
    cdr:  Optional[Any] = None   # CDRResult  | None  (includes .morph)

    # ── Vascular biomarker accessors ─────────────────────────────────────────

    @property
    def avr(self) -> Optional[float]:
        return self.pvbm.avrk if self.pvbm is not None else None

    @property
    def crae(self) -> Optional[float]:
        if self.pvbm is not None and self.pvbm.artery_calibre is not None:
            return self.pvbm.artery_calibre.craek
        return None

    @property
    def crve(self) -> Optional[float]:
        if self.pvbm is not None and self.pvbm.vein_calibre is not None:
            return self.pvbm.vein_calibre.crvek
        return None

    @property
    def fractal_d0(self) -> Optional[float]:
        if self.pvbm is not None and self.pvbm.vessels is not None:
            return self.pvbm.vessels.fractal_d0
        return None

    @property
    def vessel_density_norm(self) -> Optional[float]:
        if self.pvbm is not None and self.pvbm.vessels is not None:
            return self.pvbm.vessels.vessel_density_norm
        return None

    # ── CDR accessors ────────────────────────────────────────────────────────

    @property
    def area_cdr(self) -> Optional[float]:
        return self.cdr.area_cdr if self.cdr is not None else None

    @property
    def linear_cdr(self) -> Optional[float]:
        return self.cdr.linear_cdr if self.cdr is not None else None

    @property
    def vcdr(self) -> Optional[float]:
        if self.cdr is not None and self.cdr.morph is not None:
            return self.cdr.morph.vcdr
        return None

    @property
    def rdr(self) -> Optional[float]:
        if self.cdr is not None and self.cdr.morph is not None:
            return self.cdr.morph.rdr
        return None

    @property
    def df_angle_deg(self) -> Optional[float]:
        if self.cdr is not None and self.cdr.morph is not None:
            return self.cdr.morph.df_angle_deg
        return None

    @property
    def df_distance_norm(self) -> Optional[float]:
        if self.cdr is not None and self.cdr.morph is not None:
            return self.cdr.morph.df_distance_norm
        return None

    def to_flat_dict(self, prefix: str = "") -> dict:
        pre = f"{prefix}_" if prefix else ""
        d: dict = {
            f"{pre}avr":                 self.avr,
            f"{pre}crae":                self.crae,
            f"{pre}crve":                self.crve,
            f"{pre}area_cdr":            self.area_cdr,
            f"{pre}linear_cdr":          self.linear_cdr,
            f"{pre}vcdr":                self.vcdr,
            f"{pre}rdr":                 self.rdr,
            f"{pre}df_angle_deg":        self.df_angle_deg,
            f"{pre}df_distance_norm":    self.df_distance_norm,
            f"{pre}fractal_d0":          self.fractal_d0,
            f"{pre}vessel_density_norm": self.vessel_density_norm,
        }
        if self.pvbm is not None:
            for k, v in self.pvbm.to_flat_dict().items():
                if k != "image_id":
                    d[f"{pre}pvbm_{k}"] = v
        if self.cdr is not None:
            for k, v in self.cdr.to_flat_dict().items():
                if k != "image_id":
                    d[f"{pre}cdr_{k}"] = v
        return d


# ── Per-eye shared data ───────────────────────────────────────────────────────

@dataclass
class BaseEyeData:
    """
    Shared causal node set for one eye.

    This is the unit of observation for the causal DAG — one row in the
    analysis table corresponds to one BaseEyeData (or subclass) instance.

    Fields
    ------
    laterality  : "OD" (right) or "OS" (left)
    iop         : intraocular pressure in mmHg — primary causal node
    vf_md       : visual field mean deviation (dB) — causal outcome proxy
                  GRAPE: computed from 61-point map
                  PAPILA: directly provided as VF_MD scalar
    fundus      : image reference for pipeline processing
    computed    : pipeline-derived biomarkers (CDR, AVR, fractal, ...)
    """
    laterality: Optional[Laterality] = None
    iop:        Optional[float] = None       # mmHg
    vf_md:      Optional[float] = None       # mean deviation dB; None if not measured

    fundus:   BaseFundusImage    = field(default_factory=BaseFundusImage)
    computed: BaseComputedMetrics = field(default_factory=BaseComputedMetrics)

    def causal_node_dict(self, prefix: str = "") -> dict:
        """
        Return only the shared causal DAG node values — safe to merge across
        datasets into a common analysis DataFrame.
        """
        pre = f"{prefix}_" if prefix else ""
        d: dict = {
            f"{pre}laterality":   self.laterality,
            f"{pre}iop":          self.iop,
            f"{pre}vf_md":        self.vf_md,
            f"{pre}cfp_file":     self.fundus.filename,
        }
        d.update(self.computed.to_flat_dict(prefix=pre.rstrip("_")))
        return d


# ── Participant (cross-dataset shared container) ──────────────────────────────

@dataclass
class BaseParticipant:
    """
    Top-level shared participant model.

    Subclassed by GRAPEParticipant and PAPILAParticipant which add
    dataset-specific fields.  All fields defined here are populated for
    every dataset.

    Cross-eye asymmetry features are computed here because they are shared
    causal nodes meaningful for both datasets.  They are populated by
    calling compute_asymmetry() after eyes have been loaded.
    """
    subject_id:  str = ""
    dataset:     Dataset = "GRAPE"

    demographics: BaseDemographics = field(default_factory=BaseDemographics)
    diagnosis:    BaseDiagnosis    = field(default_factory=BaseDiagnosis)

    # Keyed "OD" / "OS" — values are BaseEyeData or dataset subclass
    eyes: dict[str, Any] = field(default_factory=dict)

    # ── Cross-eye asymmetry (DAG nodes) ──────────────────────────────────────
    # Populated by compute_asymmetry() once both eyes are loaded.
    # abs(OD - OS) for each metric.
    iop_asymmetry:   Optional[float] = None
    vf_md_asymmetry: Optional[float] = None
    cdr_asymmetry:   Optional[float] = None   # linear_cdr asymmetry

    def compute_asymmetry(self) -> None:
        """
        Compute cross-eye asymmetry features from the loaded eye data.
        Safe to call even if only one eye is present (fields stay None).
        """
        od = self.eyes.get("OD")
        os = self.eyes.get("OS")

        if od is not None and os is not None:
            if od.iop is not None and os.iop is not None:
                self.iop_asymmetry = abs(od.iop - os.iop)

            if od.vf_md is not None and os.vf_md is not None:
                self.vf_md_asymmetry = abs(od.vf_md - os.vf_md)

            od_cdr = od.computed.linear_cdr if od.computed else None
            os_cdr = os.computed.linear_cdr if os.computed else None
            if od_cdr is not None and os_cdr is not None:
                self.cdr_asymmetry = abs(od_cdr - os_cdr)

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def od(self) -> Optional[Any]:
        return self.eyes.get("OD")

    @property
    def os(self) -> Optional[Any]:
        return self.eyes.get("OS")

    @property
    def age(self) -> Optional[int]:
        return self.demographics.age

    @property
    def gender(self) -> Optional[str]:
        return self.demographics.gender

    @property
    def has_glaucoma(self) -> Optional[bool]:
        return self.diagnosis.has_glaucoma

    @property
    def glaucoma_label(self) -> Optional[GlaucomaLabel]:
        return self.diagnosis.glaucoma_label

    # ── Serialisation ─────────────────────────────────────────────────────────

    def causal_node_dict(self) -> dict:
        """
        Flatten only the shared causal DAG nodes — safe to concatenate
        across datasets into a common analysis DataFrame.
        One row = one participant (uses baseline / single-visit eye data).
        For longitudinal data (GRAPE), call at each visit instead.
        """
        d: dict = {
            "subject_id":         self.subject_id,
            "dataset":            self.dataset,
            "age":                self.demographics.age,
            "gender":             self.demographics.gender,
            "glaucoma_label":     self.diagnosis.glaucoma_label,
            "has_glaucoma":       self.diagnosis.has_glaucoma,
            "source_category":    self.diagnosis.source_category,
            "iop_asymmetry":      self.iop_asymmetry,
            "vf_md_asymmetry":    self.vf_md_asymmetry,
            "cdr_asymmetry":      self.cdr_asymmetry,
        }
        for lat, eye in self.eyes.items():
            d.update(eye.causal_node_dict(prefix=lat.lower()))
        return d
