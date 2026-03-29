"""
papila_loader.py — Load PAPILA Excel files into PAPILAParticipant records.

PAPILA provides two files per eye (OD / OS), each with one row per patient.
This loader merges them by subject ID and returns a dict keyed by subject ID.

File layout (after skipping the two merged-header rows)
-------------------------------------------------------
Col  Field               Notes
---  -----               -----
0    ID                  string, e.g. "#002"
1    Age                 int
2    Gender              0=M, 1=F
3    Diagnosis           0=Healthy, 1=Glaucoma, 2=Suspect
4    Refraction_D1       dioptre (sphere)
5    Refraction_D2       dioptre (cylinder)
6    Astigmatism         axis in degrees
7    Phakic              0=phakic, 1=pseudophakic
8    IOP_Pneumatic       mmHg (preferred)
9    IOP_Perkins         mmHg (alternate)
10   Pachymetry          µm
11   Axial_Length        mm
12   VF_MD               mean deviation dB (sparse — many NaN)

Fundus image naming convention (PAPILA)
---------------------------------------
Images are named by subject ID and laterality:
  RET{ID}_{laterality}.jpg   e.g. RET002_OD.jpg
The loader populates BaseFundusImage.filename using this convention.
Adjust FUNDUS_PATTERN in the caller if your directory uses a different scheme.

Usage
-----
    from papila_loader import load_papila_excel, participants_to_dataframe

    participants = load_papila_excel(
        od_path="patient_data_od.xlsx",
        os_path="patient_data_os.xlsx",   # optional
    )
    # Returns dict[str, PAPILAParticipant] keyed by subject ID (e.g. "#002")

    df = participants_to_dataframe(participants)   # shared causal nodes only
    df_full = participants_to_dataframe(participants, full=True)  # all fields
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from base_models import BaseDemographics, BaseDiagnosis, BaseFundusImage
from papila_models import (
    PAPILA_DIAGNOSIS_MAP,
    PAPILA_GENDER_MAP,
    PAPILAEyeData,
    PAPILAOcularData,
    PAPILAParticipant,
    RefractiveError,
)

# ── Constants ────────────────────────────────────────────────────────────────

COL_NAMES = [
    "ID", "Age", "Gender", "Diagnosis",
    "Refraction_D1", "Refraction_D2", "Astigmatism",
    "Phakic",
    "IOP_Pneumatic", "IOP_Perkins",
    "Pachymetry", "Axial_Length",
    "VF_MD",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_int(val) -> Optional[int]:
    f = _parse_float(val)
    return int(f) if f is not None else None


def _read_papila_file(path: Path) -> pd.DataFrame:
    """
    Read one PAPILA Excel file, skipping the two merged-header rows,
    and return a clean DataFrame with COL_NAMES columns.
    """
    raw = pd.read_excel(path, skiprows=2, header=None)
    raw = raw.dropna(how="all")          # drop fully-empty rows
    raw.columns = COL_NAMES[:raw.shape[1]]
    # Drop any residual header-text rows (ID == "ID" or similar)
    raw = raw[raw["ID"].notna() & (raw["ID"].astype(str) != "ID")]
    raw = raw.reset_index(drop=True)
    return raw


def _fundus_filename(subject_id: str, laterality: str) -> str:
    """
    Derive the expected fundus image filename from subject ID and laterality.
    PAPILA convention: RET{numeric_id}_{laterality}.jpg
    e.g. "#002" → "RET002_OD.jpg"
    """
    numeric = re.sub(r"[^0-9]", "", subject_id)
    return f"RET{numeric}_{laterality}.jpg"


def _build_eye(row: pd.Series, laterality: str) -> PAPILAEyeData:
    """Build a PAPILAEyeData from one row of the cleaned DataFrame."""
    subject_id = str(row["ID"])

    # Prefer Pneumatic IOP; fall back to Perkins
    iop_pneumatic = _parse_float(row.get("IOP_Pneumatic"))
    iop_perkins   = _parse_float(row.get("IOP_Perkins"))
    iop_preferred = iop_pneumatic if iop_pneumatic is not None else iop_perkins

    ocular = PAPILAOcularData(
        pachymetry_um   = _parse_float(row.get("Pachymetry")),
        axial_length_mm = _parse_float(row.get("Axial_Length")),
        refractive_error = RefractiveError(
            dioptre_1   = _parse_float(row.get("Refraction_D1")),
            dioptre_2   = _parse_float(row.get("Refraction_D2")),
            astigmatism = _parse_float(row.get("Astigmatism")),
        ),
        phakic      = (int(row["Phakic"]) == 0) if _parse_int(row.get("Phakic")) is not None else None,
        iop_perkins = iop_perkins,
    )

    eye = PAPILAEyeData(
        laterality = laterality,
        iop        = iop_preferred,
        vf_md      = _parse_float(row.get("VF_MD")),
        fundus     = BaseFundusImage(
            filename   = _fundus_filename(subject_id, laterality),
            laterality = laterality,
        ),
        ocular = ocular,
    )
    # Trigger iop_corrected calculation
    if eye.iop is not None and ocular.pachymetry_um is not None:
        eye.iop_corrected = ocular.iop_corrected(eye.iop)

    return eye


def _build_participant(
    subject_id: str,
    od_row: Optional[pd.Series],
    os_row: Optional[pd.Series],
) -> PAPILAParticipant:
    """
    Construct a PAPILAParticipant from OD and/or OS rows.
    Demographics and diagnosis are taken from whichever row is available
    (they should be identical between the two files for the same patient).
    """
    source_row = od_row if od_row is not None else os_row

    # Demographics
    gender_raw = _parse_int(source_row.get("Gender"))
    demographics = BaseDemographics(
        age    = _parse_int(source_row.get("Age")),
        gender = PAPILA_GENDER_MAP.get(gender_raw) if gender_raw is not None else None,
    )

    # Diagnosis — harmonise to GlaucomaLabel
    dx_raw = _parse_int(source_row.get("Diagnosis"))
    diagnosis = BaseDiagnosis(
        glaucoma_label  = PAPILA_DIAGNOSIS_MAP.get(dx_raw) if dx_raw is not None else None,
        source_category = str(int(dx_raw)) if dx_raw is not None else None,
    )

    participant = PAPILAParticipant(
        subject_id   = subject_id,
        dataset      = "PAPILA",
        demographics = demographics,
        diagnosis    = diagnosis,
    )

    if od_row is not None:
        participant.eyes["OD"] = _build_eye(od_row, "OD")
    if os_row is not None:
        participant.eyes["OS"] = _build_eye(os_row, "OS")

    participant.compute_asymmetry()
    return participant


# ── Public API ────────────────────────────────────────────────────────────────

def load_papila_excel(
    od_path: str | Path,
    os_path: Optional[str | Path] = None,
) -> dict[str, PAPILAParticipant]:
    """
    Load PAPILA data from one or two Excel files.

    Parameters
    ----------
    od_path  : path to the OD (right eye) Excel file.
    os_path  : path to the OS (left eye) Excel file.  Optional — if omitted,
               only OD data is loaded.  Pass the OS file to enable cross-eye
               asymmetry features and bilateral analysis.

    Returns
    -------
    dict[str, PAPILAParticipant] keyed by subject ID string (e.g. "#002").
    """
    od_path = Path(od_path)
    od_df   = _read_papila_file(od_path)
    od_map  = {str(row["ID"]): row for _, row in od_df.iterrows()}

    os_map: dict[str, pd.Series] = {}
    if os_path is not None:
        os_df  = _read_papila_file(Path(os_path))
        os_map = {str(row["ID"]): row for _, row in os_df.iterrows()}

    all_ids = sorted(set(od_map) | set(os_map))
    participants: dict[str, PAPILAParticipant] = {}

    for sid in all_ids:
        participants[sid] = _build_participant(
            subject_id = sid,
            od_row     = od_map.get(sid),
            os_row     = os_map.get(sid),
        )

    print(
        f"Loaded {len(participants)} PAPILA participants  "
        f"({sum(1 for p in participants.values() if 'OD' in p.eyes)} OD, "
        f"{sum(1 for p in participants.values() if 'OS' in p.eyes)} OS, "
        f"{sum(1 for p in participants.values() if len(p.eyes)==2)} bilateral)"
    )
    return participants


def participants_to_dataframe(
    participants: dict[str, PAPILAParticipant],
    full: bool = False,
) -> pd.DataFrame:
    """
    Convert loaded participants to a DataFrame.

    Parameters
    ----------
    participants : output of load_papila_excel()
    full         : if False (default), emit only the shared causal DAG node
                   columns (safe to concatenate with GRAPE data).
                   If True, emit all PAPILA-specific fields as well.

    Returns
    -------
    pd.DataFrame with one row per participant.
    """
    rows = []
    for p in participants.values():
        if full:
            rows.append(p.to_flat_dict())
        else:
            rows.append(p.causal_node_dict())
    return pd.DataFrame(rows)
