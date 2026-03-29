"""
glaucoma_dag.py — Causal DAG for glaucoma progression analysis.

Defines the directed acyclic graph (DAG) over the shared and
dataset-specific feature sets from GRAPE and PAPILA.

Structure
---------
The DAG is organised into five layers:

  1. Exogenous (root nodes)
     Age, Gender, Dataset
     No parents; represent background patient characteristics.

  2. Structural confounders
     Pachymetry  → biases raw IOP measurement
     Axial_Length → biases apparent CDR from fundus images (myopic tilted disc)
     Phakic       → affects IOP measurement (IOL implant changes reading)
     These are PAPILA-specific nodes; in GRAPE the analogue is CCT.

  3. Intermediate / mediator nodes
     IOP_corrected (deconfounded IOP — PAPILA only)
     Refraction / Spherical_Equivalent (downstream of Axial_Length)

  4. Biomarker nodes (image-derived + clinical)
     Vascular:   AVR, CRAE, CRVE, Fractal_D0, Vessel_Density
     Structural: CDR (linear, vertical), RDR, RNFL (GRAPE only)
     Geometric:  DF_Angle, DF_Distance (disc-fovea geometry)

  5. Outcome nodes
     VF_MD         — mean visual field deviation (shared across datasets)
     GlaucomaLabel — harmonised diagnosis ("Healthy"/"Suspect"/"Glaucoma")
     Progression   — binary progression flag (GRAPE only, longitudinal)

Key causal pathways
-------------------
  IOP → optic nerve damage → CDR ↑, RNFL ↓, RDR ↓ → VF_MD ↓
  Age → vascular ageing → AVR ↓ → optic nerve perfusion ↓ → VF_MD ↓
  Pachymetry → IOP_measured bias (thin cornea underestimates true IOP)
  Axial_Length → disc_appearance_bias → apparent CDR ↑ (confounder)
  AVR ↔ CDR (shared pathway through optic nerve head vasculature)

Nodes present in both datasets
-------------------------------
  Age, Gender, IOP, AVR, CRAE, CRVE, Fractal_D0, Vessel_Density,
  CDR_linear, CDR_vertical, RDR, DF_Angle, VF_MD, GlaucomaLabel,
  IOP_Asymmetry, VF_MD_Asymmetry, CDR_Asymmetry

PAPILA-only nodes (conditionally observed)
------------------------------------------
  Pachymetry, IOP_corrected, Axial_Length, Refraction_SE,
  Phakic, Pachymetry_Asymmetry, AxialLength_Asymmetry

GRAPE-only nodes (conditionally observed)
-----------------------------------------
  RNFL_mean, CCT, Progression, VF_pattern (pointwise)

Usage — load and validate pipeline outputs:
  python glaucoma_dag.py \\
      --grape-csv outputs/grape_results_all.csv \\
      --papila-csv outputs/papila/papila_results_all.csv \\
      --output combined_dag_data.csv

Usage — print node detail:
  python glaucoma_dag.py --node IOP_corrected
  python glaucoma_dag.py --node CDR_vertical

Usage — causal effect estimation (requires --grape-csv / --papila-csv):
  python glaucoma_dag.py \\
      --grape-csv outputs/grape_results_all.csv \\
      --papila-csv outputs/papila/papila_results_all.csv \\
      --treatment IOP_corrected --outcome VF_MD

Usage — DAG summary only (no data):
  python glaucoma_dag.py --summary
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd


# ── Node metadata ─────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    """Metadata for one node in the causal DAG."""
    name:        str
    layer:       str           # "exogenous" | "confounder" | "intermediate" |
                               #  "biomarker" | "outcome"
    datasets:    list[str]     # ["GRAPE", "PAPILA"] or subset
    feature_key: str           # column name in the analysis DataFrame
    description: str = ""
    unit:        str = ""
    is_observed: bool = True   # False = latent node (e.g. true IOP)


# ── Node definitions ──────────────────────────────────────────────────────────

NODES: list[DAGNode] = [

    # ── 1. Exogenous ──────────────────────────────────────────────────────────
    DAGNode("Age",          "exogenous",    ["GRAPE","PAPILA"], "age",
            "Age at baseline (years)"),
    DAGNode("Gender",       "exogenous",    ["GRAPE","PAPILA"], "gender",
            "Binary sex (M/F)"),
    DAGNode("Dataset",      "exogenous",    ["GRAPE","PAPILA"], "dataset",
            "Data source — accounts for acquisition differences"),

    # ── 2. Structural confounders ─────────────────────────────────────────────
    DAGNode("Pachymetry",   "confounder",   ["PAPILA"],         "od_pachymetry_um",
            "Central corneal thickness", "µm"),
    DAGNode("CCT",          "confounder",   ["GRAPE"],          "od_cct",
            "Central corneal thickness (GRAPE equivalent of Pachymetry)", "µm"),
    DAGNode("Axial_Length", "confounder",   ["PAPILA"],         "od_axial_length_mm",
            "Eye axial length — longer = more myopic; biases disc appearance", "mm"),
    DAGNode("Phakic",       "confounder",   ["PAPILA"],         "od_phakic",
            "Lens status: True=natural lens, False=IOL implant"),

    # ── 3. Intermediate / mediator ────────────────────────────────────────────
    DAGNode("IOP",          "intermediate", ["GRAPE","PAPILA"], "od_iop",
            "Raw measured intraocular pressure", "mmHg"),
    DAGNode("IOP_corrected","intermediate", ["PAPILA"],         "od_iop_corrected",
            "Pachymetry-adjusted IOP (deconfounded); GRAPE uses raw IOP", "mmHg"),
    DAGNode("Refraction_SE","intermediate", ["PAPILA"],         "od_refraction_se",
            "Spherical equivalent refractive error", "D"),

    # ── 4. Biomarker nodes ────────────────────────────────────────────────────

    # Vascular
    DAGNode("AVR",           "biomarker",   ["GRAPE","PAPILA"], "od_avr",
            "Artery-vein ratio (Knudtson) — lower = more arteriolar narrowing"),
    DAGNode("CRAE",          "biomarker",   ["GRAPE","PAPILA"], "od_crae",
            "Central retinal artery equivalent", "µm"),
    DAGNode("CRVE",          "biomarker",   ["GRAPE","PAPILA"], "od_crve",
            "Central retinal vein equivalent", "µm"),
    DAGNode("Fractal_D0",    "biomarker",   ["GRAPE","PAPILA"], "od_fractal_d0",
            "Box-counting fractal dimension of vessel network"),
    DAGNode("Vessel_Density","biomarker",   ["GRAPE","PAPILA"], "od_vessel_density_norm",
            "Normalised vessel density in zone-B ROI"),

    # Structural disc
    DAGNode("CDR_linear",    "biomarker",   ["GRAPE","PAPILA"], "od_linear_cdr",
            "Cup-to-disc diameter ratio (area-derived)"),
    DAGNode("CDR_vertical",  "biomarker",   ["GRAPE","PAPILA"], "od_vcdr",
            "Vertical cup-to-disc ratio (bounding-box) — clinically preferred"),
    DAGNode("RDR",           "biomarker",   ["GRAPE","PAPILA"], "od_rdr",
            "Rim-to-disc ratio — lower = more rim loss"),
    DAGNode("RNFL_mean",     "biomarker",   ["GRAPE"],          "od_rnfl_mean",
            "Mean RNFL thickness from OCT", "µm"),

    # Geometric
    DAGNode("DF_Angle",      "biomarker",   ["GRAPE","PAPILA"], "od_df_angle_deg",
            "Disc-fovea angle (clockwise from positive-x, y-down)", "deg"),
    DAGNode("DF_Distance",   "biomarker",   ["GRAPE","PAPILA"], "od_df_distance_norm",
            "Disc-fovea distance normalised by image diagonal"),

    # Cross-eye asymmetry
    DAGNode("IOP_Asymmetry", "biomarker",   ["GRAPE","PAPILA"], "iop_asymmetry",
            "abs(OD IOP - OS IOP)", "mmHg"),
    DAGNode("VF_Asymmetry",  "biomarker",   ["GRAPE","PAPILA"], "vf_md_asymmetry",
            "abs(OD VF_MD - OS VF_MD)", "dB"),
    DAGNode("CDR_Asymmetry", "biomarker",   ["GRAPE","PAPILA"], "cdr_asymmetry",
            "abs(OD linear_CDR - OS linear_CDR)"),

    # ── 5. Outcome nodes ──────────────────────────────────────────────────────
    DAGNode("VF_MD",         "outcome",     ["GRAPE","PAPILA"], "od_vf_md",
            "Visual field mean deviation (PAPILA) / mean sensitivity (GRAPE — no MD in source data)", "dB"),
    DAGNode("GlaucomaLabel", "outcome",     ["GRAPE","PAPILA"], "glaucoma_label",
            "Harmonised diagnosis: Healthy / Suspect / Glaucoma"),
    DAGNode("Progression",   "outcome",     ["GRAPE"],          "any_progression",
            "Binary progression flag (PLR2/PLR3/MD) — longitudinal GRAPE only"),
]

NODE_MAP: dict[str, DAGNode] = {n.name: n for n in NODES}


# ── Edge definitions ──────────────────────────────────────────────────────────
# Each tuple: (parent, child, rationale)

EDGES: list[tuple[str, str, str]] = [

    # ── Exogenous → everything ────────────────────────────────────────────────
    ("Age",     "IOP",           "IOP tends to increase with age"),
    ("Age",     "AVR",           "Arteriolar narrowing is an age-related process"),
    ("Age",     "Fractal_D0",    "Vascular complexity decreases with age"),
    ("Age",     "RNFL_mean",     "RNFL thins physiologically with age"),
    ("Age",     "VF_MD",         "Age-related sensitivity loss independent of glaucoma"),
    ("Gender",  "IOP",           "IOP differs by sex (post-menopausal women higher)"),
    ("Gender",  "CDR_linear",    "Disc size and CDR differ slightly by sex"),
    ("Dataset", "IOP",           "Acquisition protocol differs between studies"),
    ("Dataset", "CDR_linear",    "Camera/resolution affects segmentation"),
    ("Dataset", "AVR",           "Image quality and preprocessing affect biomarkers"),

    # ── Structural confounders ─────────────────────────────────────────────────
    ("Pachymetry",   "IOP",           "Thin cornea underestimates true IOP (Goldmann bias)"),
    ("CCT",          "IOP",           "GRAPE equivalent: CCT confounds IOP measurement"),
    ("Phakic",       "IOP",           "IOL implants alter IOP measurement dynamics"),
    ("Axial_Length", "CDR_linear",    "Longer eyes have tilted discs — mimics large CDR"),
    ("Axial_Length", "CDR_vertical",  "Same tilt effect on vertical CDR"),
    ("Axial_Length", "Refraction_SE", "Axial length is the main determinant of refractive error"),
    ("Axial_Length", "DF_Angle",      "Tilted disc changes disc-fovea geometry"),

    # ── IOP → optic nerve damage pathway ─────────────────────────────────────
    ("IOP",           "CDR_linear",   "Elevated IOP expands cup by damaging rim tissue"),
    ("IOP",           "CDR_vertical", "Inferior pole preferentially damaged → vertical CDR ↑"),
    ("IOP",           "RDR",          "Rim loss is direct consequence of IOP-driven damage"),
    ("IOP",           "RNFL_mean",    "IOP causes RGC axon loss → RNFL thinning"),
    ("IOP",           "VF_MD",        "IOP → optic nerve damage → VF loss (mediated)"),
    ("IOP_corrected", "CDR_linear",   "Deconfounded IOP — cleaner effect on disc"),
    ("IOP_corrected", "CDR_vertical", ""),
    ("IOP_corrected", "RNFL_mean",    ""),
    ("IOP_corrected", "VF_MD",        ""),

    # IOP correction pathway
    ("Pachymetry",   "IOP_corrected", "Pachymetry used to compute corrected IOP"),
    ("IOP",          "IOP_corrected", "Corrected IOP derived from raw IOP"),

    # ── Vascular pathway ──────────────────────────────────────────────────────
    ("AVR",          "VF_MD",         "Arteriolar narrowing reduces optic nerve perfusion"),
    ("AVR",          "RNFL_mean",     "Vascular supply sustains RGC axons"),
    ("CRAE",         "AVR",           "AVR = CRAE / CRVE"),
    ("CRVE",         "AVR",           "AVR = CRAE / CRVE"),
    ("IOP",          "AVR",           "Elevated IOP compresses retinal arterioles"),
    ("IOP_corrected","AVR",           "Same via true IOP"),
    ("Fractal_D0",   "VF_MD",         "Reduced vascular complexity → impaired perfusion"),
    ("Vessel_Density","VF_MD",        "Lower vessel density in zone-B → reduced supply"),
    ("Fractal_D0",   "AVR",           "Fractal dimension reflects arteriolar narrowing"),

    # ── Structural disc → functional outcome ──────────────────────────────────
    ("CDR_linear",   "VF_MD",         "Cup enlargement reflects RGC loss → VF defect"),
    ("CDR_vertical", "VF_MD",         "Vertical CDR more sensitive to arcuate VF loss"),
    ("RDR",          "VF_MD",         "Rim loss directly reduces axon count"),
    ("RNFL_mean",    "VF_MD",         "RNFL thickness is a direct proxy for axon count"),
    ("CDR_linear",   "CDR_vertical",  "Vertical CDR is a projection of overall CDR"),

    # ── Asymmetry nodes ───────────────────────────────────────────────────────
    ("IOP",          "IOP_Asymmetry", "Asymmetry computed from bilateral IOP"),
    ("IOP_Asymmetry","VF_MD",         "Unilateral IOP elevation drives asymmetric VF loss"),
    ("CDR_linear",   "CDR_Asymmetry", "CDR asymmetry is itself a glaucoma signal"),
    ("CDR_Asymmetry","GlaucomaLabel", "Asymmetric CDR is clinically diagnostic"),
    ("VF_MD",        "VF_Asymmetry",  "Computed from bilateral VF_MD"),
    ("VF_Asymmetry", "GlaucomaLabel", "Asymmetric VF loss distinguishes glaucoma from other causes"),

    # ── Outcomes ──────────────────────────────────────────────────────────────
    ("VF_MD",        "GlaucomaLabel", "Functional loss informs diagnostic classification"),
    ("CDR_linear",   "GlaucomaLabel", "Structural damage informs diagnosis"),
    ("CDR_vertical", "GlaucomaLabel", ""),
    ("RNFL_mean",    "GlaucomaLabel", "RNFL thinning informs diagnosis (GRAPE)"),
    ("VF_MD",        "Progression",   "Worsening VF_MD defines progression (GRAPE)"),
    ("RNFL_mean",    "Progression",   "RNFL loss is a progression criterion"),

    # ── Geometric nodes ───────────────────────────────────────────────────────
    ("DF_Angle",     "CDR_linear",    "Disc orientation affects apparent CDR measurement"),
    ("DF_Distance",  "CDR_linear",    "Disc-fovea distance relates to disc position anomalies"),
]


# ── DAG class ─────────────────────────────────────────────────────────────────

class GlaucomaDAG:
    """
    Causal DAG for glaucoma analysis over GRAPE + PAPILA data.
    """

    def __init__(self, nodes: list[DAGNode], edges: list[tuple]):
        self.G = nx.DiGraph()

        for n in nodes:
            self.G.add_node(
                n.name,
                layer=n.layer,
                datasets=n.datasets,
                feature_key=n.feature_key,
                description=n.description,
                unit=n.unit,
            )

        for parent, child, *_ in edges:
            if parent in self.G and child in self.G:
                self.G.add_edge(parent, child)
            else:
                missing = [x for x in (parent, child) if x not in self.G]
                print(f"  [DAG warning] unknown node(s): {missing}")

        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if not nx.is_directed_acyclic_graph(self.G):
            cycles = list(nx.simple_cycles(self.G))
            raise ValueError(f"DAG contains cycles: {cycles}")
        print(
            f"DAG validated: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges, acyclic ✓"
        )

    # ── Causal query helpers ──────────────────────────────────────────────────

    def parents(self, node: str) -> list[str]:
        return list(self.G.predecessors(node))

    def children(self, node: str) -> list[str]:
        return list(self.G.successors(node))

    def ancestors(self, node: str) -> set[str]:
        return nx.ancestors(self.G, node)

    def descendants(self, node: str) -> set[str]:
        return nx.descendants(self.G, node)

    def backdoor_paths(self, treatment: str, outcome: str) -> list[list]:
        undirected = self.G.to_undirected()
        paths = list(nx.all_simple_paths(undirected, treatment, outcome))
        return [p for p in paths if len(p) > 1 and p[1] in self.parents(treatment)]

    def minimal_adjustment_set(self, treatment: str, outcome: str) -> set[str]:
        parents_of_treatment = set(self.parents(treatment))
        descendants_of_treatment = self.descendants(treatment)
        return parents_of_treatment - descendants_of_treatment - {outcome}

    def dataset_subgraph(self, dataset: str) -> nx.DiGraph:
        nodes = [
            n for n, data in self.G.nodes(data=True)
            if dataset in data.get("datasets", [])
        ]
        return self.G.subgraph(nodes)

    # ── DoWhy integration ─────────────────────────────────────────────────────

    def to_dowhy_model(self, df: pd.DataFrame, treatment: str, outcome: str):
        """
        Build a DoWhy CausalModel for a specific (treatment, outcome) query.
        df should be the output of load_pipeline_outputs() — columns are
        feature_key values (e.g. "od_avr"), renamed internally to node names.
        """
        from dowhy import CausalModel
        renamed = df.rename(columns={n.feature_key: n.name for n in NODES})
        return CausalModel(
            data=renamed,
            treatment=treatment,
            outcome=outcome,
            graph=self._to_gml_string(),
        )

    def _to_gml_string(self) -> str:
        lines = ["graph [", "  directed 1"]
        for i, node in enumerate(self.G.nodes()):
            lines.append(f'  node [ id {i} label "{node}" ]')
        node_ids = {n: i for i, n in enumerate(self.G.nodes())}
        for src, dst in self.G.edges():
            lines.append(f'  edge [ source {node_ids[src]} target {node_ids[dst]} ]')
        lines.append("]")
        return "\n".join(lines)

    # ── Summary / inspection ──────────────────────────────────────────────────

    def summary(self) -> None:
        print(f"\n{'─'*60}")
        print(f"Glaucoma Causal DAG")
        print(f"{'─'*60}")
        print(f"  Nodes : {self.G.number_of_nodes()}")
        print(f"  Edges : {self.G.number_of_edges()}")
        print()

        by_layer: dict[str, list[str]] = {}
        for n, d in self.G.nodes(data=True):
            by_layer.setdefault(d["layer"], []).append(n)

        for layer in ["exogenous","confounder","intermediate","biomarker","outcome"]:
            nodes = by_layer.get(layer, [])
            if nodes:
                print(f"  [{layer}]")
                for n in nodes:
                    meta = NODE_MAP[n]
                    ds = "/".join(meta.datasets)
                    print(f"    {n:<22} ({ds})"
                          + (f"  {meta.unit}" if meta.unit else ""))
        print()

        print("  Key causal paths to VF_MD:")
        for path_nodes in [
            ["IOP_corrected","CDR_vertical","VF_MD"],
            ["IOP_corrected","RNFL_mean","VF_MD"],
            ["IOP_corrected","AVR","VF_MD"],
            ["Age","AVR","VF_MD"],
            ["Pachymetry","IOP_corrected","CDR_vertical","VF_MD"],
            ["Axial_Length","CDR_linear","VF_MD"],
        ]:
            valid = all(
                self.G.has_edge(path_nodes[i], path_nodes[i+1])
                for i in range(len(path_nodes)-1)
            )
            print(f"    {'✓' if valid else '✗'} {' → '.join(path_nodes)}")
        print(f"{'─'*60}\n")

    def node_info(self, node: str) -> None:
        if node not in NODE_MAP:
            print(f"Unknown node '{node}'. Available: {sorted(NODE_MAP)}")
            return
        meta = NODE_MAP[node]
        print(f"\nNode: {node}")
        print(f"  Layer:       {meta.layer}")
        print(f"  Datasets:    {', '.join(meta.datasets)}")
        print(f"  Feature key: {meta.feature_key}  (CSV column name)")
        print(f"  Description: {meta.description}"
              + (f"  [{meta.unit}]" if meta.unit else ""))
        print(f"  Parents:     {self.parents(node) or '—'}")
        print(f"  Children:    {self.children(node) or '—'}")
        adj = self.minimal_adjustment_set(node, "VF_MD")
        print(f"  Adjustment set (→VF_MD): {sorted(adj) or '—'}")


# ── Factory ───────────────────────────────────────────────────────────────────

def build_dag() -> GlaucomaDAG:
    return GlaucomaDAG(NODES, EDGES)


# ── Pipeline output loading ───────────────────────────────────────────────────

# Mapping from GRAPE CSV column names (unprefixed) to DAG feature keys (od_-prefixed).
# PAPILA columns are already od_-prefixed so need no renaming.
GRAPE_COLUMN_RENAMES: dict[str, str] = {
    "iop":                 "od_iop",
    "vf_md":               "od_vf_md",
    "avr":                 "od_avr",
    "crae":                "od_crae",
    "crve":                "od_crve",
    "fractal_d0":          "od_fractal_d0",
    "vessel_density_norm": "od_vessel_density_norm",
    "linear_cdr":          "od_linear_cdr",
    "vcdr":                "od_vcdr",
    "rdr":                 "od_rdr",
    "df_angle_deg":        "od_df_angle_deg",
    "df_distance_norm":    "od_df_distance_norm",
    "area_cdr":            "od_area_cdr",
    "cfp_file":            "od_cfp_file",
    # Note: 'laterality' is kept as-is (not renamed) so _compute_grape_cdr_asymmetry
    # can reference it, and so downstream code can always find the laterality column.
}


def _normalise_papila(df: pd.DataFrame) -> pd.DataFrame:
    """
    PAPILA produces two rows per subject (one OD image, one OS image), where
    each row only carries pipeline metrics for the eye that was processed.
    Collapse to one row per subject by forward-filling od_* from the OD row
    and os_* from the OS row, then deduplicating on subject_id.

    Shared columns (age, gender, glaucoma_label, asymmetry fields, etc.) are
    taken from whichever row is non-null — they are identical between the two rows.
    """
    if "subject_id" not in df.columns:
        return df

    od_cols = [c for c in df.columns if c.startswith("od_")]
    os_cols = [c for c in df.columns if c.startswith("os_")]
    shared  = [c for c in df.columns if not c.startswith("od_") and not c.startswith("os_")]

    rows = []
    for subj_id, grp in df.groupby("subject_id", sort=False):
        merged = {}
        # Shared columns: first non-null value
        for col in shared:
            vals = grp[col].dropna()
            merged[col] = vals.iloc[0] if len(vals) > 0 else None
        # od_ columns: take from the OD image row
        od_row = grp[grp["image_id"].str.upper().str.contains("OD", na=False)] if "image_id" in grp.columns else grp
        os_row = grp[grp["image_id"].str.upper().str.contains("OS", na=False)] if "image_id" in grp.columns else grp
        for col in od_cols:
            vals = od_row[col].dropna() if len(od_row) > 0 else pd.Series(dtype=float)
            merged[col] = vals.iloc[0] if len(vals) > 0 else None
        for col in os_cols:
            vals = os_row[col].dropna() if len(os_row) > 0 else pd.Series(dtype=float)
            merged[col] = vals.iloc[0] if len(vals) > 0 else None
        rows.append(merged)

    return pd.DataFrame(rows)


def _compute_grape_cdr_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CDR asymmetry for GRAPE from the CSV directly.
    The pipeline batch backfill doesn't work for GRAPE because CDR values
    live on GRAPEEyeExam (inside exam_record) not participant.eyes.

    GRAPE is longitudinal: match OD and OS images by subject + visit number,
    both encoded in the image_id stem (e.g. "100od2" = subject 100, OD, visit 2).
    """
    import re as _re
    if "linear_cdr" not in df.columns or "image_id" not in df.columns:
        return df

    # Parse subject number, laterality, and visit from image_id
    parsed = df["image_id"].str.extract(r"^(\d+)(od|os)(\d+)$", flags=_re.IGNORECASE)
    df = df.copy()
    df["_subj"]  = parsed[0].astype(float)
    df["_lat"]   = parsed[1].str.upper()
    df["_visit"] = parsed[2].astype(float)

    od  = (df[df["_lat"] == "OD"][["_subj", "_visit", "linear_cdr"]]
           .rename(columns={"linear_cdr": "_cdr_od"}))
    os_ = (df[df["_lat"] == "OS"][["_subj", "_visit", "linear_cdr"]]
           .rename(columns={"linear_cdr": "_cdr_os"}))

    asym = od.merge(os_, on=["_subj", "_visit"], how="inner")
    asym["cdr_asymmetry"] = (asym["_cdr_od"] - asym["_cdr_os"]).abs()

    df = df.drop(columns=["cdr_asymmetry"], errors="ignore")
    # Re-parse into df so we can merge on _subj+_visit
    parsed2 = df["image_id"].str.extract(r"^(\d+)(od|os)(\d+)$", flags=_re.IGNORECASE)
    df["_subj"]  = parsed2[0].astype(float)
    df["_visit"] = parsed2[2].astype(float)
    df = df.merge(
        asym[["_subj", "_visit", "cdr_asymmetry"]],
        on=["_subj", "_visit"],
        how="left",
    )
    df = df.drop(columns=["_subj", "_visit"], errors="ignore")

    n_filled = df["cdr_asymmetry"].notna().sum()
    print(f"  GRAPE CDR asymmetry computed for {n_filled}/{len(df)} rows "
          f"(visit-matched OD+OS pairs)")
    return df


def load_pipeline_outputs(
    grape_csv: Optional[Path] = None,
    papila_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and combine GRAPE and/or PAPILA pipeline results CSVs into a single
    DataFrame aligned with the DAG node feature keys (all od_-prefixed).

    GRAPE columns are unprefixed in the CSV (e.g. "iop", "avr") and are renamed
    to od_-prefixed DAG keys on load. PAPILA columns are already od_-prefixed.

    PAPILA produces two rows per subject (one OD image, one OS image). These are
    collapsed to one row per subject so asymmetry fields and bilateral metrics are
    available on every row.

    Returns one row per participant, with a 'dataset' column.
    """
    frames = []

    if grape_csv is not None:
        grape_csv = Path(grape_csv)
        if not grape_csv.exists():
            sys.exit(f"[ERROR] GRAPE CSV not found: {grape_csv}")
        df = pd.read_csv(grape_csv)
        if "dataset" not in df.columns:
            df["dataset"] = "GRAPE"
        # Compute CDR asymmetry before renaming (uses 'laterality' and 'linear_cdr')
        df = _compute_grape_cdr_asymmetry(df)
        # Rename unprefixed columns to od_-prefixed DAG keys
        df = df.rename(columns={k: v for k, v in GRAPE_COLUMN_RENAMES.items()
                                 if k in df.columns})
        frames.append(df)
        print(f"Loaded GRAPE  : {len(df):>4} rows  ({grape_csv})")

    if papila_csv is not None:
        papila_csv = Path(papila_csv)
        if not papila_csv.exists():
            sys.exit(f"[ERROR] PAPILA CSV not found: {papila_csv}")
        df = pd.read_csv(papila_csv)
        if "dataset" not in df.columns:
            df["dataset"] = "PAPILA"
        # Collapse 2 rows/subject (OD image + OS image) into 1 row/subject
        n_before = len(df)
        df = _normalise_papila(df)
        print(f"Loaded PAPILA : {len(df):>4} rows  ({papila_csv})  "
              f"[collapsed from {n_before} image rows]")
        frames.append(df)

    if not frames:
        sys.exit("[ERROR] No CSV files provided. Use --grape-csv and/or --papila-csv.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Combined      : {len(combined):>4} rows  "
          f"({combined['dataset'].value_counts().to_dict()})")
    return combined


def coverage_report(df: pd.DataFrame) -> None:
    """
    Print per-node data coverage: how many rows have a non-null value,
    broken down by dataset. Flags nodes with <50% coverage as warnings.
    """
    datasets = sorted(df["dataset"].dropna().unique()) if "dataset" in df.columns else []
    n_total  = len(df)

    print(f"\n{'─'*70}")
    print(f"DAG Node Coverage Report  ({n_total} total rows)")
    print(f"{'─'*70}")

    header = f"  {'Node':<22} {'Feature key':<28} {'Total':>6}"
    for ds in datasets:
        header += f"  {ds:>8}"
    print(header)
    print(f"  {'─'*22} {'─'*28} {'─'*6}" + "  ────────" * len(datasets))

    layer_order = ["exogenous","confounder","intermediate","biomarker","outcome"]
    by_layer: dict[str, list[DAGNode]] = {}
    for node in NODES:
        by_layer.setdefault(node.layer, []).append(node)

    for layer in layer_order:
        layer_nodes = by_layer.get(layer, [])
        if not layer_nodes:
            continue
        print(f"\n  [{layer}]")
        for node in layer_nodes:
            col = node.feature_key
            if col not in df.columns:
                total_pct = 0.0
                ds_pcts = {ds: 0.0 for ds in datasets}
            else:
                total_pct = df[col].notna().mean() * 100
                ds_pcts = {}
                for ds in datasets:
                    mask = df["dataset"] == ds if "dataset" in df.columns else slice(None)
                    sub = df.loc[mask, col] if isinstance(mask, pd.Series) else df[col]
                    ds_pcts[ds] = sub.notna().mean() * 100 if len(sub) > 0 else 0.0

            warn = "⚠" if total_pct < 50 and col in df.columns else " "
            missing = " (not in CSV)" if col not in df.columns else ""

            line = (f"  {warn} {node.name:<22} {col:<28} "
                    f"{total_pct:5.0f}%")
            for ds in datasets:
                line += f"  {ds_pcts.get(ds, 0.0):6.0f}%"
            print(line + missing)

    print(f"\n  ⚠ = present in CSV but <50% non-null")
    print(f"{'─'*70}\n")


def _numeric_cols(df: pd.DataFrame, nodes: list[str]) -> list[str]:
    """Return feature_key columns for the given node names that are numeric."""
    cols = []
    for name in nodes:
        if name in NODE_MAP:
            col = NODE_MAP[name].feature_key
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
    return cols


def descriptive_stats(df: pd.DataFrame) -> None:
    """Print mean ± std for each numeric DAG node, per dataset."""
    print(f"\n{'─'*70}")
    print("Descriptive Statistics (mean ± std, numeric DAG nodes only)")
    print(f"{'─'*70}")

    datasets = sorted(df["dataset"].dropna().unique()) if "dataset" in df.columns else ["all"]

    for layer in ["intermediate","biomarker","outcome"]:
        layer_nodes = [n for n in NODES if n.layer == layer]
        numeric_rows = []
        for node in layer_nodes:
            col = node.feature_key
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            row = f"  {node.name:<22}"
            for ds in datasets:
                sub = df.loc[df["dataset"] == ds, col] if "dataset" in df.columns else df[col]
                sub = sub.dropna()
                if len(sub) == 0:
                    row += f"  {'n/a':>16}"
                else:
                    row += f"  {sub.mean():7.3f} ± {sub.std():6.3f}"
            numeric_rows.append(row)

        if numeric_rows:
            header = f"  {'Node':<22}" + "".join(f"  {ds:>16}" for ds in datasets)
            print(f"\n  [{layer}]")
            print(header)
            for r in numeric_rows:
                print(r)

    print(f"{'─'*70}\n")


# ── Causal effect estimation ──────────────────────────────────────────────────

def estimate_effect(
    dag: GlaucomaDAG,
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    method: str = "backdoor.linear_regression",
) -> None:
    """
    Identify and estimate the causal effect of treatment on outcome using DoWhy.

    Parameters
    ----------
    dag       : GlaucomaDAG
    df        : Combined pipeline output DataFrame (feature_key columns)
    treatment : DAG node name (e.g. "IOP_corrected")
    outcome   : DAG node name (e.g. "VF_MD")
    method    : DoWhy estimation method string
    """
    try:
        from dowhy import CausalModel
    except ImportError:
        sys.exit("[ERROR] dowhy not installed. Run: pip install dowhy")

    if treatment not in NODE_MAP:
        sys.exit(f"[ERROR] Unknown treatment node '{treatment}'. "
                 f"Available: {sorted(NODE_MAP)}")
    if outcome not in NODE_MAP:
        sys.exit(f"[ERROR] Unknown outcome node '{outcome}'. "
                 f"Available: {sorted(NODE_MAP)}")

    t_col = NODE_MAP[treatment].feature_key
    o_col = NODE_MAP[outcome].feature_key

    for col, label in [(t_col, "treatment"), (o_col, "outcome")]:
        if col not in df.columns:
            sys.exit(f"[ERROR] {label} column '{col}' not found in data. "
                     f"Check that the relevant pipeline has been run.")

    # Keep only rows with both treatment and outcome observed
    analysis_df = df[[t_col, o_col] + [
        NODE_MAP[n].feature_key for n in dag.minimal_adjustment_set(treatment, outcome)
        if NODE_MAP[n].feature_key in df.columns
    ]].dropna(subset=[t_col, o_col])

    n_dropped = len(df) - len(analysis_df)
    print(f"\n{'─'*60}")
    print(f"Causal Effect Estimation")
    print(f"  Treatment : {treatment}  [{t_col}]")
    print(f"  Outcome   : {outcome}  [{o_col}]")
    print(f"  Rows used : {len(analysis_df)}  ({n_dropped} dropped for missing treatment/outcome)")
    adj = dag.minimal_adjustment_set(treatment, outcome)
    print(f"  Adjustment set: {sorted(adj) or '∅'}")
    print(f"{'─'*60}")

    if len(analysis_df) < 10:
        print(f"[WARNING] Only {len(analysis_df)} rows with complete treatment+outcome data. "
              "Estimates may be unreliable.")

    model = dag.to_dowhy_model(analysis_df, treatment, outcome)

    print("\nIdentifying causal effect...")
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified)

    print(f"\nEstimating effect (method: {method})...")
    estimate = model.estimate_effect(
        identified,
        method_name=method,
        target_units="ate",
    )
    print(estimate)

    print(f"\nRefutation (placebo treatment)...")
    try:
        refute = model.refute_estimate(
            identified, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
        )
        print(refute)
    except Exception as e:
        print(f"  Refutation failed: {e}")

    print(f"{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="glaucoma_dag.py",
        description="Glaucoma causal DAG — build, inspect, and analyse pipeline outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Load outputs, print coverage report, save combined CSV
              python glaucoma_dag.py \\
                  --grape-csv outputs/grape_results_all.csv \\
                  --papila-csv outputs/papila/papila_results_all.csv \\
                  --output combined_dag_data.csv

              # Coverage + descriptive stats
              python glaucoma_dag.py \\
                  --grape-csv outputs/grape_results_all.csv \\
                  --papila-csv outputs/papila/papila_results_all.csv \\
                  --stats

              # Causal effect estimation
              python glaucoma_dag.py \\
                  --grape-csv outputs/grape_results_all.csv \\
                  --papila-csv outputs/papila/papila_results_all.csv \\
                  --treatment IOP_corrected --outcome VF_MD

              # Inspect a specific DAG node (no data needed)
              python glaucoma_dag.py --node CDR_vertical

              # Print DAG structure only
              python glaucoma_dag.py --summary
        """),
    )

    # Data inputs
    data = parser.add_argument_group("data inputs")
    data.add_argument(
        "--grape-csv", type=Path, metavar="CSV",
        help="Path to grape_results_all.csv from run_pipeline.py",
    )
    data.add_argument(
        "--papila-csv", type=Path, metavar="CSV",
        help="Path to papila_results_all.csv from run_papila_pipeline.py",
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=Path, metavar="CSV",
        help="Save combined analysis-ready DataFrame to this CSV",
    )

    # Actions
    actions = parser.add_argument_group("actions")
    actions.add_argument(
        "--summary", action="store_true",
        help="Print DAG structure and exit (no data needed)",
    )
    actions.add_argument(
        "--node", metavar="NODE_NAME",
        help="Print detail for a specific DAG node and exit (e.g. IOP_corrected)",
    )
    actions.add_argument(
        "--stats", action="store_true",
        help="Print descriptive statistics per node per dataset",
    )
    actions.add_argument(
        "--treatment", metavar="NODE_NAME",
        help="DAG node name to use as treatment for causal effect estimation",
    )
    actions.add_argument(
        "--outcome", metavar="NODE_NAME",
        default="VF_MD",
        help="DAG node name to use as outcome (default: VF_MD)",
    )
    actions.add_argument(
        "--method", metavar="METHOD",
        default="backdoor.linear_regression",
        help="DoWhy estimation method (default: backdoor.linear_regression)",
    )

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    dag  = build_dag()

    # ── Node detail (no data needed) ──────────────────────────────────────────
    if args.node:
        dag.node_info(args.node)
        return

    # ── DAG summary only ──────────────────────────────────────────────────────
    if args.summary:
        dag.summary()
        return

    # ── All other actions require at least one CSV ────────────────────────────
    if not args.grape_csv and not args.papila_csv:
        # Default: print summary and hint
        dag.summary()
        print("Tip: pass --grape-csv and/or --papila-csv to load pipeline outputs.")
        return

    # Load and combine CSVs
    print()
    df = load_pipeline_outputs(args.grape_csv, args.papila_csv)

    # Always print coverage
    coverage_report(df)

    # Descriptive stats
    if args.stats:
        descriptive_stats(df)

    # Save combined CSV
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved combined data: {args.output}  ({len(df)} rows × {len(df.columns)} cols)")

    # Causal effect estimation
    if args.treatment:
        estimate_effect(dag, df, args.treatment, args.outcome, args.method)


if __name__ == "__main__":
    main()
