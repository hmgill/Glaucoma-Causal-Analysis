"""
run_papila_pipeline.py — Run VascX + PVBM + CDR pipelines for PAPILA images.

Image naming convention: RET{id}OD.jpg / RET{id}OS.jpg (or with underscores).
Clinical data is split across two Excel files — one for OD, one for OS.
Both files can be provided in a single run to enable bilateral asymmetry.

Output directory layout (outputs/papila/RET002_OD/):
  artery_vein.png, vessels.png, disc.png
  artery_crossings_artery.png, vein_crossings_vein.png
  quality.csv, fovea.csv
  segmap.png, disc_mask.png, cup_mask.png
  zones.png
  results.csv

Usage:
  # Both eyes (recommended — enables bilateral asymmetry)
  python run_papila_pipeline.py \\
      --od-excel datasets/papila/patient_data_od.xlsx \\
      --os-excel datasets/papila/patient_data_os.xlsx \\
      --image-dir datasets/papila/fundus_images \\
      --output-dir outputs/papila

  # OD only
  python run_papila_pipeline.py \\
      --od-excel datasets/papila/patient_data_od.xlsx \\
      --image-dir datasets/papila/fundus_images

  # Specific subjects
  python run_papila_pipeline.py \\
      --od-excel datasets/papila/patient_data_od.xlsx \\
      --os-excel datasets/papila/patient_data_os.xlsx \\
      --image-dir datasets/papila/fundus_images \\
      --ids 002 004 005
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from cdr_pipeline import CDRPipeline
from disc_morphology import compute_disc_morphology
from papila_loader import load_papila_excel
from papila_models import PAPILAEyeData, PAPILAParticipant
from pvbm_pipeline import PVBMPipeline
from pvbm_viz import plot_zones
from vascx_pipeline import VascXPipeline


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_stem(stem: str) -> str:
    """Lowercase + strip underscores: 'RET002_OD' -> 'ret002od'."""
    return stem.lower().replace("_", "")


def build_disk_index(image_dir: Path) -> dict[str, Path]:
    """
    Scan image_dir and return normalised-stem -> real path.
    RET002OD.jpg, RET002_OD.jpg, ret002od.png all map to 'ret002od'.
    """
    index: dict[str, Path] = {}
    for p in image_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            key = _normalise_stem(p.stem)
            if key in index:
                print(f"[WARNING] Duplicate normalised stem '{key}': "
                      f"{index[key].name} vs {p.name} — keeping first")
            else:
                index[key] = p
    return index


def build_image_index(
    participants: dict[str, PAPILAParticipant],
) -> dict[str, tuple[PAPILAParticipant, PAPILAEyeData]]:
    """
    Build normalised-stem -> (participant, eye) for every eye in participants.
    Key is derived from the clinical record: RET{id}_{laterality} normalised,
    e.g. 'ret002od', 'ret002os'. This matches the disk filenames directly.
    """
    index: dict[str, tuple[PAPILAParticipant, PAPILAEyeData]] = {}
    for p in participants.values():
        numeric = re.sub(r"[^0-9]", "", str(p.subject_id))
        for lat, eye in p.eyes.items():
            key = _normalise_stem(f"RET{numeric}_{lat}")
            index[key] = (p, eye)
    return index


def resolve_stems(
    args: argparse.Namespace,
    image_index: dict[str, tuple],
    disk_index: dict[str, Path],
) -> list[str]:
    """
    Return normalised stems to process.

    --ids / --id-file accept bare numbers ('002'), RET-prefixed ('RET002'),
    or full stems ('RET002OD', 'RET002_OD'). Bare numbers and RET-only prefixes
    expand to all available eyes for that subject.
    """
    def on_disk(stem: str) -> bool:
        return _normalise_stem(stem) in disk_index

    def expand(s: str) -> list[str]:
        """Expand one user-supplied ID to all matching normalised stems."""
        norm = _normalise_stem(s)
        # Already a full stem with laterality suffix
        if norm in image_index:
            return [norm]
        # Bare number or 'ret'+digits — expand to both eyes
        numeric = re.sub(r"[^0-9]", "", norm)
        candidates = [k for k in image_index if re.sub(r"[^0-9]", "", k) == numeric]
        return candidates

    if args.ids:
        stems = []
        for s in args.ids:
            expanded = expand(s)
            if not expanded:
                print(f"[WARNING] '{s}' not found in clinical index")
            stems.extend(expanded)
        missing = [s for s in stems if not on_disk(s)]
        if missing:
            print(f"[WARNING] {len(missing)} stem(s) not found on disk: {missing[:5]}")
        return [s for s in stems if on_disk(s)]

    if args.id_file:
        id_file = Path(args.id_file)
        if not id_file.exists():
            sys.exit(f"[ERROR] ID file not found: {id_file}")
        raw = id_file.read_text().splitlines()
        entries = [l.strip() for l in raw if l.strip() and not l.startswith("#")]
        stems = []
        for s in entries:
            expanded = expand(s)
            if not expanded:
                print(f"[WARNING] '{s}' not found in clinical index")
            stems.extend(expanded)
        missing = [s for s in stems if not on_disk(s)]
        if missing:
            print(f"[WARNING] {len(missing)} stem(s) not found on disk: {missing[:5]}")
        return [s for s in stems if on_disk(s)]

    # Default: all index stems with a matching file on disk
    found     = [s for s in sorted(image_index) if on_disk(s)]
    not_found = [s for s in sorted(image_index) if not on_disk(s)]
    if not_found:
        print(f"[INFO] {len(not_found)} clinical record(s) have no image on disk "
              f"(first 5: {not_found[:5]})")
    return found


# ── Single-image pipeline ─────────────────────────────────────────────────────

def run_single(
    image_stem: str,
    image_index: dict[str, tuple[PAPILAParticipant, PAPILAEyeData]],
    disk_index: dict[str, Path],
    output_root: Path,
    vascx: VascXPipeline,
    pvbm: PVBMPipeline,
    cdr: CDRPipeline,
    visualise: bool = True,
) -> dict:
    participant, eye = image_index[image_stem]
    image_path = disk_index[image_stem]

    # Output directory named from clinical key (e.g. RET002_OD), not disk stem
    numeric = re.sub(r"[^0-9]", "", str(participant.subject_id))
    out_name = f"RET{numeric}_{eye.laterality}"
    out_dir  = output_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {out_name} ---")
    print(f"  Subject:    {participant.subject_id}  (file: {image_path.name})")
    print(f"  Age/Gender: {participant.demographics.age} / {participant.demographics.gender}")
    print(f"  Diagnosis:  {participant.diagnosis.glaucoma_label} "
          f"(raw: {participant.diagnosis.source_category})")
    print(f"  IOP:        {eye.iop} mmHg  (corrected: {eye.iop_corrected})")
    print(f"  Pachymetry: {eye.ocular.pachymetry_um} µm")
    print(f"  Axial len:  {eye.ocular.axial_length_mm} mm")
    print(f"  VF_MD:      {eye.vf_md} dB")

    # VascX writes to output_root/image_path.stem; redirect into out_dir
    vascx.output_root = out_dir.parent
    vascx_result = vascx.run(image_path)
    vascx_wrote = out_dir.parent / image_path.stem
    if vascx_wrote.resolve() != out_dir.resolve() and vascx_wrote.exists():
        import shutil
        for f in vascx_wrote.iterdir():
            shutil.move(str(f), str(out_dir / f.name))
        vascx_wrote.rmdir()
    vascx_result.output_dir = out_dir

    vascx_result.save_artery_mask(crossings="artery")
    vascx_result.save_vein_mask(crossings="vein")

    pvbm_result = pvbm.run(vascx_result)

    cdr.output_root = out_dir.parent
    cdr_result = cdr.run(image_path)
    cdr_wrote = out_dir.parent / image_path.stem
    if cdr_wrote.resolve() != out_dir.resolve() and cdr_wrote.exists():
        import shutil
        for f in cdr_wrote.iterdir():
            dest = out_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
        cdr_wrote.rmdir()
    cdr_result.output_dir = out_dir
    cdr_result.disc_mask_path = out_dir / "disc_mask.png"
    cdr_result.cup_mask_path  = out_dir / "cup_mask.png"
    cdr_result.segmap_path    = out_dir / "segmap.png"

    fundus_arr = np.array(Image.open(image_path))
    cdr_result.morph = compute_disc_morphology(
        cdr_result  = cdr_result,
        image_shape = fundus_arr.shape[:2],
        fovea_xy    = vascx_result.fovea,
    )

    if visualise:
        fig = plot_zones(
            fundus      = fundus_arr,
            disc_mask   = vascx_result.load_disc_mask(),
            artery_mask = vascx_result.load_artery_mask(),
            vein_mask   = vascx_result.load_vein_mask(),
            cx          = pvbm_result.disc_center[0],
            cy          = pvbm_result.disc_center[1],
            radius      = pvbm_result.disc_radius,
            save_path   = str(out_dir / "zones.png"),
        )
        plt.close(fig)

    eye.computed.pvbm = pvbm_result
    eye.computed.cdr  = cdr_result

    if vascx_result.quality is not None:
        q = vascx_result.quality
        print(f"  Quality:    {q.get('quality','?')}  "
              f"(good={q.get('quality_good', float('nan')):.3f}  "
              f"usable={q.get('quality_usable', float('nan')):.3f}  "
              f"reject={q.get('quality_reject', float('nan')):.3f})")

    c = eye.computed
    print(f"  AVR:         {f'{c.avr:.3f}' if c.avr is not None else 'n/a'}")
    print(f"  Linear CDR:  {f'{c.linear_cdr:.3f}' if c.linear_cdr is not None else 'n/a'}")
    print(f"  Vertical CDR:{f'{c.vcdr:.3f}' if c.vcdr is not None else 'n/a'}")
    print(f"  RDR:         {f'{c.rdr:.3f}' if c.rdr is not None else 'n/a'}")

    row = _build_results_row(participant, eye, out_name, vascx_result)
    pd.DataFrame([row]).to_csv(out_dir / "results.csv", index=False)
    print(f"  Saved:       {out_dir}/results.csv")

    return row


def _build_results_row(
    participant: PAPILAParticipant,
    eye: PAPILAEyeData,
    image_id: str,
    vascx_result,
) -> dict:
    lat = (eye.laterality or "OD").upper()
    lat_lower = lat.lower()
    row: dict = {
        "image_id":               image_id,
        "subject_id":             participant.subject_id,
        "dataset":                participant.dataset,
        "laterality":             lat,
        "age":                    participant.demographics.age,
        "gender":                 participant.demographics.gender,
        "glaucoma_label":         participant.diagnosis.glaucoma_label,
        "source_category":        participant.diagnosis.source_category,
        "iop_asymmetry":          participant.iop_asymmetry,
        "vf_md_asymmetry":        participant.vf_md_asymmetry,
        "cdr_asymmetry":          participant.cdr_asymmetry,
        "pachymetry_asymmetry":   participant.pachymetry_asymmetry,
        "axial_length_asymmetry": participant.axial_length_asymmetry,
    }
    # Write under the actual laterality prefix (os_iop, os_avr, ...)
    row.update(eye.to_flat_dict(prefix=lat_lower))
    # Also write under the DAG-aligned "od_" prefix so the coverage report and
    # causal analysis work regardless of which eye was imaged. Both eyes measure
    # the same DAG nodes — the laterality column records which eye it actually is.
    if lat_lower != "od":
        row.update(eye.to_flat_dict(prefix="od"))
    if vascx_result.quality is not None:
        q = vascx_result.quality
        row["quality_good"]   = q.get("quality_good")
        row["quality_usable"] = q.get("quality_usable")
        row["quality_reject"] = q.get("quality_reject")
        row["quality"]        = q.get("quality")
    return row


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    stems: list[str],
    image_index: dict[str, tuple[PAPILAParticipant, PAPILAEyeData]],
    disk_index: dict[str, Path],
    output_root: Path,
    vascx: VascXPipeline,
    pvbm: PVBMPipeline,
    cdr: CDRPipeline,
    participants: dict[str, PAPILAParticipant],
    visualise: bool = True,
) -> pd.DataFrame:
    results = []
    skipped = []

    for i, stem in enumerate(stems):
        print(f"\n[{i+1}/{len(stems)}] {stem}")
        try:
            row = run_single(
                stem, image_index, disk_index, output_root,
                vascx, pvbm, cdr, visualise=visualise,
            )
            participant, _ = image_index[stem]
            results.append((stem, participant, row))
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            skipped.append(stem)

    # Recompute asymmetry once both eyes have CDR values; backfill into rows
    for p in participants.values():
        p.compute_asymmetry()

    rows = []
    for stem, participant, row in results:
        row["cdr_asymmetry"]           = participant.cdr_asymmetry
        row["iop_corrected_asymmetry"] = getattr(participant, "iop_corrected_asymmetry", None)
        row["pachymetry_asymmetry"]    = getattr(participant, "pachymetry_asymmetry", None)
        row["axial_length_asymmetry"]  = getattr(participant, "axial_length_asymmetry", None)
        rows.append(row)
        numeric = re.sub(r"[^0-9]", "", str(participant.subject_id))
        _, eye  = image_index[stem]
        out_dir  = output_root / f"RET{numeric}_{eye.laterality}"
        csv_path = out_dir / "results.csv"
        if csv_path.exists():
            pd.DataFrame([row]).to_csv(csv_path, index=False)

    if rows:
        df = pd.DataFrame(rows)
        combined_path = output_root / "papila_results_all.csv"
        df.to_csv(combined_path, index=False)
        print(f"\n{'─'*50}")
        print(f"Batch complete.")
        print(f"  Processed : {len(rows)}")
        print(f"  Skipped   : {len(skipped)}")
        print(f"  Combined  : {combined_path}")
        if skipped:
            print(f"  Skipped   : {skipped}")
        return df
    else:
        print("\nNo images processed successfully.")
        return pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_papila_pipeline.py",
        description="Run VascX + PVBM + CDR pipelines for PAPILA fundus images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Both eyes at once (enables bilateral asymmetry)
              python run_papila_pipeline.py \\
                  --od-excel datasets/papila/patient_data_od.xlsx \\
                  --os-excel datasets/papila/patient_data_os.xlsx \\
                  --image-dir datasets/papila/fundus_images

              # OD only
              python run_papila_pipeline.py \\
                  --od-excel datasets/papila/patient_data_od.xlsx \\
                  --image-dir datasets/papila/fundus_images

              # Specific subjects (bare numbers or RET-prefixed; both eyes processed)
              python run_papila_pipeline.py \\
                  --od-excel datasets/papila/patient_data_od.xlsx \\
                  --os-excel datasets/papila/patient_data_os.xlsx \\
                  --image-dir datasets/papila/fundus_images \\
                  --ids 002 004 005
        """),
    )

    parser.add_argument("--od-excel", type=Path,
                        help="PAPILA OD Excel file (patient_data_od.xlsx)")
    parser.add_argument("--os-excel", type=Path,
                        help="PAPILA OS Excel file (patient_data_os.xlsx)")
    parser.add_argument("--image-dir", required=True, type=Path,
                        help="Directory containing PAPILA fundus images (OD and OS)")

    target = parser.add_mutually_exclusive_group()
    target.add_argument("--ids", nargs="+", metavar="ID",
                        help="Subject IDs to process (e.g. 002 004 or RET002); "
                             "processes all available eyes per subject")
    target.add_argument("--id-file", type=Path, metavar="FILE",
                        help="Text file of subject IDs, one per line (# = comment)")

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/papila"),
                        help="Root output directory (default: outputs/papila)")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device (default: cuda if available, else cpu)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip zone visualisation")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.od_excel and not args.os_excel:
        sys.exit("[ERROR] At least one of --od-excel or --os-excel is required.")
    if args.od_excel and not args.od_excel.exists():
        sys.exit(f"[ERROR] OD Excel not found: {args.od_excel}")
    if args.os_excel and not args.os_excel.exists():
        sys.exit(f"[ERROR] OS Excel not found: {args.os_excel}")
    if not args.image_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {args.image_dir}")

    device      = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    od_excel = args.od_excel if args.od_excel and args.od_excel.exists() else None
    os_excel = args.os_excel if args.os_excel and args.os_excel.exists() else None

    participants = load_papila_excel(od_path=od_excel, os_path=os_excel)
    image_index  = build_image_index(participants)
    disk_index   = build_disk_index(args.image_dir)

    print(f"\nClinical index : {len(image_index)} records  "
          f"({'OD+OS' if od_excel and os_excel else 'OD' if od_excel else 'OS'})")
    print(f"Images on disk : {len(disk_index)} files")
    for stem in sorted(image_index)[:6]:
        p, eye = image_index[stem]
        found  = "✓" if stem in disk_index else "✗ (not on disk)"
        print(f"  {stem}  ->  id={p.subject_id}  lat={eye.laterality}"
              f"  dx={p.diagnosis.glaucoma_label}  {found}")
    if len(image_index) > 6:
        print(f"  ... and {len(image_index)-6} more")

    stems = resolve_stems(args, image_index, disk_index)
    if not stems:
        print("\n[ERROR] No images to process.")
        print(f"  Clinical records : {len(image_index)}")
        print(f"  Files on disk    : {len(disk_index)}")
        if disk_index:
            print(f"  Sample disk stems: {sorted(disk_index.keys())[:5]}")
        if image_index:
            print(f"  Sample index keys: {sorted(image_index.keys())[:5]}")
        sys.exit(1)

    print(f"\nWill process {len(stems)} image(s).")

    vascx = VascXPipeline(output_root=output_root, device=device)
    pvbm  = PVBMPipeline()
    cdr   = CDRPipeline(output_root=output_root, device=device)

    run_batch(
        stems        = stems,
        image_index  = image_index,
        disk_index   = disk_index,
        output_root  = output_root,
        vascx        = vascx,
        pvbm         = pvbm,
        cdr          = cdr,
        participants = participants,
        visualise    = not args.no_viz,
    )


if __name__ == "__main__":
    main()
