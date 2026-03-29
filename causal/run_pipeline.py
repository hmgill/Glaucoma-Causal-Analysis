"""
run_pipeline.py — Run VascX + PVBM + CDR pipelines for GRAPE images.

Image ID convention (GRAPE): {subject}_{laterality}_{visit}.jpg
  e.g.  "1_OD_2.jpg"  ->  subject 1, right eye, visit 2

Output directory layout (outputs/1_OD_2/):
  artery_vein.png               -- raw AV label map (0/1/2/3)
  vessels.png                   -- binary all-vessel mask
  disc.png                      -- VascX disc mask
  artery_crossings_artery.png   -- binary artery mask (crossings included)
  vein_crossings_vein.png       -- binary vein mask (crossings included)
  quality.csv                   -- quality_good / quality_usable / quality_reject / quality
  fovea.csv
  segmap.png / disc_mask.png / cup_mask.png
  zones.png                     -- zone A / B / C visualisation
  results.csv                   -- flat record with all metrics

Usage — directory mode (default, runs all images found in IMAGE_DIR):
  python run_pipeline.py \\
      --excel datasets/grape/vf_and_clinical_information.xlsx \\
      --image-dir datasets/grape/CFPs \\
      --output-dir outputs

Usage — ID list mode:
  python run_pipeline.py \\
      --excel datasets/grape/vf_and_clinical_information.xlsx \\
      --image-dir datasets/grape/CFPs \\
      --ids 1_OD_2 1_OS_2 2_OD_1

Usage — ID file mode (one stem per line, # = comment):
  python run_pipeline.py \\
      --excel datasets/grape/vf_and_clinical_information.xlsx \\
      --image-dir datasets/grape/CFPs \\
      --id-file my_subset.txt

Other options:
  --output-dir outputs     root directory for per-image subdirs (default: outputs)
  --device     cpu         override GPU device
  --no-viz                 skip zone visualisation (faster for large batches)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from cdr_pipeline import CDRPipeline
from disc_morphology import compute_disc_morphology
from grape_loader import load_grape_excel
from grape_models import GRAPEEyeExam, GRAPEParticipant
from pvbm_pipeline import PVBMPipeline
from pvbm_viz import plot_zones
from vascx_pipeline import VascXPipeline


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── Disk and clinical index helpers ───────────────────────────────────────────

def _normalise_stem(stem: str) -> str:
    """
    Canonical stem key used for disk <-> clinical index matching.
    Lowercases and strips all underscores so that e.g. "RET002_OD"
    and "RET002OD" both map to the same key "ret002od".
    """
    return stem.lower().replace("_", "")


def build_disk_index(image_dir: Path) -> dict[str, Path]:
    """
    Scan image_dir and return a normalised-stem -> real path mapping.
    Normalisation lowercases and strips underscores, so RET002OD,
    RET002_OD, ret002od, etc. all resolve to the same key "ret002od".
    Extension is ignored — RET002_OD.jpg and RET002_OD.png map to the
    same entry (last one wins if both exist; log a warning).
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
    participants: dict[int, GRAPEParticipant],
) -> dict[str, tuple[GRAPEParticipant, GRAPEEyeExam]]:
    """
    Returns dict keyed by normalised stem (_normalise_stem applied).
    e.g. both "1_OD_2" and "1od2" -> key "1od2"
    Walks the full longitudinal exam_record for every participant.
    """
    index: dict[str, tuple[GRAPEParticipant, GRAPEEyeExam]] = {}
    for p in participants.values():
        for exam in p.exam_record.exams:
            for eye in exam.eyes.values():
                if eye.fundus.filename:
                    key = _normalise_stem(Path(eye.fundus.filename).stem)
                    index[key] = (p, eye)
    return index


def resolve_stems(
    args: argparse.Namespace,
    image_index: dict[str, tuple],
    disk_index: dict[str, Path],
) -> list[str]:
    """
    Determine which image stems to process.

    Matching is case-insensitive and extension-agnostic — the filename stored
    in the data model does not need to match the extension on disk.

    Priority:
      1. --ids     : explicit stems on the command line
      2. --id-file : stems read from a text file (one per line)
      3. default   : all clinical index stems that have a matching file on disk
    """
    def in_index(stem: str) -> bool:
        return _normalise_stem(stem) in image_index

    def on_disk(stem: str) -> bool:
        return _normalise_stem(stem) in disk_index

    if args.ids:
        stems = args.ids
        unknown = [s for s in stems if not in_index(s)]
        missing = [s for s in stems if in_index(s) and not on_disk(s)]
        if unknown:
            print(f"[WARNING] {len(unknown)} ID(s) not in clinical index: {unknown[:5]}")
        if missing:
            print(f"[WARNING] {len(missing)} ID(s) not found on disk: {missing[:5]}")
        # Return normalised keys so downstream lookups always hit image_index
        return [_normalise_stem(s) for s in stems if in_index(s) and on_disk(s)]

    if args.id_file:
        id_file = Path(args.id_file)
        if not id_file.exists():
            sys.exit(f"[ERROR] ID file not found: {id_file}")
        raw = id_file.read_text().splitlines()
        stems = [l.strip() for l in raw if l.strip() and not l.startswith("#")]
        unknown = [s for s in stems if not in_index(s)]
        missing = [s for s in stems if in_index(s) and not on_disk(s)]
        if unknown:
            print(f"[WARNING] {len(unknown)} ID(s) not in clinical index: {unknown[:5]}")
        if missing:
            print(f"[WARNING] {len(missing)} ID(s) not found on disk: {missing[:5]}")
        return [_normalise_stem(s) for s in stems if in_index(s) and on_disk(s)]

    # Default: all index stems (already normalised) that have a file on disk
    found     = [s for s in sorted(image_index) if on_disk(s)]
    not_found = [s for s in sorted(image_index) if not on_disk(s)]
    if not_found:
        print(f"[INFO] {len(not_found)} clinical record(s) have no matching image "
              f"on disk (first 5: {not_found[:5]})")
    return found


# ── Single-image pipeline ─────────────────────────────────────────────────────

def run_single(
    image_stem: str,
    image_index: dict[str, tuple[GRAPEParticipant, GRAPEEyeExam]],
    disk_index: dict[str, Path],
    output_root: Path,
    vascx: VascXPipeline,
    pvbm: PVBMPipeline,
    cdr: CDRPipeline,
    visualise: bool = True,
) -> dict:
    """
    Run full pipeline for one GRAPE image.
    Attaches results to GRAPEEyeExam.computed and returns a flat results dict.
    Uses disk_index to resolve the real path (extension-agnostic).
    """
    participant, eye_exam = image_index[image_stem]

    # Resolve actual path from disk — handles any extension
    image_path = disk_index.get(_normalise_stem(image_stem))
    if image_path is None:
        raise FileNotFoundError(
            f"No image file found for stem '{image_stem}' in image directory."
        )

    print(f"\n--- {image_stem} ---")
    print(f"  Subject:    {participant.subject_id}  "
          f"({participant.demographics.gender}, age {participant.demographics.age})")
    print(f"  Diagnosis:  {participant.diagnosis.glaucoma_category}  "
          f"label={participant.diagnosis.glaucoma_label}  "
          f"progressor={participant.is_progressor}")
    print(f"  Eye:        {eye_exam.laterality}  "
          f"IOP={eye_exam.iop} mmHg  "
          f"VF mean={eye_exam.visual_field.mean_sensitivity:.1f} dB")

    # Use the real filename stem (from disk) as the output directory name,
    # not the normalised key — avoids creating a separate "ret002od" dir.
    out_dir = output_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # VascX
    # Pass out_dir as a one-shot output_root so VascX writes:
    #   out_dir / image_path.stem / *.png  <- still one level deep
    # To write flat we override output_root temporarily.
    vascx.output_root = out_dir.parent
    vascx_result = vascx.run(image_path)
    vascx_result.save_artery_mask(crossings="artery")
    vascx_result.save_vein_mask(crossings="vein")

    # PVBM
    pvbm_result = pvbm.run(vascx_result)

    # CDR
    cdr.output_root = out_dir.parent
    cdr_result = cdr.run(image_path)

    # Disc morphology
    fundus_arr = np.array(Image.open(image_path))
    cdr_result.morph = compute_disc_morphology(
        cdr_result  = cdr_result,
        image_shape = fundus_arr.shape[:2],
        fovea_xy    = vascx_result.fovea,
    )

    # Zone visualisation
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

    # Attach to data model
    eye_exam.computed.pvbm = pvbm_result
    eye_exam.computed.cdr  = cdr_result

    # Print quality
    if vascx_result.quality is not None:
        q = vascx_result.quality
        print(f"  Quality:    {q.get('quality','?')}  "
              f"(good={q.get('quality_good', float('nan')):.3f}  "
              f"usable={q.get('quality_usable', float('nan')):.3f}  "
              f"reject={q.get('quality_reject', float('nan')):.3f})")

    # Print summary metrics
    c = eye_exam.computed
    print(f"  AVR:         {f'{c.avr:.3f}' if c.avr is not None else 'n/a'}")
    print(f"  Linear CDR:  {f'{c.linear_cdr:.3f}' if c.linear_cdr is not None else 'n/a'}")
    print(f"  Vertical CDR:{f'{c.vcdr:.3f}' if c.vcdr is not None else 'n/a'}")
    print(f"  RDR:         {f'{c.rdr:.3f}' if c.rdr is not None else 'n/a'}")

    row = _build_results_row(participant, eye_exam, image_stem, vascx_result)
    pd.DataFrame([row]).to_csv(out_dir / "results.csv", index=False)
    print(f"  Saved:       {out_dir}/results.csv")

    return row


def _build_results_row(
    participant: GRAPEParticipant,
    eye_exam: GRAPEEyeExam,
    image_stem: str,
    vascx_result,
) -> dict:
    """Flat results dict: participant context + eye data + pipeline metrics."""
    lat = (eye_exam.laterality or "od").lower()   # "od" or "os"

    row: dict = {
        "image_id":          image_stem,
        "subject_id":        participant.subject_id,
        "dataset":           participant.dataset,
        "age":               participant.demographics.age,
        "gender":            participant.demographics.gender,
        "glaucoma_label":    participant.diagnosis.glaucoma_label,
        "glaucoma_category": participant.diagnosis.glaucoma_category,
        "any_progression":   participant.is_progressor,
        "progression_plr2":  participant.diagnosis.progression.plr2,
        "progression_plr3":  participant.diagnosis.progression.plr3,
        "progression_md":    participant.diagnosis.progression.md,
        "iop_asymmetry":     participant.iop_asymmetry,
        "vf_md_asymmetry":   participant.vf_md_asymmetry,
        "cdr_asymmetry":     participant.cdr_asymmetry,  # may be None until batch end
    }

    # Write under actual laterality prefix (od_iop / os_iop) and also always
    # under "od_" so DAG feature keys are populated for every row.
    row.update(eye_exam.to_flat_dict(prefix=lat))
    if lat != "od":
        row.update(eye_exam.to_flat_dict(prefix="od"))

    # Static per-eye clinical measurements (CCT, RNFL) live on the participant
    # baseline_ocular, not on EyeExam. Same dual-prefix approach.
    baseline = participant.baseline_ocular.get(eye_exam.laterality or "OD")
    if baseline is not None:
        row.update(baseline.to_flat_dict(prefix=lat))
        if lat != "od":
            row.update(baseline.to_flat_dict(prefix="od"))

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
    image_index: dict[str, tuple[GRAPEParticipant, GRAPEEyeExam]],
    disk_index: dict[str, Path],
    output_root: Path,
    vascx: VascXPipeline,
    pvbm: PVBMPipeline,
    cdr: CDRPipeline,
    participants: dict[int, GRAPEParticipant],
    visualise: bool = True,
) -> pd.DataFrame:
    """
    Run the pipeline for a list of image stems.
    Returns a combined DataFrame and writes grape_results_all.csv.
    """
    results  = []   # list of (image_stem, participant, row_dict)
    skipped  = []

    for i, stem in enumerate(stems):
        print(f"\n[{i+1}/{len(stems)}] {stem}")
        try:
            row = run_single(
                stem, image_index, disk_index, output_root,
                vascx, pvbm, cdr, visualise=visualise,
            )
            participant, _ = image_index[stem]
            results.append((stem, participant, row))
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            skipped.append(stem)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            skipped.append(stem)

    # Recompute asymmetry now that both eyes may have computed CDR values.
    # Then backfill cdr_asymmetry into each row and re-write per-image CSVs.
    for p in participants.values():
        p.compute_asymmetry()

    rows = []
    for stem, participant, row in results:
        row["cdr_asymmetry"] = participant.cdr_asymmetry
        rows.append(row)
        # Update the per-image results.csv with the now-populated cdr_asymmetry
        image_path = disk_index.get(_normalise_stem(stem))
        if image_path is not None:
            out_dir = output_root / image_path.stem
            csv_path = out_dir / "results.csv"
            if csv_path.exists():
                pd.DataFrame([row]).to_csv(csv_path, index=False)

    if rows:
        df = pd.DataFrame(rows)
        combined_path = output_root / "grape_results_all.csv"
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
        prog="run_pipeline.py",
        description="Run VascX + PVBM + CDR pipelines for GRAPE fundus images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # All images in directory (default behaviour)
              python run_pipeline.py \\
                  --excel datasets/grape/vf_and_clinical_information.xlsx \\
                  --image-dir datasets/grape/CFPs

              # Specific IDs on the command line
              python run_pipeline.py \\
                  --excel datasets/grape/vf_and_clinical_information.xlsx \\
                  --image-dir datasets/grape/CFPs \\
                  --ids 1_OD_2 1_OS_2 2_OD_1

              # IDs from a text file (one stem per line, # = comment)
              python run_pipeline.py \\
                  --excel datasets/grape/vf_and_clinical_information.xlsx \\
                  --image-dir datasets/grape/CFPs \\
                  --id-file my_subset.txt
        """),
    )

    parser.add_argument(
        "--excel", required=True, type=Path,
        help="Path to GRAPE Excel file (vf_and_clinical_information.xlsx)",
    )
    parser.add_argument(
        "--image-dir", required=True, type=Path,
        help="Directory containing GRAPE fundus images",
    )

    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--ids", nargs="+", metavar="STEM",
        help="One or more image stems to process (e.g. 1_OD_2 1_OS_3)",
    )
    target.add_argument(
        "--id-file", type=Path, metavar="FILE",
        help="Text file of image stems to process (one per line; # = comment)",
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Root output directory (default: outputs)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device string (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip zone visualisation (saves time in large batches)",
    )

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.excel.exists():
        sys.exit(f"[ERROR] Excel file not found: {args.excel}")
    if not args.image_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {args.image_dir}")

    device      = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    # Load clinical data and build both indexes
    participants  = load_grape_excel(args.excel)
    image_index   = build_image_index(participants)
    disk_index    = build_disk_index(args.image_dir)

    print(f"\nClinical index : {len(image_index)} records")
    print(f"Images on disk : {len(disk_index)} files")
    for stem in sorted(image_index)[:5]:
        p, eye = image_index[stem]
        on_disk = "✓" if _normalise_stem(stem) in disk_index else "✗ (not on disk)"
        print(f"  {stem}  ->  subject={p.subject_id}  lat={eye.laterality}  {on_disk}")
    if len(image_index) > 5:
        print(f"  ... and {len(image_index)-5} more")

    stems = resolve_stems(args, image_index, disk_index)
    if not stems:
        print("\n[ERROR] No images to process.")
        print(f"  Clinical records : {len(image_index)}")
        print(f"  Files on disk    : {len(disk_index)}")
        if disk_index:
            print(f"  Sample disk stems (lowercase): {sorted(disk_index.keys())[:5]}")
        if image_index:
            print(f"  Sample index stems           : {sorted(image_index.keys())[:5]}")
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
