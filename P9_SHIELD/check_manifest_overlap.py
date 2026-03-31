"""
check_manifest_overlap.py
--------------------------
Validates overlap between tcga_clinical_backbone.tsv and the
675-network GDC manifests (tcga_gbm_tumor_manifest.json,
tcga_lgg_tumor_manifest.json).

Outputs:
  - overlap_report.tsv   : one row per network file with match status
  - missing_in_backbone.txt : network barcodes absent from backbone
  - missing_in_manifest.txt : backbone barcodes absent from manifests

Run from inside P9_SHIELD/:
    python check_manifest_overlap.py
"""

import json
import pandas as pd
from pathlib import Path

BACKBONE   = Path("tcga_clinical_backbone.tsv")
GBM_MF     = Path("tcga_gbm_tumor_manifest.json")
LGG_MF     = Path("tcga_lgg_tumor_manifest.json")
OUT_REPORT = Path("overlap_report.tsv")
OUT_MISS_B = Path("missing_in_backbone.txt")
OUT_MISS_M = Path("missing_in_manifest.txt")


def tcga12(x: str) -> str:
    s = str(x).strip()
    return s[:12] if len(s) >= 12 else s


# ── 1. Load backbone ──────────────────────────────────────────────────────────
backbone = pd.read_csv(BACKBONE, sep="\t")
backbone["Patient_ID"] = backbone["Patient_ID"].map(tcga12)
backbone_ids = set(backbone["Patient_ID"].dropna())
print(f"Backbone patients       : {len(backbone_ids)}")


# ── 2. Parse manifests ────────────────────────────────────────────────────────
def parse_manifest(path: Path) -> list[dict]:
    """
    GDC manifest JSON shape assumed:
      [ { "cases": [{"submitter_id": "TCGA-XX-XXXX-..."}],
          "file_name": "...",
          "file_id": "..." }, ... ]
    Returns list of dicts with keys: barcode12, file_name, file_id, cohort
    """
    cohort = "GBM" if "gbm" in path.name.lower() else "LGG"
    with open(path) as f:
        data = json.load(f)

    records = []
    for entry in data:
        cases = entry.get("cases", [])
        submitter = cases[0].get("submitter_id", "") if cases else ""
        records.append({
            "barcode12": tcga12(submitter),
            "file_name": entry.get("file_name", ""),
            "file_id":   entry.get("file_id", ""),
            "cohort":    cohort,
        })
    return records


records = []
for mf in (GBM_MF, LGG_MF):
    if not mf.exists():
        print(f"WARNING: {mf} not found — skipping")
        continue
    r = parse_manifest(mf)
    print(f"{mf.name}: {len(r)} entries")
    records.extend(r)

manifest_df = pd.DataFrame(records)
manifest_ids = set(manifest_df["barcode12"].dropna())
print(f"Manifest network files  : {len(manifest_df)}")
print(f"Unique manifest barcodes: {len(manifest_ids)}")


# ── 3. Overlap ────────────────────────────────────────────────────────────────
matched          = backbone_ids & manifest_ids
missing_backbone = manifest_ids - backbone_ids   # in manifest, not backbone
missing_manifest = backbone_ids - manifest_ids   # in backbone, not manifest

print(f"\n── Overlap report ──────────────────────────")
print(f"Matched (both)          : {len(matched)}")
print(f"In manifest, not backbone: {len(missing_backbone)}")
print(f"In backbone, not manifest: {len(missing_manifest)}")
pct = 100 * len(matched) / len(manifest_ids) if manifest_ids else 0
print(f"Manifest match rate     : {pct:.1f}%")


# ── 4. Annotated report ───────────────────────────────────────────────────────
manifest_df["in_backbone"] = manifest_df["barcode12"].isin(backbone_ids)

# Attach Grade and OS_days for matched rows
backbone_slim = backbone[["Patient_ID", "Grade", "OS_days", "Event"]].copy()
manifest_df = manifest_df.merge(
    backbone_slim.rename(columns={"Patient_ID": "barcode12"}),
    on="barcode12", how="left"
)

manifest_df.to_csv(OUT_REPORT, sep="\t", index=False)
print(f"\nWrote {OUT_REPORT}")


# ── 5. Missing-barcode dumps ──────────────────────────────────────────────────
OUT_MISS_B.write_text("\n".join(sorted(missing_backbone)))
OUT_MISS_M.write_text("\n".join(sorted(missing_manifest)))
print(f"Wrote {OUT_MISS_B}  ({len(missing_backbone)} barcodes)")
print(f"Wrote {OUT_MISS_M}  ({len(missing_manifest)} barcodes)")


# ── 6. Grade breakdown for matched networks ───────────────────────────────────
matched_df = manifest_df[manifest_df["in_backbone"]]
print("\nGrade breakdown for matched network files:")
print(matched_df["Grade"].value_counts(dropna=False).to_string())

print("\nSurvival completeness for matched rows:")
print(f"  OS_days non-null : {matched_df['OS_days'].notna().sum()}")
print(f"  Event non-null   : {matched_df['Event'].notna().sum()}")
