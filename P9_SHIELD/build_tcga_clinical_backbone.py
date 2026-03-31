import pandas as pd
from pathlib import Path

TCGA_CDR = Path("TCGA-CDR-SupplementalTableS1 (1).xlsx")
WHO2021 = Path("Matrix_WHO2021.csv")
OUTFILE = Path("tcga_clinical_backbone.tsv")

def tcga12(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s[:12] if len(s) >= 12 else s

# 1. Load TCGA-CDR survival backbone
xl = pd.ExcelFile(TCGA_CDR)
sheet = next((s for s in xl.sheet_names if "cdr" in s.lower()), xl.sheet_names[0])
cdr = pd.read_excel(TCGA_CDR, sheet_name=sheet)

# Expected TCGA-CDR-style columns
# Adjust only if your local file differs
id_col = "bcr_patient_barcode"
os_time_col = "OS.time"
os_event_col = "OS"
age_col = "age_at_initial_pathologic_diagnosis"

cdr["Patient_ID"] = cdr[id_col].map(tcga12)
cdr = cdr[["Patient_ID", id_col, os_time_col, os_event_col, age_col]].copy()
cdr = cdr.rename(columns={
    os_time_col: "OS_days",
    os_event_col: "Event",
    age_col: "Age"
})

# 2. Load WHO2021 classification
who = pd.read_csv(WHO2021)
who["Patient_ID"] = who["Patient_ID"].map(tcga12)

# Keep the simplified label as the main grade/category field
who_keep = who[[
    "Patient_ID",
    "TCGA-histological.type",
    "classification.2021_complete.labels",
    "classification.2021_simplified.labels"
]].copy()

who_keep = who_keep.rename(columns={
    "classification.2021_simplified.labels": "Grade",
    "classification.2021_complete.labels": "WHO2021_complete",
    "TCGA-histological.type": "TCGA_histology"
})

# 3. Merge
merged = cdr.merge(who_keep, on="Patient_ID", how="left")

# 4. Optional derived covariates
# Simple IDH extraction from WHO2021_complete if you need it immediately
merged["IDH_mutation_status"] = merged["WHO2021_complete"].astype(str).str.contains("IDHmut", case=False, na=False)
merged["IDH_mutation_status"] = merged["IDH_mutation_status"].map({True: "Mutant", False: "Wildtype"})

# 5. Clean impossible rows
merged = merged[merged["Patient_ID"].notna()].copy()

# 6. Write
merged.to_csv(OUTFILE, sep="\t", index=False)

print(f"Wrote {OUTFILE}")
print(f"N rows: {len(merged)}")
print("Grade counts:")
print(merged["Grade"].value_counts(dropna=False).head(10))
print("Missing WHO2021 labels:", merged["Grade"].isna().sum())
