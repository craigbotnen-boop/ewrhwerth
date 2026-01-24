#!/bin/bash
#SBATCH --job-name=killswitch_ensemble
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=regular
# Adjust partition name for your cluster (e.g., regular, compute, physics)

# ==============================================================================
# ENSEMBLE KILL-SWITCH: Audit-Grade Phase-Reconstruction Recovery
# ==============================================================================
# This job runs a parameter sweep over:
#   - RANN: Multiple independent random catalog realizations (0, 1, 2)
#   - SEEDS: Multiple HON-1 phase randomization seeds (0-9)
#   - SMOOTH: BAO reconstruction smoothing scales (15, 20 Mpc/h)
#
# Total runs: 3 RANN × 10 seeds × 2 smoothing = 60 independent measurements
#
# This ensemble approach immunizes the verdict against "luck of the draw"
# and provides statistical uncertainty quantification for the Recovery Score.
# ==============================================================================

set -euo pipefail

echo "========================================"
echo "Ensemble Kill-Switch: Phase-Reconstruction Recovery"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "========================================"

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
# Uncomment and adjust for your cluster's module system
# module purge
# module load python/3.10
# module load gcc/11.2.0

# If using conda/mamba
# source ~/.bashrc
# conda activate desi-recon

# Thread control for numpy/scipy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Verify Python environment
python --version
echo "Python executable: $(which python)"

# Verify critical dependencies
python -c "import numpy, scipy, matplotlib, pyrecon" || {
    echo "ERROR: Missing required Python packages"
    exit 1
}

# ------------------------------------------------------------------------------
# Directory Setup
# ------------------------------------------------------------------------------
mkdir -p logs out_grid results

# ------------------------------------------------------------------------------
# Input Data Paths
# ------------------------------------------------------------------------------
# CRITICAL: Update these paths to point to your actual DESI DR1 LSS catalogs
# Use the 'clustering' products as recommended by DESI DR1 documentation

DATA_POS="/path/to/desi/dr1/lss/clustering_data_positions.npy"

# DESI provides 18 independent random realizations (RANN=0..17)
# We test RANN=0,1,2 for computational feasibility while maintaining diversity
RAND_POS_0="/path/to/desi/dr1/lss/clustering_random_RANN0_positions.npy"
RAND_POS_1="/path/to/desi/dr1/lss/clustering_random_RANN1_positions.npy"
RAND_POS_2="/path/to/desi/dr1/lss/clustering_random_RANN2_positions.npy"

# Optional: weights (FKP, completeness corrections, etc.)
DATA_W="/path/to/desi/dr1/lss/clustering_data_weights.npy"  # Optional
RAND_W="/path/to/desi/dr1/lss/clustering_random_weights.npy"  # Optional

# Verify critical input files exist
if [ ! -f "$DATA_POS" ]; then
    echo "ERROR: Data positions file not found: $DATA_POS"
    echo "Please update DATA_POS path in this script to point to your DESI DR1 LSS catalogs"
    exit 1
fi

# ------------------------------------------------------------------------------
# Analysis Parameters (DESI DR1 typical values)
# ------------------------------------------------------------------------------
NMESH=128           # Mesh resolution (128^3 is standard for surveys)
BOXSIZE=4000.0      # Survey box size in Mpc/h (adjust to your footprint)
BOXCENTER=0.0       # Box center coordinate
Q=0.97              # Top quantile for web voxels
CONNECTIVITY=6      # Graph connectivity (6-neighbor)
MIN_RAND=1.0        # Selection function threshold (expected randoms per cell)

# Reconstruction parameters (DESI DR1 at z~0.5 typical values)
F=0.8               # Growth rate f = d ln D / d ln a ~ Ω_m^0.55
BIAS=2.0            # Linear bias (tracer-dependent; LRG~2.0, ELG~1.3, QSO~2.5)
RECISO=""           # Empty = RecSym (default); set to "--reciso" for RecIso

# ------------------------------------------------------------------------------
# Ensemble Grid Definition
# ------------------------------------------------------------------------------
RANNS=(0 1 2)              # Random catalog realizations
SMOOTHS=(15 20)            # BAO smoothing scales (Mpc/h)
SEEDS=(0 1 2 3 4 5 6 7 8 9)  # HON-1 phase randomization seeds

TOTAL_RUNS=$((${#RANNS[@]} * ${#SMOOTHS[@]} * ${#SEEDS[@]}))
echo ""
echo "Ensemble Grid Configuration:"
echo "  RANN realizations: ${RANNS[@]}"
echo "  Smoothing scales: ${SMOOTHS[@]} Mpc/h"
echo "  HON-1 seeds: ${SEEDS[@]}"
echo "  Total runs: ${TOTAL_RUNS}"
echo ""

# ------------------------------------------------------------------------------
# Run Ensemble Grid
# ------------------------------------------------------------------------------
RUN_COUNT=0
FAILED_COUNT=0

for R in "${RANNS[@]}"; do
  # Select appropriate random catalog for this RANN
  RAND_POS_VAR="RAND_POS_${R}"
  RAND_POS="${!RAND_POS_VAR}"

  if [ ! -f "$RAND_POS" ]; then
    echo "WARNING: Random positions file not found for RANN=${R}: $RAND_POS"
    echo "Skipping RANN=${R} runs..."
    continue
  fi

  for SM in "${SMOOTHS[@]}"; do
    for S in "${SEEDS[@]}"; do
      RUN_COUNT=$((RUN_COUNT + 1))
      OUT="out_grid/killswitch_RANN${R}_seed${S}_smooth${SM}.json"

      echo "[${RUN_COUNT}/${TOTAL_RUNS}] Running: RANN=${R} seed=${S} smooth=${SM} Mpc/h"
      echo "  Output: ${OUT}"

      # Build command with optional weight files
      CMD="python phase_reconstruction_recovery.py \
        --data-pos \"${DATA_POS}\" \
        --rand-pos \"${RAND_POS}\" \
        --nmesh ${NMESH} \
        --boxsize ${BOXSIZE} \
        --boxcenter ${BOXCENTER} \
        --q ${Q} \
        --connectivity ${CONNECTIVITY} \
        --min-rand ${MIN_RAND} \
        --hon1-seed ${S} \
        --f ${F} \
        --bias ${BIAS} \
        --smooth ${SM} \
        ${RECISO} \
        --out \"${OUT}\""

      # Add optional weight files if they exist
      if [ -f "$DATA_W" ]; then
        CMD="${CMD} --data-w \"${DATA_W}\""
      fi
      if [ -f "$RAND_W" ]; then
        CMD="${CMD} --rand-w \"${RAND_W}\""
      fi

      # Execute with error handling
      if eval $CMD 2>&1 | tee "logs/run_R${R}_s${S}_sm${SM}.log"; then
        echo "  ✓ Success"
      else
        echo "  ✗ FAILED"
        FAILED_COUNT=$((FAILED_COUNT + 1))
      fi
      echo ""
    done
  done
done

echo "========================================"
echo "Ensemble Grid Complete"
echo "  Successful runs: $((RUN_COUNT - FAILED_COUNT))/${RUN_COUNT}"
echo "  Failed runs: ${FAILED_COUNT}"
echo "========================================"

if [ $FAILED_COUNT -gt 0 ]; then
  echo "WARNING: Some runs failed. Check individual log files in logs/"
  echo "The summary will be computed from successful runs only."
fi

# ------------------------------------------------------------------------------
# Generate Ensemble Summary
# ------------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Generating ensemble summary and verdict..."
echo "========================================"

python summarize_killswitch_ensemble.py \
  --input-dir out_grid \
  --output results/killswitch_ensemble_summary.json

SUMMARY_EXIT=$?

if [ $SUMMARY_EXIT -eq 0 ]; then
  echo ""
  echo "========================================"
  echo "Ensemble Analysis Complete"
  echo "Results: results/killswitch_ensemble_summary.json"
  echo "End time: $(date)"
  echo "========================================"
else
  echo ""
  echo "ERROR: Summary generation failed with exit code ${SUMMARY_EXIT}"
  exit $SUMMARY_EXIT
fi

# ------------------------------------------------------------------------------
# Archive Results
# ------------------------------------------------------------------------------
ARCHIVE_DIR="killswitch_ensemble_${SLURM_JOB_ID}"
mkdir -p "$ARCHIVE_DIR"
cp -r out_grid "$ARCHIVE_DIR/"
cp -r results "$ARCHIVE_DIR/"
cp -r logs "$ARCHIVE_DIR/"

echo ""
echo "Results archived to: $ARCHIVE_DIR"
echo "Job ID: $SLURM_JOB_ID"
