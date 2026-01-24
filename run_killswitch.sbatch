#!/bin/bash
#SBATCH --job-name=killswitch
#SBATCH --output=killswitch_%j.out
#SBATCH --error=killswitch_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=regular
# Adjust partition name for your cluster (e.g., regular, compute, physics)

# ==============================================================================
# PHASE-RECONSTRUCTION RECOVERY KILL-SWITCH
# ==============================================================================
# This job runs the final diagnostic to determine if cosmic web spectral
# dimension anomalies are driven by **phase coherence** or are artifacts.
#
# Three conditions tested:
#   DATA   - Original survey density field
#   HON-1  - Phase-randomized (destroys phase info, preserves P(k))
#   RECON  - BAO-reconstructed (attempts to recover phase information)
#
# Recovery Score = 100 * (d_s*[RECON] - d_s*[HON1]) / (d_s*[DATA] - d_s*[HON1])
#
# Verdict:
#   >50%  → RECOVERY: The discovery stands. Phases are load-bearing.
#   20-50% → INCONCLUSIVE: Partial phase dependence.
#   <20%  → KILL: Retire the theory. Phases don't matter.
# ==============================================================================

set -e  # Exit on error

echo "========================================"
echo "Kill-Switch Test: Phase-Reconstruction Recovery"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
# Uncomment and adjust for your cluster's module system
# module purge
# module load python/3.10
# module load gcc/11.2.0
# module load openmpi/4.1.2

# If using conda/mamba
# source ~/.bashrc
# conda activate cosmo-env

# Verify Python environment
python --version
echo "Python executable: $(which python)"

# ------------------------------------------------------------------------------
# Input Data Paths
# ------------------------------------------------------------------------------
# CRITICAL: Update these paths to point to your actual survey data files
DATA_POS="/path/to/your/survey_data_positions.npy"
RAND_POS="/path/to/your/survey_random_positions.npy"
DATA_W="/path/to/your/survey_data_weights.npy"  # Optional
RAND_W="/path/to/your/survey_random_weights.npy"  # Optional

# Verify input files exist
if [ ! -f "$DATA_POS" ]; then
    echo "ERROR: Data positions file not found: $DATA_POS"
    exit 1
fi
if [ ! -f "$RAND_POS" ]; then
    echo "ERROR: Random positions file not found: $RAND_POS"
    exit 1
fi

# ------------------------------------------------------------------------------
# Analysis Parameters
# ------------------------------------------------------------------------------
NMESH=128           # Mesh resolution (higher = more detail, more memory)
BOXSIZE=2000.0      # Survey box size in Mpc/h
BOXCENTER=0.0       # Box center coordinate
Q=0.97              # Top quantile for web voxels (0.95-0.98 typical)
CONNECTIVITY=6      # Graph connectivity (6 or 26)
HON1_SEED=42        # Random seed for phase randomization

# Reconstruction parameters (adjust for your survey's cosmology)
F=0.8               # Growth rate f = d ln D / d ln a
BIAS=2.0            # Linear bias b (galaxy/halo type dependent)
SMOOTH=15.0         # Smoothing radius in Mpc/h for reconstruction

# Output
OUTJSON="killswitch_result_${SLURM_JOB_ID}.json"

# ------------------------------------------------------------------------------
# Run the Kill-Switch Test
# ------------------------------------------------------------------------------
echo ""
echo "Running phase-reconstruction recovery analysis..."
echo "  Mesh: ${NMESH}^3"
echo "  Box size: ${BOXSIZE} Mpc/h"
echo "  Web quantile: ${Q}"
echo "  Reconstruction: f=${F}, bias=${BIAS}, smooth=${SMOOTH} Mpc/h"
echo ""

time python phase_reconstruction_recovery.py \
    --data-pos "$DATA_POS" \
    --rand-pos "$RAND_POS" \
    $([ -f "$DATA_W" ] && echo "--data-w $DATA_W") \
    $([ -f "$RAND_W" ] && echo "--rand-w $RAND_W") \
    --nmesh $NMESH \
    --boxsize $BOXSIZE \
    --boxcenter $BOXCENTER \
    --q $Q \
    --connectivity $CONNECTIVITY \
    --hon1-seed $HON1_SEED \
    --f $F \
    --bias $BIAS \
    --smooth $SMOOTH \
    --out "$OUTJSON"

EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo ""
    echo "ERROR: Analysis failed with exit code $EXITCODE"
    echo "Check error log: killswitch_${SLURM_JOB_ID}.err"
    exit $EXITCODE
fi

# ------------------------------------------------------------------------------
# Generate Verdict
# ------------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Generating verdict..."
echo "========================================"

python analyze_killswitch_results.py --input "$OUTJSON"

# ------------------------------------------------------------------------------
# Archive Results
# ------------------------------------------------------------------------------
ARCHIVE_DIR="killswitch_archive_${SLURM_JOB_ID}"
mkdir -p "$ARCHIVE_DIR"
mv "$OUTJSON" "$ARCHIVE_DIR/"
mv phase_reconstruction_recovery.png "$ARCHIVE_DIR/" 2>/dev/null || true

echo ""
echo "========================================"
echo "Kill-Switch Test Complete"
echo "Results archived to: $ARCHIVE_DIR"
echo "End time: $(date)"
echo "========================================"
