#!/bin/bash
#SBATCH --job-name=validation_gate_v2
#SBATCH --output=logs/validation_gate_%j.out
#SBATCH --error=logs/validation_gate_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=regular

# ==============================================================================
# ValidationGate v2.0.5: MPI-Distributed Archival Measurement
# ==============================================================================
# Rigorous spectral dimension estimation with:
# - Stationary distribution floor guards
# - Sliding window longest valid window (LVW) search
# - 1000+ distributed random walk realizations
# - Audit ledger with complete provenance
#
# Archival Anchors:
#   - Sierpiński Carpet: d_s ≈ 1.6735 (Barlow & Bass, 1999)
#   - Sierpiński Gasket: d_s ≈ 1.3652 (Hambly et al., 1994)
# ==============================================================================

set -euo pipefail

echo "========================================"
echo "ValidationGate v2.0.5: Archival Spectral Dimension"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "========================================"

# Directory setup
mkdir -p logs results

# Environment setup
# Uncomment for your cluster
# module load python/3.10
# module load openmpi/4.1.2
# conda activate fractal-env

# Verify Python environment
python --version
python -c "import numpy, scipy" || {
    echo "ERROR: Missing numpy/scipy. Install with:"
    echo "pip install -r requirements_fractal.txt"
    exit 1
}

# ==============================================================================
# Analysis Parameters
# ==============================================================================

# Fractal configuration
FRACTAL_TYPE="carpet"  # carpet or gasket
LEVEL=4                # Carpet: L=3^level, Gasket: N~3^level

# ValidationGate parameters
N_REALIZATIONS=1000    # Total realizations (distributed across ranks)
MAX_STEPS=5000        # Steps per random walk
FLOOR_FACTOR=50.0     # Floor guard: P(t) > 50 * π_stationary
R2_THRESHOLD=0.99     # R² threshold for valid window

# Output
OUTPUT="results/validation_gate_${FRACTAL_TYPE}_L${LEVEL}_audit.csv"

# ==============================================================================
# Single-Node Execution (for testing)
# ==============================================================================

echo ""
echo "Running ValidationGate v2.0.5 (single-node mode)..."
echo "  Fractal: ${FRACTAL_TYPE}"
echo "  Level: ${LEVEL}"
echo "  Realizations: ${N_REALIZATIONS}"
echo "  Floor factor: ${FLOOR_FACTOR}x"
echo "  R² threshold: ${R2_THRESHOLD}"
echo ""

time python validation_gate_v2.py \
    --fractal ${FRACTAL_TYPE} \
    --level ${LEVEL} \
    --n-realizations ${N_REALIZATIONS} \
    --max-steps ${MAX_STEPS} \
    --floor-factor ${FLOOR_FACTOR} \
    --r2-threshold ${R2_THRESHOLD} \
    --output "${OUTPUT}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "ValidationGate v2.0.5 Complete"
    echo "Output: ${OUTPUT}"
    echo "End time: $(date)"
    echo "========================================"
    echo ""
    echo "INTERPRETATION GUIDE:"
    echo "---------------------"
    echo ""
    echo "Status Codes:"
    echo "  PASS: Measured d_s within 5% of archival anchor"
    echo "  WARNING: Deviation 5-10% (CURVATURE_FLAG)"
    echo "  FAIL: Deviation > 10% (ANCHOR_VIOLATION)"
    echo ""
    echo "Floor Ratio Diagnostic:"
    echo "  >50: Walk is in scaling regime (hierarchical probing)"
    echo "  10-50: Marginal (may be seeing finite-size effects)"
    echo "  <10: Invalid (too close to stationary distribution)"
    echo ""
    echo "Archival Anchors:"
    echo "  Sierpiński Carpet: d_s = 1.6735 ± 0.05"
    echo "  Sierpiński Gasket: d_s = 1.3652 ± 0.05"
    echo ""
    echo "Next Steps:"
    echo "1. Check ${OUTPUT} for detailed diagnostics"
    echo "2. If PASS: proceed to ensemble sweep (vary L)"
    echo "3. If WARNING/FAIL: investigate floor_ratio, increase n_realizations"
    echo "4. Generate comparative plots across multiple L values"
else
    echo "ERROR: ValidationGate failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi

# ==============================================================================
# Optional: MPI-Distributed Mode (for production)
# ==============================================================================

# Uncomment for MPI execution with mpi4py
# Note: Requires validation_gate_v2_mpi.py wrapper (to be implemented)

# echo ""
# echo "Running MPI-distributed ValidationGate..."
# mpirun -n $SLURM_NTASKS python validation_gate_v2_mpi.py \
#     --fractal ${FRACTAL_TYPE} \
#     --level ${LEVEL} \
#     --n-realizations-per-rank $((N_REALIZATIONS / SLURM_NTASKS)) \
#     --max-steps ${MAX_STEPS} \
#     --floor-factor ${FLOOR_FACTOR} \
#     --r2-threshold ${R2_THRESHOLD} \
#     --output "${OUTPUT}"

# ==============================================================================
# Generate Summary Report
# ==============================================================================

echo ""
echo "Generating summary report..."

python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('${OUTPUT}')
    print('\n' + '='*60)
    print('AUDIT LEDGER SUMMARY')
    print('='*60)
    print(df.to_string(index=False))
    print('\n' + '='*60)
    print('KEY METRICS:')
    print('='*60)
    print(f'Measured d_s: {df[\"d_s\"].values[0]:.4f}')
    print(f'Expected d_s: {df[\"d_s_expected\"].values[0]:.4f}')
    print(f'Deviation: {df[\"deviation\"].values[0]:.2%}')
    print(f'Winner window: [{df[\"winner_window_start\"].values[0]}, {df[\"winner_window_end\"].values[0]}]')
    print(f'N points: {df[\"n_points\"].values[0]}')
    print(f'R²: {df[\"r2\"].values[0]:.4f}')
    print(f'Floor ratio: {df[\"floor_ratio_min\"].values[0]:.1f}')
    print(f'Status: {df[\"status\"].values[0]}')
    if df['flag'].values[0]:
        print(f'⚠️  Flag: {df[\"flag\"].values[0]}')
    print('='*60)
except Exception as e:
    print(f'Could not generate summary: {e}', file=sys.stderr)
" || echo "Install pandas for summary: pip install pandas"
