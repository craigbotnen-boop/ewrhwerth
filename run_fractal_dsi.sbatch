#!/bin/bash
#SBATCH --job-name=fractal_dsi
#SBATCH --output=logs/fractal_dsi_%j.out
#SBATCH --error=logs/fractal_dsi_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=regular

# ==============================================================================
# Fractal Discrete Scale Invariance (DSI) Analysis
# ==============================================================================
# Comparative analysis of log-periodic oscillations in entropy gradients
# across Sierpiński Carpet, Sierpiński Gasket, and Euclidean baselines.
#
# Expected Signatures:
#   - Carpet: Δlog(r) ≈ log(3) ≈ 1.099 (scaling factor λ=3)
#   - Gasket: Δlog(r) ≈ log(2) ≈ 0.693 (scaling factor λ=2)
#   - Euclidean: No log-periodic oscillations (control)
#
# Output: fractal_dsi_comparison.png (3-panel plot)
# ==============================================================================

set -euo pipefail

echo "========================================"
echo "Fractal DSI Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Directory setup
mkdir -p logs results

# Environment setup
# Uncomment and adjust for your cluster
# module load python/3.10
# source ~/.bashrc
# conda activate fractal-env

# Verify Python environment
python --version
python -c "import numpy, scipy, matplotlib" || {
    echo "ERROR: Missing required packages. Install with:"
    echo "pip install -r requirements_fractal.txt"
    exit 1
}

# ==============================================================================
# Analysis Parameters
# ==============================================================================

# Carpet level (L = 3^level grid, recommended: 4-5)
# Level 4: L=81,  ~6561 cells,  ~5168 filled
# Level 5: L=243, ~59049 cells, ~32896 filled (needs ~8GB RAM)
CARPET_LEVEL=4

# Gasket level (N ≈ 3^level nodes, recommended: 6-7)
# Level 6: ~729 nodes
# Level 7: ~2187 nodes
GASKET_LEVEL=6

# Maximum BFS radius (graph distance)
# Should be ~0.3 * L for carpet to avoid edge effects
R_MAX=80

# Output filename
OUTPUT="results/fractal_dsi_comparison_L${CARPET_LEVEL}_G${GASKET_LEVEL}.png"

# ==============================================================================
# Run Analysis
# ==============================================================================

echo ""
echo "Running DSI comparative analysis..."
echo "  Carpet level: ${CARPET_LEVEL} (L = $((3**CARPET_LEVEL)))"
echo "  Gasket level: ${GASKET_LEVEL}"
echo "  Max radius: ${R_MAX}"
echo ""

time python fractal_dsi_analysis.py \
    --carpet-level ${CARPET_LEVEL} \
    --gasket-level ${GASKET_LEVEL} \
    --r-max ${R_MAX} \
    --output "${OUTPUT}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Analysis Complete"
    echo "Output: ${OUTPUT}"
    echo "End time: $(date)"
    echo "========================================"
    echo ""
    echo "INTERPRETATION GUIDE:"
    echo "---------------------"
    echo "Panel A: Raw force ratios F(r) = S(r+1) - S(r)"
    echo "  - Euclidean should show smooth power-law decay"
    echo "  - Fractals may show visible modulations"
    echo ""
    echo "Panel B: Detrended residuals (smoking gun)"
    echo "  - Look for sinusoidal oscillations"
    echo "  - Carpet: wavelength ~ log(3) ≈ 1.099"
    echo "  - Gasket: wavelength ~ log(2) ≈ 0.693"
    echo ""
    echo "Panel C: Crossing interval histogram"
    echo "  - Peaks should align with expected periods"
    echo "  - Random noise would show flat distribution"
    echo ""
    echo "If Panel B shows clear oscillations with correct periods:"
    echo "  → This is the DSI signature for PRE/JSTAT"
    echo "  → Proceed with Monte Carlo null testing"
else
    echo "ERROR: Analysis failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi
