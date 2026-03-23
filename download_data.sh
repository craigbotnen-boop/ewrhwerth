#!/usr/bin/env bash
# P1 Data Download Script — ACT DR6 + DESI DR1 v1.5
# Downloads kappa/curl alms, lensing mask, LRG catalogs, and simulations.
set -euo pipefail

DATA_DIR="${P1_DATA_DIR:-./P1_Data}"
mkdir -p "$DATA_DIR/sims"
cd "$DATA_DIR"

NERSC_BASE="https://portal.nersc.gov/project/act/dr6_lensing_v1/maps/baseline"
DESI_BASE="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"

# --- ACT DR6 Lensing (Baseline) ---
echo "[1/4] ACT DR6 kappa alms..."
curl -L -C - -o "kappa_alm_data_act_dr6_lensing_v1_baseline.fits" \
    "$NERSC_BASE/kappa_alm_data_act_dr6_lensing_v1_baseline.fits"

echo "[2/4] ACT DR6 lensing mask (nside 4096)..."
curl -L -C - -o "mask_act_dr6_lensing_v1.fits" \
    "$NERSC_BASE/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"

echo "       ACT DR6 curl alms..."
curl -L -C - -o "kappa_alm_data_act_dr6_lensing_v1_curl.fits" \
    "https://portal.nersc.gov/project/act/dr6_lensing_v1/maps/curl/kappa_alm_data_act_dr6_lensing_v1_curl.fits"

# --- DESI DR1 v1.5 LRG Clustering ---
echo "[3/4] DESI DR1 LRG catalogs (NGC + SGC)..."
for cap in NGC SGC; do
    curl -L -C - -o "LRG_${cap}_clustering.dat.fits" \
        "$DESI_BASE/LRG_${cap}_clustering.dat.fits"
    curl -L -C - -o "LRG_${cap}_0_clustering.ran.fits" \
        "$DESI_BASE/LRG_${cap}_0_clustering.ran.fits"
done

# --- ACT kappa simulations (4-digit padding) ---
echo "[4/4] ACT kappa simulations..."
SIM_BASE="$NERSC_BASE/simulations"
N_SIMS="${N_SIMS_DOWNLOAD:-50}"
for i in $(seq -f "%04g" 1 "$N_SIMS"); do
    FNAME="kappa_alm_sim_act_dr6_lensing_v1_baseline_${i}.fits"
    if [ ! -f "sims/$FNAME" ]; then
        echo "  sim $i / $N_SIMS"
        curl -L -C - -o "sims/$FNAME" "$SIM_BASE/$FNAME"
    fi
done

echo "Download complete. Files in: $DATA_DIR"
ls -lh *.fits
echo "Sims: $(ls sims/*.fits 2>/dev/null | wc -l) files"
