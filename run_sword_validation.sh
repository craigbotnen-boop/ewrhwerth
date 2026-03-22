#!/bin/bash
set -e

echo "==================================================="
echo " SWORD P1 Observational Pipeline: Validation Run"
echo "==================================================="

# 0. Import gate — abort early if pymaster/healpy unavailable
echo ""
echo "-> Running import gate check..."
python check_imports.py

# 1. Define Paths (Adjust if necessary)
# Using absolute paths found on disk relative to Downloads
DOWNLOADS="/mnt/c/Users/magika/Downloads"
ACT_KAPPA_ALM="$DOWNLOADS/dr6_lensing_maps/maps/baseline/kappa_alm_data_act_dr6_lensing_v1_baseline.fits"
ACT_CURL_ALM="$DOWNLOADS/dr6_lensing_maps/maps/curl/kappa_alm_data_act_dr6_lensing_v1_curl.fits"
ACT_MASK="$DOWNLOADS/dr6_lensing_maps/maps/baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"
DESI_NPZ="$DOWNLOADS/dump/desi_dr1_tomo_maps/DESI_LRG_z0.40_0.55_nside512.npz"

# 2. Generate SHA-256 Hashes for Provenance
echo "-> Hashing input files for cryptographic receipt..."
HASH_KAPPA=$(sha256sum "$ACT_KAPPA_ALM" | awk '{print $1}')
HASH_CURL=$(sha256sum "$ACT_CURL_ALM" | awk '{print $1}')
HASH_MASK=$(sha256sum "$ACT_MASK" | awk '{print $1}')
HASH_DESI=$(sha256sum "$DESI_NPZ" | awk '{print $1}')

echo "   [KAPPA] $HASH_KAPPA"
echo "   [CURL]  $HASH_CURL"
echo "   [MASK]  $HASH_MASK"
echo "   [DESI]  $HASH_DESI"

# Export hashes as environment variables so the Python script can grab them for the JSON
export PROV_HASH_KAPPA=$HASH_KAPPA
export PROV_HASH_CURL=$HASH_CURL
export PROV_HASH_MASK=$HASH_MASK
export PROV_HASH_DESI=$HASH_DESI

# 3. Execute ALM to Map Conversion
echo ""
echo "-> Executing ALM to Map Conversion (nside=512)..."
python alm_to_map_v2.py

# 4. Execute NaMaster Cross-Correlation
echo ""
echo "-> Executing NaMaster Cross-Correlation (Control Channel)..."
python desi_act_validation_v2.py

echo ""
echo "==================================================="
echo " Pipeline Complete. Check validation summary JSON."
echo "==================================================="
