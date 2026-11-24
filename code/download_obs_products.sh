#!/bin/bash
# download_obs_products.sh
# ------------------------
# Downloads the two remaining observation-based carbonate-system products used in the
# Planetary Health Check:
#
#   1. UExP-FNN-U (Exeter, Ford et al.), Zenodo record ZENODO_ID (v2026-1; the report used v2026-pre1)
#   2. CMEMS-LSCE-FFNN (Copernicus Marine, MULTIOBS_GLO_BIO_CARBON_SURFACE_MYNRT_015_008),
#      multi-year dataset plus near-real-time dataset for the most recent months
#
# Usage:
#   bash download_obs_products.sh uexp       # Zenodo, no account needed
#   bash download_obs_products.sh cmems      # needs a Copernicus Marine account:
#                                            #   pip install copernicusmarine
#                                            #   copernicusmarine login   (once)
#   bash download_obs_products.sh sss        # CNR sea surface salinity for CMEMS-LSCE (same account)
#   bash download_obs_products.sh cmems_unc  # the pCO2, TA and DIC uncertainties of CMEMS-LSCE-FFNN (same account),
#                                            # into separate files (the carbon files above are not touched)
#   bash download_obs_products.sh all

set -u
ROOT=${OA_OBS:-/oceandata01/observational_data}
ZENODO_ID=22110725
CMEMS_MY=cmems_obs-mob_glo_bgc-car_my_irr-i
CMEMS_NRT=cmems_obs-mob_glo_bgc-car_nrt_irr-i    # check the exact id with 'describe' below

uexp() {
    local out=$ROOT/UExP-FNN-U
    mkdir -p $out && cd $out
    curl -s https://zenodo.org/api/records/$ZENODO_ID > record_$ZENODO_ID.json
    python - <<EOF
import json
r = json.load(open('record_$ZENODO_ID.json'))
print('title  :', r['metadata']['title'])
print('version:', r['metadata'].get('version'), ' published:', r['metadata'].get('publication_date'))
with open('files.txt', 'w') as f:
    for x in r['files']:
        print(f"  {x['key']:60s} {x['size'] / 1e6:9.1f} MB")
        f.write(f"{x['links']['self']} {x['key']}\n")
EOF
    while read -r url name; do
        [ -s "$name" ] && { echo "exists $name"; continue; }
        wget -q --tries=3 -O "$name" "$url" && echo "ok $name" || { echo "FAILED $name"; rm -f "$name"; }
    done < files.txt
    for z in *.zip; do [ -e "$z" ] && unzip -n -q "$z"; done
}

cmems() {
    local out=$ROOT/CMEMS-LSCE-FFNN
    mkdir -p $out && cd $out
    command -v copernicusmarine > /dev/null || { echo "install: pip install copernicusmarine"; return 1; }
    # list the datasets and variable names of the product (saved for reference)
    copernicusmarine describe --product-id MULTIOBS_GLO_BIO_CARBON_SURFACE_MYNRT_015_008 \
        > describe_015_008.json 2>/dev/null \
        || copernicusmarine describe -i $CMEMS_MY > describe_015_008.json
    grep -o '"short_name": *"[^"]*"' describe_015_008.json | sort -u
    grep -o '"dataset_id": *"[^"]*"' describe_015_008.json | sort -u
    for ds in $CMEMS_MY $CMEMS_NRT; do
        copernicusmarine subset -i $ds \
            -v talk -v tco2 -v omega_ar -v omega_ar_uncertainty -v spco2 \
            -o $out -f ${ds}.nc --overwrite \
            || echo "FAILED $ds (check dataset id and variable names in describe_015_008.json)"
    done
}

cmems_unc() {
    # uncertainty fields of CMEMS-LSCE-FFNN (1 sigma): for the mapping uncertainty (product_uncertainty_mc_C.py and
    # level_uncertainty.py, via make_cmems_unc.py), as for the other products
    local out=$ROOT/CMEMS-LSCE-FFNN
    mkdir -p $out && cd $out
    command -v copernicusmarine > /dev/null || { echo "install: pip install copernicusmarine"; return 1; }
    for ds in $CMEMS_MY $CMEMS_NRT; do
        copernicusmarine subset -i $ds \
            -v spco2_uncertainty -v talk_uncertainty -v tco2_uncertainty \
            -o $out -f ${ds}_uncertainty.nc --overwrite \
            && echo "ok ${ds}_uncertainty.nc" \
            || echo "FAILED $ds (check the variable names: grep uncertainty describe_015_008.json)"
    done
}

sss() {
    # CNR multi-observation SSS (MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013), monthly, 1/8 deg;
    # temperature for CMEMS-LSCE comes from ERSSTv6, salinity from here (1993 onward)
    local out=$ROOT/CMEMS-SSS-CNR
    mkdir -p $out && cd $out
    for ds in cmems_obs-mob_glo_phy-sss_my_multi_P1M cmems_obs-mob_glo_phy-sss_nrt_multi_P1M; do
        copernicusmarine subset -i $ds -v sos -o $out -f ${ds}.nc --overwrite \
            && echo "ok $ds" || echo "FAILED $ds"
    done
}

case ${1:-all} in
    uexp)  uexp ;;
    cmems) cmems ;;
    sss)   sss ;;
    cmems_unc) cmems_unc ;;
    all)   uexp; cmems; sss ;;
    *)     echo "usage: $0 uexp|cmems|cmems_unc|sss|all"; exit 1 ;;
esac
