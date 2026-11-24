#!/bin/bash
# prepare_inputs.sh
# -----------------
# 1. Links the CMIP6 files used in this study into data/raw/<exp>/ for every
#    historical and piControl run with thetao, so, dissic and talk. These files
#    are already on r360x180 (remapdis) and standard depth levels (intlevelx,
#    top level 5 m), so no further processing is needed.
# 2. Remaps the GLODAPv2.2016b surface Si and PO4 climatology to the same
#    r360x180 grid and fills gaps by nearest neighbour -> data/obs/.

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)          # the production folder (the parent of code/)
SRC=${OA_MODELS:-/oceandata02/terhaar/Data/model_data}/CMIP_DATA/CMIP6_DATA
# the GLODAPv2.2016b mapped climatology (Lauvset et al. 2016), as downloaded from glodap.info
GLODAP_DIR=${OA_OBS2:-/oceandata02/terhaar/Data/observation_based}/GLODAPv2
GRID=r360x180
VARS="thetao so dissic talk"

mkdir -p $ROOT/data/raw $ROOT/data/obs/raw $ROOT/logs
LOG=$ROOT/logs/prepare_inputs_$(date +%Y%m%d_%H%M).log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

[ -d "$GLODAP_DIR" ] || { log "GLODAP_DIR not set or missing: $GLODAP_DIR"; exit 1; }

for EXP in historical piControl; do
    [ $EXP = historical ] && SUB=historical || SUB=picontrol
    mkdir -p $ROOT/data/raw/$EXP
    for f in $SRC/$SUB/dissic_Omon_*_${EXP}_*.nc; do
        IFS=_ read -r _ _ MODEL _ MEMBER _ _ <<< "$(basename "$f" .nc)"
        files=()
        for v in $VARS; do
            g=$(ls $SRC/$SUB/${v}_Omon_${MODEL}_${EXP}_${MEMBER}_*.nc 2>/dev/null | head -1)
            [ -z "$g" ] && { log "skip  $MODEL $EXP $MEMBER (no $v)"; files=(); break; }
            files+=("$g")
        done
        [ ${#files[@]} -eq 4 ] || continue
        ln -sf "${files[@]}" $ROOT/data/raw/$EXP/
        log "link  $MODEL $EXP $MEMBER"
    done
done

for var in silicate PO4; do
    in=$GLODAP_DIR/GLODAPv2.2016b.$var.nc
    out=$ROOT/data/obs/${var}_GLODAPv2.2016b_surface_${GRID}.nc
    ln -sf "$in" $ROOT/data/obs/raw/
    [ -s "$out" ] && { log "exists $(basename "$out")"; continue; }
    cdo -s -f nc4 setmisstonn -remapdis,$GRID -sellevidx,1 -selname,$var "$in" "$out" >>"$LOG" 2>&1 \
        && log "cdo   $(basename "$out")" || log "FAILED $(basename "$out")"
done
log "done"
