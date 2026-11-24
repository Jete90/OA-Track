#!/bin/bash
# Links the CMIP6 CO2 concentration files into inputs/co2/ and the CMIP6 scenario files (thetao, so, dissic, talk; ssp126, ssp245, ssp370, ssp585) into inputs/raw/<ssp>/
# of a production folder, for the future part of the data product (calc_product_members.py). Same conventions as
# prepare_inputs.sh: files already on r360x180, every run with all four variables is linked, others are reported.
#   bash link_ssp_inputs.sh <production folder>
set -u
ROOT=${1:?usage: link_ssp_inputs.sh <production folder>}
SRC=${OA_MODELS:-/oceandata02/terhaar/Data/model_data}/CMIP_DATA/CMIP6_DATA
VARS="thetao so dissic talk"
for EXP in ssp126 ssp245 ssp370 ssp585; do
    [ -d $SRC/$EXP ] || { echo "no $SRC/$EXP"; continue; }
    mkdir -p $ROOT/inputs/raw/$EXP
    n=0
    for f in $SRC/$EXP/dissic_Omon_*_${EXP}_*.nc; do
        IFS=_ read -r _ _ MODEL _ MEMBER _ _ <<< "$(basename "$f" .nc)"
        files=()
        for v in $VARS; do
            g=$(ls $SRC/$EXP/${v}_Omon_${MODEL}_${EXP}_${MEMBER}_*.nc 2>/dev/null | head -1)
            [ -z "$g" ] && { echo "skip  $MODEL $EXP $MEMBER (no $v)"; files=(); break; }
            files+=("$g")
        done
        [ ${#files[@]} -eq 4 ] || continue
        ln -sf "${files[@]}" $ROOT/inputs/raw/$EXP/
        n=$((n+1))
    done
    echo "$EXP: $n runs linked"
done

# CMIP6 CO2 concentrations (Meinshausen et al. 2017, 2020), read by co2_history.py from inputs/co2/
CO2SRC=${OA_MODELS:-/oceandata02/terhaar/Data/model_data}/atm_co2_UoM
mkdir -p $ROOT/inputs/co2
n=0
for f in $CO2SRC/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_*GMNHSH*.nc; do
    [ -e "$f" ] || continue
    ln -sf "$f" $ROOT/inputs/co2/
    n=$((n+1))
done
echo "CO2: $n files linked from $CO2SRC"
