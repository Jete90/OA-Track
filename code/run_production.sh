#!/bin/bash
# run_production.sh: the whole chain of one run, from the raw model output to the data product with its add-on,
# unattended, stopping at the first failing step (nothing runs on stale input). Start from code/ in env_jens:
#   mkdir -p ../logs/u74; JOBS=4 nohup ./run_production.sh u74 > ../logs/u74/run_production.out 2>&1 &
# (stdout to run_production.out, not to run_production.log: log() appends to the master log itself)
# Steps (each with its own log in logs/<run>/, the master log run_production.log):
#   1 rerun.sh            the main analysis: raw Omega, products, drift, bias adjustment, preindustrial states,
#                         mechanism states, the emergent constraint on the own and common masks, Jiang et al.
#   2 rerun_product.sh    the data product: smoothed model states, the products' carbonate system, the constrained
#                         states of every reference (the long step, references in parallel), the product files
#   3 run_structural.sh   the two adjustments: polar test, feedback pattern and series, Barrett sampling, central and
#                         six sensitivity runs of structural_adjustments.py
#   4 make_product_alkalinity.py   the add-on per reference (in parallel), then the combined files
#   5 patch_product_xco2.py        the CO2 of the product and scenario years
#   6 summarize_product.py, diagnose_product.py   the text summaries and the check of the files
# Afterwards: the notebook (OA_boundary_figures_*.ipynb) top to bottom, and the tar of data/<run>/figures/manuscript.
# JOBS: processes in parallel inside the steps (default 4); REFS to restrict the product steps; SKIP="1 2" to skip
# steps already complete (their outputs are then used as they are). PARALLEL_REFS=0: the references of step 2 one
# after the other (about 5 GB per process, JOBS processes per reference).
set -e
RUN=$1
[ -n "$RUN" ] || { echo "usage: ./run_production.sh <run>  (runs in oa_config.py)"; exit 1; }
cd "$(dirname "$0")"
export OA_RUN=$RUN
export JOBS=${JOBS:-4}
BASE=$(python -c "import oa_config; print(oa_config.BASE)")
DATA=$(python -c "import oa_config; print(oa_config.DATA)")
L=$BASE/logs/$RUN
mkdir -p "$L"
REFS=${REFS:-"OceanSODA UExP JMA CMEMS Jiang"}
SKIP=${SKIP:-""}
log() { echo "$(date '+%F %T') $*" | tee -a "$L/run_production.log"; }
skip() { [[ " $SKIP " == *" $1 "* ]]; }
chmod +x rerun.sh rerun_product.sh run_structural.sh 2>/dev/null || true

log "run $RUN in $BASE ($(python -c 'import oa_config; print(oa_config.VERSION)')); JOBS $JOBS; REFS $REFS; SKIP '${SKIP}'"
log "area weights: $(python -c 'import oa_config; print(oa_config.WEIGHTS_TEXT)')"
if [ "$(python -c 'import oa_config; print(oa_config.WEIGHTS)')" = ocean ]; then
    [ -f "$BASE/inputs/ESACCI_water_bodies/ocean_fraction_1deg.nc" ] || { log "inputs/ESACCI_water_bodies/ocean_fraction_1deg.nc missing (OA_WEIGHTS=ocean)"; exit 1; }
fi
python -c "import co2_history as c; h = c.CO2History(); assert not h.missing, 'CO2 files missing: ' + str(h.missing)" \
    || { log "CMIP6 CO2 files missing in $BASE/inputs/co2 (co2_history.py, link_ssp_inputs.sh)"; exit 1; }
python -c "import co2_history as c; assert c.noaa(), 'NOAA record missing'" || { log "NOAA CO2 record missing (inputs/co2_annmean_gl.txt)"; exit 1; }

if skip 1; then log "1 main analysis skipped (SKIP)"; else
    log "1 main analysis: rerun.sh $RUN"
    ./rerun.sh $RUN > "$L/rerun.out" 2>&1 || { log "1 FAILED, see $L/rerun.out and $L/rerun.log"; exit 1; }
    log "1 done: $(tail -n 1 $L/rerun.log)"
fi
[ -f "$DATA/constraint_results.pkl" ] || { log "no $DATA/constraint_results.pkl"; exit 1; }

if skip 2; then log "2 data product skipped (SKIP)"; else
    log "2 data product: rerun_product.sh $RUN (references in parallel)"
    PR=${PARALLEL_REFS:-1}; [ "$PR" = 0 ] && PR=""
    PARALLEL_REFS="$PR" REFS="$REFS" ./rerun_product.sh $RUN > "$L/rerun_product.out" 2>&1 || { log "2 FAILED, see $L/rerun_product.out and $L/rerun_product.log"; exit 1; }
    log "2 done: $(tail -n 1 $L/rerun_product.log)"
fi
for r in $REFS; do
    [ -f "$DATA/product/carbonate_system_$r.nc" ] || { log "no $DATA/product/carbonate_system_$r.nc"; exit 1; }
done

if skip 3; then log "3 structural adjustments skipped (SKIP)"; else
    log "3 structural adjustments: run_structural.sh all"
    ./run_structural.sh all > "$L/run_structural.log" 2>&1 || { log "3 FAILED, see $L/run_structural.log"; exit 1; }
    log "3 done: $(grep -E 'whole ocean .* points' $L/structural_adjustments.log | tail -n 1)"
fi
[ -f "$DATA/structural/structural_adjustments.npz" ] || { log "no structural_adjustments.npz"; exit 1; }

if skip 4; then log "4 alkalinity add-on skipped (SKIP)"; else
    log "4 alkalinity add-on: references in parallel, then the combined files"
    pids=(); fail=0
    for r in $REFS; do
        python -u make_product_alkalinity.py --reference $r > "$L/product_alkalinity_$r.log" 2>&1 & pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || fail=1; done
    [ $fail -eq 0 ] || { log "4 FAILED for a reference, see $L/product_alkalinity_<ref>.log"; exit 1; }
    if [ -f "$DATA/product/carbonate_system_common.nc" ]; then
        python -u make_product_alkalinity.py --combined > "$L/product_alkalinity_combined.log" 2>&1 || { log "4 FAILED (combined), see $L/product_alkalinity_combined.log"; exit 1; }
    else
        log "4 no combined files (REFS restricted?): combined add-on skipped"
    fi
    log "4 done"
fi

if skip 5; then log "5 xco2 patch skipped (SKIP)"; else
    log "5 the CO2 of the product and scenario years: patch_product_xco2.py"
    python -u patch_product_xco2.py > "$L/patch_product_xco2.log" 2>&1 || { log "5 FAILED, see $L/patch_product_xco2.log"; exit 1; }
    log "5 done"
fi

if skip 6; then log "6 summaries skipped (SKIP)"; else
    log "6 summaries and the check of the files"
    python -u summarize_product.py > "$L/summarize_product.log" 2>&1 || log "6 summarize_product.py failed, see the log (not fatal)"
    python -u diagnose_product.py --models > "$L/diagnose_product.log" 2>&1 || log "6 diagnose_product.py failed, see the log (not fatal)"
    log "6 done; sections A to F of $L/diagnose_product.log should report none or explain what is left"
fi
log "all done: now the notebook, then tar czf ~/oa_figures.tgz -C $DATA/figures manuscript"
