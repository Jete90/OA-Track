#!/bin/bash
# run_structural.sh: the structural adjustments of the decline, in order (start from code/):
#   OA_RUN=u74 nohup bash run_structural.sh > ../logs/u74/run_structural.log 2>&1 &
#   OA_RUN=u74 nohup bash run_structural.sh polar > ../logs/u74/run_structural.log 2>&1 &   (the pattern and the sampling as they are)
#   OA_RUN=u74 nohup bash run_structural.sh structural > ../logs/u74/run_structural.log 2>&1 &   (step 3 only)
# 1 test_polar_shift_models.py (writes polar_models.npz) and feedback_pattern.py side by side (independent);
#   with "polar" only test_polar_shift_models.py; skipped with "structural"
# 2 barrett_sampling.py (prints p with the pattern); skipped with "polar" and "structural"
# 3 structural_adjustments.py (central; reads polar_models.npz, feedback_pattern.npz, barrett_sampling.npz) and eight
#   pattern sensitivities side by side: each of the four feedback models alone, three smoothings, uniform
# Stops at the first failing step, so that nothing runs on stale input.
set -e
cd "$(dirname "$0")"
export OA_RUN=${OA_RUN:-u74}
MODE=${1:-all}
L=../logs/$OA_RUN
mkdir -p "$L"

if [ "$MODE" = structural ]; then
    echo "$(date '+%F %T') step 1 and 2 skipped (polar test, pattern and sampling kept)"
elif [ "$MODE" = polar ]; then
    echo "$(date '+%F %T') step 1: test_polar_shift_models.py (pattern and sampling kept)"
    python -u test_polar_shift_models.py > "$L/test_polar_shift_models.log" 2>&1 || { echo "test_polar_shift_models.py failed"; exit 1; }
else
    echo "$(date '+%F %T') step 1: test_polar_shift_models.py, feedback_pattern.py"
    python -u test_polar_shift_models.py > "$L/test_polar_shift_models.log" 2>&1 & p1=$!
    python -u feedback_pattern.py > "$L/feedback_pattern.log" 2>&1 & p2=$!
    wait $p1 || { echo "test_polar_shift_models.py failed"; exit 1; }
    wait $p2 || { echo "feedback_pattern.py failed"; exit 1; }
fi
grep -q "^written .*polar_models.npz" "$L/test_polar_shift_models.log" || { echo "polar_models.npz not written"; exit 1; }
if [ "$MODE" = all ]; then
    echo "$(date '+%F %T') step 2: barrett_sampling.py"
    python -u barrett_sampling.py > "$L/barrett_sampling.log" 2>&1
fi
for f in polar_models feedback_pattern barrett_sampling; do
    [ -f "../data/$OA_RUN/structural/$f.npz" ] || { echo "../data/$OA_RUN/structural/$f.npz missing"; exit 1; }
done

echo "$(date '+%F %T') step 3: structural_adjustments.py, central and sensitivities"
pids=()
python -u structural_adjustments.py > "$L/structural_adjustments.log" 2>&1 & pids+=($!)
for p in pattern_esm2m pattern_GFDL-CM4 pattern_GFDL-ESM4 pattern_UKESM1-0-LL pattern_s0 pattern_s2 pattern_s10 uniform; do
    python -u structural_adjustments.py --pattern $p > "$L/structural_adjustments_$p.log" 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo "a structural_adjustments.py run failed, see the logs"; exit 1; }
echo "$(date '+%F %T') done"
