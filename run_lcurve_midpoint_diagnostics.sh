#!/usr/bin/env bash

set -u

source /home/firedrake/firedrake/bin/activate

REPO="/home/firedrake/icepack/icepack-inverse"

CONFIG="$REPO/production_workflow/amundsen_production_config.json"

OUT="$REPO/production_runs/lcurve_midpoint_diagnostics_20260817_a"

mkdir -p "$OUT"

run_point () {
    REGC="$1"
    RUNID="$2"

    echo
    echo "===================================================="
    echo "Running diagnostic L-curve point"
    echo "reg_C = $REGC"
    echo "run_id = $RUNID"
    echo "===================================================="
    echo

    /home/firedrake/firedrake/bin/python \
        "$REPO/production_workflow/production_amundsen.py" \
        lcurve-point \
        --config "$CONFIG" \
        --repo-root "$REPO" \
        --output-root "$OUT" \
        --run-id "$RUNID" \
        --reg-c "$REGC"

    RC=$?

    echo
    echo "Finished reg_C=$REGC with exit code $RC"
    echo

    return $RC
}


run_point 0.01189207115 diag_regc_0p011892
run_point 0.01414213562 diag_regc_0p014142
run_point 0.01681792831 diag_regc_0p016818

echo
echo "All requested diagnostic points attempted."