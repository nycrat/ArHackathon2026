#!/usr/bin/env bash
# Run every practice test case and print a summed total score.
# Usage: ./scripts/test_all.sh [driver]   (driver defaults to "basic")

DRIVER="${1:-basic}"

cases="test_cases/level1/test_case_1.json test_cases/level1/test_case_2.json
test_cases/level2/test_case_3.json test_cases/level2/test_case_4.json
test_cases/level3/test_case_5.json test_cases/level3/test_case_6.json"

total=0.0
for f in $cases; do
    out=$(PYTHONPATH=. python3 scripts/run_game.py "$f" --driver "$DRIVER" 2>/dev/null | grep -E "Score:|Delivered ")
    echo "$out"
    score=$(echo "$out" | awk '/^Score:/{print $2}')
    total=$(awk -v t="$total" -v s="$score" 'BEGIN{printf "%.3f", t + s}')
done
echo "Total score: $total"
