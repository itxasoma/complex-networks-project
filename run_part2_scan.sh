#!/bin/bash
set -euo pipefail

# Run from the repository root
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
if [ -f "$SCRIPT_DIR/Makefile" ]; then
  ROOT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../Makefile" ]; then
  ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
else
  ROOT_DIR="$PWD"
fi
cd "$ROOT_DIR"

mkdir -p part2/results/raw

GAMMAS=(3.5 2.5)
SIZES=(10000 30000 50000 100000 300000 500000 1000000)

NRUNS=${NRUNS:-2000}

LMIN_35=${LMIN_35:-0.05}
LMAX_35=${LMAX_35:-0.20}
NLAM_35=${NLAM_35:-25}

LMIN_25=${LMIN_25:-0.005}
LMAX_25=${LMAX_25:-0.08}
NLAM_25=${NLAM_25:-25}

ONLY_GAMMA=${ONLY_GAMMA:-}
ONLY_N=${ONLY_N:-}
DRY_RUN=${DRY_RUN:-0}

printf 'Starting Part 2 scan at %s\n' "$(date)"
printf 'Repository root: %s\n' "$ROOT_DIR"
printf 'NRUNS=%s\n' "$NRUNS"

for GAMMA in "${GAMMAS[@]}"; do
  if [ -n "$ONLY_GAMMA" ] && [ "$GAMMA" != "$ONLY_GAMMA" ]; then
    continue
  fi

  case "$GAMMA" in
    3.5)
      LMIN="$LMIN_35"
      LMAX="$LMAX_35"
      NLAM="$NLAM_35"
      ;;
    2.5)
      LMIN="$LMIN_25"
      LMAX="$LMAX_25"
      NLAM="$NLAM_25"
      ;;
    *)
      echo "Unsupported GAMMA=$GAMMA"
      exit 1
      ;;
  esac

  for N in "${SIZES[@]}"; do
    if [ -n "$ONLY_N" ] && [ "$N" != "$ONLY_N" ]; then
      continue
    fi

    OUTFILE="part2/results/raw/part2_N${N}_g${GAMMA}.dat"
    CMD=(make run2 N="$N" GAMMA="$GAMMA" NRUNS="$NRUNS" LMIN="$LMIN" LMAX="$LMAX" NLAM="$NLAM" OUT2="$OUTFILE")

    echo
    printf 'Running gamma=%s N=%s range=[%s,%s] nlam=%s\n' "$GAMMA" "$N" "$LMIN" "$LMAX" "$NLAM"
    printf 'Output: %s\n' "$OUTFILE"

    if [ "$DRY_RUN" = "1" ]; then
      printf 'DRY_RUN: %s\n' "${CMD[*]}"
    else
      "${CMD[@]}"
    fi
  done
done

printf '\nFinished Part 2 scan at %s\n' "$(date)"
