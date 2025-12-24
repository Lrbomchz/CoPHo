#!/usr/bin/env bash
set -euo pipefail

# Common hyper-parameters (you can change these if needed)
PHASE_S=0.6
PHASE_E=1.0

# All four experiments (Table 2-community in manuscript)
targets=(clustering assortativity transitivity density)

for tgt in "${targets[@]}"; do
  echo "==============================="
  echo "Running experiment with condition_target=${tgt}"
  echo "  homo_condition_phase_s = ${PHASE_S}"
  echo "  homo_condition_phase_e = ${PHASE_E}"
  echo "==============================="

  python run_condi_pipeline.py \
    --homo_condition_phase_s "${PHASE_S}" \
    --homo_condition_phase_e "${PHASE_E}" \
    --condition_target "${tgt}"
done