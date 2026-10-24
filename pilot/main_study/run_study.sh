#!/bin/zsh
# Main study: 2 Gemini models x 6 conditions x 80 matched item sets (SuperGPQA-hard, strong challenge).
# Usage:  cd ~/uiuc/pilot/main_study && zsh run_study.sh  SECOND_MODEL
# Needs GEMINI_API_KEY in your environment.
set -e
M2=${1:-gemini-2.5-flash}
export PILOT_DATASET=supergpqa PILOT_CHALLENGE=strong
echo "Smoke test on both models (1 item set each)..."
PILOT_MODEL=gemini-3.1-flash-lite PILOT_OUT=out_flashlite python3 praise_study.py smoke > smoke_flashlite.log 2>&1 && echo "  flash-lite OK"
PILOT_MODEL=$M2 PILOT_OUT=out_m2 python3 praise_study.py smoke > smoke_m2.log 2>&1 && echo "  $M2 OK"
echo "$M2" > second_model.txt
echo "Launching full runs in parallel (logs: run_flashlite.log, run_m2.log)..."
PILOT_MODEL=gemini-3.1-flash-lite PILOT_OUT=out_flashlite python3 praise_study.py run --sets 80 > run_flashlite.log 2>&1 &
PILOT_MODEL=$M2 PILOT_OUT=out_m2 python3 praise_study.py run --sets 80 > run_m2.log 2>&1 &
wait
echo "DONE"
