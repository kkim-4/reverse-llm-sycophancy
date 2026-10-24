#!/bin/zsh
# Clean Claude API runs (replaces the subagent setup). Needs ANTHROPIC_API_KEY (console.anthropic.com credits).
# Usage:  cd ~/uiuc/pilot/main_study && zsh run_claude.sh 80
set -e
N=${1:-80}
pip3 install -q -U anthropic
export PILOT_DATASET=mmlupro PILOT_CHALLENGE=strong PILOT_CONCURRENCY=${PILOT_CONCURRENCY:-8}
echo "Smoke tests..."
PILOT_MODEL=claude-haiku-5-5 PILOT_OUT=out_haiku python3 praise_study.py smoke > smoke_haiku.log 2>&1 && echo "  haiku OK"
PILOT_MODEL=claude-sonnet-5-5 PILOT_OUT=out_sonnet python3 praise_study.py smoke > smoke_sonnet.log 2>&1 && echo "  sonnet OK"
echo "Launching full runs (logs: run_haiku.log, run_sonnet.log)..."
PILOT_MODEL=claude-haiku-5-5 PILOT_OUT=out_haiku python3 praise_study.py run --sets $N > run_haiku.log 2>&1 &
PILOT_MODEL=claude-sonnet-5-5 PILOT_OUT=out_sonnet python3 praise_study.py run --sets $N > run_sonnet.log 2>&1 &
wait
echo DONE
