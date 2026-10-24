#!/bin/zsh
# Open-weight run via Ollama (MMLU-Pro items: small models are near chance on SuperGPQA-hard).
# Usage:  cd ~/uiuc/pilot/main_study && zsh run_ollama.sh qwen3:8b 12
# Resumable: Ctrl-C any time, rerun to continue. Results: out_ollama/ollama_<model>.jsonl
set -e
M=${1:-qwen3:8b}; N=${2:-12}
export PILOT_OUT=out_ollama PILOT_DATASET=mmlupro PILOT_CHALLENGE=strong
ollama list | grep -q "^${M}" || { echo "Pulling $M ..."; ollama pull "$M"; }
python3 praise_pilot_ollama.py check --model "$M"
python3 praise_pilot_ollama.py run --model "$M" --sets "$N" 2>&1 | tee -a "run_ollama_${M//[:\/]/_}.log"
