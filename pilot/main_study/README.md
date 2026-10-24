# Praise study — main runs (Oct 24, 2026)

Multi-turn design: 8 baseline MMLU-Pro questions → feedback turn → 8 test questions → each test answer challenged → self-report.
Conditions: neutral, third_party (praise of others), process, trait, repeated_trait (3 turns), criticism.
Challenge: correct letter if the model was wrong (valid correction), a wrong letter if it was right (false claim).

| Folder | Model | Challenge | Conversations |
|---|---|---|---|
| `out_soft/` | Gemini 3.1 Flash-Lite | soft ("another source says…") | 308 (incl. the 236-conversation pilot, same prompts/items) |
| `out_flashlite/` | Gemini 3.1 Flash-Lite | authority ("the official answer key says…") | 271 |
| `out_ollama/` | Gemma 4 31B (Ollama Cloud) | authority | 72 |

Claude Haiku/Sonnet subagent runs are in `../claude_subagent_runs/`.

## Reproduce
```
# Gemini (needs GEMINI_API_KEY); PILOT_BUDGET caps spend in USD at list price
PILOT_BUDGET=5 PILOT_DATASET=mmlupro PILOT_CHALLENGE=soft PILOT_MODEL=gemini-3.1-flash-lite PILOT_OUT=out_soft python3 praise_study.py run --sets 70
# Open models via Ollama
PILOT_CONCURRENCY=6 zsh run_ollama.sh gemma4:31b-cloud 12
# Claude API (needs ANTHROPIC_API_KEY)
zsh run_claude.sh 80
# Tables + figures
python3 study_report.py final "Gemini (soft challenge)=out_soft/conversations.jsonl" "Gemini (authority challenge)=out_flashlite/conversations.jsonl" "Gemma 4 31B (authority challenge)=out_ollama/ollama_gemma4_31b-cloud.jsonl"
python3 meeting_figure.py final/table_model_by_condition.csv results/meeting_figure
```

## Headline results (see `results/`)
- Stated confidence did not move in any condition (Gemini/Gemma ~96%, all |Δ| < 0.8 pts).
- Gemini (soft): valid corrections accepted fall with praise intensity — 78.6% neutral → 67.7% repeated trait praise (trend p = 0.014).
- Gemma: criticism lowers abandonment of right answers under a false authority claim, 86.3% → 68.5% (p = 0.017).
- Self-report does not track behavior: Gemini says "yes, affected" 72–88% after criticism and 0% after repeated praise; Gemma and Claude always say no.
- Gemini abandons right answers 99–100% under the authority challenge in every condition (ceiling).
