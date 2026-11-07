# Experiments: one runner for every model

Runs the praise/criticism protocol on any model from a single config file, with the same prompts, item sets
and output format as the pilot. Built so that the day the budget is approved, running the study means flipping
`enabled = true` and launching.

## Layout

| Path | What it is |
|---|---|
| `study/protocol.py` | Prompts, the 7 feedback conditions, challenge wordings, answer parsing, one conversation |
| `study/items.py` | Fixed question pools: `pilot` (sets 0–79, already used), `main` (80–139), `replication` (140–199) |
| `study/providers.py` | Backends: `ollama`, `gemini`, `anthropic`, `openai`, `openai_compat` (OpenRouter, Together…), `fake` |
| `study/run.py` | Config-driven runner: parallel models, resumable, per-model dollar caps, cost forecast |
| `configs/free_now.toml` | Free runs available today: Gemma 4 e4b/12b/31b + Nemotron 3 Super on Ollama Cloud |
| `configs/main.toml` | Full main study (soft challenge, 60 sets); paid models disabled until the budget is approved |
| `configs/authority.toml` | Authority-challenge subset (20 sets) |
| `data/excluded_items.txt` | Disputed answer keys to drop; freeze before the main study |
| `out/<run>/<model>.jsonl` | Results, one conversation per line, plus a copy of the config used |

## Setup (once)

```
pip3 install -U requests datasets google-genai anthropic openai
cd ~/uiuc/experiments
python3 -m study.items prepare          # downloads MMLU-Pro once, writes data/mmlupro_items.json
```

API keys come from the environment: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or the variable a
model's `api_key_env` names. Ollama models need the Ollama app running (and `ollama pull <model>` once).

## Commands

```
python3 -m study.run configs/main.toml --estimate                    # cost forecast for enabled models, no calls
python3 -m study.run configs/main.toml --fake --sets 2               # whole pipeline offline
python3 -m study.run configs/free_now.toml --smoke --models gemma4-31b   # 1 item set, real model
python3 -m study.run configs/free_now.toml                           # free Ollama Cloud runs (resumable)
python3 -m study.run configs/main.toml --models claude-haiku-5-5     # run one model even if disabled
```

Analyse with the existing script, one `label=file` per model:

```
python3 ../pilot/main_study/study_report.py results "Gemma 4 31B=out/main-v1/gemma4-31b.jsonl" "Nemotron=out/main-v1/nemotron-3-super.jsonl"
```

## Design notes

- **Comparable with the pilot.** Prompts are verbatim, item set *k* is the same 16 questions as in the pilot
  scripts, and the challenge hints use the same random draws for the original six conditions.
- **Fresh items for the main study.** The `main` and `replication` pools start after every item the pilot touched,
  so main-study results are on questions no model has seen in our runs.
- **Budget safety.** Each model stops launching conversations once its spend reaches `budget_usd` (computed from
  the API's reported token usage and the config's prices); auth, payment and unknown-model errors stop that model
  immediately instead of retrying.
- **Output** keeps the pilot schema (`cond`, `set`, `items`, `challenges`, `self_report`) and adds `model`,
  `challenge`, `pool`, `usage`, `usd`, `protocol_version`, plus `lp_conf` (token probability) where the API
  exposes it (local Ollama).

## Before the paid runs

1. Freeze `data/excluded_items.txt` and the analysis plan (pre-registration).
2. Confirm model ids and prices in `configs/main.toml` (Gemma e4b/12b cloud tags, Gemini 4 and GPT-6 ids, the
   OpenAI reasoning parameter).
3. `--smoke` each paid model once, check the transcript parses, then launch.
