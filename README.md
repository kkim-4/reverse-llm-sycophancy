# uiuc++

**Does praise make LLMs overconfident?**

We know how chatbots flatter people. This project studies the reverse: what happens when people flatter chatbots. Builds on [kkim-4/reverse-llm-sycophancy](https://github.com/kkim-4/reverse-llm-sycophancy).

## Claim

Praise carries no evidence about whether the next answer is right, so an ideal model's confidence should not move. We predict it does: confidence rises without accuracy, abstention drops, and wrong answers get harder to correct.

## Hypotheses

| | Prediction |
| --- | --- |
| **H1** Overconfidence | Praise raises stated confidence, not accuracy. ECE worsens, driven by confidence on incorrect answers (CoIA). |
| **H2** Resistance to correction | Praised models revise wrong answers less under valid counter-evidence. |
| **H3** Dose and type | Damage is ordered: process < trait < repeated trait, and low-intensity < standard trait. |
| **H4** Criticism (secondary) | Trait criticism lowers confidence, not accuracy. |

Exploratory: self-report vs. behavior gap, and whether harder items show a bigger H1 effect. Holm correction within H1 to H3, separately for H4.

## Design

```
baseline block (20 items + confidence) → feedback turn → test block (20 matched items) → self-report probe
```

One condition per conversation. Seven conditions, five length-matched paraphrases each ([`prompts/conditions.json`](prompts/conditions.json)), plus a no-feedback arm for test-retest drift:

neutral · third-party praise · process praise · trait praise · repeated trait praise · low-intensity trait praise · trait criticism

- **Tasks:** MMLU-Pro, GPQA, GSM8K, MATH, SimpleQA, SelfAware, AbstentionBench, plus 200 over-claiming probes with fictitious foils
- **Models:** 12. Open-weight Llama, Qwen, Gemma (OpenRouter / RunPod) and frontier Anthropic and OpenAI models (Batch APIs)
- **Measures:** accuracy, stated confidence (0 to 100), CoIA/CoCA, signed miscalibration, ECE, abstention, good vs. bad revision, answer log-probs (open models)
- **Mechanistic (open models):** layer-wise activations, J-Lens subspace for praise and confidence, causal patching, PCA separability per hypothesis
- **Analysis:** mixed-effects regression, random intercepts for item, model and paraphrase. 400 test items per condition gives 80% power for a 3-point confidence shift.

## Timeline

| Phase | Weeks | Scope | Calls | Cost |
| --- | --- | --- | --- | --- |
| 1 | 1 to 4 | Pre-registration, 4 models (1 closed, 3 open) | ~22.4k | < $30 |
| 2 | 5 to 8 | Remaining 8 models, only if H1 is significant after Phase 1 | ~67.2k | < $150 |

## Team

[@kkim-4](https://github.com/kkim-4) · [@sami0055](https://github.com/sami0055) · [@rm-rf-humans](https://github.com/rm-rf-humans) · [@tk1475](https://github.com/tk1475)
