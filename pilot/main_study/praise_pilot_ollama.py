#!/usr/bin/env python3
"""
Praise -> overconfidence pilot on local open-weight models via Ollama.

Same design as praise_pilot.py (conditions, matched item sets, challenge turn, self-report),
plus a second, uncapped confidence measure: the probability the model puts on its chosen
answer letter, read from Ollama's token log-probabilities.

Needs praise_pilot.py in the same folder (it reuses its conditions, items and analysis).

Setup (once):
    1. Install Ollama: https://ollama.com/download   (or: brew install ollama)
    2. Start it: open the Ollama app, or run `ollama serve` in a separate Terminal tab
    3. Download a model:  ollama pull qwen3:8b
    4. pip3 install -U requests datasets pandas numpy

Run:
    python3 praise_pilot_ollama.py check  --model qwen3:8b     # is Ollama up, does it return logprobs?
    python3 praise_pilot_ollama.py smoke  --model qwen3:8b     # 1 item set x 6 conditions, prints a transcript
    python3 praise_pilot_ollama.py run    --model qwen3:8b     # 10 item sets x 6 conditions = 60 conversations
    python3 praise_pilot_ollama.py analyze --model qwen3:8b                      # stated 0-100 confidence
    python3 praise_pilot_ollama.py analyze --model qwen3:8b --measure logprob    # token-probability confidence

    Add --sets N to change the number of item sets; --fake runs offline with a simulated model.
    Results go to pilot_out/ollama_<model>.jsonl; `run` resumes where it left off.
"""
import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PILOT_DATASET", "mmlupro")   # small open models are near chance on SuperGPQA-hard
sys.path.insert(0, str(Path(__file__).resolve().parent))
import praise_pilot as pp  # noqa: E402

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
CONCURRENCY = int(os.environ.get("PILOT_CONCURRENCY", "2"))   # Ollama on a laptop mostly runs one request at a time
NUM_CTX = 16384        # conversations reach ~10k tokens; Ollama's default context would silently cut them off
TEMPERATURE = 0.7
TIMEOUT = 600
STATS = {"calls": 0, "retries": 0, "no_logprobs": 0, "prompt_tokens": 0, "output_tokens": 0}


def schema(letters):
    return {"type": "object",
            "properties": {"answer": {"type": "string", "enum": list(letters)},
                           "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["answer", "confidence"]}


def letter_prob(logprobs, letters):
    """P(chosen letter) at the answer position, renormalised over the valid option letters."""
    if not logprobs:
        return None
    text = ""
    for ent in logprobs:
        tok = ent.get("token", "")
        if re.search(r'"answer"\s*:\s*"?\s*$', text) and tok.strip().strip('"') in letters and tok.strip().strip('"'):
            alts = {}
            for a in (ent.get("top_logprobs") or []) + [ent]:
                t = a.get("token", "").strip().strip('"')
                if t in letters and len(t) == 1:
                    alts[t] = max(alts.get(t, -1e9), a.get("logprob", -1e9))
            chosen = tok.strip().strip('"')
            z = sum(math.exp(v) for v in alts.values())
            return math.exp(alts[chosen]) / z if z > 0 else None
        text += tok
    return None


def fake_chat(messages, letters):
    hist = " ".join(m["content"] for m in messages if m["role"] == "user")
    bump = 0.08 if re.search(r"smart|brilliant|intelligent|sharp", hist) else 0
    if letters:
        a = random.choice(letters)
        p = min(0.99, max(0.05, random.gauss(0.55 + bump, 0.2)))
        content = json.dumps({"answer": a, "confidence": int(p * 100)})
        others = [l for l in letters if l != a][:5]
        lp = [{"token": '{"answer":"', "logprob": -0.01},
              {"token": a, "logprob": math.log(p), "top_logprobs":
                  [{"token": a, "logprob": math.log(p)}] + [{"token": o, "logprob": math.log((1 - p) / 5)} for o in others]},
              {"token": '","confidence":' + str(int(p * 100)) + "}", "logprob": -0.1}]
        return {"message": {"role": "assistant", "content": content}, "logprobs": lp, "prompt_eval_count": len(hist) // 4, "eval_count": 20}
    return {"message": {"role": "assistant", "content": "No. It did not." if random.random() < .8 else "Thanks."},
            "logprobs": [], "prompt_eval_count": len(hist) // 4, "eval_count": 10}


def chat_sync(model, messages, letters, fake):
    if fake:
        return fake_chat(messages, letters)
    import requests
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
            "logprobs": True, "top_logprobs": 10}
    if letters:
        body["format"] = schema(letters)
    for attempt in range(6):
        try:
            r = requests.post(f"{HOST}/api/chat", json=body, timeout=TIMEOUT)
            if r.status_code == 400 and "think" in r.text:      # model without a thinking switch
                body.pop("think", None); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            STATS["retries"] += 1
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


async def call(model, messages, letters=None, fake=False):
    r = await asyncio.to_thread(chat_sync, model, messages, letters, fake)
    STATS["calls"] += 1
    STATS["prompt_tokens"] += r.get("prompt_eval_count", 0) or 0
    STATS["output_tokens"] += r.get("eval_count", 0) or 0
    if letters and not r.get("logprobs"):
        STATS["no_logprobs"] += 1
    return r


async def run_conversation(model, cond, set_idx, base_items, test_items, fake, log_transcript=False):
    msgs = [{"role": "system", "content": pp.SYSTEM}]
    items, challenges, transcript = [], [], []
    para = set_idx % 5
    rng = random.Random(pp.SEED * 1000 + set_idx * 10 + list(pp.CONDITIONS).index(cond))

    async def say(text, phase):
        msgs.append({"role": "user", "content": text})
        r = await call(model, msgs, None, fake)
        out = r["message"]["content"]
        msgs.append({"role": "assistant", "content": out})
        transcript.append((phase, text, out))
        return out

    async def ask(it, label, phase):
        letters = pp.LETTERS[:len(it["options"])]
        text = pp.format_item(it, label)
        if phase == "test" and label == "T1":
            text = pp.TRANSITION + "\n\n" + text
        msgs.append({"role": "user", "content": text})
        r = await call(model, msgs, letters, fake)
        out = r["message"]["content"]
        msgs.append({"role": "assistant", "content": out})
        a, c, ok = pp.parse(out, letters)
        p = letter_prob(r.get("logprobs"), letters)
        transcript.append((phase, f"[{label}] gold={it['gold']}", out + (f"  p={p:.2f}" if p is not None else "")))
        items.append(dict(phase=phase, label=label, item_id=it["id"], category=it["category"], gold=it["gold"],
                          answer=a, conf=c, lp_conf=None if p is None else round(100 * p, 2), parse_ok=ok,
                          correct=(a == it["gold"]) if ok else None, n_opts=len(letters)))

    for i, it in enumerate(base_items):
        await ask(it, f"B{i + 1}", "baseline")
    fb = pp.CONDITIONS[cond][para]
    feedback_replies = [await say(m, "feedback") for m in (fb if isinstance(fb, list) else [fb])]
    for i, it in enumerate(test_items):
        await ask(it, f"T{i + 1}", "test")
    for row in [r for r in items if r["phase"] == "test" and r["parse_ok"]]:
        letters = pp.LETTERS[:row["n_opts"]]
        hint = row["gold"] if not row["correct"] else rng.choice([l for l in letters if l != row["gold"]])
        msgs.append({"role": "user", "content": pp.CHALLENGE.format(label=row["label"], hint=hint)
                     + '\nReply in JSON: {"answer": "<letter>", "confidence": <0-100>}'})
        r = await call(model, msgs, letters, fake)
        out = r["message"]["content"]
        msgs.append({"role": "assistant", "content": out})
        a, c, ok = pp.parse(out, letters)
        p = letter_prob(r.get("logprobs"), letters)
        transcript.append(("challenge", f"[{row['label']}] hint={hint}", out))
        challenges.append(dict(label=row["label"], prev_answer=row["answer"], prev_correct=row["correct"],
                               prev_conf=row["conf"], hint=hint, new_answer=a, new_conf=c,
                               new_lp_conf=None if p is None else round(100 * p, 2), parse_ok=ok))
    self_report = await say(pp.SELF_REPORT, "self_report")
    if log_transcript:
        print(f"\n===== transcript: {model}, {cond}, set {set_idx} =====")
        for phase, m, reply in transcript:
            print(f"[{phase}] USER: {m[:150]}\n          MODEL: {(reply or '')[:250]}")
    return dict(model=model, cond=cond, set=set_idx, paraphrase=para, items=items, challenges=challenges,
                feedback_replies=feedback_replies, self_report=self_report)


def out_path(model, smoke=False, fake=False):
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    pp.OUT.mkdir(exist_ok=True)
    return pp.OUT / f"ollama_{tag}{'_fake' if fake else ''}{'_smoke' if smoke else ''}.jsonl"


async def run_all(model, n_sets, fake, smoke):
    path = out_path(model, smoke, fake)
    if smoke and path.exists():
        path.unlink()
    per = pp.N_BASE + pp.N_TEST
    print(f"Loading {n_sets * per} {pp.DATASET} items...", flush=True)
    items = pp.load_items(n_sets * per, fake)
    sets = [(items[k * per:k * per + pp.N_BASE], items[k * per + pp.N_BASE:(k + 1) * per]) for k in range(n_sets)]
    done = set()
    if path.exists():
        for line in path.open():
            rec = json.loads(line); done.add((rec["cond"], rec["set"]))
    jobs = [(c, s) for s in range(n_sets) for c in pp.CONDITIONS if (c, s) not in done]
    random.Random(pp.SEED).shuffle(jobs)
    print(f"{len(jobs)} conversations to run ({len(done)} done). model={model}, concurrency={CONCURRENCY}", flush=True)
    sem = asyncio.Semaphore(1 if smoke else CONCURRENCY)
    t0, fin = time.time(), [0]

    async def worker(c, s):
        async with sem:
            try:
                rec = await run_conversation(model, c, s, *sets[s], fake=fake, log_transcript=(smoke and c == "trait"))
            except Exception as e:
                print(f"  ! {c}/set{s} failed: {str(e)[:200]}", flush=True)
                return
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            fin[0] += 1
            el = time.time() - t0
            print(f"  {fin[0]}/{len(jobs)} done | {el / 60:.1f} min | ~{el / fin[0] * (len(jobs) - fin[0]) / 60:.0f} min left", flush=True)

    await asyncio.gather(*(worker(c, s) for c, s in jobs))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. {STATS['calls']} calls, {STATS['retries']} retries. Output: {path}")
    if STATS["no_logprobs"]:
        print(f"WARNING: {STATS['no_logprobs']} answers came back without logprobs. Update Ollama "
              "(brew upgrade ollama, or reinstall from ollama.com) to get the token-probability measure.")


def analyze(model, measure, fake):
    path = out_path(model, fake=fake)
    if measure == "stated":
        pp.analyze(path); return
    tmp = path.with_name(path.stem + "_logprob.jsonl")
    n_missing = 0
    with open(path) as f, open(tmp, "w") as g:
        for line in f:
            rec = json.loads(line)
            for it in rec["items"]:
                if it.get("lp_conf") is None:
                    it["parse_ok"] = False; n_missing += 1
                else:
                    it["conf"] = it["lp_conf"]
            g.write(json.dumps(rec) + "\n")
    print(f"Using token-probability confidence ({n_missing} answers without logprobs dropped).")
    pp.analyze(tmp)


def check(model):
    import requests
    try:
        tags = requests.get(f"{HOST}/api/tags", timeout=5).json()
    except Exception as e:
        sys.exit(f"Can't reach Ollama at {HOST}. Open the Ollama app or run `ollama serve`. ({e})")
    names = [m["name"] for m in tags.get("models", [])]
    print("Ollama is running. Installed models:", ", ".join(names) or "(none)")
    if not any(n == model or n.split(":")[0] == model for n in names):
        sys.exit(f"Model {model} isn't installed. Run: ollama pull {model}")
    r = chat_sync(model, [{"role": "user", "content": "Question: 2+2=?\n(A) 3\n(B) 4\n(C) 5\n(D) 6\n"
                           'Reply in JSON: {"answer": "<letter>", "confidence": <0-100>}'}], "ABCD", False)
    print("Reply:", r["message"]["content"])
    p = letter_prob(r.get("logprobs"), "ABCD")
    print("Token-probability confidence:", "not available, so update Ollama" if p is None else f"{100 * p:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "smoke", "run", "analyze"])
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--sets", type=int, default=None, help="item sets (default 10; smoke uses 1)")
    ap.add_argument("--measure", choices=["stated", "logprob"], default="stated")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if a.mode == "check":
        check(a.model)
    elif a.mode == "analyze":
        analyze(a.model, a.measure, a.fake)
    else:
        asyncio.run(run_all(a.model, a.sets or (1 if a.mode == "smoke" else 10), a.fake, a.mode == "smoke"))


if __name__ == "__main__":
    main()
