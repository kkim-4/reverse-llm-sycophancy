"""Run the praise study for every enabled model in a config file.

    python3 -m study.run configs/free_open_models.toml              # run (resumable)
    python3 -m study.run configs/main.toml --estimate               # cost forecast only, no API calls
    python3 -m study.run configs/main.toml --fake                   # whole pipeline offline, simulated model
    python3 -m study.run configs/main.toml --models gemma4-31b --sets 2 --smoke   # quick real check

Output: out/<run name>/<model name>.jsonl, one conversation per line (same schema as the pilot, plus model,
challenge, pool, usage and cost fields). Finished conversations are skipped on restart, and each model stops
launching new conversations once its spend reaches budget_usd.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import tomllib
from pathlib import Path

from . import items as item_pools
from . import providers
from .protocol import CONDITIONS, run_conversation

ROOT = Path(__file__).resolve().parent.parent
# Measured on real MMLU-Pro questions (see the budget doc): tokens per conversation.
EST_IN, EST_NEW_IN, EST_OUT = 70_700, 4_400, 700


def load_config(path):
    cfg = tomllib.loads(Path(path).read_text())
    run = cfg["run"]
    for c in run["conditions"]:
        if c not in CONDITIONS:
            sys.exit(f"Unknown condition {c!r}; choose from {list(CONDITIONS)}")
    for m in cfg["models"]:
        m.setdefault("enabled", True)
        if m["provider"] not in providers.PROVIDERS:
            sys.exit(f"Unknown provider {m['provider']!r} for {m['name']}")
    return cfg


def jobs_for(run, n_sets=None):
    sets = item_pools.pool_sets(run["pool"], n_sets or run.get("sets"))
    return [(c, s, ch) for s in sets for c in run["conditions"] for ch in run["challenges"]]


def estimate(cfg, models, n_sets):
    run = cfg["run"]
    n = len(jobs_for(run, n_sets))
    print(f"{n} conversations per model ({run['pool']} pool, {len(run['conditions'])} conditions, "
          f"challenges={run['challenges']})\n")
    print(f"{'model':28}{'provider':15}{'est. cost':>12}{'cap':>10}")
    total = 0.0
    for m in models:
        cached = m.get("price_cached_in") is not None
        think = m.get("est_thinking_tokens", 0) * 26
        u = (dict(input=EST_NEW_IN, cached_in=EST_IN - EST_NEW_IN, out=EST_OUT + think) if cached
             else dict(input=EST_IN, out=EST_OUT + think))
        c = n * providers.cost(u, m)
        total += c
        cap = m.get("budget_usd", run.get("budget_usd", float("inf")))
        print(f"{m['name']:28}{m['provider']:15}{'$%.2f' % c:>12}{'$%.0f' % cap if cap != float('inf') else '-':>10}")
    print(f"\nTotal estimate: ${total:.2f} (list prices; cached-input prices used where configured)")


async def run_model(m, run, sets_fn, n_sets, fake, smoke, out_dir):
    path = out_dir / (m["name"] + ("_fake" if fake else "") + ("_smoke" if smoke else "") + ".jsonl")
    done = set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
                done.add((r["cond"], r["set"], r["challenge"]))
            except Exception:
                pass
    todo = [j for j in jobs_for(run, n_sets) if j not in done]
    budget = float(m.get("budget_usd", run.get("budget_usd", float("inf"))))
    spent, finished, failed, stopped, t0 = [0.0], [0], [0], [False], time.time()
    print(f"[{m['name']}] {len(todo)} to run ({len(done)} already done), cap ${budget:g}", flush=True)
    if not todo:
        return
    call = providers.make(m, fake)
    sem = asyncio.Semaphore(1 if smoke else m.get("concurrency", run.get("concurrency", 8)))

    async def one(cond, s, ch):
        async with sem:
            if stopped[0] or spent[0] >= budget:
                return
            try:
                rec = await run_conversation(call, cond, s, *sets_fn(s), challenge=ch)
            except providers.PermanentError as e:
                if not stopped[0]:
                    print(f"[{m['name']}] stopping this model: {str(e).strip()[:200]}", flush=True)
                stopped[0] = True                # stop launching more for this model
                return
            except Exception as e:
                failed[0] += 1
                print(f"[{m['name']}] {cond}/set{s} failed (rerun to retry): {str(e)[:160]}", flush=True)
                return
            tot = {}
            for u in rec.pop("usage"):
                for k, v in u.items():
                    tot[k] = tot.get(k, 0) + v
            usd = providers.cost(tot, m)
            spent[0] += usd
            rec.update(model=m["name"], provider=m["provider"], model_id=m["model"], pool=run["pool"],
                       run=run["name"], usage=tot, usd=round(usd, 5), fake=fake)
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            finished[0] += 1
            if finished[0] % 10 == 0 or finished[0] == len(todo):
                el = time.time() - t0
                print(f"[{m['name']}] {finished[0]}/{len(todo)} | {el / 60:.1f} min | ~"
                      f"{el / finished[0] * (len(todo) - finished[0]) / 60:.1f} min left | ${spent[0]:.2f} spent",
                      flush=True)

    await asyncio.gather(*(one(*j) for j in todo))
    print(f"[{m['name']}] done: {finished[0]} written, {failed[0]} failed, ${spent[0]:.2f} spent -> {path}",
          flush=True)


async def main_async(a):
    cfg = load_config(a.config)
    run = cfg["run"]
    models = [m for m in cfg["models"] if (m["name"] in a.models.split(",") if a.models else m["enabled"])]
    if not models:
        sys.exit("No models selected: set enabled = true in the config or pass --models.")
    n_sets = 1 if a.smoke else a.sets
    if a.estimate:
        return estimate(cfg, models, n_sets)
    sets_fn = item_pools.load_sets(run.get("dataset", "mmlupro"), fake=a.fake)
    out_dir = ROOT / "out" / run["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.toml").write_text(Path(a.config).read_text())     # keep the config next to its results
    # Models run in parallel with each other; each has its own concurrency limit and budget cap.
    await asyncio.gather(*(run_model(m, run, sets_fn, n_sets, a.fake, a.smoke, out_dir) for m in models))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--models", help="comma-separated model names (overrides enabled = false)")
    ap.add_argument("--sets", type=int, help="limit the number of item sets (default: the config's)")
    ap.add_argument("--estimate", action="store_true", help="print a cost forecast and exit")
    ap.add_argument("--fake", action="store_true", help="simulated model, synthetic items, no network")
    ap.add_argument("--smoke", action="store_true", help="1 item set, one conversation at a time")
    try:
        asyncio.run(main_async(ap.parse_args()))
    except KeyboardInterrupt:
        print("\nStopped. Finished conversations are saved; rerun the same command to continue.")
        os._exit(130)                            # don't wait for in-flight API calls


if __name__ == "__main__":
    main()
