"""Question pools.

Items come from one fixed shuffle of the benchmark (seed 1234, same filters as the pilot), so set k always holds
the same 16 questions:

  pilot        sets   0-79   already used by the Oct 2026 pilot runs (Gemini, Gemma, Nemotron, Claude)
  main         sets  80-139  main study, never seen in any earlier run
  replication  sets 140-199  fresh items for the robustness/replication runs

Sets from 80 on skip any item listed in data/excluded_items.txt (disputed answer keys), so the exclusion list must
be frozen before the main study starts. Run once on a machine with internet access:

    python3 -m study.items prepare            # downloads the benchmark, writes data/mmlupro_items.json
    python3 -m study.items show --set 80      # print one item set
"""
import argparse
import json
import random
from pathlib import Path

from .protocol import LETTERS, N_BASE, N_TEST, SEED

DATA = Path(__file__).resolve().parent.parent / "data"
PER_SET = N_BASE + N_TEST
PILOT_SETS = 80
POOLS = {"pilot": (0, 80), "main": (80, 60), "replication": (140, 60)}   # (first set, number of sets)
N_PREPARE = 200 * PER_SET + 400                                          # all pools + headroom for exclusions


def _filtered(dataset):
    from datasets import load_dataset
    if dataset == "supergpqa":
        ds = load_dataset("m-a-p/SuperGPQA", split="train").filter(lambda r: r["difficulty"] == "hard")
    else:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    for i in idx:
        r = ds[i]
        opts = [str(o) for o in r["options"]]
        if len(opts) < 4 or len(opts) > len(LETTERS) or any(o.strip() in ("", "N/A") for o in opts):
            continue
        if len(r["question"]) + sum(map(len, opts)) > 2000:
            continue
        if dataset == "supergpqa":
            gold, iid, cat = r["answer_letter"].strip().upper(), r["uuid"], r["discipline"]
        else:
            gold, iid, cat = LETTERS[r["answer_index"]], str(r["question_id"]), r["category"]
        if gold not in LETTERS[:len(opts)]:
            continue
        yield dict(id=iid, question=r["question"], options=opts, gold=gold, category=cat)


def prepare(dataset="mmlupro", n=N_PREPARE):
    DATA.mkdir(exist_ok=True)
    items = []
    for it in _filtered(dataset):
        items.append(it)
        if len(items) == n:
            break
    path = DATA / f"{dataset}_items.json"
    path.write_text(json.dumps(items))
    print(f"wrote {len(items)} items to {path}")


def fake_items(n=N_PREPARE):
    rng = random.Random(SEED)
    out = []
    for i in range(n):
        k = rng.choice([4, 10])
        out.append(dict(id=f"fake{i}", question=f"Synthetic question {i}?", options=[f"option {j}" for j in range(k)],
                        gold=LETTERS[rng.randrange(k)], category="fake"))
    return out


def excluded():
    p = DATA / "excluded_items.txt"
    if not p.exists():
        return set()
    return {ln.split("#")[0].strip() for ln in p.read_text().splitlines() if ln.split("#")[0].strip()}


def load_sets(dataset="mmlupro", fake=False):
    """Return a function set_idx -> (base_items, test_items)."""
    if fake:
        items = fake_items()
    else:
        path = DATA / f"{dataset}_items.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing. Run once: python3 -m study.items prepare --dataset {dataset}")
        items = json.loads(path.read_text())
    pilot, later = items[:PILOT_SETS * PER_SET], items[PILOT_SETS * PER_SET:]
    ex = excluded()
    later = [it for it in later if it["id"] not in ex]

    def get(k):
        src, j = (pilot, k) if k < PILOT_SETS else (later, k - PILOT_SETS)
        chunk = src[j * PER_SET:(j + 1) * PER_SET]
        if len(chunk) < PER_SET:
            raise SystemExit(f"Not enough prepared items for set {k}; rerun `prepare` with a larger n.")
        return chunk[:N_BASE], chunk[N_BASE:]
    return get


def pool_sets(pool, n_sets=None):
    first, n = POOLS[pool]
    return list(range(first, first + (n if n_sets is None else min(n, n_sets))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "show"])
    ap.add_argument("--dataset", default="mmlupro")
    ap.add_argument("--set", type=int, default=80)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.dataset)
    else:
        b, t = load_sets(a.dataset)(a.set)
        for it in b + t:
            print(it["id"], it["category"], it["gold"], it["question"][:90].replace("\n", " "))
