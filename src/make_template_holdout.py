"""
Қалқан — разбиение по шаблонам (template holdout).

Зачем. Корпус синтетический: десятки тысяч реплик собраны из примерно 600
шаблонов. Если делить диалоги случайно, одни и те же формулировки попадут и в
обучение, и в тест — модель запомнит фразы и покажет красивые, но бессмысленные
метрики.

Что делает скрипт. Делит сами ШАБЛОНЫ на три непересекающиеся части:
    train     — обучение
    val       — выбор эпохи и калибровка порога (v2)
    holdout   — финальный тест
Каждая часть порождает свои диалоги только из своих формулировок.

Почему val тоже по шаблонам (v2). В v1 валидация была случайной долей
диалогов из обучающих шаблонов: модель узнавала заученные фразы, на валидации
была уверена на 100% и выбирала порог ≈ 0 — на новых формулировках это давало
FPR 0.30–0.37. Порог, подобранный на невиданных формулировках, переносится.

Выход:
    corpus_train.jsonl      — split=train и split=val (разные шаблоны)
    corpus_holdout.jsonl    — тест на невиданных формулировках
"""

import argparse
import copy
import json
import random
from pathlib import Path

import generate_corpus as gc

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "generated"

# Ключи, значения которых — списки готовых реплик
UTTERANCE_KEYS = {"ru", "kk"}
SEED_OFFSET = {"train": 0, "holdout": 1, "val": 2}


def three_way(items, local: random.Random, frac_hold: float, frac_val: float):
    """Сначала отделяем holdout, затем от остатка — val. Пул не должен опустеть:
    у совсем маленьких пулов (1–2 реплики) части пересекаются — это учитывается
    в отчёте о пересечении шаблонов."""
    items = sorted(items)
    local.shuffle(items)
    n_hold = max(1, int(round(len(items) * frac_hold))) if len(items) > 1 else 0
    hold, rest = items[:n_hold], (items[n_hold:] or items)
    n_val = max(1, int(round(len(rest) * frac_val))) if len(rest) > 1 else 0
    val, train = rest[:n_val], (rest[n_val:] or rest)
    return {"train": train, "val": val or rest, "holdout": hold or items}


def split_pools(node, rng: random.Random, frac_hold: float, frac_val: float, keep: str, path=()):
    """Рекурсивно пройти фразобанк и в каждом списке реплик оставить нужную часть."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in UTTERANCE_KEYS and isinstance(v, list) and v and isinstance(v[0], str):
                # детерминированно: порядок перемешивания зависит от пути в дереве
                local = random.Random(f"{'/'.join(path)}/{k}|{rng.randint(0, 10**9)}")
                out[k] = three_way(v, local, frac_hold, frac_val)[keep]
            else:
                out[k] = split_pools(v, rng, frac_hold, frac_val, keep, path + (str(k),))
        return out
    if isinstance(node, list):
        return [split_pools(x, rng, frac_hold, frac_val, keep, path) for x in node]
    return node


def build(tax, keep: str, n: int, seed: int, frac_hold: float, frac_val: float):
    tax2 = copy.deepcopy(tax)
    for key, root in (("pb_scam", "scam"), ("pb_benign", "benign"), ("topics", "topics")):
        rng = random.Random(f"{seed}|{key}")       # одинаковый для всех частей → согласованное деление
        tax2[key] = split_pools(tax[key], rng, frac_hold, frac_val, keep, (root,))
    gen = gc.DialogueGenerator(tax2, seed=seed + SEED_OFFSET[keep])
    r = random.Random(seed + SEED_OFFSET[keep])
    return gen.sample(n, r, tax["benign"]["corpus_balance"]["scam_share"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3400)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-holdout", type=int, default=1000)
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="доля шаблонов, отложенных в holdout")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="доля оставшихся шаблонов, отложенных в val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tax = gc.load_taxonomy()
    parts = {k: build(tax, k, n, args.seed, args.holdout_frac, args.val_frac)
             for k, n in (("train", args.n_train), ("val", args.n_val), ("holdout", args.n_holdout))}

    for i, d in enumerate(parts["train"] + parts["val"]):
        d["dialogue_id"] = f"QLK-T{i:06d}"
        d["split"] = "train" if i < len(parts["train"]) else "val"
    for i, d in enumerate(parts["holdout"]):
        d["dialogue_id"] = f"QLK-H{i:06d}"
        d["split"] = "holdout"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("corpus_train.jsonl", parts["train"] + parts["val"]),
                       ("corpus_holdout.jsonl", parts["holdout"])):
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    tpl = {k: {t["template_id"] for d in v for t in d["turns"]} for k, v in parts.items()}
    for k, v in parts.items():
        print(f"{k:8s} {len(v):5d} диалогов, шаблонов {len(tpl[k])}")
    print()
    for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
        ov = tpl[a] & tpl[b]
        print(f"Пересечение шаблонов {a}∩{b}: {len(ov)} ({len(ov) / len(tpl[b]):.1%} шаблонов {b})")
    print("  Остаточное пересечение — короткие реплики из пулов в 1–2 варианта")
    print("  («Да, слушаю»), которые нельзя поделить без опустошения пула.")


if __name__ == "__main__":
    main()
