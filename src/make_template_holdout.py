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

Групповое разбиение (v3). В расширенном фразобанке у одной мысли бывает
несколько близких формулировок. Если «Продиктуйте код из SMS» попадёт в train,
а «Назовите код из SMS» — в holdout, тест проверяет запоминание, а не перенос.
Поэтому внутри пула близкие реплики (сходство символьных 4-грамм по Жаккару
≥ --group-thr, без слотов и пунктуации) объединяются в группу, и делятся
группы, а не отдельные реплики.

Выход:
    corpus_train.jsonl      — split=train и split=val (разные шаблоны)
    corpus_holdout.jsonl    — тест на невиданных формулировках
"""

import argparse
import copy
import json
import random
import re
from pathlib import Path

import generate_corpus as gc

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "generated"

# Ключи, значения которых — списки готовых реплик
UTTERANCE_KEYS = {"ru", "kk"}
SEED_OFFSET = {"train": 0, "holdout": 1, "val": 2}
GROUP_THR = 0.5
STATS = {"pools": 0, "items": 0, "groups": 0, "merged": 0, "unsplittable": 0}


def _shingles(text: str, n: int = 4) -> set:
    t = re.sub(r"\{[a-z_]+\}", " ", text.lower())
    t = " " + re.sub(r"[^\w]+", " ", t).strip() + " "
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def group_items(items, thr: float):
    """Близкие формулировки → одна группа (объединение по порогу сходства).
    Порядок детерминирован: группы сортируются по первому элементу."""
    items = sorted(items)
    sh = [_shingles(x) for x in items]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if thr <= 1.0:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                inter = len(sh[i] & sh[j])
                if inter and inter / len(sh[i] | sh[j]) >= thr:
                    parent[find(j)] = find(i)
    groups = {}
    for i, x in enumerate(items):
        groups.setdefault(find(i), []).append(x)
    return sorted(groups.values(), key=lambda g: g[0])


def three_way(items, local: random.Random, frac_hold: float, frac_val: float, thr: float = GROUP_THR):
    """Сначала отделяем holdout, затем от остатка — val. Делятся ГРУППЫ близких
    реплик. Пул не должен опустеть: у совсем маленьких пулов (1–2 группы) части
    пересекаются — это учитывается в отчёте о пересечении шаблонов."""
    groups = group_items(items, thr)
    local.shuffle(groups)
    n_hold = max(1, int(round(len(groups) * frac_hold))) if len(groups) > 1 else 0
    hold, rest = groups[:n_hold], (groups[n_hold:] or groups)
    n_val = max(1, int(round(len(rest) * frac_val))) if len(rest) > 1 else 0
    val, train = rest[:n_val], (rest[n_val:] or rest)
    flat = lambda gs: [x for g in gs for x in g]
    return {"train": flat(train), "val": flat(val or rest), "holdout": flat(hold or groups)}, groups


def split_pools(node, rng: random.Random, frac_hold: float, frac_val: float, keep: str,
                thr: float, path=(), stats=None):
    """Рекурсивно пройти фразобанк и в каждом списке реплик оставить нужную часть."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in UTTERANCE_KEYS and isinstance(v, list) and v and isinstance(v[0], str):
                # детерминированно: порядок перемешивания зависит от пути в дереве
                local = random.Random(f"{'/'.join(path)}/{k}|{rng.randint(0, 10**9)}")
                parts, groups = three_way(v, local, frac_hold, frac_val, thr)
                out[k] = parts[keep]
                if stats is not None:
                    stats["pools"] += 1
                    stats["items"] += len(v)
                    stats["groups"] += len(groups)
                    stats["merged"] += len(v) - len(groups)
                    stats["unsplittable"] += len(groups) < 3
            else:
                out[k] = split_pools(v, rng, frac_hold, frac_val, keep, thr, path + (str(k),), stats)
        return out
    if isinstance(node, list):
        return [split_pools(x, rng, frac_hold, frac_val, keep, thr, path, stats) for x in node]
    return node


SPLIT_KEYS = (("pb_scam", "scam"), ("pb_benign", "benign"), ("topics", "topics"), ("fillers", "fillers"))


def build(tax, keep: str, n: int, seed: int, frac_hold: float, frac_val: float, thr: float,
          stats=None):
    tax2 = copy.deepcopy(tax)
    for key, root in SPLIT_KEYS:
        if not tax.get(key):
            continue
        rng = random.Random(f"{seed}|{key}")       # одинаковый для всех частей → согласованное деление
        tax2[key] = split_pools(tax[key], rng, frac_hold, frac_val, keep, thr, (root,), stats)
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
    ap.add_argument("--no-ext", action="store_true",
                    help="фразобанк v2 без расширения (*_ext.yaml, fillers.yaml)")
    ap.add_argument("--group-thr", type=float, default=GROUP_THR,
                    help="порог сходства для группы близких реплик; >1 — без группировки (как в v2)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    tax = gc.load_taxonomy(ext=not args.no_ext)
    parts = {k: build(tax, k, n, args.seed, args.holdout_frac, args.val_frac, args.group_thr,
                      STATS if k == "train" else None)
             for k, n in (("train", args.n_train), ("val", args.n_val), ("holdout", args.n_holdout))}

    for i, d in enumerate(parts["train"] + parts["val"]):
        d["dialogue_id"] = f"QLK-T{i:06d}"
        d["split"] = "train" if i < len(parts["train"]) else "val"
    for i, d in enumerate(parts["holdout"]):
        d["dialogue_id"] = f"QLK-H{i:06d}"
        d["split"] = "holdout"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("corpus_train.jsonl", parts["train"] + parts["val"]),
                       ("corpus_holdout.jsonl", parts["holdout"])):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    s = STATS
    print(f"Фразобанк: {'v2 (без расширения)' if args.no_ext else 'v3 (с расширением)'}; "
          f"пулов {s['pools']}, реплик {s['items']}, групп {s['groups']} "
          f"(близких формулировок объединено: {s['merged']}); "
          f"пулов меньше чем из 3 групп: {s['unsplittable']}")
    tpl = {k: {t["template_id"] for d in v for t in d["turns"]} for k, v in parts.items()}
    for k, v in parts.items():
        print(f"{k:8s} {len(v):5d} диалогов, шаблонов {len(tpl[k])}")
    print()
    for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
        ov = tpl[a] & tpl[b]
        print(f"Пересечение шаблонов {a}∩{b}: {len(ov)} ({len(ov) / len(tpl[b]):.1%} шаблонов {b})")
    print("  Остаточное пересечение — пулы из 1–2 групп, которые нельзя поделить")
    print("  без опустошения пула.")


if __name__ == "__main__":
    main()
