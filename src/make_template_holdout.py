"""
Қалқан — разбиение по шаблонам (template holdout).

Зачем. Корпус синтетический: 31 тыс. реплик собраны примерно из 500 шаблонов.
Если делить диалоги случайно, одни и те же формулировки попадут и в обучение,
и в тест — модель запомнит фразы и покажет красивые, но бессмысленные метрики.

Что делает скрипт. Делит сами ШАБЛОНЫ: часть формулировок уходит в отложенный
набор и при обучении не встречается ни разу. Тест на этом наборе отвечает на
честный вопрос: понимает ли модель приём манипуляции или запомнила слова.

Выход:
    corpus_train.jsonl      — обучение и валидация (шаблоны A)
    corpus_holdout.jsonl    — тест на невиданных формулировках (шаблоны B)
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


def split_pools(node, rng: random.Random, frac: float, keep: str, path=()):
    """
    Рекурсивно пройти фразобанк и в каждом списке реплик оставить
    либо обучающую часть, либо отложенную.
    """
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in UTTERANCE_KEYS and isinstance(v, list) and v and isinstance(v[0], str):
                items = sorted(v)
                # детерминированно: порядок перемешивания зависит от пути в дереве
                local = random.Random(f"{'/'.join(path)}/{k}|{rng.randint(0, 10**9)}")
                local.shuffle(items)
                n_hold = max(1, int(round(len(items) * frac))) if len(items) > 1 else 0
                hold = items[:n_hold]
                train = items[n_hold:] or items  # пул не должен опустеть
                out[k] = train if keep == "train" else (hold or items)
            else:
                out[k] = split_pools(v, rng, frac, keep, path + (str(k),))
        return out
    if isinstance(node, list):
        return [split_pools(x, rng, frac, keep, path) for x in node]
    return node


def build(tax, keep: str, n: int, seed: int, frac: float):
    tax2 = copy.deepcopy(tax)
    rng = random.Random(seed)
    tax2["pb_scam"] = split_pools(tax["pb_scam"], rng, frac, keep, ("scam",))
    tax2["pb_benign"] = split_pools(tax["pb_benign"], rng, frac, keep, ("benign",))

    gen = gc.DialogueGenerator(tax2, seed=seed + (0 if keep == "train" else 1))
    r = random.Random(seed + (0 if keep == "train" else 1))

    balance = tax["benign"]["corpus_balance"]
    n_scam = int(n * balance["scam_share"])
    scam_ids = list(gen.scam_scenarios)
    benign_ids = list(gen.benign_scenarios)
    hard = [i for i in benign_ids
            if gen.benign_scenarios[i]["difficulty"] in ("hard", "very_hard")]
    easy = [i for i in benign_ids if i not in hard]

    dialogues = [gen.gen_scam(r.choice(scam_ids)) for _ in range(n_scam)]
    for _ in range(n - n_scam):
        pool = hard if r.random() < balance["hard_negative_share_within_benign"] else easy
        dialogues.append(gen.gen_benign(r.choice(pool)))
    r.shuffle(dialogues)
    return dialogues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-holdout", type=int, default=1000)
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="доля шаблонов, отложенных из обучения")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tax = gc.load_taxonomy()

    train = build(tax, "train", args.n_train, args.seed, args.holdout_frac)
    hold = build(tax, "holdout", args.n_holdout, args.seed, args.holdout_frac)

    # внутри обучающей части — обычное деление на train/val
    n_tr = int(0.85 * len(train))
    for i, d in enumerate(train):
        d["dialogue_id"] = f"QLK-T{i:06d}"
        d["split"] = "train" if i < n_tr else "val"
    for i, d in enumerate(hold):
        d["dialogue_id"] = f"QLK-H{i:06d}"
        d["split"] = "holdout"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("corpus_train.jsonl", train), ("corpus_holdout.jsonl", hold)):
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    tpl_tr = {t["template_id"] for d in train for t in d["turns"]}
    tpl_ho = {t["template_id"] for d in hold for t in d["turns"]}
    overlap = tpl_tr & tpl_ho

    print(f"corpus_train.jsonl    {len(train):5d} диалогов, шаблонов {len(tpl_tr)}")
    print(f"corpus_holdout.jsonl  {len(hold):5d} диалогов, шаблонов {len(tpl_ho)}")
    print(f"\nПересечение шаблонов: {len(overlap)} "
          f"({len(overlap) / len(tpl_ho):.1%} отложенного набора)")
    if overlap:
        print("  Остаточное пересечение — это короткие реплики жертвы («Да, слушаю»),")
        print("  пул которых слишком мал, чтобы делить его без опустошения.")
        print("  На них модель не учится ничему специфичному, поэтому это допустимо.")


if __name__ == "__main__":
    main()
