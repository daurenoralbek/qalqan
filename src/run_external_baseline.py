"""
Қалқан — rule-baseline на внешних (реальных) звонках.

Дешёвая проверка ДО обучения модели: переносится ли детектор с синтетического
корпуса на живую речь. Сравниваем с тем же бейзлайном на шаблонном holdout.

Что считается для каждого набора:
  * precision / recall / F1 / FPR по сетке порогов — как в run_baseline.py;
  * ROC AUC по максимальному накопленному риску за диалог;
  * доля диалогов, где сработал хоть один красный флаг (по классам) —
    главный индикатор «словарь не переносится»;
  * где сработала тревога: номер реплики, доля пройденного разговора,
    секунда от начала (если есть таймкоды).

Контроль: те же наборы на языке оригинала (ko/en). Правила написаны для ru/kk
и там обязаны молчать — иначе в переводе что-то протекает.

    python run_external_baseline.py
"""

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score

import redflags

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "external" / "work"
HOLDOUT = ROOT / "data" / "generated" / "corpus_holdout.jsonl"
OUT = ROOT / "results" / "external_baseline_rules.json"
THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def med(xs):
    return round(st.median(xs), 3) if xs else None


def evaluate(dialogues, score_fn, thresholds=THRESHOLDS):
    """
    score_fn(turns) -> список накопленного риска после каждой реплики.
    Годится и для правил, и для обученной модели (шаг 2).
    """
    curves = [score_fn(d["turns"]) for d in dialogues]
    y = [1 if d["label"] == "scam" else 0 for d in dialogues]
    peak = [max(c) if c else 0.0 for c in curves]

    res = {
        "n": len(dialogues), "n_scam": sum(y), "n_benign": len(y) - sum(y),
        "roc_auc": round(roc_auc_score(y, peak), 4) if 0 < sum(y) < len(y) else None,
        "sweep": [],
    }
    for th in thresholds:
        tp = fp = tn = fn = 0
        at_turn, at_frac, at_sec, lead = [], [], [], []
        for d, c, lab in zip(dialogues, curves, y):
            idx = next((i for i, s in enumerate(c) if s >= th), -1)
            if lab:
                if idx >= 0:
                    tp += 1
                    at_turn.append(idx)
                    at_frac.append((idx + 1) / len(c))
                    if d["turns"][idx].get("start") is not None:
                        at_sec.append(d["turns"][idx]["start"])
                    if d.get("action_turn_idx", -1) >= 0:
                        lead.append(d["action_turn_idx"] - idx)
                else:
                    fn += 1
            else:
                fp += idx >= 0
                tn += idx < 0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        res["sweep"].append({
            "threshold": th,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "fpr": round(fp / (fp + tn), 4) if fp + tn else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "median_alert_turn": med(at_turn),
            "median_alert_frac": med(at_frac),
            "median_alert_sec": med(at_sec),
            "median_lead_turns": med(lead),
            "share_alerted_before_action": round(sum(l > 0 for l in lead) / len(lead), 4) if lead else None,
        })
    return res


def flag_stats(dialogues):
    """Сколько диалогов каждого класса содержат хоть один флаг и какие именно."""
    any_flag = Counter()
    per_flag = {"scam": Counter(), "benign": Counter()}
    n = Counter(d["label"] for d in dialogues)
    for d in dialogues:
        seen = {f for t in d["turns"] for f in redflags.detect(t["text"])}
        if seen:
            any_flag[d["label"]] += 1
        per_flag[d["label"]].update(seen)
    return {
        "any_flag_share": {k: round(any_flag[k] / n[k], 4) for k in n},
        "dialogues_with_flag": {k: dict(v.most_common()) for k, v in per_flag.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    base = redflags.RuleBaseline()
    sets = {"holdout_synthetic": load(HOLDOUT)}
    for p in sorted(WORK.glob("*.jsonl")):
        name = p.stem
        if name.endswith("_sample") or name.startswith("youtube_kz_asr"):
            continue
        sets[name] = load(p)

    report = {}
    print(f"{'набор':22s} {'n':>5s} {'AUC':>6s} │ {'порог 0.6: P':>12s} {'R':>6s} {'FPR':>6s} "
          f"│ {'флаг есть: scam':>15s} {'benign':>7s} │ алерт: реплика / доля / сек")
    print("─" * 120)
    for name, D in sets.items():
        r = evaluate(D, base.score_dialogue)
        r.update(flag_stats(D))
        report[name] = r
        s = r["sweep"][0]
        af = r["any_flag_share"]
        auc = "—" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
        fpr = "—" if s["fpr"] is None else f"{s['fpr']:.3f}"
        print(f"{name:22s} {r['n']:5d} {auc:>6s} │ {s['precision']:12.3f} {s['recall']:6.3f} "
              f"{fpr:>6s} │ {af.get('scam', 0):15.1%} {af.get('benign', 0):7.1%} │ "
              f"{s['median_alert_turn']} / {s['median_alert_frac']} / {s['median_alert_sec']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nСохранено → {out}")


if __name__ == "__main__":
    main()
