"""
Қалқан — оценка обученного детектора против rule-baseline.

Одни и те же наборы, одни и те же определения метрик (run_baseline.py):
  * шаблонный holdout (синтетика, формулировки не видны при обучении) —
    здесь известен момент целевого действия, поэтому считается главное:
    доля тревог, прозвучавших ДО запроса SMS-кода / перевода / AnyDesk;
  * внешние наборы на реальных звонках (data/external/work/*.jsonl) —
    переносится ли модель с синтетики на живую речь.

Точки сравнения:
  calibrated     порог и накопитель, выбранные на ВАЛИДАЦИИ (честная, развёртываемая)
  @rules_recall  порог модели, дающий на этом наборе recall правил → сравниваем FPR
  @rules_fpr     порог модели, дающий на этом наборе FPR правил   → сравниваем recall
  (последние две — «оракульные» точки на кривой, только для сравнения кривых)

Предсказания кэшируются в <model_dir>/preds/ (models/ — вне git).

    python eval_detector.py --model-dir ../models/xlmr-lora
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

import detector as D
import redflags

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "external" / "work"
HOLDOUT = ROOT / "data" / "generated" / "corpus_holdout.jsonl"
RULE_TH = 0.6          # лучший по F1 порог правил (results/baseline_rules_holdout.json)


def available_sets():
    sets = {"holdout": HOLDOUT}
    for name in ("korccvi_ru", "en_bank_ru", "en_topic_ru", "youtube_kz_orig", "kazllm_kk"):
        if (WORK / f"{name}.jsonl").exists():
            sets[name] = WORK / f"{name}.jsonl"
    return sets


def auc(dialogues, curves):
    y = [1 if d["label"] == "scam" else 0 for d in dialogues]
    if not 0 < sum(y) < len(y):
        return None
    return round(roc_auc_score(y, [max(c) if c else 0.0 for c in curves]), 4)


def matched_threshold(dialogues, curves, target, what):
    """Порог, при котором recall ≥ target (what='recall') или FPR ≤ target (what='fpr')."""
    cand = sorted({round(x, 6) for c in curves for x in c}, reverse=True)
    best = None
    for th in cand:
        op = D.operating_point(dialogues, curves, th)
        if what == "recall" and op["recall"] >= target:
            return op
        if what == "fpr":
            if op["fpr"] is not None and op["fpr"] <= target:
                best = op
            else:
                break
    return best


def breakdown(dialogues, curves, th):
    """Recall по сценариям мошенничества, FPR по легитимным сценариям и по языку."""
    by = defaultdict(lambda: [0, 0])
    for d, c in zip(dialogues, curves):
        fired = D.first_alert(c, th) >= 0
        for key in (f"scenario:{d.get('scenario_id')}", f"lang:{d.get('language_profile', d.get('lang'))}:{d['label']}"):
            by[key][0] += fired
            by[key][1] += 1
    return {k: {"fired": v[0], "n": v[1], "share": round(v[0] / v[1], 3)} for k, v in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "xlmr-lora"))
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="первые N диалогов каждого набора (проба)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir)
    model, tok, meta = D.load_trained(model_dir, device)
    enc = D.PrefixEncoder(tok, meta["max_len"])
    acc = meta["accumulator"]
    rules = redflags.RuleBaseline(threshold=RULE_TH)
    (model_dir / "preds").mkdir(exist_ok=True)

    sets = available_sets()
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets}

    report = {"model_dir": str(model_dir), "accumulator": acc, "rules_threshold": RULE_TH, "sets": {}}
    head = (f"{'набор':16s} {'n':>5s} │ {'AUC модель':>10s} {'правила':>8s} │ "
            f"{'модель (порог с val): P':>24s} {'R':>6s} {'FPR':>6s} {'заранее':>8s} │ "
            f"{'правила @0.6: P':>16s} {'R':>6s} {'FPR':>6s} {'заранее':>8s}")
    print(head)
    print("─" * len(head))
    for name, path in sets.items():
        data = D.load_jsonl(path)
        if args.limit and len(data) > args.limit:
            data = data[:args.limit // 2] + data[-(args.limit // 2):]
        # ключ кэша — состав набора: kazllm_kk, например, дописывается в фоне
        sig = hashlib.md5("|".join(d["dialogue_id"] for d in data).encode()).hexdigest()[:8]
        cache = model_dir / "preds" / f"{name}_{len(data)}_{sig}.json"
        if cache.exists():
            raw = json.load(open(cache, encoding="utf-8"))
        else:
            raw = D.score_dialogues(model, enc, data, batch_size=args.batch, device=device)
            json.dump(raw, open(cache, "w", encoding="utf-8"))
        curves = [D.accumulate(c, acc["mode"], acc["a"]) for c in raw]
        rcurves = [rules.score_dialogue(d["turns"]) for d in data]

        m_op = D.operating_point(data, curves, acc["threshold"])
        r_op = D.operating_point(data, rcurves, RULE_TH)
        entry = {
            "n": len(data), "n_scam": sum(d["label"] == "scam" for d in data),
            "model": {"roc_auc": auc(data, curves), "calibrated": m_op},
            "rules": {"roc_auc": auc(data, rcurves), f"@{RULE_TH}": r_op},
        }
        # точки на кривой «при recall / FPR правил» имеют смысл, только если правила
        # хоть что-то ловят
        if 0 < entry["n_scam"] < len(data) and r_op["recall"] > 0:
            entry["model"]["@rules_recall"] = matched_threshold(data, curves, r_op["recall"], "recall")
            entry["model"]["@rules_fpr"] = matched_threshold(data, curves, r_op["fpr"], "fpr")
        if name == "holdout":
            entry["model"]["breakdown"] = breakdown(data, curves, acc["threshold"])
            entry["rules"]["breakdown"] = breakdown(data, rcurves, RULE_TH)
        report["sets"][name] = entry

        def fmt(op):
            e = op["share_alerted_before_action"]
            fpr = "—" if op["fpr"] is None else f"{op['fpr']:.3f}"
            return (f"{op['precision']:.3f} {op['recall']:6.3f} {fpr:>6s} "
                    f"{'—' if e is None else f'{e:.0%}':>8s}")
        ma, ra = entry["model"]["roc_auc"], entry["rules"]["roc_auc"]
        print(f"{name:16s} {len(data):5d} │ {'—' if ma is None else f'{ma:.3f}':>10s} "
              f"{'—' if ra is None else f'{ra:.3f}':>8s} │ {fmt(m_op):>45s} │ {fmt(r_op):>37s}")

    for name, e in report["sets"].items():
        m = e["model"]
        if "@rules_recall" in m and m["@rules_recall"]:
            rr, rf = m["@rules_recall"], m.get("@rules_fpr")
            print(f"  {name}: при recall правил {e['rules'][f'@{RULE_TH}']['recall']:.3f} FPR модели "
                  f"{rr['fpr']:.3f} (правила {e['rules'][f'@{RULE_TH}']['fpr']:.3f})"
                  + (f"; при FPR правил recall модели {rf['recall']:.3f}" if rf else ""))

    out = Path(args.out) if args.out else ROOT / "results" / f"detector_eval_{model_dir.name}.json"
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nСохранено → {out}")


if __name__ == "__main__":
    main()
