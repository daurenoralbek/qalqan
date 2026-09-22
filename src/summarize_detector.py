"""
Қалқан — сводка по нескольким сидам: модель против правил.

Берёт results/detector_eval_<model>.json (eval_detector.py) для набора моделей
(по умолчанию xlmr-lora-s*), считает среднее и разброс по сидам и печатает
таблицу в Markdown — её можно вставлять в карточку датасета и техприложение.

    python summarize_detector.py --pattern "xlmr-lora-s*"
"""

import argparse
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

SET_TITLES = {
    "holdout": "шаблонный holdout (синтетика)",
    "korccvi_ru": "реальный вишинг, Корея → ru",
    "en_bank_ru": "реальные скам-звонки vs банк, en → ru",
    "en_topic_ru": "реальные скам-звонки vs та же тема, en → ru",
    "kazllm_kk": "те же реальные звонки → kk (KazLLM)",
    "youtube_kz_orig": "реальные звонки казахстанцам, ru (Whisper)",
}


def ms(xs, pct=False):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.3f}")
    if len(xs) == 1:
        return f(xs[0])
    sd = st.stdev(xs)
    return f"{f(st.mean(xs))} ± {f(sd) if not pct else f'{100 * sd:.0f}%'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="xlmr-lora-s*")
    ap.add_argument("--res-dir", default=str(RES), help="папка с detector_eval_*.json")
    ap.add_argument("--out", default=None, help="по умолчанию <res-dir>/detector_summary.md")
    args = ap.parse_args()
    res = Path(args.res_dir)
    args.out = args.out or str(res / "detector_summary.md")

    files = sorted(glob.glob(str(res / f"detector_eval_{args.pattern}.json")))
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    if not runs:
        raise SystemExit(f"нет файлов results/detector_eval_{args.pattern}.json")
    sets = [s for s in SET_TITLES if all(s in r["sets"] for r in runs)]

    lines = [f"Моделей (сидов): {len(runs)} — " + ", ".join(Path(f).stem.replace('detector_eval_', '') for f in files),
             "",
             "Порог модели выбран на валидации (FPR ≤ 5%), правила — порог 0.6 (лучший F1 на holdout).",
             "«Заранее» — доля мошеннических диалогов, где тревога прозвучала ДО запроса целевого действия",
             "(только holdout: во внешних данных момент действия не размечен).",
             "",
             "| Набор | n | ROC AUC модель | ROC AUC правила | Recall модель | Recall правила | FPR модель | FPR правила | Заранее модель | Заранее правила |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for s in sets:
        m = [r["sets"][s]["model"] for r in runs]
        rl = runs[0]["sets"][s]["rules"]                       # правила детерминированы
        rop = rl[[k for k in rl if k.startswith("@")][0]]
        row = {
            "n": runs[0]["sets"][s]["n"],
            "auc_model": [x["roc_auc"] for x in m], "auc_rules": rl["roc_auc"],
            "recall_model": [x["calibrated"]["recall"] for x in m], "recall_rules": rop["recall"],
            "fpr_model": [x["calibrated"]["fpr"] for x in m], "fpr_rules": rop["fpr"],
            "early_model": [x["calibrated"]["share_alerted_before_action"] for x in m],
            "early_rules": rop["share_alerted_before_action"],
        }
        summary[s] = row
        lines.append(
            f"| {SET_TITLES[s]} | {row['n']} | {ms(row['auc_model'])} | {ms([row['auc_rules']])} | "
            f"{ms(row['recall_model'])} | {ms([row['recall_rules']])} | {ms(row['fpr_model'])} | "
            f"{ms([row['fpr_rules']])} | {ms(row['early_model'], True)} | {ms([row['early_rules']], True)} |")

    # сравнение на кривой: FPR модели при recall правил (holdout)
    if "holdout" in sets:
        rr = [r["sets"]["holdout"]["model"].get("@rules_recall") for r in runs]
        rf = [r["sets"]["holdout"]["model"].get("@rules_fpr") for r in runs]
        rl = runs[0]["sets"]["holdout"]["rules"]
        rop = rl[[k for k in rl if k.startswith("@")][0]]
        lines += ["",
                  f"На holdout при том же recall, что у правил ({rop['recall']:.3f}), FPR модели "
                  f"{ms([x['fpr'] for x in rr if x])} против {rop['fpr']:.3f} у правил; "
                  f"при том же FPR ({rop['fpr']:.3f}) recall модели {ms([x['recall'] for x in rf if x])} "
                  f"против {rop['recall']:.3f}."]

    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    json.dump(summary, open(Path(args.out).with_suffix(".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nСохранено → {args.out}")


if __name__ == "__main__":
    main()
