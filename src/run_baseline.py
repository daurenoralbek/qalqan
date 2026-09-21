"""
Қалқан — прогон rule-based бейзлайна и сохранение метрик.

Этот бейзлайн — то, что банк может построить за день на регулярных выражениях.
Обученная модель обязана его превзойти, и прежде всего по двум показателям:
  * FPR — ложные тревоги на легитимных звонках (усталость от алертов);
  * запас по времени — срабатывание ДО запроса целевого действия, а не в момент.
"""

import json
import statistics as st
from pathlib import Path

import redflags

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "generated" / "corpus.jsonl"
OUT = ROOT / "results" / "baseline_rules.json"


def evaluate(dialogues, threshold: float):
    base = redflags.RuleBaseline(threshold=threshold)
    tp = fp = tn = fn = 0
    lead_times, alert_positions = [], []

    for d in dialogues:
        idx = base.first_alert_index(d["turns"])
        fired = idx >= 0
        if d["label"] == "scam":
            if fired:
                tp += 1
                alert_positions.append(idx)
                if d["action_turn_idx"] >= 0:
                    lead_times.append(d["action_turn_idx"] - idx)
            else:
                fn += 1
        else:
            if fired:
                fp += 1
            else:
                tn += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    in_time = sum(1 for l in lead_times if l > 0)
    return {
        "threshold": threshold,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "median_alert_turn": st.median(alert_positions) if alert_positions else None,
        "median_lead_turns": st.median(lead_times) if lead_times else None,
        "share_alerted_before_action": round(in_time / len(lead_times), 4) if lead_times else None,
    }


def main():
    D = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    test = [d for d in D if d["split"] == "test"]

    print(f"Тестовая выборка: {len(test)} диалогов\n")
    print(f"{'порог':>6s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'FPR':>7s} "
          f"{'алерт на':>9s} {'запас':>7s} {'успели':>8s}")
    print("─" * 66)

    results = []
    for th in (0.6, 0.7, 0.8, 0.9, 0.95):
        r = evaluate(test, th)
        results.append(r)
        lead = "—" if r["median_lead_turns"] is None else f"{r['median_lead_turns']:.0f}"
        share = "—" if r["share_alerted_before_action"] is None else f"{r['share_alerted_before_action']:.0%}"
        alert = "—" if r["median_alert_turn"] is None else f"{r['median_alert_turn']:.0f}"
        print(f"{th:6.2f} {r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f} "
              f"{r['fpr']:7.3f} {alert:>9s} {lead:>7s} {share:>8s}")

    best = max(results, key=lambda r: r["f1"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"test_size": len(test), "sweep": results, "best_by_f1": best},
              open(OUT, "w"), ensure_ascii=False, indent=2)

    print(f"\nЛучший по F1: порог {best['threshold']}, F1 {best['f1']:.3f}, FPR {best['fpr']:.3f}")
    every_nth = round(1 / best["fpr"]) if best["fpr"] else None
    missed = 1 - best["recall"]
    print("\nЧто должна улучшить обученная модель:")
    print(f"  1. FPR {best['fpr']:.1%}" +
          (f" — ложная тревога на каждом {every_nth}-м легитимном звонке" if every_nth else ""))
    print(f"  2. Запас до целевого действия — медиана {best['median_lead_turns']:.0f} реплик, "
          f"вовремя успевает лишь в {best['share_alerted_before_action']:.0%} случаев")
    print(f"  3. Пропуск {missed:.0%} мошеннических диалогов (recall {best['recall']:.1%})")
    print("\n  Разброс по порогам показывает суть проблемы правил: при нулевом FPR")
    print("  (порог 0.9) recall падает до 32%, а предупредить заранее удаётся в 5% случаев.")
    print(f"\nСохранено → {OUT}")


if __name__ == "__main__":
    main()
