"""
Қалқан — проверка качества корпуса.

Главный вопрос: нет ли в корпусе тривиальных артефактов, по которым модель
решит задачу, не научившись ничему полезному (длина диалога, один стоп-флаг).
"""

import json
import statistics as st
from collections import Counter
from pathlib import Path

import redflags

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "generated" / "corpus.jsonl"


def load():
    return [json.loads(l) for l in open(CORPUS, encoding="utf-8")]


def section(title):
    print(f"\n{'─' * 68}\n{title}\n{'─' * 68}")


def main():
    D = load()
    print(f"Всего диалогов: {len(D)}")
    print(f"Всего реплик:   {sum(len(d['turns']) for d in D)}")

    section("Баланс классов и разбиение")
    print("Метки:      ", dict(Counter(d["label"] for d in D)))
    print("Split:      ", dict(Counter(d["split"] for d in D)))
    print("Языки:      ", dict(Counter(d["language_profile"] for d in D)))
    print("Профили:    ", dict(Counter(d["victim_profile"] for d in D)))

    section("Покрытие сценариев")
    for code, n in sorted(Counter(d["scenario_code"] for d in D).items(),
                          key=lambda x: -x[1]):
        print(f"  {code:32s} {n:5d}")

    section("АРТЕФАКТ №1 — длина диалога")
    ls = [len(d["turns"]) for d in D if d["label"] == "scam"]
    lb = [len(d["turns"]) for d in D if d["label"] == "benign"]
    print(f"  scam:   медиана {st.median(ls):5.1f}  диапазон {min(ls)}–{max(ls)}")
    print(f"  benign: медиана {st.median(lb):5.1f}  диапазон {min(lb)}–{max(lb)}")
    overlap = len(set(ls) & set(lb)) / len(set(ls) | set(lb))
    print(f"  пересечение диапазонов длин: {overlap:.0%}", end="  ")
    print("OK" if overlap > 0.5 else "ПРОБЛЕМА: длина разделяет классы")

    section("АРТЕФАКТ №2 — доля мошеннических диалогов, оборванных до S5")
    term = sum(1 for d in D if d["label"] == "scam" and d["terminated_early_at"])
    scam_n = sum(1 for d in D if d["label"] == "scam")
    print(f"  оборвано жертвой: {term}/{scam_n} = {term / scam_n:.0%}")
    print("  (нужно заметно > 0: иначе 'дошёл до конца' = 'мошенничество')")

    section("Rule-based бейзлайн (порог 0.8)")
    base = redflags.RuleBaseline(threshold=0.8)
    tp = fp = tn = fn = 0
    alerts_scam, early = [], []
    for d in D:
        idx = base.first_alert_index(d["turns"])
        fired = idx >= 0
        if d["label"] == "scam":
            if fired:
                tp += 1
                alerts_scam.append(idx)
                if d["action_turn_idx"] >= 0:
                    early.append(d["action_turn_idx"] - idx)
            else:
                fn += 1
        else:
            fp += 1 if fired else 0
            tn += 0 if fired else 1

    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    print(f"  precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}   FPR {fpr:.3f}")
    if alerts_scam:
        print(f"  первое срабатывание: медиана на реплике {st.median(alerts_scam):.0f}")
    if early:
        pos = sum(1 for e in early if e > 0)
        print(f"  запас до целевого действия: медиана {st.median(early):.0f} реплик, "
              f"успели заранее в {pos / len(early):.0%} случаев")

    section("Ложные срабатывания по трудным негативам")
    hard = Counter()
    for d in D:
        if d["label"] == "benign" and base.first_alert_index(d["turns"]) >= 0:
            hard[d["scenario_code"]] += 1
    tot = Counter(d["scenario_code"] for d in D if d["label"] == "benign")
    for code, n in tot.items():
        print(f"  {code:28s} FP {hard.get(code, 0):4d}/{n:4d} = {hard.get(code, 0) / n:.1%}")

    section("Распределение красных флагов")
    fl = Counter()
    for d in D:
        for t in d["turns"]:
            fl.update(t["red_flags"])
    for f, n in fl.most_common():
        print(f"  {f:30s} {n:6d}")

    section("Примеры")
    for want in ("scam", "benign"):
        d = next(x for x in D if x["label"] == want and len(x["turns"]) >= 8)
        print(f"\n[{d['label'].upper()}] {d['scenario_code']} "
              f"({d['language_profile']}, жертва: {d['victim_profile']})")
        for t in d["turns"][:10]:
            mark = "!" if t["rule_risk"] >= 0.8 else " "
            who = "звонящий" if t["speaker"] == "caller" else "абонент "
            print(f"  {mark}[{t['stage']}] {who}: {t['text'][:76]}")


if __name__ == "__main__":
    main()
