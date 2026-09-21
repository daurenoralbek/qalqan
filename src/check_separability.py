"""
Қалқан — проверка нетривиальности корпуса.

Вопрос, на который отвечает скрипт: может ли модель решить задачу,
просто выучив словарь красных флагов? Если да — корпус бесполезен,
потому что на реальных звонках такой детектор развалится.

Критерий: каждый значимый флаг должен встречаться в ОБОИХ классах.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "generated" / "corpus.jsonl"


def main():
    D = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]

    per_class = defaultdict(Counter)          # flag -> {scam: n, benign: n}
    dial_with_flag = defaultdict(Counter)     # flag -> сколько диалогов
    for d in D:
        seen = set()
        for t in d["turns"]:
            for f in t["red_flags"]:
                per_class[f][d["label"]] += 1
                seen.add(f)
        for f in seen:
            dial_with_flag[f][d["label"]] += 1

    print(f"{'флаг':30s} {'scam':>7s} {'benign':>7s} {'доля benign':>12s}  вердикт")
    print("─" * 78)

    trivial = []
    for f in sorted(per_class, key=lambda x: -sum(per_class[x].values())):
        s, b = per_class[f]["scam"], per_class[f]["benign"]
        share = b / (s + b) if (s + b) else 0
        if b == 0:
            verdict, flagged = "только scam — тривиальный признак", True
        elif share < 0.03:
            verdict, flagged = "почти только scam", True
        else:
            verdict, flagged = "встречается в обоих классах", False
        if flagged:
            trivial.append(f)
        print(f"{f:30s} {s:7d} {b:7d} {share:11.1%}  {verdict}")

    print("\n" + "─" * 78)
    n_scam = sum(1 for d in D if d["label"] == "scam")
    n_ben = sum(1 for d in D if d["label"] == "benign")
    ben_with_any = sum(1 for d in D if d["label"] == "benign"
                       and any(t["red_flags"] for t in d["turns"]))
    scam_no_flag = sum(1 for d in D if d["label"] == "scam"
                       and not any(t["red_flags"] for t in d["turns"]))

    print(f"Легитимных диалогов хотя бы с одним флагом: "
          f"{ben_with_any}/{n_ben} = {ben_with_any / n_ben:.1%}")
    print(f"Мошеннических диалогов вообще без флагов:   "
          f"{scam_no_flag}/{n_scam} = {scam_no_flag / n_scam:.1%}")
    print(f"\nФлагов, которые сами по себе выдают класс: {len(trivial)}")
    if trivial:
        print("  " + ", ".join(trivial))
        print("\n  Это не обязательно ошибка: «безопасный счёт» и AnyDesk в легитимном")
        print("  звонке действительно не звучат. Но чем таких флагов меньше,")
        print("  тем честнее задача.")


if __name__ == "__main__":
    main()
