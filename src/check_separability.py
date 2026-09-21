"""
Қалқан — проверка нетривиальности корпуса.

Вопрос, на который отвечает скрипт: может ли модель решить задачу, НЕ поняв
приём манипуляции? Если да — корпус бесполезен: на реальных звонках такой
детектор развалится (это и произошло с корпусом v1, см. topics.yaml).

Три проверки:
  1. Красные флаги — каждый значимый флаг должен встречаться в ОБОИХ классах,
     иначе модель выучит словарь.
  2. Темы (v2) — у каждой темы должны быть оба класса примерно поровну,
     иначе модель выучит тему («звонок про банк = мошенничество»).
  3. Проба «начало разговора» (v2) — простейшая модель (логистическая
     регрессия на символьных n-граммах) учится на первых k репликах train
     и проверяется на шаблонном holdout. Для инкрементальной детекции первая
     реплика (приветствие) НЕ должна выдавать класс: в v1 её AUC был 0.911.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "generated"
CORPUS = GEN / "corpus.jsonl"
TRAIN = GEN / "corpus_train.jsonl"
HOLDOUT = GEN / "corpus_holdout.jsonl"

FIRST_TURN_AUC_MAX = 0.65      # выше — приветствие выдаёт класс
TOPIC_SHARE_RANGE = (0.35, 0.65)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def check_flags(D):
    per_class = defaultdict(Counter)          # flag -> {scam: n, benign: n}
    for d in D:
        for t in d["turns"]:
            for f in t["red_flags"]:
                per_class[f][d["label"]] += 1

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

    n_scam = sum(1 for d in D if d["label"] == "scam")
    n_ben = sum(1 for d in D if d["label"] == "benign")
    ben_with_any = sum(1 for d in D if d["label"] == "benign"
                       and any(t["red_flags"] for t in d["turns"]))
    scam_no_flag = sum(1 for d in D if d["label"] == "scam"
                       and not any(t["red_flags"] for t in d["turns"]))
    print(f"\nЛегитимных диалогов хотя бы с одним флагом: "
          f"{ben_with_any}/{n_ben} = {ben_with_any / n_ben:.1%}")
    print(f"Мошеннических диалогов вообще без флагов:   "
          f"{scam_no_flag}/{n_scam} = {scam_no_flag / n_scam:.1%}")
    print(f"Флагов, которые сами по себе выдают класс: {len(trivial)} из {len(per_class)}")
    if trivial:
        print("  " + ", ".join(trivial))
        print("  Это не обязательно ошибка: «безопасный счёт» в легитимном звонке")
        print("  действительно не звучит. Но чем таких флагов меньше, тем честнее задача.")
    return trivial


def check_topics(D):
    if "topic" not in D[0]:
        print("  в корпусе нет поля topic (корпус v1) — проверка пропущена")
        return []
    c = defaultdict(Counter)
    for d in D:
        c[d["topic"]][d["label"]] += 1
    bad = []
    print(f"{'тема':18s} {'scam':>6s} {'benign':>7s} {'доля scam':>10s}")
    for t in sorted(c):
        s, b = c[t]["scam"], c[t]["benign"]
        share = s / (s + b)
        ok = TOPIC_SHARE_RANGE[0] <= share <= TOPIC_SHARE_RANGE[1]
        bad += [] if ok else [t]
        print(f"{t:18s} {s:6d} {b:7d} {share:10.1%}  {'' if ok else '← тема выдаёт класс'}")
    return bad


def probe_start(train, holdout):
    """AUC простейшей модели по первым k репликам (train → шаблонный holdout)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    res = {}
    for k in (1, 2, 3, 99):
        xtr = [" ".join(t["text"] for t in d["turns"][:k]) for d in train]
        xho = [" ".join(t["text"] for t in d["turns"][:k]) for d in holdout]
        ytr = [d["label"] == "scam" for d in train]
        yho = [d["label"] == "scam" for d in holdout]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        lr = LogisticRegression(max_iter=2000, C=4).fit(vec.fit_transform(xtr), ytr)
        res[k] = roc_auc_score(yho, lr.predict_proba(vec.transform(xho))[:, 1])
        label = "весь диалог" if k == 99 else f"первые {k} реплик" if k > 1 else "только реплика 0"
        print(f"  {label:18s} AUC {res[k]:.3f}")
    return res


def main():
    D = load(CORPUS)
    print("═" * 78 + "\n1. КРАСНЫЕ ФЛАГИ (corpus.jsonl)\n" + "═" * 78)
    trivial = check_flags(D)

    print("\n" + "═" * 78 + "\n2. ТЕМЫ: не выдаёт ли тема класс (corpus.jsonl)\n" + "═" * 78)
    bad_topics = check_topics(D)

    print("\n" + "═" * 78 + "\n3. ПРОБА «НАЧАЛО РАЗГОВОРА»: TF-IDF + логрегрессия, "
          "train → шаблонный holdout\n" + "═" * 78)
    train = [d for d in load(TRAIN) if d["split"] == "train"]
    res = probe_start(train, load(HOLDOUT))

    print("\n" + "═" * 78 + "\nИТОГ\n" + "═" * 78)
    print(f"  флагов только в одном классе:  {len(trivial)}")
    print(f"  тем, выдающих класс:           {len(bad_topics)}" + (f"  {bad_topics}" if bad_topics else ""))
    verdict = "OK" if res[1] <= FIRST_TURN_AUC_MAX else "ПРОБЛЕМА: приветствие выдаёт класс"
    print(f"  AUC по реплике 0:              {res[1]:.3f} (порог {FIRST_TURN_AUC_MAX})  {verdict}")
    print(f"  AUC простейшей модели по всему диалогу: {res[99]:.3f} — нижняя планка для обученной модели")


if __name__ == "__main__":
    main()
