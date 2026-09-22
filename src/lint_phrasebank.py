"""
Қалқан — проверка фразобанка (v3).

Запускать после любой правки phrasebank_*.yaml, topics*.yaml, fillers.yaml —
в том числе после вычитки казахских реплик носителем.

Проверки:
  1. Пути в *_ext.yaml существуют в базовом файле (опечатка в ключе стадии
     молча создала бы пул, который генератор никогда не прочитает).
  2. Слоты {…} — только из тех, что заполняет SlotContext.
  3. Нет повторов внутри пула (с учётом расширения).
  4. В казахских пулах нет русских строк: нет ни одной казахской буквы
     (ә і ң ғ ү ұ қ ө һ), зато есть ё/щ/ъ или частое русское слово.
     Смешанная речь («подозрительная операция болды») проходит.
  5. В нейтральных репликах (fillers.yaml) нет ни одного красного флага.
  6. Сводка: размеры пулов и число групп близких формулировок
     (как их видит make_template_holdout.py).
Код выхода 1, если есть ошибки (1–5).
"""

import re
import sys
from collections import Counter
from pathlib import Path

import yaml

import generate_corpus as gc
import redflags
from make_template_holdout import GROUP_THR, group_items

TAX = gc.TAX
KNOWN_SLOTS = {
    "victim_name", "patronymic", "agent_name", "bank", "telecom", "gov_org", "authority",
    "pension_fund", "city", "district", "amount_small", "amount_medium", "amount_large",
    "loan_amount", "card_last4", "sms_code", "case_number", "app_name",
    "time_pressure_window", "investment_asset", "return_promise", "link",
}
KK_LETTERS = re.compile(r"[әіңғүұқөһӘІҢҒҮҰҚӨҺ]")
RU_ONLY = re.compile(r"[ёщъЁЩЪ]")          # ы и э есть и в казахском
RU_WORDS = {"и", "в", "на",   # «не», «да» — ещё и казахские слова
            "что", "это", "вы", "вас", "вам", "я", "с", "по", "для", "как",
            "но", "нет", "если", "уже", "только", "сейчас", "пожалуйста", "здравствуйте",
            "спасибо", "хорошо", "ваш", "ваша", "мне", "меня", "есть", "будет", "диктую"}
PAIRS = (("phrasebank_scam.yaml", "phrasebank_scam_ext.yaml"),
         ("phrasebank_benign.yaml", "phrasebank_benign_ext.yaml"),
         ("topics.yaml", "topics_ext.yaml"))


def pools(node, path=()):
    """Все списки реплик: (путь, язык, список)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("ru", "kk") and isinstance(v, list) and v and isinstance(v[0], str):
                yield path, k, v
            else:
                yield from pools(v, path + (str(k),))


def missing_paths(base, ext, path=()):
    if not isinstance(ext, dict):
        return
    for k, v in ext.items():
        if not isinstance(base, dict) or k not in base:
            yield "/".join(path + (str(k),))
        else:
            yield from missing_paths(base[k], v, path + (str(k),))


def main():
    errors = []
    for base_name, ext_name in PAIRS:
        if not (TAX / ext_name).exists():
            continue
        base = yaml.safe_load(open(TAX / base_name, encoding="utf-8"))
        ext = yaml.safe_load(open(TAX / ext_name, encoding="utf-8"))
        for p in missing_paths(base, ext):
            errors.append(f"{ext_name}: путь {p} отсутствует в {base_name}")

    tax = gc.load_taxonomy(ext=True)
    trees = {"scam": tax["pb_scam"], "benign": tax["pb_benign"], "topics": tax["topics"],
             "fillers": tax["fillers"]}
    sizes = Counter()
    n_lines = Counter()
    for root, tree in trees.items():
        for path, lang, items in pools(tree, (root,)):
            where = "/".join(path) + f"/{lang}"
            n_lines[lang] += len(items)
            for x in items:
                for s in re.findall(r"\{([a-z_]+)\}", x):
                    if s not in KNOWN_SLOTS:
                        errors.append(f"{where}: неизвестный слот {{{s}}} в «{x}»")
                words = set(re.findall(r"\w+", re.sub(r"\{[a-z_]+\}", " ", x).lower()))
                if lang == "kk" and not KK_LETTERS.search(x) and (RU_ONLY.search(x) or words & RU_WORDS):
                    errors.append(f"{where}: русская строка в казахском пуле? «{x}»")
            for x, c in Counter(items).items():
                if c > 1:
                    errors.append(f"{where}: повтор «{x}»")
            if root == "fillers":
                for x in items:
                    if redflags.detect(x):
                        errors.append(f"{where}: нейтральная реплика с флагом {redflags.detect(x)}: «{x}»")
            n_groups = len(group_items(items, GROUP_THR))
            sizes[("групп", min(n_groups, 6))] += 1

    print(f"Реплик: ru {n_lines['ru']}, kk {n_lines['kk']}")
    print("Пулов по числу групп близких формулировок (6 = шесть и больше):")
    print("  " + ", ".join(f"{k[1]}: {v}" for k, v in sorted(sizes.items())))
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    print("\nОшибок нет.")


if __name__ == "__main__":
    main()
