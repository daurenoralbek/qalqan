"""
Қалқан — подготовка внешнего теста на реальных (не синтетических) звонках.

Зачем. Корпус Қалқан синтетический. Прежде чем тратить дни на обучение,
нужно дёшево проверить: не разваливается ли детектор на живой речи —
с перебиваниями, оговорками, ошибками ASR и формулировками, которых нет
в наших шаблонах.

Источники (подробности и лицензии — data/external/README.md):

  korccvi       реальные вишинговые звонки, опубликованные финрегулятором Кореи
                (FSS), против обычных звонков в госорганы/магазины (AI Hub).
                ASR с диаризацией и таймкодами. Язык: ko.
  en_bank       реальные звонки скамеров со скам-бейтерами (YouTube) против
                реальных звонков в банковскую поддержку (HarperValleyBank).
                Сплошной текст ASR без разметки говорящих. Язык: en.
  en_topic      те же реальные скам-звонки против легитимных звонков
                НА ТУ ЖЕ ТЕМУ, написанных вручную (самые трудные негативы). en.

Принцип: внутри каждого источника оба класса проходят ОДИН и тот же конвейер
(одинаковый ASR, одинаковая нарезка, один переводчик). Иначе детектор может
выучить «переведено с корейского» вместо «мошенничество».

Выход: data/external/work/<source>_orig.jsonl — в формате, совместимом
с корпусом (turns[].text), язык оригинала. Перевод — translate_external.py.
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "external" / "raw"
WORK = ROOT / "data" / "external" / "work"

# Ранняя детекция — это про начало разговора. Длинные звонки (у FSS бывают
# по 20+ минут) обрезаем: и перевод дешевле, и метрика «успел ли заранее»
# от хвоста разговора не зависит.
MAX_TURNS = 40
# Для сплошного текста без разметки говорящих: псевдо-реплика из N слов.
# ~20 слов — типичная длина реплики звонящего в корпусе Қалқан.
CHUNK_WORDS = 20
# Предел длины склеенной реплики в записях с таймкодами (KorCCVi).
MAX_TURN_SEC = 20


def dialogue(did, source, label, lang, turns, **meta):
    n_orig = len(turns)
    turns = turns[:MAX_TURNS]
    for i, t in enumerate(turns):
        t["idx"] = i
    return {
        "dialogue_id": did,
        "source": source,
        "label": label,
        "orig_lang": lang,
        "lang": lang,
        "translation": None,
        # стадии и момент целевого действия во внешних данных не размечены
        "action_turn_idx": -1,
        "n_turns_orig": n_orig,
        "truncated": n_orig > MAX_TURNS,
        "split": "external",
        **meta,
        "turns": turns,
    }


# ─────────────────────────────────────────────────────────────────────────
def load_korccvi(n_per_class: int, rng: random.Random):
    segs = defaultdict(list)
    for part in ("train", "val", "test"):
        path = RAW / "korccvi" / f"{part}_segment_manifest_merged.jsonl"
        for line in open(path, encoding="utf-8"):
            s = json.loads(line)
            segs[s["call_id"]].append(s)

    by_label = defaultdict(list)
    for cid, ss in segs.items():
        by_label[ss[0]["label"]].append(cid)

    out = []
    for lab, name in ((1, "scam"), (0, "benign")):
        ids = sorted(by_label[lab])
        rng.shuffle(ids)
        for cid in ids[:n_per_class]:
            ss = sorted(segs[cid], key=lambda s: s["start"])
            # склеиваем подряд идущие сегменты одного говорящего в реплику, но не
            # длиннее MAX_TURN_SEC: у трети записей FSS диаризация видит одного
            # говорящего, и без ограничения весь звонок стал бы одной репликой
            turns = []
            for s in ss:
                text = s["text"].strip()
                if not text:
                    continue
                if (turns and turns[-1]["speaker"] == s["speaker"]
                        and s["end"] - turns[-1]["start"] <= MAX_TURN_SEC):
                    turns[-1]["text"] += " " + text
                    turns[-1]["end"] = s["end"]
                else:
                    turns.append({"speaker": s["speaker"], "text": text,
                                  "start": s["start"], "end": s["end"]})
            out.append(dialogue(f"EXT-KOR-{cid}", "korccvi", name, "ko", turns,
                                origin="FSS" if lab == 1 else "AIHub"))
    return out


# ─────────────────────────────────────────────────────────────────────────
def chunk_words(text: str, n: int = CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def load_rajkirant(fname: str, source: str):
    out = []
    for r in csv.DictReader(open(RAW / "rajkirant" / fname, encoding="utf-8")):
        text = re.sub(r"\s+", " ", r["text"]).strip()
        turns = [{"speaker": "unknown", "text": c, "start": None, "end": None}
                 for c in chunk_words(text)]
        label = "scam" if r["label"] == "scam" else "benign"
        meta = {"origin": r["source"]}
        if "topic" in r:
            meta.update(topic=r["topic"], pair_id=r["pair_id"])
        out.append(dialogue(f"EXT-{source.upper()}-{r['id']}", source, label, "en",
                            turns, **meta))
    return out


def load_youtube_kz():
    """Реальные звонки мошенников казахстанцам (YouTube → Whisper), только размеченные
    фрагменты самого звонка. Сегмент Whisper = реплика (говорящие не размечены)."""
    asr_path = WORK / "youtube_kz_asr.jsonl"
    ann_path = ROOT / "data" / "external" / "youtube_kz_annotations.json"
    if not asr_path.exists():
        return []
    asr = {json.loads(l)["video_id"]: json.loads(l) for l in open(asr_path, encoding="utf-8")}
    out = []
    for c in json.load(open(ann_path, encoding="utf-8"))["calls"]:
        rec = asr[c["video_id"]]
        a, b = c["segments"]
        turns = [{"speaker": "unknown", "text": s["text"], "start": s["start"], "end": s["end"]}
                 for s in rec["segments"][a:b + 1]]
        out.append(dialogue(f"EXT-YTKZ-{c['video_id']}", "youtube_kz", "scam",
                            rec["detected_lang"], turns, origin=rec["url"],
                            scheme=c["scheme"], note=c["note"], asr=rec["asr_model"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--korccvi-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    WORK.mkdir(parents=True, exist_ok=True)
    sets = {
        "korccvi": load_korccvi(args.korccvi_per_class, rng),
        "en_bank": load_rajkirant("scambait_bank_422.csv", "en_bank"),
        "en_topic": load_rajkirant("scambait_synthetic_196.csv", "en_topic"),
        "youtube_kz": load_youtube_kz(),
    }
    for name, D in sets.items():
        if not D:
            continue
        with open(WORK / f"{name}_orig.jsonl", "w", encoding="utf-8") as f:
            for d in D:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_s = sum(d["label"] == "scam" for d in D)
        n_t = sum(len(d["turns"]) for d in D)
        n_tr = sum(d["truncated"] for d in D)
        print(f"{name:10s} {len(D):4d} диалогов (scam {n_s}, benign {len(D) - n_s}), "
              f"реплик {n_t}, обрезано до {MAX_TURNS}: {n_tr}")


if __name__ == "__main__":
    main()
