"""
Қалқан — сравнение конфигураций распознавания речи для живого режима.

Запись режется на реплики тем же VAD, что и в demo/live.py, каждая реплика
распознаётся отдельно (как в живом звонке), тексты склеиваются и сравниваются
с эталоном. Печатаются доля ошибок по словам (WER), по символам (CER) и время
на реплику — то есть цена качества.

Эталон:
  * для записей из youtube_kz — расшифровка faster-whisper large-v3, сделанная
    заранее (data/external/work/youtube_kz_asr.jsonl). Это не идеальная истина,
    а лучшее, что у нас есть: метрика показывает, насколько быстрая конфигурация
    отстаёт от самой сильной модели;
  * для собственных записей (demo/audio/<имя>.wav + <имя>.txt) — настоящий
    эталонный текст, который читал диктор.

    python src/asr_bench.py --configs small,small-beam5
    python src/asr_bench.py --audio demo/audio/kk_test.wav --ref demo/audio/kk_test.txt
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
import live as L  # noqa: E402

WORK = ROOT / "data" / "external" / "work"
RAW = ROOT / "data" / "external" / "raw" / "youtube_ru"

# Термины, которые модель путает: банки, ведомства, слова схем. Whisper принимает
# такую подсказку как начало предыдущего текста и держит их в словаре.
PROMPT_RU = ("Разговор с банком. Kaspi, Halyk, Forte, Jusan, Отбасы, БЖЗҚ, ЕНПФ, eGov, "
             "Activ, Beeline, Tele2, Казахтелеком. Код из SMS, безопасный счёт, "
             "мобильный банк, удалённый доступ, AnyDesk, заявка на кредит, ИИН.")
PROMPT_KK = ("Банкпен әңгіме. Kaspi, Halyk, Отбасы, БЖЗҚ, eGov, Activ, Beeline, Tele2, "
             "Қазақтелеком. SMS-тегі код, қауіпсіз шот, мобильді банк, қашықтан қосылу, "
             "несиеге өтінім, ЖСН, жеке куәлік.")

CONFIGS = {
    # имя: (модель, beam, подсказка, потоков)
    "small":            ("small", 1, None, 4),
    "small-beam5":      ("small", 5, None, 4),
    "small-prompt":     ("small", 1, "domain", 4),
    "small-beam5-prompt": ("small", 5, "domain", 4),
    "medium":           ("medium", 1, None, 4),
    "medium-prompt":    ("medium", 1, "domain", 4),
    "turbo":            ("large-v3-turbo", 1, None, 4),
    "turbo-prompt":     ("large-v3-turbo", 1, "domain", 4),
    "large-v3":         ("large-v3", 1, None, 4),
    # двуязычная CTC-модель NVIDIA (казахский + русский), ONNX
    "nemo":             ("nemo", None, None, 4),
    "nemo-fp32":        ("nemo-fp32", None, None, 4),
    # казахский Whisper turbo (дообучен на KSC2 и др.), конвертирован в CTranslate2
    "kk-turbo":         ("models/asr/kazakh-turbo-ct2", 1, None, 4),
}


def norm(t: str) -> str:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit(a, b) -> int:
    """Расстояние Левенштейна по элементам последовательностей."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def rates(hyp: str, ref: str):
    h, r = norm(hyp), norm(ref)
    hw, rw = h.split(), r.split()
    wer = edit(hw, rw) / max(1, len(rw))
    cer = edit(h, r) / max(1, len(r))
    return wer, cer, len(rw)


def utterances(path, min_silence_ms=600):
    """Реплики ровно так, как их видит живой режим."""
    audio = L.load_audio(path)
    seg = L.Segmenter(min_silence_ms=min_silence_ms)
    out, blk = [], 640
    import threading
    stop = threading.Event()
    for i in range(0, len(audio), blk):
        out += seg.feed(audio[i:i + blk], wall=i / L.SR)
    out += seg.feed(__import__("numpy").zeros(L.SR, "float32"), wall=len(audio) / L.SR)
    del stop
    return audio, out


def run(name, utts, lang):
    model, beam, prompt, threads = CONFIGS[name]
    if model.startswith("nemo"):
        import asr_nemo
        asr = asr_nemo.NemoASR(quant="int8" if model == "nemo" else "", threads=threads)
    else:
        asr = L.ASR(model, cpu_threads=threads, beam_size=beam)
    p = (PROMPT_KK if lang == "kk" else PROMPT_RU) if prompt == "domain" else None
    texts, t0 = [], time.time()
    for u in utts:
        text, _, _ = asr(u.audio, lang, p)
        if text:
            texts.append(text)
    return " ".join(texts), (time.time() - t0) / max(1, len(utts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="small,small-prompt,small-beam5-prompt")
    ap.add_argument("--audio", nargs="*", help="свои файлы; иначе записи youtube_kz")
    ap.add_argument("--ref", nargs="*", help="эталонные тексты к --audio")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--fleurs", type=int, default=0,
                    help="проверка на казахском: N записей FLEURS с эталонными расшифровками")
    ap.add_argument("--limit", type=int, default=0, help="взять первые N реплик записи")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = []            # (имя, путь, эталонный текст)
    if args.fleurs:
        import csv
        hub = Path.home() / ".cache/huggingface/hub/datasets--google--fleurs/snapshots"
        tsv = next(hub.glob("*/data/kk_kz/dev.tsv"))
        wavs = {p.name: p for p in (ROOT / "data/external/raw/fleurs_kk").rglob("*.wav")}
        for r in list(csv.reader(open(tsv, encoding="utf-8"), delimiter="\t"))[:args.fleurs]:
            if r[1] in wavs:
                items.append((r[0], wavs[r[1]], r[3]))       # id, wav, нормализованный эталон
        args.lang = "kk"
    elif args.audio:
        refs = args.ref or []
        for i, a in enumerate(args.audio):
            ref = Path(refs[i]).read_text(encoding="utf-8") if i < len(refs) else ""
            items.append((Path(a).stem, Path(a), ref))
    else:
        for d in map(json.loads, open(WORK / "youtube_kz_asr.jsonl", encoding="utf-8")):
            f = RAW / f"{d['video_id']}.m4a"
            if f.exists():
                items.append((d["video_id"], f, " ".join(s["text"] for s in d["segments"])))

    report = []
    for name, path, ref in items:
        if args.fleurs:                       # одна запись FLEURS = одна фраза, VAD не нужен
            audio = L.load_audio(path)
            utts = [L.Utterance(audio, 0.0, len(audio) / L.SR, 0.0, 0.0)]
        else:
            audio, utts = utterances(path)
        if args.limit:
            utts = utts[:args.limit]
        if not args.fleurs:
            print(f"\n=== {name}: {len(audio) / L.SR:.0f} с, {len(utts)} реплик"
                  f"{'' if ref else ' (эталона нет)'}", flush=True)
        for cfg in args.configs.split(","):
            hyp, sec = run(cfg, utts, args.lang)
            if ref:
                wer, cer, nref = rates(hyp, ref)
                if not args.fleurs:
                    print(f"  {cfg:20s} WER {wer:5.1%}  CER {cer:5.1%}  {sec:4.1f} с/реплика", flush=True)
                report.append({"запись": name, "конфигурация": cfg, "wer": round(wer, 4),
                               "cer": round(cer, 4), "сек_на_реплику": round(sec, 2),
                               "слов_в_эталоне": nref})
            else:
                print(f"  {cfg:20s} {sec:4.1f} с/реплика │ {hyp[:110]}", flush=True)
    if report:
        import statistics as st
        print("\nСреднее" + (f" по {len(items)} фразам FLEURS (казахский):" if args.fleurs
                                else " по записям:"))
        for cfg in args.configs.split(","):
            rows = [r for r in report if r["конфигурация"] == cfg]
            if rows:
                w = sum(r["слов_в_эталоне"] for r in rows)
                wer = sum(r["wer"] * r["слов_в_эталоне"] for r in rows) / max(1, w)
                cer = sum(r["cer"] * r["слов_в_эталоне"] for r in rows) / max(1, w)
                print(f"  {cfg:20s} WER {wer:5.1%}  CER {cer:5.1%}  "
                      f"{st.mean(r['сек_на_реплику'] for r in rows):4.1f} с/фраза")
    if args.out and report:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("Сохранено →", args.out)


if __name__ == "__main__":
    main()
