"""
Қалқан — расшифровка публичных записей звонков мошенников (YouTube) Whisper'ом.

Это единственный источник, где звучит НАСТОЯЩАЯ речь мошенников, звонящих
в Казахстан, на русском и казахском — без машинного перевода. Записей мало
(единицы), поэтому это не статистический тест, а проверка на живом
материале: сработают ли правила на реальной формулировке, а не на шаблоне.

Аудио лежит в data/external/raw/youtube_ru/ (в git не попадает),
на выходе — data/external/work/youtube_kz_asr.jsonl: сегменты Whisper
с таймкодами. Где в ролике сам звонок, а где комментарий автора —
размечается вручную по этому файлу.

    python transcribe_youtube.py --threads 4
"""

import argparse
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "external" / "raw" / "youtube_ru"
WORK = ROOT / "data" / "external" / "work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         cpu_threads=args.threads)
    out_path = WORK / "youtube_kz_asr.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["video_id"] for l in open(out_path, encoding="utf-8")}

    # короткие ролики первыми — быстрее увидеть результат
    audios = sorted(RAW.glob("*.m4a"), key=lambda p: p.stat().st_size)
    for audio in audios:
        vid = audio.stem
        if vid in done:
            continue
        info = json.load(open(RAW / f"{vid}.info.json", encoding="utf-8"))
        t0 = time.time()
        # язык не фиксируем: в казахстанских звонках речь смешанная kk/ru
        segs, meta = model.transcribe(str(audio), beam_size=5, vad_filter=True,
                                      condition_on_previous_text=False)
        segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segs]
        rec = {
            "video_id": vid,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": info.get("title"),
            "channel": info.get("channel"),
            "duration": info.get("duration"),
            "asr_model": f"faster-whisper {args.model} int8",
            "detected_lang": meta.language,
            "lang_prob": round(meta.language_probability, 3),
            "segments": segs,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dt = time.time() - t0
        dur = info.get("duration") or 1
        print(f"{vid}  {dur} с аудио за {dt:.0f} с (×{dt / dur:.2f}), "
              f"язык {meta.language} {meta.language_probability:.2f}, сегментов {len(segs)}",
              flush=True)


if __name__ == "__main__":
    main()
