"""
Қалқан — замер задержки живого режима на аудиофайле.

Файл проигрывается с реальной скоростью (как звонок по линии) через тот же
конвейер, что и страница «Живой звонок»: VAD → Whisper → детектор. Для каждой
реплики печатается, через сколько секунд после конца фразы риск появился бы
на экране.

    python demo/live_bench.py --file demo/audio/call.m4a --lang ru
    python demo/live_bench.py --file ... --asr large-v3-turbo --out results/live_latency.json

Замер честен только на свободном процессоре: параллельная оценка или
обучение увеличивают задержку распознавания.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

import detector as D  # noqa: E402
import live as L  # noqa: E402
import redflags  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="начать с секунды")
    ap.add_argument("--max", type=float, default=None, help="взять не больше N секунд")
    ap.add_argument("--lang", default=None, help="ru / kk; по умолчанию — автоопределение")
    ap.add_argument("--asr", default="nemo", help="nemo (kk+ru) или модель Whisper: small, medium, large-v3…")
    ap.add_argument("--asr-threads", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--min-silence-ms", type=int, default=600)
    ap.add_argument("--model-dir", default=None, help="по умолчанию — как в демо (common.MODEL_DIR)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-text", action="store_true",
                    help="сохранить в --out и расшифровку (по умолчанию нет: чужие записи не публикуем)")
    args = ap.parse_args()

    if args.model_dir is None:
        cur = HERE.parent / "models" / "v31" / "xlmr-lora-s42"
        args.model_dir = os.environ.get("QALQAN_MODEL_DIR", str(cur if cur.exists() else HERE.parent / "models" / "xlmr-lora-s42"))
    torch.set_num_threads(args.torch_threads)
    model, tok, meta = D.load_trained(args.model_dir)
    enc, acc = D.PrefixEncoder(tok, meta["max_len"]), meta["accumulator"]

    @torch.inference_mode()
    def score(turns):
        x = torch.tensor([enc.prefixes(turns)[-1]])
        return float(torch.softmax(model(input_ids=x, attention_mask=torch.ones_like(x)).logits.float(), -1)[0, 1])

    t0 = time.time()
    asr = L.make_asr(args.asr, threads=args.asr_threads)
    print(f"распознавание {args.asr} загружено за {time.time() - t0:.0f} с; детектор {Path(args.model_dir).name}")
    src = L.FileSource(args.file, start_s=args.start, max_s=args.max)
    print(f"звук {len(src.audio) / L.SR:.0f} с, проигрываю с реальной скоростью…\n")

    s = L.LiveSession(src, asr, score, lambda p: D.accumulate(p, acc["mode"], acc["a"]),
                      rules=redflags.RuleBaseline(threshold=0.6), language=args.lang,
                      segmenter=L.Segmenter(min_silence_ms=args.min_silence_ms)).start()
    shown = 0
    while s.running:
        time.sleep(0.2)
        turns = s.snapshot()["turns"]
        for t in turns[shown:]:
            print(f"{t.idx + 1:3d} [{t.start_s:6.1f}–{t.end_s:6.1f} с] риск {t.risk:4.0%} · "
                  f"на экране через {t.latency_s:4.1f} с (пауза {t.endpoint_wait_s:.1f}, "
                  f"очередь {t.queue_s:.1f}, ASR {t.asr_s:.1f}, детектор {t.score_s * 1000:.0f} мс) │ {t.text[:80]}")
        shown = len(turns)
    if s.error:
        print("ОШИБКА:", s.error)
    turns = s.snapshot()["turns"]
    summ = L.latency_summary(turns)
    th = acc["threshold"]
    alert = D.first_alert([t.risk for t in turns], th)
    print("\nИтог:", json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in summ.items()},
                                ensure_ascii=False))
    print(f"Тревога модели (порог {th:.2f}): " +
          (f"реплика {alert + 1}, {turns[alert].end_s:.0f}-я секунда звука" if alert >= 0 else "нет"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"file": Path(args.file).name, "asr": args.asr, "lang": args.lang,
                   "model": Path(args.model_dir).name, "cpu_threads": {"asr": args.asr_threads,
                                                                        "torch": args.torch_threads},
                   "min_silence_ms": args.min_silence_ms, "summary": summ,
                   "alert_turn": alert,
                   "turns": [{k: v for k, v in asdict(t).items() if args.keep_text or k not in ("text", "flags")}
                             for t in turns]},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("Сохранено →", args.out)


if __name__ == "__main__":
    main()
