"""
Қалқан — машинный перевод внешнего теста (NLLB-200, локально на CPU).

Правила redflags.py и будущая модель работают на русском и казахском,
а реальные записи мошеннических звонков в открытом доступе — корейские
и английские. Переводим локально: условия KorCCVi запрещают передавать
данные третьим лицам, поэтому облачные API здесь не используются.

Честная оговорка: перевод меняет формулировки. Это не «русская живая речь»,
а «структура реального разговора, пересказанная переводчиком». Для правил
это даже жёстче, чем реальная речь: переводчик не знает наших регулярок.

Кэш переводов (work/mt_cache_*.json) позволяет прервать и продолжить.

    python translate_external.py --sets korccvi en_bank en_topic --tgt ru
"""

import argparse
import json
import re
import time
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "external" / "work"
MODEL = "facebook/nllb-200-distilled-600M"
# Та же модель, сконвертированная в CTranslate2 int8 — в ~4 раза быстрее на CPU:
#   ct2-transformers-converter --model facebook/nllb-200-distilled-600M #       --output_dir models/nllb-600m-ct2-int8 --quantization int8
CT2_DIR = ROOT / "models" / "nllb-600m-ct2-int8"

NLLB_CODE = {"ko": "kor_Hang", "en": "eng_Latn", "ru": "rus_Cyrl", "kk": "kaz_Cyrl"}
MAX_PIECE_TOKENS = 160


def split_pieces(text: str):
    """Разбить реплику на предложения: NLLB обучен на предложениях и на длинных
    абзацах начинает пропускать куски."""
    parts = re.split(r"(?<=[.?!。])\s+", text.strip())
    return [p for p in parts if p.strip()]


class Translator:
    def __init__(self, src: str, tgt: str, threads: int, batch: int, beams: int):
        self.tok = AutoTokenizer.from_pretrained(MODEL, src_lang=NLLB_CODE[src])
        self.model = ctranslate2.Translator(str(CT2_DIR), device="cpu", compute_type="int8",
                                            intra_threads=threads)
        self.tgt_code = NLLB_CODE[tgt]
        self.batch, self.beams = batch, beams
        self.cache_path = WORK / f"mt_cache_{src}_{tgt}.json"
        self.cache = (json.load(open(self.cache_path, encoding="utf-8"))
                      if self.cache_path.exists() else {})

    def cut(self, piece: str):
        """Слишком длинный кусок без пунктуации (ASR) режем по словам."""
        if len(self.tok(piece, add_special_tokens=False)["input_ids"]) <= MAX_PIECE_TOKENS:
            return [piece]
        out, cur = [], []
        for w in piece.split():
            cur.append(w)
            if len(self.tok(" ".join(cur), add_special_tokens=False)["input_ids"]) > MAX_PIECE_TOKENS * 0.8:
                out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
        return out

    def save(self):
        json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    def translate_all(self, pieces):
        todo = sorted({p for p in pieces if p not in self.cache}, key=len)
        t0, done = time.time(), 0
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            src_tokens = [self.tok.convert_ids_to_tokens(
                self.tok(c, truncation=True, max_length=MAX_PIECE_TOKENS + 8)["input_ids"])
                for c in chunk]
            res = self.model.translate_batch(
                src_tokens, target_prefix=[[self.tgt_code]] * len(chunk),
                beam_size=self.beams, max_decoding_length=int(MAX_PIECE_TOKENS * 1.8),
                repetition_penalty=1.1)
            for src, r in zip(chunk, res):
                ids = self.tok.convert_tokens_to_ids(r.hypotheses[0][1:])
                self.cache[src] = self.tok.decode(ids, skip_special_tokens=True)
            done += len(chunk)
            if (i // self.batch) % 20 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                eta = (len(todo) - done) / rate if rate else 0
                print(f"  {done}/{len(todo)} кусков, {rate:.1f}/с, осталось ~{eta / 60:.0f} мин",
                      flush=True)
                self.save()
        self.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["korccvi", "en_bank", "en_topic"])
    ap.add_argument("--tgt", default="ru", choices=["ru", "kk"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--beams", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="первые N/2 и последние N/2 диалогов (проба)")
    args = ap.parse_args()

    by_src = {}
    for name in args.sets:
        D = [json.loads(l) for l in open(WORK / f"{name}_orig.jsonl", encoding="utf-8")]
        if args.limit:
            D = D[:args.limit // 2] + D[-(args.limit // 2):]   # оба класса
        by_src.setdefault(D[0]["orig_lang"], []).append((name, D))

    for src, sets in by_src.items():
        print(f"[{src} → {args.tgt}] загрузка {MODEL}", flush=True)
        tr = Translator(src, args.tgt, args.threads, args.batch, args.beams)

        plan = {}   # (set, dialogue, turn) -> список кусков
        for name, D in sets:
            for di, d in enumerate(D):
                for ti, t in enumerate(d["turns"]):
                    plan[(name, di, ti)] = [c for p in split_pieces(t["text"]) for c in tr.cut(p)]
        pieces = [p for pcs in plan.values() for p in pcs]
        uniq = set(pieces)
        print(f"  кусков всего {len(pieces)}, уникальных {len(uniq)}, "
              f"уже в кэше {sum(p in tr.cache for p in uniq)}", flush=True)
        tr.translate_all(pieces)

        for name, D in sets:
            suffix = f"_{args.tgt}" + ("_sample" if args.limit else "")
            with open(WORK / f"{name}{suffix}.jsonl", "w", encoding="utf-8") as f:
                for di, d in enumerate(D):
                    for ti, t in enumerate(d["turns"]):
                        t["text_orig"] = t["text"]
                        t["text"] = " ".join(tr.cache[p] for p in plan[(name, di, ti)])
                    d["lang"] = args.tgt
                    d["translation"] = f"{MODEL} ct2-int8 ({src}→{args.tgt}, beams={args.beams})"
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"  → {name}{suffix}.jsonl", flush=True)


if __name__ == "__main__":
    main()
