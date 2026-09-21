"""
Қалқан — перевод внешнего теста на казахский моделью ISSAI KazLLM
(LLama-3.1-KazLLM-1.0-8B, квант Q4_K_M, локально через llama.cpp на CPU).

Зачем. Публичных записей мошеннических звонков на казахском нет —
это зафиксировано в data/external/README.md. Единственный честный способ
проверить детектор на казахском в структуре РЕАЛЬНОГО разговора —
перевести реальные звонки. KazLLM — казахстанская модель (ISSAI, NU),
её казахский заметно естественнее, чем у NLLB; для сравнения тот же
поднабор переводится и NLLB (translate_external.py --tgt kk).

Корейского в KazLLM нет (kk/ru/en/tr), поэтому KorCCVi переводим через
русский: ko →(NLLB)→ ru →(KazLLM)→ kk. Английские звонки — напрямую en → kk.

Данные локальные, в облако не уходят (условия KorCCVi).

    python translate_kazllm.py --per-class 40
"""

import argparse
import json
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "external" / "work"
LLAMA = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
GGUF_REPO = "issai/LLama-3.1-KazLLM-1.0-8B-GGUF4"
GGUF_FILE = "checkpoints_llama8b_031224_18900-Q4_K_M.gguf"
PORT = 8088
TURNS_PER_REQUEST = 8

SYSTEM = (
    "Ты профессиональный переводчик на казахский язык. Тебе дают реплики "
    "телефонного разговора. Переводи разговорную речь естественно, как говорят "
    "в Казахстане. Сохраняй смысл, ничего не добавляй и не убирай, не отвечай "
    "на содержание и не комментируй. Выводи только перевод."
)

SRC_NAME = {"ru": "русского", "en": "английского"}


def start_server(threads: int):
    from huggingface_hub import hf_hub_download
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE)
    proc = subprocess.Popen(
        [str(LLAMA), "-m", gguf, "-c", "8192", "-t", str(threads), "--port", str(PORT),
         "-np", "1", "--temp", "0.2"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(2)
    proc.kill()
    raise RuntimeError("llama-server не поднялся")


def chat(messages, max_tokens: int) -> str:
    body = json.dumps({"messages": messages, "temperature": 0.2, "max_tokens": max_tokens,
                       "cache_prompt": True}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def translate_block(lines, src_lang):
    numbered = "\n".join(f"{i + 1}) {t}" for i, t in enumerate(lines))
    user = (f"Переведи каждую строку с {SRC_NAME[src_lang]} на казахский. "
            f"Сохрани нумерацию: ровно {len(lines)} строк в формате «N) перевод».\n\n{numbered}")
    max_tokens = 64 + 4 * sum(len(t.split()) for t in lines) * 2
    out = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
               max_tokens)
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\)\s*(.*)", line)
        if m:
            got[int(m.group(1))] = m.group(2).strip()
    if len(got) == len(lines) and all(got.get(i + 1) for i in range(len(lines))):
        return [got[i + 1] for i in range(len(lines))]
    # нумерация поехала — переводим по одной строке
    res = []
    for t in lines:
        o = chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Переведи с {SRC_NAME[src_lang]} на казахский:\n{t}"}],
                 64 + 8 * len(t.split()))
        res.append(o.strip().splitlines()[0] if o.strip() else "")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # (файл-источник, язык источника): KorCCVi — уже русский перевод, EN — оригинал
    plan = [("korccvi_ru.jsonl", "ru", "korccvi"), ("en_bank_orig.jsonl", "en", "en_bank")]
    out_path = WORK / "kazllm_kk.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["dialogue_id"] for l in open(out_path, encoding="utf-8")}

    rng = random.Random(args.seed)
    todo = []
    for fname, src_lang, _ in plan:
        D = [json.loads(l) for l in open(WORK / fname, encoding="utf-8")]
        for lab in ("scam", "benign"):
            X = [d for d in D if d["label"] == lab]
            rng.shuffle(X)
            todo += [(d, src_lang) for d in X[:args.per_class]]
    todo = [(d, s) for d, s in todo if d["dialogue_id"] not in done]
    print(f"к переводу {len(todo)} диалогов (уже готово {len(done)})", flush=True)
    if not todo:
        return

    proc = start_server(args.threads)
    try:
        t0 = time.time()
        for k, (d, src_lang) in enumerate(todo):
            texts = [t["text"] for t in d["turns"]]
            kk = []
            for i in range(0, len(texts), TURNS_PER_REQUEST):
                kk += translate_block(texts[i:i + TURNS_PER_REQUEST], src_lang)
            for t, k_text in zip(d["turns"], kk):
                t.setdefault("text_orig", t["text"])
                t["text_pivot"] = t["text"] if src_lang == "ru" else None
                t["text"] = k_text
            d["lang"] = "kk"
            d["translation"] = (f"KazLLM-1.0-8B Q4_K_M ({'ko→ru NLLB → ' if src_lang == 'ru' else ''}"
                                f"{src_lang}→kk)")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo)} {d['dialogue_id']} — {el / (k + 1):.0f} с/диалог, "
                  f"осталось ~{el / (k + 1) * (len(todo) - k - 1) / 60:.0f} мин", flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
