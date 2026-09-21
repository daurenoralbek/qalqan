"""
Қалқан — перевод внешнего теста на казахский моделью ISSAI KazLLM
(LLama-3.1-KazLLM-1.0-8B, квант Q4_K_M, локально через llama.cpp на CPU).

Зачем. Публичных записей мошеннических звонков на казахском нет —
это зафиксировано в data/external/README.md. Единственный честный способ
проверить детектор на казахском в структуре РЕАЛЬНОГО разговора —
перевести реальные звонки. KazLLM — казахстанская модель (ISSAI, NU).

Корейского в KazLLM нет (kk/ru/en/tr), поэтому KorCCVi переводим через
русский: ko →(NLLB)→ ru →(KazLLM)→ kk. Английские звонки — напрямую en → kk.

Инструкция модели — на казахском: с русской инструкцией KazLLM отвечает
по-русски (первая проба так и «перевела» — переписала русский текст).
Каждая строка проверяется на казахские буквы; не прошедшие проверку
переспрашиваются по одной, итог помечается в kk_ok.

Данные локальные, в облако не уходят (условия KorCCVi).
Скорость на CPU ~2–4 мин на диалог, поэтому берём начало разговора
(--max-turns) и поднабор (--per-class).

    python translate_kazllm.py --per-class 25 --max-turns 20
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
    "Сен кәсіби аудармашысың. Саған телефон әңгімесінің репликалары беріледі. "
    "Әр репликаны қазақ тіліне аудар. Ауызекі сөйлеуді табиғи, Қазақстанда "
    "сөйлейтіндей аудар. Мағынасын сақта, ештеңе қоспа және алып тастама, "
    "мазмұнына жауап берме, түсініктеме жазба. Тек қазақша аударманы шығар."
)
SRC_NAME = {"ru": "орыс", "en": "ағылшын"}

# Образец формата и языка ответа (нейтральная фраза, не из тестовых данных)
FEWSHOT_SRC = {
    "ru": ["Добрый день, чем могу помочь?", "Я хотел бы узнать баланс карты."],
    "en": ["Good afternoon, how can I help you?", "I would like to check my card balance."],
}
FEWSHOT_KK = ["Қайырлы күн, қалай көмектесе аламын?", "Мен картамның балансын білгім келеді."]

KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


def looks_kazakh(text: str) -> bool:
    """Казахский текст почти всегда содержит специфические буквы; русский — нет."""
    if len(text.split()) <= 2:       # «Иә», «Жоқ», «OK» — не проверяем
        return True
    return any(ch in KK_LETTERS for ch in text)


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


def numbered(lines):
    return "\n".join(f"{i + 1}) {t}" for i, t in enumerate(lines))


def user_prompt(lines, src_lang):
    return (f"Төмендегі әр жолды {SRC_NAME[src_lang]} тілінен қазақ тіліне аудар. "
            f"Нөмірлеуді сақта: дәл {len(lines)} жол, «N) аударма» форматында.\n\n"
            f"{numbered(lines)}")


def ask(lines, src_lang):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(FEWSHOT_SRC[src_lang], src_lang)},
        {"role": "assistant", "content": numbered(FEWSHOT_KK)},
        {"role": "user", "content": user_prompt(lines, src_lang)},
    ]
    out = chat(messages, 64 + 6 * sum(len(t.split()) for t in lines))
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\)\s*(.*)", line)
        if m:
            got[int(m.group(1))] = m.group(2).strip()
    return [got.get(i + 1, "") for i in range(len(lines))]


def translate_block(lines, src_lang):
    """Блоком (быстрее, контекст разговора сохраняется); строки, где нумерация
    поехала или ответ не по-казахски, — повторно по одной."""
    res = ask(lines, src_lang)
    ok = []
    for i, (src, kk) in enumerate(zip(lines, res)):
        if not kk or not looks_kazakh(kk):
            kk = ask([src], src_lang)[0]
        ok.append(bool(kk) and looks_kazakh(kk))
        res[i] = kk
    return res, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--max-turns", type=int, default=20,
                    help="переводим начало разговора — ранняя детекция про него")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # (файл-источник, язык источника): KorCCVi — уже русский перевод, EN — оригинал
    plan = [("korccvi_ru.jsonl", "ru"), ("en_bank_orig.jsonl", "en")]
    out_path = WORK / "kazllm_kk.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["dialogue_id"] for l in open(out_path, encoding="utf-8")}

    rng = random.Random(args.seed)
    todo = []
    for fname, src_lang in plan:
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
            d["turns"] = d["turns"][:args.max_turns]
            texts = [t["text"] for t in d["turns"]]
            kk, ok = [], []
            for i in range(0, len(texts), TURNS_PER_REQUEST):
                r, o = translate_block(texts[i:i + TURNS_PER_REQUEST], src_lang)
                kk += r
                ok += o
            for t, k_text, k_ok in zip(d["turns"], kk, ok):
                t.setdefault("text_orig", t["text"])
                t["text_pivot"] = t["text"] if src_lang == "ru" else None
                t["text"] = k_text
                t["kk_ok"] = k_ok
            d["kk_ok_share"] = round(sum(ok) / len(ok), 3) if ok else None
            d["lang"] = "kk"
            pivot = "ko→ru NLLB → " if src_lang == "ru" else ""
            d["translation"] = (f"KazLLM-1.0-8B Q4_K_M ({pivot}{src_lang}→kk), "
                                f"первые {args.max_turns} реплик")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
            el = time.time() - t0
            left = el / (k + 1) * (len(todo) - k - 1) / 60
            print(f"  {k + 1}/{len(todo)} {d['dialogue_id']} kk_ok {d['kk_ok_share']} — "
                  f"{el / (k + 1):.0f} с/диалог, осталось ~{left:.0f} мин", flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
