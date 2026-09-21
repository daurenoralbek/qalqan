"""
Қалқан — сборка демо для Hugging Face Space (Docker, CPU).

Что публикуется: код демо, detector.py, redflags.py, 4 синтетических примера
из шаблонного holdout и LoRA-адаптер (обучен только на синтетическом корпусе
Қалқан). Реальные звонки (KorCCVi, YouTube, переводы) НЕ публикуются.

    python demo/make_space.py                                   # собрать demo/space_build/
    python demo/make_space.py --push --repo <user>/qalqan-demo  # + выложить (нужен write-токен HF)
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
BUILD = DEMO / "space_build"
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "qalqan_meta.json"]

README = """---
title: Қалқан — детектор мошеннических звонков
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Предупреждает о мошенничестве по ходу звонка (kk / ru)
---

# Қалқан

Инкрементальный детектор мошеннических звонков для Казахстана: после каждой
реплики модель (XLM-RoBERTa + LoRA) оценивает риск по всему разговору и
поднимает тревогу **до** того, как человек продиктует код или переведёт деньги.
Казахский, русский и смешанная речь.

Модель обучена только на синтетическом корпусе Қалқан (11 тем звонков,
у каждой — мошеннический и легитимный сценарий). На реальных звонках
(переводы публичных записей) ROC AUC 0.71–0.75 против 0.50 у правил на
красных флагах.

Исследовательский прототип (конкурс АБРК / КБТУ / Nazarbayev University,
трек AI for Finance). Не является финансовой или юридической консультацией.
"""

DOCKERFILE = """FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

USER user
# базовая модель — в образ, чтобы Space не качал её при каждом старте
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \\
    AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base'); \\
    M.from_pretrained('FacebookAI/xlm-roberta-base', num_labels=2)"

COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", \\
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
"""

# те же версии, на которых демо проверено локально (адаптер сохранён peft 0.2x)
REQUIREMENTS = """torch==2.11.0
transformers==5.17.0
peft==0.21.0
streamlit==1.64.0
altair>=5
pandas
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "xlmr-lora-s42"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default=None, help="например, daurenoralbek/qalqan-demo")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    BUILD.mkdir(exist_ok=True)
    for f in BUILD.iterdir():
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    (BUILD / "model").mkdir()
    (BUILD / ".streamlit").mkdir()

    shutil.copy(DEMO / "app.py", BUILD / "app.py")
    shutil.copy(DEMO / "examples.json", BUILD / "examples.json")
    shutil.copy(DEMO / ".streamlit" / "config.toml", BUILD / ".streamlit" / "config.toml")
    for f in ("detector.py", "redflags.py"):
        shutil.copy(ROOT / "src" / f, BUILD / f)
    for f in ADAPTER_FILES:
        shutil.copy(Path(args.model_dir) / f, BUILD / "model" / f)
    (BUILD / "README.md").write_text(README, encoding="utf-8")
    (BUILD / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (BUILD / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    # в метаданных адаптера не должно быть локальных путей
    meta = json.load(open(BUILD / "model" / "qalqan_meta.json", encoding="utf-8"))
    meta.pop("args", None)
    meta.pop("platform", None)
    json.dump(meta, open(BUILD / "model" / "qalqan_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    files = sorted(str(p.relative_to(BUILD)) for p in BUILD.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in BUILD.rglob("*") if p.is_file())
    print(f"собрано → {BUILD} ({size / 1e6:.1f} МБ):")
    for f in files:
        print("   ", f)

    if args.push:
        from huggingface_hub import HfApi
        if not args.repo:
            raise SystemExit("укажите --repo <user>/qalqan-demo")
        api = HfApi()
        api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=args.private,
                        exist_ok=True)
        api.upload_folder(folder_path=str(BUILD), repo_id=args.repo, repo_type="space",
                          commit_message="Қалқан demo")
        print(f"опубликовано: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
