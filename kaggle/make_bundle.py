"""
Қалқан — сборка пакета для обучения на GPU Kaggle.

Что уходит на Kaggle: ТОЛЬКО наш синтетический корпус (corpus_train/holdout,
лицензия проекта) и код. Внешние наборы на реальных звонках туда НЕ уходят
(условия KorCCVi запрещают передачу) — модель скачивается обратно и
оценивается на них локально (eval_detector.py).

    python kaggle/make_bundle.py                 # собрать kaggle/build/
    python kaggle/make_bundle.py --push          # + загрузить датасет и запустить ноутбук
                                                 #   (нужен ~/.kaggle/kaggle.json и pip install kaggle)

Без API-токена: загрузите kaggle/build/qalqan-corpus/ как New Dataset (private),
импортируйте kaggle/build/kernel/qalqan_train.ipynb, подключите датасет,
включите GPU и Internet, Run All. Подробно — kaggle/README.md.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle" / "build"
CODE = ["detector.py", "train_detector.py", "eval_detector.py", "redflags.py"]
DATA = ["corpus_train.jsonl", "corpus_holdout.jsonl"]


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(epochs: float, seeds):
    seeds_s = ", ".join(str(s) for s in seeds)
    cells = [
        cell("markdown", [
            "# Қалқан — обучение инкрементального детектора (XLM-R + LoRA)\n",
            "\n",
            "Обучение на синтетическом корпусе Қалқан, оценка на шаблонном holdout.\n",
            "Оценка на реальных звонках делается локально (данные не покидают машину).\n",
            "Нужно: GPU (T4/P100) и Internet (pip install peft, скачивание xlm-roberta-base).",
        ]),
        cell("code", [
            "import glob, os, shutil, json\n",
            "!nvidia-smi --query-gpu=name,memory.total --format=csv\n",
            "!pip install -q \"peft>=0.13\"\n",
            "IN = os.path.dirname(glob.glob('/kaggle/input/**/detector.py', recursive=True)[0])\n",
            "ROOT = '/kaggle/working/qalqan'\n",
            "for d in ('src', 'data/generated', 'results', 'models'):\n",
            "    os.makedirs(f'{ROOT}/{d}', exist_ok=True)\n",
            "for f in glob.glob(f'{IN}/*.py'):\n",
            "    shutil.copy(f, f'{ROOT}/src/')\n",
            "for f in glob.glob(f'{IN}/*.jsonl'):\n",
            "    shutil.copy(f, f'{ROOT}/data/generated/')\n",
            "print(os.listdir(f'{ROOT}/src'), os.listdir(f'{ROOT}/data/generated'))",
        ]),
        cell("code", [
            f"SEEDS = [{seeds_s}]\n",
            "for seed in SEEDS:\n",
            f"    !cd /kaggle/working/qalqan/src && python train_detector.py --epochs {epochs} --batch 32 --eval-batch 128 --seed {{seed}} --out ../models/xlmr-lora-s{{seed}}\n",
            "    !cd /kaggle/working/qalqan/src && python eval_detector.py --model-dir ../models/xlmr-lora-s{seed} --sets holdout --batch 128",
        ]),
        cell("code", [
            "# всё, что нужно забрать: адаптеры (несколько МБ), метаданные, метрики holdout\n",
            "for d in glob.glob('/kaggle/working/qalqan/models/xlmr-lora-s*'):\n",
            "    shutil.rmtree(os.path.join(d, 'preds'), ignore_errors=True)\n",
            "shutil.make_archive('/kaggle/working/qalqan_models', 'zip', '/kaggle/working/qalqan/models')\n",
            "shutil.make_archive('/kaggle/working/qalqan_results', 'zip', '/kaggle/working/qalqan/results')\n",
            "for d in sorted(glob.glob('/kaggle/working/qalqan/models/xlmr-lora-s*')):\n",
            "    m = json.load(open(os.path.join(d, 'qalqan_meta.json')))\n",
            "    print(d, 'best epoch', m['best_epoch'], m['accumulator'], m['val_at_calibration'])",
        ]),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def kaggle_user():
    p = Path.home() / ".kaggle" / "kaggle.json"
    return json.load(open(p))["username"] if p.exists() else "YOUR_KAGGLE_USERNAME"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    user = kaggle_user()
    shutil.rmtree(BUILD, ignore_errors=True)
    ds, kn = BUILD / "qalqan-corpus", BUILD / "kernel"
    ds.mkdir(parents=True)
    kn.mkdir(parents=True)
    for f in CODE:
        shutil.copy(ROOT / "src" / f, ds / f)
    for f in DATA:
        shutil.copy(ROOT / "data" / "generated" / f, ds / f)
    json.dump({"title": "qalqan-corpus", "id": f"{user}/qalqan-corpus",
               "licenses": [{"name": "CC-BY-4.0"}]},
              open(ds / "dataset-metadata.json", "w"), indent=2)
    json.dump(notebook(args.epochs, args.seeds), open(kn / "qalqan_train.ipynb", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"id": f"{user}/qalqan-train", "title": "qalqan-train",
               "code_file": "qalqan_train.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [f"{user}/qalqan-corpus"], "competition_sources": [],
               "kernel_sources": []},
              open(kn / "kernel-metadata.json", "w"), indent=2)
    print(f"собрано → {BUILD}  (пользователь Kaggle: {user})")

    if args.push:
        # CLI Kaggle на Windows строит имя временного файла из пути -p и падает
        # на путях со слэшами — поэтому запускаем из самой папки с «-p .»
        exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/qalqan-corpus"],
                                capture_output=True, text=True).returncode == 0
        cmd = (["kaggle", "datasets", "version", "-p", ".", "-m", "update"] if exists
               else ["kaggle", "datasets", "create", "-p", "."])
        subprocess.run(cmd, check=True, cwd=ds)
        for _ in range(40):                       # ждём, пока Kaggle обработает датасет
            st = subprocess.run(["kaggle", "datasets", "status", f"{user}/qalqan-corpus"],
                                capture_output=True, text=True).stdout
            if "ready" in st:
                break
            time.sleep(15)
        subprocess.run(["kaggle", "kernels", "push", "-p", "."], check=True, cwd=kn)
        print(f"ноутбук запущен: https://www.kaggle.com/code/{user}/qalqan-train")


if __name__ == "__main__":
    main()
