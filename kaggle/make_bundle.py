"""
Қалқан — сборка пакета для обучения на GPU Kaggle.

Что уходит на Kaggle: ТОЛЬКО наш синтетический корпус (corpus_train/holdout,
лицензия проекта) и код. Внешние наборы на реальных звонках туда НЕ уходят
(условия KorCCVi запрещают передачу) — модели скачиваются обратно и
оцениваются на них локально (eval_detector.py).

    python kaggle/make_bundle.py --seeds 42 43 44             # шаг 2: обучение детектора
    python kaggle/make_bundle.py --job federated --seeds 42   # шаг 3: федеративный эксперимент
    ... --push     # + загрузить датасет и запустить ноутбук
                   #   (нужен ~/.kaggle/kaggle.json и pip install kaggle)

Без API-токена: загрузите kaggle/build/qalqan-corpus/ как New Dataset (private),
импортируйте ноутбук из kaggle/build/kernel/, подключите датасет,
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
CODE = ["detector.py", "train_detector.py", "eval_detector.py", "redflags.py", "federated.py"]
DATA = ["corpus_train.jsonl", "corpus_holdout.jsonl"]
SRC_DIR = "/kaggle/working/qalqan/src"


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def setup_cell():
    return cell("code", [
        "import glob, os, shutil, json\n",
        "!nvidia-smi --query-gpu=name,memory.total --format=csv\n",
        "!pip install -q \"peft>=0.13\"\n",
        "# свежий peft отказывается работать со старой torchao из образа Kaggle,\n",
        "# а torchao нам не нужна вовсе\n",
        "!pip uninstall -y -q torchao\n",
        "IN = os.path.dirname(glob.glob('/kaggle/input/**/detector.py', recursive=True)[0])\n",
        "ROOT = '/kaggle/working/qalqan'\n",
        "for d in ('src', 'data/generated', 'results', 'models'):\n",
        "    os.makedirs(f'{ROOT}/{d}', exist_ok=True)\n",
        "for f in glob.glob(f'{IN}/*.py'):\n",
        "    shutil.copy(f, f'{ROOT}/src/')\n",
        "for f in glob.glob(f'{IN}/*.jsonl'):\n",
        "    shutil.copy(f, f'{ROOT}/data/generated/')\n",
        "print(os.listdir(f'{ROOT}/src'), os.listdir(f'{ROOT}/data/generated'))",
    ])


def run_cell(job, epochs, seeds, splits):
    seeds_s = ", ".join(str(s) for s in seeds)
    if job == "train":
        return cell("code", [
            f"SEEDS = [{seeds_s}]\n",
            "for seed in SEEDS:\n",
            f"    !cd {SRC_DIR} && python train_detector.py --epochs {epochs} --batch 32 "
            "--eval-batch 128 --seed {seed} --out ../models/xlmr-lora-s{seed}\n",
            f"    !cd {SRC_DIR} && python eval_detector.py --model-dir ../models/xlmr-lora-s{{seed}} "
            "--sets holdout --batch 128",
        ])
    splits_s = ", ".join(repr(x) for x in splits)
    return cell("code", [
        f"SPLITS = [{splits_s}]\n",
        f"SEEDS = [{seeds_s}]\n",
        "for seed in SEEDS:\n",
        "    for split in SPLITS:\n",
        f"        !cd {SRC_DIR} && python federated.py --split {{split}} --seed {{seed}} "
        f"--rounds {int(epochs)} --local-epochs 1\n",
        "        for d in sorted(glob.glob(f'/kaggle/working/qalqan/models/fed-{split}-s{seed}-*')):\n",
        f"            !cd {SRC_DIR} && python eval_detector.py --model-dir {{d}} --sets holdout --batch 128",
    ])


def collect_cell():
    return cell("code", [
        "# всё, что нужно забрать: адаптеры (несколько МБ), метаданные, метрики holdout\n",
        "for d in glob.glob('/kaggle/working/qalqan/models/*'):\n",
        "    shutil.rmtree(os.path.join(d, 'preds'), ignore_errors=True)\n",
        "shutil.make_archive('/kaggle/working/qalqan_models', 'zip', '/kaggle/working/qalqan/models')\n",
        "shutil.make_archive('/kaggle/working/qalqan_results', 'zip', '/kaggle/working/qalqan/results')\n",
        "for d in sorted(glob.glob('/kaggle/working/qalqan/models/*')):\n",
        "    m = json.load(open(os.path.join(d, 'qalqan_meta.json')))\n",
        "    print(d, m.get('best_epoch', m.get('method')), m['accumulator'], m['val_at_calibration'])",
    ])


def notebook(job, epochs, seeds, splits):
    title = ("обучение инкрементального детектора (XLM-R + LoRA)" if job == "train"
             else "федеративный эксперимент: central vs local vs FedAvg")
    cells = [
        cell("markdown", [
            f"# Қалқан — {title}\n",
            "\n",
            "Обучение на синтетическом корпусе Қалқан, оценка на шаблонном holdout.\n",
            "Оценка на реальных звонках делается локально (данные не покидают машину).\n",
            "Нужно: GPU и Internet (pip install peft, скачивание xlm-roberta-base).",
        ]),
        setup_cell(),
        run_cell(job, epochs, seeds, splits),
        collect_cell(),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def kaggle_user():
    p = Path.home() / ".kaggle" / "kaggle.json"
    return json.load(open(p))["username"] if p.exists() else "YOUR_KAGGLE_USERNAME"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="train", choices=["train", "federated"])
    ap.add_argument("--epochs", type=float, default=3,
                    help="train: эпох; federated: раундов (по 1 локальной эпохе)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--splits", nargs="+", default=["random", "profile", "topic"])
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    user = kaggle_user()
    kname = "qalqan-train" if args.job == "train" else "qalqan-federated"
    ds, kn = BUILD / "qalqan-corpus", BUILD / "kernel"
    for d in (ds, kn):          # чистим содержимое, а не папку: на Windows её может держать процесс
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    for f in CODE:
        shutil.copy(ROOT / "src" / f, ds / f)
    for f in DATA:
        shutil.copy(ROOT / "data" / "generated" / f, ds / f)
    json.dump({"title": "qalqan-corpus", "id": f"{user}/qalqan-corpus",
               "licenses": [{"name": "CC-BY-4.0"}]},
              open(ds / "dataset-metadata.json", "w"), indent=2)
    json.dump(notebook(args.job, args.epochs, args.seeds, args.splits),
              open(kn / f"{kname}.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"id": f"{user}/{kname}", "title": kname, "code_file": f"{kname}.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": True,
               "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [f"{user}/qalqan-corpus"], "competition_sources": [],
               "kernel_sources": []},
              open(kn / "kernel-metadata.json", "w"), indent=2)
    print(f"собрано → {BUILD}  (ноутбук {kname}, пользователь Kaggle: {user})")

    if args.push:
        # CLI Kaggle на Windows строит имя временного файла из пути -p и падает
        # на путях со слэшами — поэтому запускаем из самой папки с «-p .»
        exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/qalqan-corpus"],
                                capture_output=True, text=True).returncode == 0
        cmd = (["kaggle", "datasets", "version", "-p", ".", "-m", "update"] if exists
               else ["kaggle", "datasets", "create", "-p", "."])
        subprocess.run(cmd, check=True, cwd=ds)
        time.sleep(10)
        for _ in range(40):                       # ждём, пока Kaggle обработает датасет
            st = subprocess.run(["kaggle", "datasets", "status", f"{user}/qalqan-corpus"],
                                capture_output=True, text=True).stdout
            if "ready" in st:
                break
            time.sleep(15)
        subprocess.run(["kaggle", "kernels", "push", "-p", "."], check=True, cwd=kn)
        print(f"ноутбук запущен: https://www.kaggle.com/code/{user}/{kname}")


if __name__ == "__main__":
    main()
