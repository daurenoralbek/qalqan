"""
Қалқан — обучение инкрементального детектора (XLM-RoBERTa + LoRA).

Обучение — на corpus_train.jsonl (split=train), выбор эпохи и калибровка
накопителя/порога — на split=val. Шаблонный holdout и внешние наборы
в обучении не участвуют вовсе (оценка — eval_detector.py).

Работает и на CPU (прототип, --train-limit для дымового теста),
и на GPU (Kaggle/Colab: fp16 включается автоматически).

    python train_detector.py --train-limit 60 --val-limit 40 --epochs 0.2 --max-len 128   # дымовой тест
    python train_detector.py --epochs 3                                                   # полное обучение
"""

import argparse
import json
import platform
import random
import time
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

import detector as D

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "generated" / "corpus_train.jsonl"


def val_report(model, enc, val, device, batch):
    raw = D.score_dialogues(model, enc, val, batch_size=batch, device=device)
    y = [1 if d["label"] == "scam" else 0 for d in val]
    auc = roc_auc_score(y, [max(c) for c in raw]) if 0 < sum(y) < len(y) else None
    # по-репличный AUC: насколько хорошо отделены отдельные префиксы
    ty = [yy for yy, c in zip(y, raw) for _ in c]
    tp = [p for c in raw for p in c]
    turn_auc = roc_auc_score(ty, tp) if 0 < sum(ty) < len(ty) else None
    return raw, {"dialogue_auc": round(auc, 4) if auc else None,
                 "turn_auc": round(turn_auc, 4) if turn_auc else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=D.MODEL_NAME)
    ap.add_argument("--train-file", default=str(TRAIN))
    ap.add_argument("--out", default=str(ROOT / "models" / "xlmr-lora"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-fpr", type=float, default=0.05,
                    help="целевой FPR на валидации для выбора порога")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"

    data = D.load_jsonl(args.train_file)
    train = [d for d in data if d["split"] == "train"]
    val = [d for d in data if d["split"] == "val"]
    rng = random.Random(args.seed)
    if args.train_limit:
        rng.shuffle(train)
        train = train[:args.train_limit]
    if args.val_limit:
        rng.shuffle(val)
        val = val[:args.val_limit]

    model, tok = D.build_model(args.model, lora_r=args.lora_r)
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    enc = D.PrefixEncoder(tok, args.max_len)
    ds = D.PrefixDataset(train, enc)
    print(f"устройство {device}{' fp16' if amp else ''}, потоков {torch.get_num_threads()}")
    print(f"обучаемых параметров {n_train / 1e6:.2f} M из {n_all / 1e6:.0f} M "
          f"({100 * n_train / n_all:.2f}%)")
    print(f"диалогов train {len(train)}, val {len(val)}; префиксов train {len(ds)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    t_start = time.time()
    n_rounds = max(1, int(round(args.epochs))) if args.epochs >= 1 else 1
    per_round = args.epochs / n_rounds
    for ep in range(n_rounds):
        print(f"\nэпоха {ep + 1}/{n_rounds}")
        stats = D.train_epochs(model, ds, epochs=per_round, lr=args.lr, batch_size=args.batch,
                               device=device, seed=args.seed + ep, amp=amp)
        raw, rep = val_report(model, enc, val, device, args.eval_batch)
        cal = D.calibrate(val, raw, max_fpr=args.max_fpr)
        rep.update(epoch=ep + 1, train_seconds=stats["seconds"], calibration=cal)
        history.append(rep)
        v = cal["val"]
        print(f"  val: AUC диалог {rep['dialogue_auc']}, AUC реплика {rep['turn_auc']}; "
              f"накопитель {cal['mode']}({cal['a']}), порог {cal['threshold']:.3f} → "
              f"P {v['precision']:.3f} R {v['recall']:.3f} FPR {v['fpr']:.3f}, "
              f"заранее {cal['val_early_share_of_actioned']:.0%} (от диалогов с целевым действием)")
        key = (cal["val_early_share_of_actioned"], v["recall"], rep["dialogue_auc"] or 0)
        if best is None or key > best[0]:
            best = (key, ep + 1, D.trainable_state(model), cal)

    # лучшая эпоха → сохраняем адаптер и метаданные
    D.load_trainable_state(model, best[2])
    model.save_pretrained(str(out))
    meta = {
        "base_model": args.model,
        "max_len": args.max_len,
        "best_epoch": best[1],
        "accumulator": {k: best[3][k] for k in ("mode", "a", "threshold", "max_fpr")},
        "val_at_calibration": best[3]["val"],
        "history": history,
        "args": vars(args),
        "n_train_dialogues": len(train), "n_val_dialogues": len(val), "n_train_prefixes": len(ds),
        "trainable_params": n_train, "total_params": n_all,
        "device": device, "torch": torch.__version__, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    json.dump(meta, open(out / "qalqan_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nлучшая эпоха {best[1]}; сохранено → {out}")


if __name__ == "__main__":
    main()
