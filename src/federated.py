"""
Қалқан — федеративный эксперимент: как банки обучают детектор вместе,
не передавая друг другу записи разговоров.

Три «банка» получают непересекающиеся части обучающего корпуса. Сравниваем
при РАВНОМ вычислительном бюджете (3 прохода по данным):

  central   все данные в одном месте, 3 эпохи                  — верхняя граница
  local_k   банк k обучается только на своих данных, 3 эпохи   — «каждый сам по себе»
  fedavg    3 раунда: каждый банк 1 эпоха локально от общей модели,
            сервер усредняет обновления (взвешенно по объёму данных)

Банки обмениваются только обучаемыми параметрами — LoRA-адаптерами и головой
классификатора (~1.5 млн чисел, ~6 МБ за раунд), а не разговорами. Это та же
схема, что в AI-in-the-Loop (PoPETs 2026), но с LoRA вместо полной модели.

Разбиения по банкам (--split):
  random     IID: случайно
  profile    по профилю клиента (victim_profile): пожилые+доверчивые / сомневающиеся /
             подозрительные — у разных банков разная клиентская база
  topic      по темам звонков: каждый банк видел только часть тем (например,
             только банковские и телеком-схемы) — у каждой темы оба класса.
             Самый жёсткий случай: сможет ли общая модель ловить схему,
             которой в «своём» банке не было?

Все модели сохраняются в формате train_detector.py → оцениваются eval_detector.py
(holdout — на Kaggle, реальные звонки — локально).

    python federated.py --split profile --rounds 3
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

import detector as D

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "generated" / "corpus_train.jsonl"

PROFILE_BANK = {"elderly": 0, "compliant": 0, "hesitant": 1, "suspicious": 2}
# v2: банки специализируются по темам звонков (у каждой темы оба класса)
TOPIC_BANK = {"bank": 0, "credit": 0, "telecom": 0, "telecom_security": 0,
              "pension": 1, "invest": 1, "tech_support": 1,
              "relative": 2, "delivery": 2, "clinic": 2, "job": 2}


def split_banks(dialogues, mode: str, k: int, seed: int):
    rng = random.Random(seed)
    banks = [[] for _ in range(k)]
    ds = list(dialogues)
    rng.shuffle(ds)
    for i, d in enumerate(ds):
        if mode == "random":
            b = i % k
        elif mode == "profile":
            b = PROFILE_BANK[d["victim_profile"]]
        elif mode == "topic":
            b = TOPIC_BANK[d["topic"]]
        else:
            raise ValueError(mode)
        banks[b].append(d)
    return banks


def val_eval(model, enc, val, device, batch, max_fpr):
    raw = D.score_dialogues(model, enc, val, batch_size=batch, device=device)
    y = [1 if d["label"] == "scam" else 0 for d in val]
    auc = roc_auc_score(y, [max(c) for c in raw])
    cal = D.calibrate(val, raw, max_fpr=max_fpr)
    return round(auc, 4), cal


def save(model, out: Path, args, name, cal, extra):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    meta = {"base_model": args.model, "max_len": args.max_len, "method": name,
            "accumulator": {k: cal[k] for k in ("mode", "a", "threshold", "max_fpr")},
            "val_at_calibration": cal["val"], "args": vars(args), **extra}
    json.dump(meta, open(out / "qalqan_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def fresh_model(args, device):
    torch.manual_seed(args.seed)             # одинаковая инициализация LoRA у всех методов
    model, tok = D.build_model(args.model, lora_r=args.lora_r)
    return model.to(device), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=D.MODEL_NAME)
    ap.add_argument("--split", default="profile", choices=["random", "profile", "topic"])
    ap.add_argument("--banks", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local-epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--methods", nargs="+", default=["central", "local", "fedavg"])
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default=str(ROOT / "models"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    data = D.load_jsonl(TRAIN)
    train = [d for d in data if d["split"] == "train"]
    val = [d for d in data if d["split"] == "val"]
    rng = random.Random(args.seed)
    if args.train_limit:
        rng.shuffle(train)
        train = train[:args.train_limit]
    if args.val_limit:
        rng.shuffle(val)
        val = val[:args.val_limit]

    banks = split_banks(train, args.split, args.banks, args.seed)
    total_epochs = args.rounds * args.local_epochs     # одинаковый бюджет для всех методов
    report = {"split": args.split, "device": device, "total_epochs": total_epochs,
              "banks": [{"dialogues": len(b), "scam": sum(d["label"] == "scam" for d in b),
                         "scenarios": sorted({d["scenario_id"] for d in b if d["label"] == "scam"}),
                         "topics": sorted({d["topic"] for d in b}),
                         "profiles": sorted({d["victim_profile"] for d in b})} for b in banks],
              "methods": {}}
    for i, b in enumerate(report["banks"]):
        print(f"банк {i}: {b['dialogues']} диалогов, scam {b['scam']}, темы {b['topics']}, "
              f"профили {b['profiles']}")

    tag = f"fed-{args.split}-s{args.seed}"
    kw = dict(lr=args.lr, batch_size=args.batch, device=device, amp=amp)

    # ── централизованно ──────────────────────────────────────────────
    if "central" in args.methods:
        print(f"\n[central] {total_epochs} эпох на всех данных")
        model, tok = fresh_model(args, device)
        enc = D.PrefixEncoder(tok, args.max_len)
        t0 = time.time()
        D.train_epochs(model, D.PrefixDataset(train, enc), epochs=total_epochs, seed=args.seed, **kw)
        auc, cal = val_eval(model, enc, val, device, args.eval_batch, args.max_fpr)
        save(model, Path(args.out_root) / f"{tag}-central", args, "central", cal,
             {"val_auc": auc, "seconds": round(time.time() - t0)})
        report["methods"]["central"] = {"val_auc": auc, "val": cal["val"]}
        print(f"  val AUC {auc}, R {cal['val']['recall']:.3f} при FPR {cal['val']['fpr']:.3f}")
        del model

    # ── каждый банк сам по себе ──────────────────────────────────────
    if "local" in args.methods:
        for k, bank in enumerate(banks):
            print(f"\n[local_{k}] {total_epochs} эпох только на данных банка {k}")
            model, tok = fresh_model(args, device)
            enc = D.PrefixEncoder(tok, args.max_len)
            t0 = time.time()
            D.train_epochs(model, D.PrefixDataset(bank, enc), epochs=total_epochs,
                           seed=args.seed + k, **kw)
            auc, cal = val_eval(model, enc, val, device, args.eval_batch, args.max_fpr)
            save(model, Path(args.out_root) / f"{tag}-local{k}", args, f"local_{k}", cal,
                 {"val_auc": auc, "bank": k, "seconds": round(time.time() - t0)})
            report["methods"][f"local_{k}"] = {"val_auc": auc, "val": cal["val"]}
            print(f"  val AUC {auc}, R {cal['val']['recall']:.3f} при FPR {cal['val']['fpr']:.3f}")
            del model

    # ── FedAvg ───────────────────────────────────────────────────────
    if "fedavg" in args.methods:
        print(f"\n[fedavg] {args.rounds} раундов × {args.local_epochs} локальных эпох")
        model, tok = fresh_model(args, device)
        enc = D.PrefixEncoder(tok, args.max_len)
        bank_ds = [D.PrefixDataset(b, enc) for b in banks]
        sizes = [len(ds) for ds in bank_ds]
        global_state = D.trainable_state(model)
        n_params = sum(v.numel() for v in global_state.values())
        rounds, t0 = [], time.time()
        for r in range(args.rounds):
            states = []
            for k, ds in enumerate(bank_ds):
                D.load_trainable_state(model, global_state)          # банк получает общую модель
                D.train_epochs(model, ds, epochs=args.local_epochs,
                               seed=args.seed + 100 * r + k, **kw)   # учится у себя
                states.append(D.trainable_state(model))              # отдаёт только обновление
            global_state = {n: sum(s[n] * sz for s, sz in zip(states, sizes)) / sum(sizes)
                            for n in global_state}                    # сервер усредняет
            D.load_trainable_state(model, global_state)
            auc, cal = val_eval(model, enc, val, device, args.eval_batch, args.max_fpr)
            rounds.append({"round": r + 1, "val_auc": auc, "val": cal["val"]})
            print(f"  раунд {r + 1}: val AUC {auc}, R {cal['val']['recall']:.3f} "
                  f"при FPR {cal['val']['fpr']:.3f}")
        mb = n_params * 4 / 1e6
        comm = {"trainable_params": n_params, "mb_per_update": round(mb, 2),
                "total_mb_all_banks": round(mb * 2 * args.banks * args.rounds, 1)}
        save(model, Path(args.out_root) / f"{tag}-fedavg", args, "fedavg", cal,
             {"val_auc": auc, "rounds": rounds, "communication": comm,
              "seconds": round(time.time() - t0)})
        report["methods"]["fedavg"] = {"val_auc": auc, "val": cal["val"], "rounds": rounds,
                                       "communication": comm}

    out = ROOT / "results" / f"federated_{args.split}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nСохранено → {out}")


if __name__ == "__main__":
    main()
