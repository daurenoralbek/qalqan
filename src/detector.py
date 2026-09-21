"""
Қалқан — инкрементальный детектор: XLM-RoBERTa + LoRA.

Постановка. После каждой реплики t модель видит префикс разговора — последние
реплики до t включительно (обрезка слева по бюджету токенов) — и выдаёт
    p_t = P(мошенничество | реплики 0..t).
Обучение: каждый префикс каждого диалога — отдельный пример с меткой диалога.
Ранние префиксы мошеннического и легитимного звонка неотличимы («Здравствуйте,
это банк»), и модель честно выучивает там p ≈ 0.5 — решение приходит по мере
накопления признаков, а не по словарю.

Накопительный риск r_t поверх p_t (как в RuleBaseline, но без ручных весов):
    raw       r_t = p_t
    ewma      r_t = a·p_t + (1−a)·r_{t−1}        сглаживает одиночные всплески
    maxdecay  r_t = max(p_t, a·r_{t−1})          «запоминает» тревогу
Тип накопителя, a и порог подбираются на валидации (calibrate).

Роли говорящих в модель НЕ подаются: во внешних данных (ASR реальных звонков)
их нет, а детектор должен на них переноситься.

Этот модуль общий для обучения (train_detector.py), оценки (eval_detector.py)
и федеративного эксперимента (шаг 3): trainable_state / load_trainable_state
отдают и принимают только LoRA-веса и голову классификатора.
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "FacebookAI/xlm-roberta-base"


def load_jsonl(path) -> List[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# ─────────────────────────────────────────────────────────────────────────
# Кодирование префиксов
# ─────────────────────────────────────────────────────────────────────────
class PrefixEncoder:
    """Превращает диалог в последовательность префиксов input_ids."""

    def __init__(self, tokenizer, max_len: int = 256):
        self.tok = tokenizer
        self.max_len = max_len
        self.cls, self.sep = tokenizer.cls_token_id, tokenizer.sep_token_id

    def turn_ids(self, turns: Sequence[dict]) -> List[List[int]]:
        texts = [t["text"] or "" for t in turns]
        return self.tok(texts, add_special_tokens=False)["input_ids"] if texts else []

    def prefixes(self, turns: Sequence[dict]) -> List[List[int]]:
        ids = self.turn_ids(turns)
        budget = self.max_len - 2                      # [CLS] ... [SEP]
        out = []
        for t in range(len(ids)):
            cur = ids[t][-budget:]                     # текущая реплика целиком или её хвост
            ctx = list(cur)
            for j in range(t - 1, -1, -1):             # добавляем предыдущие, пока влезают
                piece = ids[j] + [self.sep]
                if len(ctx) + len(piece) > budget:
                    break
                ctx = piece + ctx
            out.append([self.cls] + ctx + [self.sep])
        return out


class PrefixDataset(Dataset):
    """Все префиксы всех диалогов: (input_ids, label, dialogue_idx, turn_idx)."""

    def __init__(self, dialogues: Sequence[dict], encoder: PrefixEncoder):
        self.pad_id = encoder.tok.pad_token_id
        self.items = []
        for di, d in enumerate(dialogues):
            y = 1 if d["label"] == "scam" else 0
            for ti, ids in enumerate(encoder.prefixes(d["turns"])):
                self.items.append((ids, y, di, ti))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    L = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, *_rest) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return ids, att, y


def length_bucketed_batches(dataset: PrefixDataset, batch_size: int, rng: random.Random):
    """Батчи из примеров близкой длины (меньше паддинга — в разы быстрее на CPU),
    порядок батчей перемешан."""
    order = sorted(range(len(dataset)), key=lambda i: len(dataset.items[i][0]) + rng.random())
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


# ─────────────────────────────────────────────────────────────────────────
# Модель
# ─────────────────────────────────────────────────────────────────────────
def build_model(model_name: str = MODEL_NAME, lora_r: int = 16, lora_alpha: int = 32,
                lora_dropout: float = 0.1):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha,
                     lora_dropout=lora_dropout, target_modules=["query", "key", "value"])
    return get_peft_model(base, cfg), tok


def load_trained(model_dir, device="cpu"):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = json.load(open(Path(model_dir) / "qalqan_meta.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(meta["base_model"])
    base = AutoModelForSequenceClassification.from_pretrained(meta["base_model"], num_labels=2)
    model = PeftModel.from_pretrained(base, str(model_dir)).to(device).eval()
    return model, tok, meta


def trainable_state(model) -> Dict[str, torch.Tensor]:
    """Только то, что обучается: LoRA-адаптеры и голова. Это и есть «обновление»,
    которым банки обмениваются в федеративном обучении — не сырые разговоры."""
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def load_trainable_state(model, state: Dict[str, torch.Tensor]):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].device))


# ─────────────────────────────────────────────────────────────────────────
# Скоринг
# ─────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def score_dialogues(model, encoder: PrefixEncoder, dialogues: Sequence[dict],
                    batch_size: int = 32, device="cpu", progress: bool = False) -> List[List[float]]:
    """p_t для каждой реплики каждого диалога."""
    model.eval()
    flat = []                                       # (ids, dialogue, turn)
    for di, d in enumerate(dialogues):
        for ti, ids in enumerate(encoder.prefixes(d["turns"])):
            flat.append((ids, di, ti))
    order = sorted(range(len(flat)), key=lambda i: len(flat[i][0]))
    probs = [0.0] * len(flat)
    pad = encoder.tok.pad_token_id
    for bi in range(0, len(order), batch_size):
        idx = order[bi:bi + batch_size]
        ids, att, _ = collate([(flat[i][0], 0) for i in idx], pad)
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 1].tolist()
        for i, pi in zip(idx, p):
            probs[i] = pi
        if progress and (bi // batch_size) % 50 == 0:
            print(f"    скоринг {bi + len(idx)}/{len(flat)}", flush=True)
    curves = [[0.0] * len(d["turns"]) for d in dialogues]
    for (ids, di, ti), p in zip(flat, probs):
        curves[di][ti] = p
    return curves


# ─────────────────────────────────────────────────────────────────────────
# Накопительный риск и калибровка на валидации
# ─────────────────────────────────────────────────────────────────────────
def accumulate(p: Sequence[float], mode: str = "raw", a: float = 1.0) -> List[float]:
    r, out = None, []
    for x in p:
        if r is None or mode == "raw":
            r = x
        elif mode == "ewma":
            r = a * x + (1 - a) * r
        elif mode == "maxdecay":
            r = max(x, a * r)
        else:
            raise ValueError(mode)
        out.append(r)
    return out


def first_alert(curve: Sequence[float], th: float) -> int:
    return next((i for i, s in enumerate(curve) if s >= th), -1)


def operating_point(dialogues, curves, th):
    """Метрики одного порога — те же определения, что в run_baseline.py."""
    tp = fp = tn = fn = 0
    lead, at = [], []
    for d, c in zip(dialogues, curves):
        i = first_alert(c, th)
        if d["label"] == "scam":
            if i >= 0:
                tp += 1
                at.append(i)
                if d.get("action_turn_idx", -1) >= 0:
                    lead.append(d["action_turn_idx"] - i)
            else:
                fn += 1
        else:
            fp += i >= 0
            tn += i < 0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": round(th, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "fpr": round(fp / (fp + tn), 4) if fp + tn else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "median_alert_turn": _med(at), "median_lead_turns": _med(lead),
        "share_alerted_before_action": round(sum(l > 0 for l in lead) / len(lead), 4) if lead else None,
    }


def _med(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def threshold_for_fpr(dialogues, curves, max_fpr: float) -> float:
    """Минимальный порог, при котором FPR (по диалогам) не превышает max_fpr."""
    peaks = sorted((max(c) if c else 0.0) for d, c in zip(dialogues, curves) if d["label"] != "scam")
    if not peaks:
        return 0.5
    k = int(math.floor(max_fpr * len(peaks)))       # сколько легитимных можем «задеть»
    # порог чуть выше (k+1)-го по величине пика среди легитимных
    th = peaks[len(peaks) - 1 - k] + 1e-6 if k < len(peaks) else 0.0
    return min(th, 1.0)


ACC_GRID = [("raw", 1.0)] + [("ewma", a) for a in (0.3, 0.5, 0.7)] + \
           [("maxdecay", a) for a in (0.7, 0.85, 0.95)]


def calibrate(dialogues, raw_curves, max_fpr: float = 0.05) -> dict:
    """
    Подбор накопителя и порога на валидации: максимизируем долю мошеннических
    диалогов, где тревога прозвучала ДО запроса целевого действия, при FPR ≤ max_fpr;
    при равенстве — выше recall.
    """
    best = None
    for mode, a in ACC_GRID:
        curves = [accumulate(c, mode, a) for c in raw_curves]
        th = threshold_for_fpr(dialogues, curves, max_fpr)
        op = operating_point(dialogues, curves, th)
        n_act = sum(1 for d in dialogues if d["label"] == "scam" and d.get("action_turn_idx", -1) >= 0)
        early = sum(1 for d, c in zip(dialogues, curves)
                    if d["label"] == "scam" and d.get("action_turn_idx", -1) >= 0
                    and 0 <= first_alert(c, th) < d["action_turn_idx"])
        key = (early / n_act if n_act else 0.0, op["recall"])
        cand = {"mode": mode, "a": a, "threshold": th, "max_fpr": max_fpr,
                "val_early_share_of_actioned": round(key[0], 4), "val": op}
        if best is None or key > best[0]:
            best = (key, cand)
    return best[1]


# ─────────────────────────────────────────────────────────────────────────
# Обучение (общее для централизованного и федеративного режима)
# ─────────────────────────────────────────────────────────────────────────
def train_epochs(model, dataset: PrefixDataset, *, epochs: float, lr: float, batch_size: int,
                 device="cpu", seed: int = 42, warmup: float = 0.06, weight_decay: float = 0.01,
                 grad_clip: float = 1.0, amp: bool = False, log_every: int = 50,
                 max_steps: Optional[int] = None, log=print) -> dict:
    """Обучить trainable-параметры модели на префиксах. Оптимизатор — свежий на
    каждый вызов (в FedAvg так и нужно: локальное обучение каждого банка)."""
    from transformers import get_linear_schedule_with_warmup

    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    per_epoch = math.ceil(len(dataset) / batch_size)
    total = int(per_epoch * epochs) if max_steps is None else min(max_steps, int(per_epoch * epochs))
    sched = get_linear_schedule_with_warmup(opt, int(total * warmup), total)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    model.train()
    step, run_loss, t0 = 0, 0.0, time.time()
    while step < total:
        for batch_idx in length_bucketed_batches(dataset, batch_size, rng):
            if step >= total:
                break
            ids, att, y = collate([dataset.items[i] for i in batch_idx], dataset.pad_id)
            ids, att, y = ids.to(device), att.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = torch.nn.functional.cross_entropy(
                    model(input_ids=ids, attention_mask=att).logits.float(), y)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            if step % log_every == 0 or step == total:
                el = time.time() - t0
                log(f"    шаг {step}/{total}  loss {run_loss / log_every if step % log_every == 0 else run_loss / (step % log_every):.4f}"
                    f"  {el / step:.2f} с/шаг, осталось ~{el / step * (total - step) / 60:.0f} мин")
                run_loss = 0.0
    model.eval()
    return {"steps": step, "seconds": round(time.time() - t0, 1)}
