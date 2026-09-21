"""
Қалқан — генератор корпуса диалогов.

Методика следует TeleAntiFraud-28k (arXiv:2503.24115): корпус строится
из документированных сценариев, а не из записей реальных разговоров, —
это снимает вопрос персональных данных и позволяет опубликовать датасет.

Отличие от простой шаблонной генерации:
  * профиль жертвы управляет сопротивлением и досрочным обрывом разговора,
    поэтому «длинный диалог» не становится тривиальным признаком мошенничества;
  * языковой профиль включает смешанную казахско-русскую речь;
  * красные флаги размечаются детектором ПО ТЕКСТУ реплики, а не приписываются
    по ярлыку сценария.

Выход: JSONL, одна строка — один диалог.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import redflags

ROOT = Path(__file__).resolve().parent.parent
TAX = ROOT / "data" / "taxonomy"


# ─────────────────────────────────────────────────────────────────────────
def load_taxonomy() -> Dict:
    return {
        "stages": yaml.safe_load(open(TAX / "stages.yaml")),
        "scam": yaml.safe_load(open(TAX / "scam_scenarios.yaml")),
        "benign": yaml.safe_load(open(TAX / "benign_scenarios.yaml")),
        "slots": yaml.safe_load(open(TAX / "slots.yaml")),
        "pb_scam": yaml.safe_load(open(TAX / "phrasebank_scam.yaml")),
        "pb_benign": yaml.safe_load(open(TAX / "phrasebank_benign.yaml")),
    }


def weighted_choice(rng: random.Random, items: List[Dict], key: str = "weight"):
    total = sum(i[key] for i in items)
    r = rng.uniform(0, total)
    acc = 0.0
    for i in items:
        acc += i[key]
        if r <= acc:
            return i
    return items[-1]


# ─────────────────────────────────────────────────────────────────────────
class SlotContext:
    """Слоты фиксируются один раз на диалог — иначе банк «меняется» по ходу разговора."""

    def __init__(self, rng: random.Random, slots: Dict):
        s = slots
        # Казахское имя чаще — это Казахстан, но и в русскоязычной речи оно уместно
        if rng.random() < 0.62:
            self.victim_name = rng.choice(s["victim_name_kk"])
        else:
            self.victim_name = rng.choice(s["victim_name_ru"])

        self.values = {
            "victim_name": self.victim_name,
            "patronymic": rng.choice(s["victim_patronymic"]),
            "agent_name": rng.choice(s["agent_name_kk"] + s["agent_name_ru"]),
            "bank": rng.choice(s["bank"]),
            "telecom": rng.choice(s["telecom"]),
            "gov_org": rng.choice(s["gov_org"]),
            "authority": rng.choice(s["authority"]),
            "pension_fund": rng.choice(s["pension_fund"]),
            "city": rng.choice(s["city_ru"]),
            "district": rng.choice(s["district"]),
            "amount_small": f"{rng.choice(s['amount_small']):,}".replace(",", " "),
            "amount_medium": f"{rng.choice(s['amount_medium']):,}".replace(",", " "),
            "amount_large": f"{rng.choice(s['amount_large']):,}".replace(",", " "),
            "loan_amount": f"{rng.choice(s['loan_amount']):,}".replace(",", " "),
            "card_last4": rng.choice(s["card_last4"]),
            "sms_code": rng.choice(s["sms_code"]),
            "case_number": rng.choice(s["case_number"]),
            "app_name": rng.choice(s["app_name"]),
            "time_pressure_window": rng.choice(s["time_pressure_window"]),
            "investment_asset": rng.choice(s["investment_asset"]),
            "return_promise": rng.choice(s["return_promise"]),
            "link": rng.choice([
                "kaspi-verify.kz/id", "halyk-confirm.com/app",
                "egov-check.net/loan", "bank-online.kz/cancel",
            ]),
        }

    def fill(self, template: str) -> str:
        out = template
        for k, v in self.values.items():
            out = out.replace("{" + k + "}", str(v))
        # незаполненные слоты не должны утекать в корпус
        out = re.sub(r"\{[a-z_]+\}", "", out)
        return re.sub(r"\s{2,}", " ", out).strip()


# ─────────────────────────────────────────────────────────────────────────
class DialogueGenerator:

    def __init__(self, tax: Dict, seed: int = 42):
        self.tax = tax
        self.rng = random.Random(seed)
        self.stage_meta = {s["id"]: s for s in tax["stages"]["stages"]}
        self.profiles = tax["slots"]["victim_profiles"]
        self.languages = tax["slots"]["language_profiles"]
        self.scam_scenarios = {s["id"]: s for s in tax["scam"]["scenarios"]}
        self.benign_scenarios = {b["id"]: b for b in tax["benign"]["benign_scenarios"]}
        self.pb_scam = tax["pb_scam"]["scenarios"]
        self.pb_benign = tax["pb_benign"]["scenarios"]

    # ── выбор языка реплики ──────────────────────────────────────────
    def pick_lang(self, profile_id: str) -> str:
        if profile_id == "ru":
            return "ru"
        if profile_id == "kk":
            return "kk"
        # code-switching: внутри одного диалога язык плавает
        return "kk" if self.rng.random() < 0.45 else "ru"

    def pick_utterance(self, pool: Optional[Dict], lang: str,
                       used: Optional[set] = None) -> Optional[str]:
        """
        Взять реплику нужного языка; при отсутствии — откатиться на другой.
        `used` не даёт повторить одну и ту же реплику подряд внутри стадии.
        """
        if not pool:
            return None
        candidates = None
        if lang in pool and pool[lang]:
            candidates = pool[lang]
        else:
            for alt in ("ru", "kk"):
                if alt in pool and pool[alt]:
                    candidates = pool[alt]
                    break
        if not candidates:
            return None
        if used is not None:
            fresh = [c for c in candidates if c not in used]
            if fresh:
                candidates = fresh
        choice = self.rng.choice(candidates)
        if used is not None:
            used.add(choice)
        return choice

    # ── мошеннический диалог ─────────────────────────────────────────
    def gen_scam(self, scenario_id: str) -> Dict:
        sc = self.scam_scenarios[scenario_id]
        pb = self.pb_scam[scenario_id]
        profile = weighted_choice(self.rng, self.profiles)
        lang_profile = weighted_choice(self.rng, self.languages)
        ctx = SlotContext(self.rng, self.tax["slots"])

        stages = [s for s in sc["stage_sequence"] if s in pb]

        # Сопротивление жертвы: часть диалогов обрывается до целевого действия.
        # Без этого длина диалога стала бы тривиальным признаком.
        terminated_at = None
        if self.rng.random() < profile["resistance"]:
            pre_action = [s for s in stages if s not in ("S5", "S6")]
            if pre_action:
                cut = self.rng.randint(1, len(pre_action))
                stages = pre_action[:cut]
                terminated_at = stages[-1]

        turns, action_turn_idx = [], -1

        for stage in stages:
            stage_pb = pb.get(stage, {})
            caller_pool = stage_pb.get("caller")
            n_caller = self.rng.choice([1, 1, 2] if stage in ("S2", "S3", "S5") else [1, 1, 1, 2])
            used: set = set()

            for _ in range(n_caller):
                lang = self.pick_lang(lang_profile["id"])
                raw = self.pick_utterance(caller_pool, lang, used)
                if not raw:
                    continue
                text = ctx.fill(raw)
                if stage == "S5" and action_turn_idx < 0:
                    action_turn_idx = len(turns)
                turns.append(self._turn(len(turns), "caller", text, stage, lang,
                                        stage_pb.get("intent", ""), raw))

            # ответ жертвы
            if self.rng.random() < 0.85:
                vr = self.tax["pb_scam"]["victim_responses"].get(stage, {})
                pool = vr.get(profile["id"]) or vr.get("any")
                lang = self.pick_lang(lang_profile["id"])
                raw = self.pick_utterance(pool, lang)
                if raw:
                    turns.append(self._turn(len(turns), "victim", ctx.fill(raw),
                                            stage, lang, "ответ жертвы", raw))

        return {
            "label": "scam",
            "scenario_id": scenario_id,
            "scenario_code": sc["code"],
            "victim_profile": profile["id"],
            "language_profile": lang_profile["id"],
            "terminated_early_at": terminated_at,
            "action_turn_idx": action_turn_idx,
            "target_action": sc.get("target_action", []),
            "turns": turns,
        }

    # ── легитимный диалог ────────────────────────────────────────────
    def gen_benign(self, scenario_id: str) -> Dict:
        bn = self.benign_scenarios[scenario_id]
        pb = self.pb_benign[scenario_id]
        profile = weighted_choice(self.rng, self.profiles)
        lang_profile = weighted_choice(self.rng, self.languages)
        ctx = SlotContext(self.rng, self.tax["slots"])

        stages = [s for s in ("S0", "S1", "S2", "SB") if s in pb]
        if "S1" in stages and self.rng.random() < 0.4:
            stages.remove("S1")

        # Подмешиваем ли в этот диалог легитимную реплику с красным флагом.
        # Без них модель выучит словарь вместо контекста.
        inject_rate = self.tax["pb_benign"].get("flag_bearing_injection_rate", 0.5)
        inject = self.rng.random() < inject_rate
        injected = False

        turns = []

        for stage in stages:
            stage_pb = pb.get(stage, {})
            caller_pool = stage_pb.get("caller")
            fb_pool = stage_pb.get("caller_flag_bearing")
            used: set = set()

            for _ in range(self.rng.choice([1, 1, 2])):
                lang = self.pick_lang(lang_profile["id"])
                raw = self.pick_utterance(caller_pool, lang, used)
                if not raw:
                    continue
                turns.append(self._turn(len(turns), "caller", ctx.fill(raw),
                                        stage, lang, stage_pb.get("intent", ""), raw))

            # flag_bearing реплика идёт после основной — как естественное продолжение
            if inject and not injected and fb_pool:
                lang = self.pick_lang(lang_profile["id"])
                raw = self.pick_utterance(fb_pool, lang)
                if raw:
                    t = self._turn(len(turns), "caller", ctx.fill(raw),
                                   stage, lang, "легитимная реплика с поверхностным маркером", raw)
                    t["flag_bearing_benign"] = True
                    turns.append(t)
                    injected = True

            if self.rng.random() < 0.8:
                lang = self.pick_lang(lang_profile["id"])
                raw = self.pick_utterance(stage_pb.get("victim"), lang)
                if raw:
                    turns.append(self._turn(len(turns), "victim", ctx.fill(raw),
                                            stage, lang, "ответ абонента", raw))

        return {
            "label": "benign",
            "scenario_id": scenario_id,
            "scenario_code": bn["code"],
            "difficulty": bn["difficulty"],
            "victim_profile": profile["id"],
            "language_profile": lang_profile["id"],
            "has_flag_bearing_turn": injected,
            "terminated_early_at": None,
            "action_turn_idx": -1,
            "target_action": [],
            "turns": turns,
        }

    def _turn(self, idx: int, speaker: str, text: str, stage: str,
              lang: str, intent: str, template: Optional[str] = None) -> Dict:
        flags = redflags.detect(text)
        return {
            "idx": idx,
            "speaker": speaker,
            "text": text,
            "stage": stage,
            "lang": lang,
            "intent": intent,
            "red_flags": flags,
            "rule_risk": redflags.risk_from_flags(flags),
            # отпечаток исходного шаблона — нужен для разбиения,
            # при котором тестовые реплики модель не видела ни разу
            "template_id": hashlib.md5((template or text).encode("utf-8")).hexdigest()[:10],
        }


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="сколько диалогов сгенерировать")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "corpus.jsonl"))
    args = ap.parse_args()

    tax = load_taxonomy()
    gen = DialogueGenerator(tax, seed=args.seed)
    rng = random.Random(args.seed)

    balance = tax["benign"]["corpus_balance"]
    n_scam = int(args.n * balance["scam_share"])
    n_benign = args.n - n_scam

    scam_ids = list(gen.scam_scenarios)
    benign_ids = list(gen.benign_scenarios)
    hard_ids = [i for i in benign_ids
                if gen.benign_scenarios[i]["difficulty"] in ("hard", "very_hard")]
    easy_ids = [i for i in benign_ids if i not in hard_ids]
    hard_share = balance["hard_negative_share_within_benign"]

    dialogues = []
    for _ in range(n_scam):
        dialogues.append(gen.gen_scam(rng.choice(scam_ids)))
    for _ in range(n_benign):
        pool = hard_ids if rng.random() < hard_share else easy_ids
        dialogues.append(gen.gen_benign(rng.choice(pool)))

    rng.shuffle(dialogues)

    # разбиение по диалогам (не по репликам), чтобы не было утечки
    n = len(dialogues)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    for i, d in enumerate(dialogues):
        d["dialogue_id"] = f"QLK-{i:06d}"
        d["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Записано {len(dialogues)} диалогов → {out}")


if __name__ == "__main__":
    main()
