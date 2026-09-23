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

v2 (topics.yaml) — тема звонка не несёт информации о классе:
  * диалог порождается «тема → класс → сценарий»: у каждой из 11 тем есть
    мошеннический и легитимный сценарий, тема выбирается одинаково для обоих
    классов;
  * приветствие (S0) и подтверждение личности (S1) — из ОБЩЕГО пула темы
    (объединение пулов обоих классов + нейтральные приветствия), ответы
    абонента на S0/S1 — тоже общие;
  * число реплик и вероятность ответа абонента на S0/S1 одинаковы для классов.
Классы расходятся со стадии S2 — там, где начинается манипуляция.

v3 (*_ext.yaml, fillers.yaml) — расширение фразобанка:
  * в каждом пуле 6–8 самостоятельных формулировок вместо 1–3, чтобы разбиение
    по шаблонам (make_template_holdout.py) делило пулы без пересечений;
  * часть казахских реплик — смешанная речь внутри одной фразы;
  * нейтральные реплики («Секунду, проверяю», «Алло, слышите?») вставляются
    после стадии с одной и той же вероятностью в обоих классах: в живом
    разговоре много реплик без информации о классе.

v3.1 — слова доверия и вежливости: в v3 «официальный», «спасибо», «хорошо»,
«всего доброго» звучали в репликах звонящего только у легитимных звонков, и
модель выучила «вежливый и официальный = не мошенник». Теперь мошенники тоже
ссылаются на «официальность», благодарят и прощаются (пул closing — после
целевого действия, как в жизни).

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

EARLY = ("S0", "S1")          # стадии с общими пулами темы
P_S1 = 0.7                    # вероятность стадии S1 — одинакова для обоих классов
P_VICTIM = 0.85               # вероятность ответа абонента — одинакова для обоих классов
P_FILLER = 0.12               # v3: нейтральная реплика после стадии — одинаково для обоих классов
P_CLOSE = 0.7                 # v3.1: мошенник вежливо прощается после целевого действия


# ─────────────────────────────────────────────────────────────────────────
def _yaml(name):
    return yaml.safe_load(open(TAX / name, encoding="utf-8"))


def merge_ext(base, ext):
    """v3: дописать реплики из *_ext.yaml в те же пулы базового фразобанка.
    Словари сливаются рекурсивно, списки реплик — конкатенацией без повторов."""
    if isinstance(base, dict) and isinstance(ext, dict):
        out = dict(base)
        for k, v in ext.items():
            out[k] = merge_ext(base[k], v) if k in base else v
        return out
    if isinstance(base, list) and isinstance(ext, list):
        return base + [x for x in ext if x not in base]
    return ext


def load_taxonomy(ext: bool = True) -> Dict:
    """ext=False — фразобанк v2 без расширения (для сравнения v2 и v3)."""
    tax = {
        "stages": _yaml("stages.yaml"),
        "scam": _yaml("scam_scenarios.yaml"),
        "benign": _yaml("benign_scenarios.yaml"),
        "slots": _yaml("slots.yaml"),
        "pb_scam": _yaml("phrasebank_scam.yaml"),
        "pb_benign": _yaml("phrasebank_benign.yaml"),
        "topics": _yaml("topics.yaml"),
        "fillers": {},
    }
    if ext:
        for key, name in (("pb_scam", "phrasebank_scam_ext.yaml"),
                          ("pb_benign", "phrasebank_benign_ext.yaml"),
                          ("topics", "topics_ext.yaml")):
            if (TAX / name).exists():
                tax[key] = merge_ext(tax[key], _yaml(name))
        if (TAX / "fillers.yaml").exists():
            tax["fillers"] = _yaml("fillers.yaml")
    return tax


def weighted_choice(rng: random.Random, items: List[Dict], key: str = "weight"):
    total = sum(i[key] for i in items)
    r = rng.uniform(0, total)
    acc = 0.0
    for i in items:
        acc += i[key]
        if r <= acc:
            return i
    return items[-1]


def merge_pools(*pools) -> Dict[str, List[str]]:
    """Объединить пулы {ru: [...], kk: [...]} без повторов, с сохранением порядка."""
    out: Dict[str, List[str]] = {}
    for pool in pools:
        for lang, items in (pool or {}).items():
            if not isinstance(items, list):
                continue
            dst = out.setdefault(lang, [])
            for x in items:
                if x not in dst:
                    dst.append(x)
    return out


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
        self.topics = tax["topics"]["topics"]
        self.topic_of = {sid: t for t, d in self.topics.items() for sid in d["scam"] + d["benign"]}
        missing = (set(self.scam_scenarios) | set(self.benign_scenarios)) - set(self.topic_of)
        if missing:
            raise ValueError(f"сценарии без темы в topics.yaml: {sorted(missing)}")
        self._cache: Dict = {}

    # ── общие пулы темы для S0/S1 ────────────────────────────────────
    def _pb(self, sid):
        return self.pb_scam.get(sid, {}) if sid.startswith("SC") else self.pb_benign.get(sid, {})

    def topic_caller_pool(self, topic: str, stage: str) -> Dict[str, List[str]]:
        key = ("caller", topic, stage)
        if key not in self._cache:
            t = self.topics[topic]
            pools = [self._pb(sid).get(stage, {}).get("caller") for sid in t["scam"] + t["benign"]]
            if stage == "S0":
                pools.append(t.get("shared_openings"))
            self._cache[key] = merge_pools(*pools)
        return self._cache[key]

    def topic_victim_pool(self, topic: str, stage: str, profile: str) -> Dict[str, List[str]]:
        key = ("victim", topic, stage, profile)
        if key not in self._cache:
            t = self.topics[topic]
            vr = self.tax["pb_scam"]["victim_responses"].get(stage, {})
            pools = [vr.get(profile) or vr.get("any")]
            pools += [self._pb(sid).get(stage, {}).get("victim") for sid in t["benign"]]
            self._cache[key] = merge_pools(*pools)
        return self._cache[key]

    def has_s1(self, topic: str) -> bool:
        return bool(self.topic_caller_pool(topic, "S1"))

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

    def _early_stage(self, turns, topic, stage, lang_profile, profile, ctx, intent):
        """S0/S1: реплики звонящего и ответ абонента — из общих пулов темы.
        Код одинаковый для обоих классов — в этом и смысл."""
        used: set = set()
        for _ in range(self.rng.choice([1, 1, 1, 2])):
            lang = self.pick_lang(lang_profile)
            raw = self.pick_utterance(self.topic_caller_pool(topic, stage), lang, used)
            if raw:
                turns.append(self._turn(len(turns), "caller", ctx.fill(raw), stage, lang, intent, raw))
        if self.rng.random() < P_VICTIM:
            lang = self.pick_lang(lang_profile)
            raw = self.pick_utterance(self.topic_victim_pool(topic, stage, profile), lang)
            if raw:
                turns.append(self._turn(len(turns), "victim", ctx.fill(raw), stage, lang,
                                        "ответ абонента", raw))

    def _maybe_filler(self, turns, lang_profile, ctx):
        """Нейтральная реплика — пул и вероятность одинаковы для обоих классов."""
        fl = self.tax.get("fillers") or {}
        if not fl or self.rng.random() >= P_FILLER:
            return
        speaker = "caller" if self.rng.random() < 0.6 else "victim"
        lang = self.pick_lang(lang_profile)
        raw = self.pick_utterance(fl.get(speaker), lang)
        if raw:
            t = self._turn(len(turns), speaker, ctx.fill(raw), turns[-1]["stage"] if turns else "S0",
                           lang, "нейтральная реплика", raw)
            t["filler"] = True
            turns.append(t)

    def _early_stages(self, topic: str) -> List[str]:
        return ["S0"] + (["S1"] if self.has_s1(topic) and self.rng.random() < P_S1 else [])

    # ── мошеннический диалог ─────────────────────────────────────────
    def gen_scam(self, scenario_id: str) -> Dict:
        sc = self.scam_scenarios[scenario_id]
        pb = self.pb_scam[scenario_id]
        topic = self.topic_of[scenario_id]
        profile = weighted_choice(self.rng, self.profiles)
        lang_profile = weighted_choice(self.rng, self.languages)
        ctx = SlotContext(self.rng, self.tax["slots"])

        stages = self._early_stages(topic) + [s for s in sc["stage_sequence"]
                                              if s not in EARLY and s in pb]

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
            if stage in EARLY:
                self._early_stage(turns, topic, stage, lang_profile["id"], profile["id"], ctx,
                                  self.stage_meta[stage]["name_ru"])
                self._maybe_filler(turns, lang_profile["id"], ctx)
                continue
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
            if self.rng.random() < P_VICTIM:
                vr = self.tax["pb_scam"]["victim_responses"].get(stage, {})
                pool = vr.get(profile["id"]) or vr.get("any")
                lang = self.pick_lang(lang_profile["id"])
                raw = self.pick_utterance(pool, lang)
                if raw:
                    turns.append(self._turn(len(turns), "victim", ctx.fill(raw),
                                            stage, lang, "ответ жертвы", raw))
            self._maybe_filler(turns, lang_profile["id"], ctx)

        # v3.1: вежливое завершение — только если разговор дошёл до целевого действия
        closing = self.tax["pb_scam"].get("closing")
        if closing and terminated_at is None and action_turn_idx >= 0 and self.rng.random() < P_CLOSE:
            lang = self.pick_lang(lang_profile["id"])
            raw = self.pick_utterance(closing.get("caller"), lang)
            if raw:
                turns.append(self._turn(len(turns), "caller", ctx.fill(raw), "S6", lang,
                                        "вежливое завершение", raw))
                if self.rng.random() < P_VICTIM:
                    lang = self.pick_lang(lang_profile["id"])
                    raw = self.pick_utterance(closing.get("victim"), lang)
                    if raw:
                        turns.append(self._turn(len(turns), "victim", ctx.fill(raw), "S6", lang,
                                                "ответ жертвы", raw))

        return {
            "label": "scam",
            "scenario_id": scenario_id,
            "scenario_code": sc["code"],
            "topic": topic,
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
        topic = self.topic_of[scenario_id]
        profile = weighted_choice(self.rng, self.profiles)
        lang_profile = weighted_choice(self.rng, self.languages)
        ctx = SlotContext(self.rng, self.tax["slots"])

        stages = self._early_stages(topic) + [s for s in ("S2", "SB") if s in pb]

        # Подмешиваем ли в этот диалог легитимную реплику с красным флагом.
        # Без них модель выучит словарь вместо контекста.
        inject_rate = self.tax["pb_benign"].get("flag_bearing_injection_rate", 0.5)
        inject = self.rng.random() < inject_rate
        injected = False

        turns = []

        for stage in stages:
            if stage in EARLY:
                self._early_stage(turns, topic, stage, lang_profile["id"], profile["id"], ctx,
                                  self.stage_meta[stage]["name_ru"])
                self._maybe_filler(turns, lang_profile["id"], ctx)
                continue
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

            if self.rng.random() < P_VICTIM:
                lang = self.pick_lang(lang_profile["id"])
                raw = self.pick_utterance(stage_pb.get("victim"), lang)
                if raw:
                    turns.append(self._turn(len(turns), "victim", ctx.fill(raw),
                                            stage, lang, "ответ абонента", raw))
            self._maybe_filler(turns, lang_profile["id"], ctx)

        return {
            "label": "benign",
            "scenario_id": scenario_id,
            "scenario_code": bn["code"],
            "topic": topic,
            "difficulty": bn["difficulty"],
            "victim_profile": profile["id"],
            "language_profile": lang_profile["id"],
            "has_flag_bearing_turn": injected,
            "terminated_early_at": None,
            "action_turn_idx": -1,
            "target_action": [],
            "turns": turns,
        }

    # ── выборка: тема → класс → сценарий ─────────────────────────────
    def sample(self, n: int, rng: random.Random, scam_share: float = 0.5) -> List[Dict]:
        """P(тема | класс) одинакова для обоих классов — тема не выдаёт класс."""
        topics = sorted(self.topics)
        n_scam = int(n * scam_share)
        out = []
        for i in range(n):
            label = "scam" if i < n_scam else "benign"
            topic = rng.choice(topics)
            sid = rng.choice(self.topics[topic][label])
            out.append(self.gen_scam(sid) if label == "scam" else self.gen_benign(sid))
        rng.shuffle(out)
        return out

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
    dialogues = gen.sample(args.n, rng, tax["benign"]["corpus_balance"]["scam_share"])

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
