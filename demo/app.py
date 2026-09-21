"""
Қалқан — демо инкрементального детектора мошеннических звонков.

Разговор вводится реплика за репликой; после каждой реплики модель
(XLM-RoBERTa + LoRA) пересчитывает риск по всему сказанному, рядом —
rule-baseline на красных флагах и объяснение через redflags.explain().

    streamlit run demo/app.py
"""

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # в Space detector.py лежит рядом
import detector as D  # noqa: E402
import redflags  # noqa: E402

MODEL_DIR = Path(os.environ.get("QALQAN_MODEL_DIR", ROOT / "models" / "xlmr-lora-s42"))
if not MODEL_DIR.exists() and (Path(__file__).parent / "model").exists():
    MODEL_DIR = Path(__file__).parent / "model"           # раскладка для Hugging Face Space
EXAMPLES = json.load(open(Path(__file__).parent / "examples.json", encoding="utf-8"))["examples"]
RULE_TH = 0.6

# цвета: первые два слота эталонной палитры (проверены валидатором), порог — приглушённый
C_MODEL, C_RULES, C_MUTED, C_TEXT2 = "#2a78d6", "#eb6834", "#8a8983", "#52514e"
SPEAKER = {"caller": ("Звонящий", "📞"), "victim": ("Абонент", "🙋")}

st.set_page_config(page_title="Қалқан — детектор мошеннических звонков", page_icon="🛡️", layout="wide")


# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Загружаю модель…")
def load():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    model, tok, meta = D.load_trained(MODEL_DIR)
    return model, D.PrefixEncoder(tok, meta["max_len"]), meta


@torch.inference_mode()
def score_last(model, enc, turns):
    """p_t для последней реплики: модель видит весь разговор до неё включительно."""
    ids = enc.prefixes(turns)[-1]
    x = torch.tensor([ids])
    logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits
    return float(torch.softmax(logits.float(), -1)[0, 1])


def state():
    ss = st.session_state
    ss.setdefault("turns", [])        # [{speaker, text}]
    ss.setdefault("p", [])            # сырые p_t модели
    ss.setdefault("example", None)    # индекс примера или None
    ss.setdefault("pos", 0)           # сколько реплик примера уже проиграно
    return ss


def add_turn(ss, speaker, text):
    model, enc, _ = load()
    ss.turns.append({"speaker": speaker, "text": text.strip()})
    ss.p.append(score_last(model, enc, ss.turns))


def reset(ss, example=None):
    ss.turns, ss.p, ss.example, ss.pos = [], [], example, 0


def plural(n, one, few, many):
    """1 реплику, 2 реплики, 5 реплик."""
    n10, n100 = n % 10, n % 100
    return one if n10 == 1 and n100 != 11 else few if 2 <= n10 <= 4 and not 12 <= n100 <= 14 else many


def play(ss, n=None):
    ex = EXAMPLES[ss.example]
    end = len(ex["turns"]) if n is None else min(len(ex["turns"]), ss.pos + n)
    while ss.pos < end:
        t = ex["turns"][ss.pos]
        add_turn(ss, t["speaker"], t["text"])
        ss.pos += 1


def apply_query(ss):
    """Ссылки для видео и проверки: ?example=1&play=all (или play=5)."""
    if ss.get("query_done"):
        return
    ss.query_done = True
    q = st.query_params
    if "example" in q:
        try:
            i = int(q["example"]) - 1
        except ValueError:
            return
        if 0 <= i < len(EXAMPLES):
            reset(ss, i)
            if "play" in q:
                play(ss, None if q["play"] == "all" else int(q["play"]))


# ─────────────────────────────────────────────────────────────────────────
def risk_chart(turns, model_r, rules_r, th, action_idx, show_rules):
    rows = []
    for i, t in enumerate(turns):
        snippet = (t["text"][:70] + "…") if len(t["text"]) > 70 else t["text"]
        rows.append({"Реплика": i + 1, "Риск": model_r[i], "Кривая": "Модель", "Текст": snippet})
        if show_rules:
            rows.append({"Реплика": i + 1, "Риск": rules_r[i], "Кривая": "Правила", "Текст": snippet})
    df = pd.DataFrame(rows)
    domain = ["Модель", "Правила"] if show_rules else ["Модель"]
    color = alt.Color("Кривая:N", scale=alt.Scale(domain=domain, range=[C_MODEL, C_RULES][:len(domain)]),
                      legend=alt.Legend(orient="top", title=None, labelColor=C_TEXT2))
    x = alt.X("Реплика:Q", axis=alt.Axis(tickMinStep=1, format="d", title="Реплика"),
              scale=alt.Scale(domain=[1, max(len(turns), 2)]))
    y = alt.Y("Риск:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", title="Риск"))
    tip = [alt.Tooltip("Кривая:N"), alt.Tooltip("Реплика:Q"), alt.Tooltip("Риск:Q", format=".0%"),
           alt.Tooltip("Текст:N")]
    base = alt.Chart(df)
    lines = base.mark_line(strokeWidth=2, interpolate="monotone").encode(x=x, y=y, color=color)
    pts = base.mark_circle(size=70, opacity=1).encode(x=x, y=y, color=color, tooltip=tip)
    hit = base.mark_circle(size=500, opacity=0).encode(x=x, y=y, tooltip=tip)   # цель наведения больше метки
    last = df[df["Реплика"] == df["Реплика"].max()]
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=12, color=C_TEXT2).encode(
        x=x, y=y, text="Кривая:N")
    layers = [lines, pts, hit, labels]
    rule_df = pd.DataFrame({"y": [th], "t": [f"порог модели {th:.0%}"]})
    layers.append(alt.Chart(rule_df).mark_rule(strokeDash=[4, 4], color=C_MUTED).encode(y="y:Q"))
    layers.append(alt.Chart(rule_df).mark_text(align="left", dy=-7, x=4, fontSize=11, color=C_TEXT2)
                  .encode(y="y:Q", text="t:N"))
    if action_idx is not None and 0 <= action_idx < len(turns):
        act = pd.DataFrame({"x": [action_idx + 1], "t": ["запрос кода / перевода"]})
        layers.append(alt.Chart(act).mark_rule(color=C_MUTED).encode(x="x:Q"))
        layers.append(alt.Chart(act).mark_text(align="right", dx=-5, y=10, fontSize=11, color=C_TEXT2)
                      .encode(x="x:Q", text="t:N"))
    return (alt.layer(*layers).properties(height=320)
            .configure_axis(gridColor="#e9e8e4", domainColor="#c9c8c2", tickColor="#c9c8c2",
                            labelColor=C_TEXT2, titleColor=C_TEXT2)
            .configure_view(strokeWidth=0))


# ─────────────────────────────────────────────────────────────────────────
def main():
    ss = state()
    model, enc, meta = load()
    acc = meta["accumulator"]
    apply_query(ss)

    # ── боковая панель ──
    with st.sidebar:
        st.header("Примеры")
        st.caption("Из отложенного набора: этих формулировок модель при обучении не видела.")
        titles = ["— свой разговор —"] + [e["title"] for e in EXAMPLES]
        cur = 0 if ss.example is None else ss.example + 1
        choice = st.selectbox("Диалог", range(len(titles)), index=cur, format_func=lambda i: titles[i])
        if choice != cur:
            reset(ss, None if choice == 0 else choice - 1)
            st.rerun()
        if ss.example is not None:
            ex = EXAMPLES[ss.example]
            left = len(ex["turns"]) - ss.pos
            c1, c2 = st.columns(2)
            if c1.button("▶ Следующая реплика", disabled=left == 0, width="stretch"):
                play(ss, 1)
                st.rerun()
            if c2.button("⏩ До конца", disabled=left == 0, width="stretch"):
                with st.spinner("Модель слушает…"):
                    play(ss)
                st.rerun()
            st.caption(f"Проиграно {ss.pos} из {len(ex['turns'])} реплик. "
                       f"Эталон: **{'мошенничество' if ex['label'] == 'scam' else 'легитимный звонок'}**")
        if st.button("↺ Начать заново", width="stretch"):
            reset(ss, ss.example)
            st.rerun()

        st.divider()
        th = st.slider("Порог тревоги модели", 0.30, 0.99, float(round(acc["threshold"], 2)), 0.01,
                       help="По умолчанию — порог, выбранный на валидации (FPR ≤ 5%).")
        show_rules = st.checkbox("Показывать правила (красные флаги)", value=True)
        with st.expander("Как это работает"):
            st.markdown(
                "После каждой реплики модель **XLM-RoBERTa + LoRA** оценивает вероятность "
                "мошенничества по всему сказанному. Она обучена на синтетическом корпусе "
                "Қалқан (11 тем, у каждой — мошеннический и легитимный сценарий), поэтому "
                "реагирует на **приём манипуляции**, а не на тему звонка.\n\n"
                "Правила — регулярные выражения по 20 красным флагам: то, что банк может "
                "собрать за день. Они срабатывают на ключевых словах — и поэтому поздно "
                "(на самой просьбе кода) или ложно (когда настоящая техподдержка говорит "
                "«не кладите трубку»).")
        st.caption(f"Модель: {MODEL_DIR.name} · накопитель {acc['mode']}")

    # ── заголовок ──
    st.title("🛡️ Қалқан")
    st.markdown("**Детектор мошеннических звонков, который предупреждает до того, как вы продиктуете код.** "
                "Казахский, русский и смешанная речь. · *Алаяқтық қоңырауларды нақты уақытта анықтау.*")

    col_chat, col_risk = st.columns([1.05, 1], gap="large")

    model_r = D.accumulate(ss.p, acc["mode"], acc["a"]) if ss.p else []
    rules_r = redflags.RuleBaseline(threshold=RULE_TH).score_dialogue(ss.turns) if ss.turns else []
    action_idx = None
    if ss.example is not None and EXAMPLES[ss.example]["action_turn_idx"] >= 0:
        action_idx = EXAMPLES[ss.example]["action_turn_idx"]

    # ── диалог ──
    with col_chat:
        st.subheader("Разговор")
        if not ss.turns:
            st.info("Выберите пример слева и нажимайте «Следующая реплика» — или введите свой "
                    "разговор ниже, реплику за репликой.")
        for i, t in enumerate(ss.turns):
            name, icon = SPEAKER[t["speaker"]]
            with st.chat_message(name, avatar=icon):
                st.markdown(t["text"])
                flags = redflags.explain(redflags.detect(t["text"]))
                note = f"риск модели после реплики: **{model_r[i]:.0%}**"
                if flags:
                    note += " · флаги: " + "; ".join(flags)
                st.caption(f"{i + 1}. {name} · {note}")
        speaker = st.radio("Кто говорит", ["caller", "victim"], horizontal=True,
                           format_func=lambda s: SPEAKER[s][0])

    text = st.chat_input("Следующая реплика разговора…")
    if text:
        if ss.example is not None:
            ss.example = None               # ручной ввод продолжает свой разговор
        add_turn(ss, speaker, text)
        st.rerun()

    # ── риск ──
    with col_risk:
        st.subheader("Риск по ходу разговора")
        if ss.turns:
            m_alert = D.first_alert(model_r, th)
            r_alert = D.first_alert(rules_r, RULE_TH)
            if m_alert >= 0:
                msg = (f"**Высокий риск мошенничества** — с реплики {m_alert + 1}. "
                       "Прервите разговор и перезвоните в организацию по официальному номеру. "
                       "Никому не называйте коды из SMS.")
                st.error(msg, icon="🚨")
                if action_idx is not None and m_alert < action_idx:
                    k = action_idx - m_alert
                    st.caption(f"Тревога прозвучала за {k} {plural(k, 'реплику', 'реплики', 'реплик')} "
                               "до запроса кода / перевода.")
            elif model_r[-1] >= 0.8 * th:
                # около 0.5 модель держится, пока «непонятно» (приветствие) — это не повод пугать
                st.warning("**Внимание:** разговор похож на мошеннический, но уверенности пока нет.",
                           icon="⚠️")
            else:
                st.success("**Признаков мошенничества не видно.**", icon="✅")

            st.altair_chart(risk_chart(ss.turns, model_r, rules_r, th, action_idx, show_rules),
                            width="stretch")

            c1, c2 = st.columns(2)
            c1.metric("Риск модели сейчас", f"{model_r[-1]:.0%}",
                      delta=f"{(model_r[-1] - model_r[-2]) * 100:+.0f} п.п." if len(model_r) > 1 else None,
                      delta_color="inverse")
            c2.metric("Правила", "тревога" if r_alert >= 0 else "молчат",
                      help=f"Порог правил {RULE_TH:.0%}" + (f", тревога с реплики {r_alert + 1}"
                                                            if r_alert >= 0 else ""))

            st.markdown("**Почему**")
            if len(ss.p) > 1:
                jumps = [ss.p[i] - ss.p[i - 1] for i in range(1, len(ss.p))]
                j = max(range(len(jumps)), key=lambda k: jumps[k]) + 1
                if jumps[j - 1] > 0.1:
                    st.markdown(f"- Модель: сильнее всего риск вырос на реплике **{j + 1}** "
                                f"(+{jumps[j - 1] * 100:.0f} п.п.): «{ss.turns[j]['text'][:90]}»")
            fl = [(i, redflags.explain(redflags.detect(t["text"]))) for i, t in enumerate(ss.turns)]
            fl = [(i, f) for i, f in fl if f]
            if fl:
                for i, f in fl:
                    st.markdown(f"- Красный флаг, реплика {i + 1}: {', '.join(f)}")
            else:
                st.markdown("- Красных флагов в тексте нет.")
            with st.expander("Таблица: риск по репликам"):
                st.dataframe(pd.DataFrame({
                    "Реплика": range(1, len(ss.turns) + 1),
                    "Кто": [SPEAKER[t["speaker"]][0] for t in ss.turns],
                    "Риск модели": [f"{r:.0%}" for r in model_r],
                    "Правила": [f"{r:.0%}" for r in rules_r],
                    "Текст": [t["text"] for t in ss.turns],
                }), hide_index=True, width="stretch")
        else:
            st.caption("Здесь появится шкала риска.")

    st.caption("Қалқан — исследовательский прототип (конкурс АБРК / КБТУ / NU, трек AI for Finance). "
               "Модель обучена только на синтетическом корпусе; на реальных звонках ROC AUC 0.71–0.75 "
               "против 0.50 у правил. Не является финансовой или юридической консультацией.")


main()
