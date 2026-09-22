"""Общее для страниц демо: модель, скоринг реплики, график риска."""

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

_V3 = ROOT / "models" / "v3" / "xlmr-lora-s42"          # корпус v3 (расширенный фразобанк)
MODEL_DIR = Path(os.environ.get("QALQAN_MODEL_DIR", _V3 if _V3.exists() else ROOT / "models" / "xlmr-lora-s42"))
if not MODEL_DIR.exists() and (Path(__file__).parent / "model").exists():
    MODEL_DIR = Path(__file__).parent / "model"           # раскладка для Hugging Face Space
RULE_TH = 0.6

# цвета: первые два слота эталонной палитры (проверены валидатором), порог — приглушённый
C_MODEL, C_RULES, C_MUTED, C_TEXT2 = "#2a78d6", "#eb6834", "#8a8983", "#52514e"


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


def plural(n, one, few, many):
    """1 реплику, 2 реплики, 5 реплик."""
    n10, n100 = n % 10, n % 100
    return one if n10 == 1 and n100 != 11 else few if 2 <= n10 <= 4 and not 12 <= n100 <= 14 else many


def risk_chart(turns, model_r, rules_r, th, action_idx, show_rules, height=320):
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
    return (alt.layer(*layers).properties(height=height)
            .configure_axis(gridColor="#e9e8e4", domainColor="#c9c8c2", tickColor="#c9c8c2",
                            labelColor=C_TEXT2, titleColor=C_TEXT2)
            .configure_view(strokeWidth=0))
