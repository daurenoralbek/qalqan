"""Қалқан — страница «Живой звонок»: звук → текст → риск в реальном времени (см. live.py)."""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

import live as L
from common import D, MODEL_DIR, RULE_TH, load, redflags, risk_chart, score_last

ROOT = Path(__file__).resolve().parent.parent
LOGS = Path(__file__).parent / "live_logs"            # расшифровки — только локально (.gitignore)
AUDIO_DIRS = [Path(__file__).parent / "audio", ROOT / "data" / "external" / "raw" / "youtube_ru"]
LANGS = {"русский": "ru", "казахский": "kk", "авто — вдвое медленнее": None}
SILENT_DB = -85.0                                      # ниже — микрофон фактически молчит


@st.cache_resource(show_spinner="Загружаю распознавание речи (первый раз — до минуты)…")
def load_asr(name):
    return L.make_asr(ENGINES[name], threads=4)


@st.cache_data(ttl=120, show_spinner=False)
def mics():
    try:
        return L.list_mics()
    except Exception:                                  # noqa: BLE001
        return []


ENGINES = {}                                           # подпись в интерфейсе → имя движка


def cached_engines():
    """Только модели, уже лежащие в кэше: ничего не скачиваем без спроса."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    kk_turbo = ROOT / "models" / "asr" / "kazakh-turbo-ct2"
    known = [("models--OpenVoiceOS--stt_kk_ru_fastconformer_hybrid_large_onnx",
              "nemo", "NVIDIA kk+ru — быстрая, двуязычная"),
             ("models--Systran--faster-whisper-small", "small", "Whisper small"),
             ("models--Systran--faster-whisper-medium", "medium", "Whisper medium"),
             ("models--mobiuslabsgmbh--faster-whisper-large-v3-turbo", "large-v3-turbo",
              "Whisper large-v3-turbo — точная, но медленная на CPU"),
             ("models--Systran--faster-whisper-large-v3", "large-v3",
              "Whisper large-v3 — самая точная, для CPU слишком медленная")]
    ENGINES.clear()
    for d, engine, label in known:
        if (hub / d).exists():
            ENGINES[label] = engine
    if kk_turbo.exists():                              # локальная, вне git
        ENGINES["Казахская Whisper turbo — точная, но 22 с на фразу"] = str(kk_turbo)
    if not ENGINES:
        ENGINES["Whisper small"] = "small"
    return list(ENGINES)


def audio_files():
    exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    return [p for d in AUDIO_DIRS if d.exists() for p in sorted(d.iterdir()) if p.suffix.lower() in exts]


def start_session(source, asr_name, language, min_silence_ms):
    ss = st.session_state
    if ss.get("live"):
        ss.live.stop()
    model, enc, meta = load()
    acc = meta["accumulator"]
    LOGS.mkdir(exist_ok=True)
    ss.live = L.LiveSession(
        source, load_asr(asr_name),
        score_fn=lambda turns: score_last(model, enc, turns),
        accumulate=lambda p: D.accumulate(p, acc["mode"], acc["a"]),
        rules=redflags.RuleBaseline(threshold=RULE_TH), language=language,
        segmenter=L.Segmenter(min_silence_ms=min_silence_ms),
        log_path=LOGS / time.strftime("live_%Y%m%d_%H%M%S.jsonl")).start()
    ss.live_source = source


def fmt_t(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


# ─────────────────────────────────────────────────────────────────────────
ss = st.session_state
model, enc, meta = load()
acc = meta["accumulator"]

with st.sidebar:
    st.header("Живой звонок")
    kind = st.radio("Источник звука", ["Микрофон", "Аудиофайл (как по линии)"],
                    help="Микрофон: телефон на громкой связи рядом с ноутбуком. "
                         "Аудиофайл проигрывается с реальной скоростью — как будто звонок идёт сейчас.")
    source = None
    if kind == "Микрофон":
        devs = mics()
        if devs:
            dev = st.selectbox("Микрофон", devs)
            source = L.MicSource(dev)
        else:
            st.error("Микрофоны не найдены.")
    else:
        files = audio_files()
        if files:
            f = st.selectbox("Файл", files, format_func=lambda p: p.name,
                             help=f"Файлы из {AUDIO_DIRS[0].relative_to(ROOT)} (свои записи) "
                                  "и из внешнего теста, если он скачан.")
            start_s = st.number_input("Начать с секунды", 0, 3600, 0, 5)
            source = ("file", f, start_s)
        else:
            st.info(f"Положите аудиофайл в {AUDIO_DIRS[0].relative_to(ROOT)}.")
    asr_name = st.selectbox("Распознавание речи", cached_engines(),
                            help="NVIDIA kk+ru обучена сразу на казахском и русском, язык выбирать не нужно "
                                 "и работает быстрее всех (~0,3 с на фразу). Модели Whisper на этом "
                                 "процессоре тратят от 2 с (small) до 20 с (large-v3) на фразу.")
    is_whisper = ENGINES.get(asr_name, "small") != "nemo"
    lang = LANGS[st.selectbox("Язык разговора", list(LANGS), disabled=not is_whisper,
                              help="Только для Whisper: автоопределение прогоняет распознавание дважды. "
                                   "Двуязычной модели язык указывать не нужно.")] if is_whisper else None
    min_sil = st.slider("Пауза, после которой реплика закончилась, мс", 300, 1200, 600, 50,
                        help="Меньше — быстрее тревога, но фразы чаще рвутся на части.")

    c1, c2 = st.columns(2)
    running = bool(ss.get("live")) and ss.live.running
    if c1.button("▶ Старт", disabled=source is None, type="primary", width="stretch"):
        if isinstance(source, tuple):
            with st.spinner("Читаю файл…"):
                source = L.FileSource(source[1], start_s=source[2])
        start_session(source, asr_name, lang, min_sil)
        st.rerun()
    if c2.button("■ Стоп", disabled=not running, width="stretch"):
        ss.live.stop()
        st.rerun()

    st.divider()
    th = st.slider("Порог тревоги модели", 0.30, 0.99, float(round(acc["threshold"], 2)), 0.01,
                   help="По умолчанию — порог, выбранный на валидации (FPR ≤ 5%).")
    show_rules = st.checkbox("Показывать правила (красные флаги)", value=True)
    st.caption(f"Модель: {MODEL_DIR.name} · распознавание: {asr_name}, локально. "
               "Звук не сохраняется; расшифровка пишется только в demo/live_logs.")

st.title("🎙️ Живой звонок")
st.markdown("Телефон на **громкой связи** рядом с ноутбуком — Қалқан слушает разговор, распознаёт речь "
            "и после каждой фразы пересчитывает риск. Кто говорит, модели знать не нужно.")


@st.fragment(run_every=1.0)
def live_panel():
    s = ss.get("live")
    if not s:
        st.info("Выберите источник звука слева и нажмите «▶ Старт». Для проверки без звонка — "
                "«Аудиофайл»: запись проиграется с реальной скоростью, как будто звонок идёт сейчас.")
        return
    snap = s.snapshot()
    turns = snap["turns"]

    # ── состояние ──
    if snap["error"]:
        st.error(f"Ошибка: {snap['error']}")
    st_line = f"**{snap['status']}** · {fmt_t(snap['elapsed'])}"
    if s.running:
        st_line += " · 🗣️ идёт речь" if snap["in_speech"] else " · тишина"
        if snap["busy"] or snap["backlog"]:
            st_line += f" · распознаю (в очереди {snap['backlog']})"
        st_line += f" · уровень {snap['level_db']:.0f} дБ"
    st.markdown(st_line)
    if (s.running and isinstance(ss.get("live_source"), L.MicSource) and snap["elapsed"] > 4
            and snap["level_db"] < SILENT_DB):
        st.warning("Микрофон отдаёт тишину. Проверьте, что он не выключен, и что Windows разрешает "
                   "доступ: Параметры → Конфиденциальность и защита → Микрофон → «Разрешить классическим "
                   "приложениям доступ к микрофону».")

    if not turns:
        st.caption("Реплики появятся здесь после первой паузы в речи.")
        return

    model_r = [t.risk for t in turns]
    rules_r = [t.rules for t in turns]
    alert = D.first_alert(model_r, th)
    r_alert = D.first_alert(rules_r, RULE_TH)

    left, right = st.columns([1.05, 1], gap="large")
    with right:
        if alert >= 0:
            st.error(f"**Высокий риск мошенничества** — с {fmt_t(turns[alert].end_s)} разговора "
                     f"(реплика {alert + 1}). Прервите разговор и перезвоните в организацию по "
                     "официальному номеру. Никому не называйте коды из SMS.", icon="🚨")
        elif model_r[-1] >= 0.8 * th:
            st.warning("**Внимание:** разговор похож на мошеннический, но уверенности пока нет.", icon="⚠️")
        else:
            st.success("**Признаков мошенничества не видно.**", icon="✅")
        st.altair_chart(risk_chart([{"text": t.text} for t in turns], model_r, rules_r, th, None,
                                   show_rules, height=280), width="stretch")
        lat = L.latency_summary(turns)
        c1, c2, c3 = st.columns(3)
        c1.metric("Риск сейчас", f"{model_r[-1]:.0%}")
        c2.metric("Задержка, медиана", f"{lat['median_s']:.1f} с",
                  help="От конца фразы до риска на экране: пауза + распознавание + детектор.")
        c3.metric("Правила", "тревога" if r_alert >= 0 else "молчат")
        st.caption(f"Из чего складывается задержка (медиана): пауза {lat['endpoint_s']:.1f} с · "
                   f"распознавание {lat['asr_s']:.1f} с · детектор {lat['score_s'] * 1000:.0f} мс · "
                   f"90% реплик — не дольше {lat['p90_s']:.1f} с.")

    with left:
        st.subheader("Разговор")
        for i, t in enumerate(turns):
            with st.chat_message("speech", avatar="🎧"):
                st.markdown(t.text)
                note = (f"{fmt_t(t.start_s)} · {t.lang} · риск **{t.risk:.0%}** · "
                        f"на экране через {t.latency_s:.1f} с")
                if t.flags:
                    note += " · флаги: " + "; ".join(t.flags)
                st.caption(f"{i + 1}. {note}")

    if not s.running:
        with st.expander("Таблица задержек"):
            st.dataframe(pd.DataFrame([{
                "Реплика": t.idx + 1, "Начало": fmt_t(t.start_s), "Длина, с": round(t.end_s - t.start_s, 1),
                "Пауза, с": round(t.endpoint_wait_s, 2), "Очередь, с": round(t.queue_s, 2),
                "Распознавание, с": round(t.asr_s, 2), "Детектор, с": round(t.score_s, 3),
                "Итого, с": round(t.latency_s, 2), "Риск": f"{t.risk:.0%}", "Текст": t.text}
                for t in turns]), hide_index=True, width="stretch")


live_panel()
