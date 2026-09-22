"""
Қалқан — живой режим: звук → реплики → текст → риск, в реальном времени.

Конвейер (всё локально, звук никуда не отправляется и не сохраняется):

    источник звука ─► VAD (Silero) ─► реплика ─► Whisper ─► текст ─► детектор ─► риск
    (микрофон или    режет поток по      закончилась   (faster-whisper)       (XLM-R + LoRA,
     файл «как по    паузам ≥ min_silence                                      весь разговор)
     линии»)

Реплика — отрезок речи между паузами. Кто говорит, детектору знать не нужно
(он обучен без ролей), поэтому оба голоса могут идти в один микрофон:
телефон на громкой связи, ноутбук слушает.

Задержка считается от момента, когда человек ДОГОВОРИЛ фразу, до появления
риска на экране: ожидание паузы + очередь + распознавание + детектор.

Без Streamlit: этот модуль используют и страница демо (live_page.py),
и замер задержки из консоли (live_bench.py).
"""

import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

SR = 16000
FRAME = 512                        # кадр Silero VAD при 16 кГц = 32 мс
FRAME_S = FRAME / SR

# Частые «галлюцинации» Whisper на тишине и шуме (обучен на субтитрах YouTube)
HALLUCINATIONS = re.compile(
    r"продолжение следует|субтитр|спасибо за просмотр|подписывайтесь|редактор|dimatorzok|"
    r"корректор|amara\.org|thanks for watching", re.I)


# ─────────────────────────────────────────────────────────────────────────
# Источники звука: отдают блоки float32 16 кГц моно
# ─────────────────────────────────────────────────────────────────────────
def list_mics() -> List[str]:
    """Аудиоустройства DirectShow (Windows) — через PyAV, без дополнительных пакетов."""
    import av
    import av.logging
    av.logging.set_level(av.logging.INFO)
    with av.logging.Capture() as logs:
        try:
            av.open("dummy", format="dshow", options={"list_devices": "true"})
        except Exception:
            pass
    names, last = [], None
    for _, _, msg in logs:
        m = re.match(r'\s*"(.+)"\s*$', msg)
        if m:
            last = m.group(1)
        elif last and "(audio" in msg:
            names.append(last)
    return names


class MicSource:
    realtime = True

    def __init__(self, device: str, buffer_ms: int = 40):
        self.device, self.buffer_ms = device, buffer_ms

    def blocks(self, stop: threading.Event):
        import av
        c = av.open(f"audio={self.device}", format="dshow",
                    options={"audio_buffer_size": str(self.buffer_ms)})
        try:
            res = av.AudioResampler(format="flt", layout="mono", rate=SR)
            for frame in c.decode(c.streams.audio[0]):
                if stop.is_set():
                    break
                for f in res.resample(frame):
                    yield f.to_ndarray().reshape(-1).astype(np.float32)
        finally:
            c.close()


def load_audio(path, sr: int = SR) -> np.ndarray:
    import av
    c = av.open(str(path))
    res = av.AudioResampler(format="flt", layout="mono", rate=sr)
    out = [f.to_ndarray().reshape(-1) for fr in c.decode(c.streams.audio[0]) for f in res.resample(fr)]
    c.close()
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


class FileSource:
    """Аудиофайл, проигрываемый «как по линии»: блоки приходят с реальной скоростью.
    speed > 1 — быстрее реального времени (для отладки; задержка тогда не показательна)."""

    def __init__(self, path, start_s: float = 0.0, max_s: Optional[float] = None,
                 speed: float = 1.0, block_ms: int = 40):
        self.path, self.speed, self.block = Path(path), speed, int(SR * block_ms / 1000)
        a = load_audio(path)
        a = a[int(start_s * SR):]
        self.audio = a[:int(max_s * SR)] if max_s else a
        self.realtime = speed <= 1.0

    def blocks(self, stop: threading.Event):
        t0 = time.perf_counter()
        for i in range(0, len(self.audio), self.block):
            if stop.is_set():
                break
            due = t0 + (i + self.block) / SR / self.speed
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            yield self.audio[i:i + self.block]
        # хвост тишины, чтобы последняя реплика закрылась по паузе
        for _ in range(int(1.5 * SR / self.block)):
            if stop.is_set():
                break
            time.sleep(self.block / SR / self.speed)
            yield np.zeros(self.block, np.float32)


# ─────────────────────────────────────────────────────────────────────────
# VAD: поток → реплики
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Utterance:
    audio: np.ndarray
    start_s: float                 # позиция в звуке сессии
    end_s: float
    speech_end_wall: float         # когда последний звук речи пришёл в систему
    endpoint_wall: float           # когда VAD решил, что реплика закончилась


class Segmenter:
    """Режет поток на реплики по паузам. Хранит звук только с начала текущей
    реплики (минус отступ), поэтому память и время не растут с длиной звонка."""

    def __init__(self, on_speech: float = 0.5, off_speech: float = 0.35, min_silence_ms: int = 600,
                 min_speech_ms: int = 350, max_utt_s: float = 12.0, pad_ms: int = 150):
        from faster_whisper.vad import get_vad_model
        self.vad = get_vad_model()
        self.on, self.off = on_speech, off_speech
        self.min_sil = int(min_silence_ms / 1000 / FRAME_S)
        self.min_speech = int(min_speech_ms / 1000 / FRAME_S)
        self.max_frames = int(max_utt_s / FRAME_S)
        self.pad = int(pad_ms / 1000 * SR)
        self.buf = np.zeros(0, np.float32)            # звук с абсолютного сэмпла buf_off
        self.buf_off = 0
        self.pending = np.zeros(0, np.float32)        # ещё не размеченный хвост (< кадра)
        self.arrival = []                             # (конец блока в сэмплах, время прихода)
        self.frame_i = 0
        self.in_speech, self.start_f, self.last_speech_f, self.silence = False, 0, 0, 0
        self.level_db = -120.0

    def _wall_at(self, sample: int) -> float:
        for end, wall in self.arrival:
            if end >= sample:
                return wall
        return self.arrival[-1][1] if self.arrival else time.time()

    def _emit(self, end_f: int, now: float) -> Optional[Utterance]:
        n_speech = end_f - self.start_f
        self.in_speech = False
        if n_speech < self.min_speech:
            return None
        a = max(self.buf_off, self.start_f * FRAME - self.pad)
        b = min(self.buf_off + len(self.buf), end_f * FRAME + self.pad)
        return Utterance(self.buf[a - self.buf_off:b - self.buf_off].copy(), a / SR, end_f * FRAME / SR,
                         self._wall_at(end_f * FRAME), now)

    def feed(self, block: np.ndarray, wall: Optional[float] = None) -> List[Utterance]:
        wall = time.time() if wall is None else wall
        self.buf = np.concatenate([self.buf, block])
        self.arrival.append((self.buf_off + len(self.buf), wall))
        self.pending = np.concatenate([self.pending, block])
        n = len(self.pending) // FRAME
        out = []
        if n:
            chunk, self.pending = self.pending[:n * FRAME], self.pending[n * FRAME:]
            self.level_db = float(20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-9))
            probs = np.asarray(self.vad(chunk)).reshape(-1)
            for p in probs:
                f = self.frame_i
                self.frame_i += 1
                if not self.in_speech:
                    if p >= self.on:
                        self.in_speech, self.start_f, self.last_speech_f, self.silence = True, f, f + 1, 0
                    continue
                if p >= self.off:
                    self.last_speech_f, self.silence = f + 1, 0
                else:
                    self.silence += 1
                if self.silence >= self.min_sil:
                    u = self._emit(self.last_speech_f, wall)
                    if u:
                        out.append(u)
                elif f + 1 - self.start_f >= self.max_frames:          # слишком длинная — режем
                    u = self._emit(f + 1, wall)
                    if u:
                        out.append(u)
                    self.in_speech, self.start_f, self.last_speech_f, self.silence = True, f + 1, f + 1, 0
        # выбросить звук, который уже не понадобится
        keep_from = (self.start_f * FRAME - self.pad) if self.in_speech else \
            (self.buf_off + len(self.buf) - self.pad - 4 * FRAME)
        cut = max(0, keep_from - self.buf_off)
        if cut:
            self.buf, self.buf_off = self.buf[cut:], self.buf_off + cut
            self.arrival = [x for x in self.arrival if x[0] >= self.buf_off] or self.arrival[-1:]
        return out


# ─────────────────────────────────────────────────────────────────────────
# Распознавание и детектор
# ─────────────────────────────────────────────────────────────────────────
class ASR:
    def __init__(self, model: str = "large-v3", compute_type: str = "int8", cpu_threads: int = 8,
                 beam_size: int = 1):
        from faster_whisper import WhisperModel
        self.name, self.beam = model, beam_size
        self.model = WhisperModel(model, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)

    def __call__(self, audio: np.ndarray, language: Optional[str] = None, prompt: Optional[str] = None):
        segs, info = self.model.transcribe(
            audio, language=language, beam_size=self.beam, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
            initial_prompt=prompt, temperature=0.0)
        parts = [s.text.strip() for s in segs if s.no_speech_prob < 0.7 and s.avg_logprob > -1.2]
        text = " ".join(p for p in parts if p and not HALLUCINATIONS.search(p)).strip()
        return text, info.language, float(info.language_probability)


@dataclass
class LiveTurn:
    idx: int
    text: str
    lang: str
    lang_prob: float
    start_s: float
    end_s: float
    p: float                         # сырая вероятность модели по префиксу
    risk: float                      # после накопителя
    rules: float
    flags: List[str] = field(default_factory=list)
    endpoint_wait_s: float = 0.0     # от конца речи до решения VAD «реплика закончилась»
    queue_s: float = 0.0
    asr_s: float = 0.0
    score_s: float = 0.0
    latency_s: float = 0.0           # от конца речи до риска на экране


class LiveSession:
    """Фоновая сессия: поток захвата+VAD и поток распознавания+детектора.
    Страница демо читает snapshot() раз в полсекунды."""

    def __init__(self, source, asr: ASR, score_fn: Callable[[List[dict]], float], accumulate: Callable,
                 rules=None, language: Optional[str] = None, segmenter: Optional[Segmenter] = None,
                 log_path: Optional[Path] = None):
        self.source, self.asr, self.score_fn, self.accumulate = source, asr, score_fn, accumulate
        self.rules, self.language = rules, language
        self.seg = segmenter or Segmenter()
        self.log_path = log_path
        self.turns: List[LiveTurn] = []
        self.p: List[float] = []
        self.q: "queue.Queue[Optional[Utterance]]" = queue.Queue()
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.status, self.error, self.busy = "запуск", None, False
        self.started = time.time()
        self.t_capture = threading.Thread(target=self._capture, daemon=True)
        self.t_work = threading.Thread(target=self._work, daemon=True)

    def start(self):
        self.t_capture.start()
        self.t_work.start()
        return self

    def stop(self):
        self.stop_ev.set()

    @property
    def running(self):
        return self.t_work.is_alive()

    def _capture(self):
        try:
            self.status = "слушаю"
            for block in self.source.blocks(self.stop_ev):
                for u in self.seg.feed(block, time.time()):
                    self.q.put(u)
            self.status = "звук закончился"
        except Exception as e:                       # noqa: BLE001 — показать на странице
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.q.put(None)

    def _work(self):
        while True:
            u = self.q.get()
            if u is None:
                break
            t0 = time.time()
            self.busy = True
            try:
                prev = " ".join(t.text for t in self.turns[-2:])[-200:] or None
                text, lang, lp = self.asr(u.audio, self.language, prev)
                t1 = time.time()
                if not text:
                    continue
                turns = [{"speaker": "caller", "text": t.text} for t in self.turns] + [{"speaker": "caller", "text": text}]
                p = self.score_fn(turns)
                t2 = time.time()
                with self.lock:
                    self.p.append(p)
                    risk = self.accumulate(self.p)[-1]
                    rules, flags = 0.0, []
                    if self.rules is not None:
                        rules = self.rules.score_dialogue(turns)[-1]
                        flags = self.rules_flags(text)
                    turn = LiveTurn(len(self.turns), text, lang, lp, u.start_s, u.end_s, p, risk, rules, flags,
                                    endpoint_wait_s=u.endpoint_wall - u.speech_end_wall,
                                    queue_s=t0 - u.endpoint_wall, asr_s=t1 - t0, score_s=t2 - t1,
                                    latency_s=t2 - u.speech_end_wall)
                    self.turns.append(turn)
                if self.log_path:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        import json
                        f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
            except Exception as e:                   # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.busy = False
        self.status = "остановлено" if self.stop_ev.is_set() else "готово"

    @staticmethod
    def rules_flags(text):
        import redflags
        return redflags.explain(redflags.detect(text))

    def snapshot(self):
        with self.lock:
            return {"turns": list(self.turns), "status": self.status, "error": self.error,
                    "busy": self.busy, "backlog": self.q.qsize(), "level_db": self.seg.level_db,
                    "in_speech": self.seg.in_speech, "elapsed": time.time() - self.started}


def latency_summary(turns: List[LiveTurn]) -> dict:
    if not turns:
        return {}
    lat = np.array([t.latency_s for t in turns])
    return {"n": len(turns), "median_s": float(np.median(lat)), "p90_s": float(np.percentile(lat, 90)),
            "max_s": float(lat.max()),
            "endpoint_s": float(np.median([t.endpoint_wait_s for t in turns])),
            "asr_s": float(np.median([t.asr_s for t in turns])),
            "score_s": float(np.median([t.score_s for t in turns])),
            "audio_s_per_turn": float(np.median([t.end_s - t.start_s for t in turns]))}
