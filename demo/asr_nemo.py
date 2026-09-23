"""
Қалқан — распознавание речи моделью NVIDIA FastConformer (казахский + русский).

Зачем. Whisper small на телефонном звуке ошибается, а казахского в его обучении
почти не было. `nvidia/stt_kk_ru_fastconformer_hybrid_large` обучена сразу на
двух наших языках, вдесятеро меньше Whisper large и работает на процессоре
быстро: это CTC-модель, декодировать нечего — один проход кодировщика.

Здесь ONNX-экспорт (OpenVoiceOS) и всё, что к нему нужно: мел-признаки ровно как
в NeMo и жадное CTC-декодирование. Ни NeMo, ни PyTorch не требуются.

Интерфейс совпадает с live.ASR, поэтому в живом режиме модели взаимозаменяемы.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np

REPO = "OpenVoiceOS/stt_kk_ru_fastconformer_hybrid_large_onnx"
SR = 16000
# Параметры препроцессора NeMo (AudioToMelSpectrogramPreprocessor)
N_FFT, WIN, HOP, N_MELS, PREEMPH = 512, 400, 160, 80, 0.97
LOG_GUARD = 2.0 ** -24


class NemoASR:
    """CTC-распознавание: казахский и русский одной моделью, без выбора языка."""

    def __init__(self, quant: str = "int8", threads: int = 4, model_dir: Optional[str] = None):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download

        d = Path(model_dir) if model_dir else Path(snapshot_download(
            REPO, allow_patterns=[f"*.{quant}.onnx" if quant else "model.onnx", "vocab.txt", "config.json"]))
        onnx = next(d.glob(f"*.{quant}.onnx")) if quant else d / "model.onnx"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx), sess_options=opts, providers=["CPUExecutionProvider"])
        self.name = f"nemo-kk-ru{'-' + quant if quant else ''}"
        self.vocab = {}
        for line in (d / "vocab.txt").read_text(encoding="utf-8").splitlines():
            tok, _, idx = line.rpartition(" ")
            self.vocab[int(idx)] = tok
        self.blank = max(self.vocab)                       # <blk> — последний
        # матрица мел-фильтров посчитана один раз (librosa.filters.mel, slaney) и лежит рядом,
        # чтобы не тащить librosa в зависимости демо
        self.mel = np.load(Path(__file__).with_name("mel_filters_80.npy"))
        self.window = np.hanning(WIN + 1)[:-1].astype(np.float32)

    # ── признаки: как в NeMo FilterbankFeatures ──────────────────────
    def features(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        x = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]])          # преэмфазис
        pad = N_FFT // 2
        x = np.pad(x, pad, mode="reflect")
        n = 1 + (len(x) - N_FFT) // HOP
        idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
        frames = x[idx + (N_FFT - WIN) // 2] * self.window
        spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=-1)) ** 2       # мощность
        m = np.log(self.mel @ spec.T + LOG_GUARD)                       # [80, T]
        m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-5)
        return m.astype(np.float32)

    # ── жадное CTC-декодирование ─────────────────────────────────────
    def decode(self, logprobs: np.ndarray) -> str:
        ids, prev, out = logprobs.argmax(-1), -1, []
        for i in ids:
            if i != prev and i != self.blank:
                out.append(self.vocab.get(int(i), ""))
            prev = i
        text = "".join(out).replace("▁", " ")
        return re.sub(r"\s+", " ", text).strip()

    def __call__(self, audio: np.ndarray, language: Optional[str] = None, prompt: Optional[str] = None):
        """language и prompt игнорируются: модель двуязычная и без подсказок."""
        f = self.features(audio)[None]
        lp = self.sess.run(None, {"audio_signal": f, "length": np.array([f.shape[-1]], np.int64)})[0][0]
        return self.decode(lp), "kk-ru", 1.0
