"""Whisper word error rate for directories of WAVs against reference texts.

Runs in the ASR env (tools/asr/setup.sh): ~/.cache/chimera-asr/.venv/bin/python judge.py
  <texts.json> <out.json> name=dir [name=dir ...]
texts.json: [{"id", "text"}]; each dir holds <id>.wav (16-bit PCM, any rate). Same model and
greedy decoding as tools/asr/speech_wer.py, so numbers are comparable with that judge.
"""
import json
import os
import re
import sys
import wave

import numpy as np
from faster_whisper import WhisperModel


def words(text):
    text = re.sub(r"[^a-z' ]+", " ", text.lower().replace("’", "'"))
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edits(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


def load16k(path):
    with wave.open(path) as w:
        ch, rate = w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    x = x.reshape(-1, ch).mean(axis=1)
    n = int(round(len(x) * 16000 / rate))
    return (np.fft.irfft(np.fft.rfft(x)[: n // 2 + 1], n) * (n / len(x))).astype(np.float32)


def main():
    texts, out, *dirs = sys.argv[1:]
    rows = json.load(open(texts))
    model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=os.cpu_count(),
                         download_root=os.path.expanduser("~/.cache/chimera-asr/models"))
    report = {}
    for spec in dirs:
        name, d = spec.split("=", 1)
        errors = total = 0
        heard = {}
        for r in rows:
            path = os.path.join(d, f"{r['id']}.wav")
            if not os.path.exists(path):
                continue
            segs, _ = model.transcribe(load16k(path), language="en", beam_size=1, temperature=0.0,
                                       condition_on_previous_text=False, vad_filter=False)
            heard[r["id"]] = " ".join(s.text.strip() for s in segs)
            ref = words(r["text"])
            errors += edits(ref, words(heard[r["id"]]))
            total += len(ref)
        report[name] = {"wer": round(100 * errors / max(total, 1), 2), "words": total, "heard": heard}
        print(f"{name}: WER {report[name]['wer']}% over {total} words", flush=True)
    json.dump(report, open(out, "w"), indent=1)


main()
