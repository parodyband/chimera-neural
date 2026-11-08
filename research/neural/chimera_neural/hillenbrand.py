"""Turn Hillenbrand et al. (1995) men's vowels into steady-vowel targets.

Source: https://github.com/santiagobarreda/hillenbrand_et_al_1995 (hosted with Jim
Hillenbrand's permission). Download it yourself into the ignored data directory; neither the
recordings nor anything derived from them goes in git.

Per token: a 150 ms window centred on the two judges' mean steady-state time, clipped to the
labelled vowel nucleus; band-limited x3 resampling (16 -> 48 kHz, done once here, so the env's
loader can keep rejecting other rates); RMS set to the synthesiser's typical vowel level
(recording gain is arbitrary, and the policy has no gain control); f0 from vowdata.dat.
Dropped: /er/ (the policy can't curl the tongue tip), tokens listeners misidentified,
and tokens with an unmeasured or out-of-range f0. Talkers, never tokens, are split:
talker rank % 10 in {0, 1} -> test, 2 -> validation, else train.
"""
import json
from pathlib import Path
import zipfile
import io

import numpy as np
import soundfile as sf

WINDOW_MS = 150
TARGET_RMS_DB = -17.0  # median steady synthetic vowel level measured through VowelEnv


def upsample3(audio):
    """Exact band-limited x3 interpolation by zero-padding the spectrum."""
    n = len(audio)
    spectrum = np.fft.rfft(audio)
    padded = np.zeros(3 * n // 2 + 1, dtype=complex)
    padded[:len(spectrum)] = spectrum
    if n % 2 == 0:
        padded[n // 2] *= 0.5  # split the old Nyquist bin between +/- frequencies
    return np.fft.irfft(padded, 3 * n) * 3


def split_for(rank):
    return "test" if rank % 10 in (0, 1) else "val" if rank % 10 == 2 else "train"


def prepare(raw, out, group="m"):
    raw, out = Path(raw), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; use a fresh directory")
    f0 = {}
    for line in (raw / "vowdata.dat").read_text(errors="replace").splitlines():
        parts = line.split()
        if parts and len(parts[0]) == 5 and parts[0][0] == group and parts[1].isdigit():
            f0[parts[0]] = float(parts[2])
    times = {}
    for line in (raw / "timedata.dat").read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 5 and len(parts[0]) == 5 and parts[0][0] == group:
            times[parts[0]] = [float(p) for p in parts[1:]]
    misid = {line.split()[0].split(".")[0] for line in (raw / "misid.dat").read_text().splitlines() if line.strip()}
    archive = zipfile.ZipFile(raw / {"m": "men.zip", "w": "women.zip"}[group])
    talkers = sorted({name[:3] for name in f0})
    manifest, dropped = {}, {"er": 0, "misidentified": 0, "f0": 0, "missing": 0}
    for split in ("train", "val", "test"):
        (out / split).mkdir(parents=True)
    for member in sorted(archive.namelist()):
        token = Path(member).stem.lower()
        if not member.lower().endswith(".wav"):
            continue
        if token[3:] == "er":
            dropped["er"] += 1
            continue
        if token in misid:
            dropped["misidentified"] += 1
            continue
        if token not in f0 or token not in times:
            dropped["missing"] += 1
            continue
        if not 70 <= f0[token] <= 400:
            dropped["f0"] += 1
            continue
        audio, fs = sf.read(io.BytesIO(archive.read(member)), always_2d=True)
        if fs != 16000:
            raise ValueError(f"{member}: expected 16 kHz, got {fs}")
        start, end, c1, c2 = times[token]
        centre = (c1 + c2) / 2 if c1 > 0 and c2 > 0 else (start + end) / 2
        lo = max(start, min(centre - WINDOW_MS / 2, end - WINDOW_MS))
        hi = min(end, lo + WINDOW_MS)
        clip = audio[int(lo * 16):int(hi * 16), 0]
        clip = upsample3(clip - clip.mean())
        clip *= 10 ** (TARGET_RMS_DB / 20) / max(np.sqrt(np.mean(clip**2)), 1e-9)
        split = split_for(talkers.index(token[:3]))
        sf.write(out / split / f"{token}.wav", clip.astype(np.float32), 48000, subtype="FLOAT")
        manifest.setdefault(split, {})[f"{token}.wav"] = f0[token]
    for split, entries in manifest.items():
        (out / split / "f0.json").write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    summary = {"source": "Hillenbrand et al. 1995 via santiagobarreda/hillenbrand_et_al_1995",
               "group": group, "window_ms": WINDOW_MS, "rms_db": TARGET_RMS_DB,
               "counts": {s: len(e) for s, e in manifest.items()}, "dropped": dropped,
               "talkers": {s: sorted({t[:3] for t in e}) for s, e in manifest.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
