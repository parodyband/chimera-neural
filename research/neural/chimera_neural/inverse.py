"""Audio -> control-track inverse model, trained on the synthesiser's own labelled speech.

Data (`chimera-learn dataset`): per utterance a 48 kHz WAV and a `.ctl` control track
(FRAME floats per 6.67 ms frame: value, weight and speed of each control key, then a
"score driving" flag). Replaying a track with `chimera-learn replay` re-speaks it.

Input features are one WavLM Base+ layer (50 Hz), chosen because self-supervised speech
features carry phonetic content and are fairly insensitive to channel and speaker. That is
what should let a model trained only on synth speech say something sensible about real
speech. Training audio is augmented (tilt, noise, reverb, band limit) before WavLM for the
same reason. Each WavLM frame predicts three control frames.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn

from .env import ROOT

N_CONTROL = 22
FRAME = 3 * N_CONTROL + 1
UP = 3  # control frames per WavLM frame (20 ms / 6.67 ms)
LAYER = 7
WAVLM = "microsoft/wavlm-base-plus"
KEYS = ["jaw", "tbX", "tbY", "ttX", "ttY", "ttCurl", "root", "lipAp", "lipProt", "lipWidth",
        "labiodental", "velum", "larynx", "epilarynx", "ventricular", "lateral", "dorsum",
        "abduction", "closure", "breath", "f0", "vibrato"]
F0 = KEYS.index("f0")


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def to16k(x, fs):
    if fs == 16000:
        return x.astype(np.float32)
    if fs != 48000:
        raise ValueError(f"expected 16 or 48 kHz, got {fs}")
    n = len(x) // 3
    return (np.fft.irfft(np.fft.rfft(x[:3 * n])[:n // 2 + 1], n) / 3).astype(np.float32)


def augment(x, rng):
    """Channel and room variety for 16 kHz audio; keeps the words."""
    spec = np.fft.rfft(x)
    hz = np.fft.rfftfreq(len(x), 1 / 16000)
    tilt = rng.uniform(-6, 6) * np.log2(np.maximum(hz, 100) / 1000)  # dB per octave around 1 kHz
    cut = rng.uniform(3800, 8000)
    shape = 10 ** (tilt / 20) / np.sqrt(1 + (hz / cut) ** 8)
    x = np.fft.irfft(spec * shape, len(x))
    if rng.uniform() < 0.5:
        rt60 = rng.uniform(0.1, 0.5)
        t = np.arange(int(0.6 * 16000)) / 16000
        ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
        ir[0] = 1 / rng.uniform(0.15, 0.5)  # direct-to-reverberant ratio
        n = len(x) + len(ir)
        x = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[:len(x)]
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    x = x / rms * 0.05 * 10 ** (rng.uniform(-10, 10) / 20)
    x += rng.standard_normal(len(x)) * 0.05 * 10 ** (-rng.uniform(20, 45) / 20)
    return x.astype(np.float32)


class Features:
    def __init__(self):
        from transformers import WavLMModel
        self.model = WavLMModel.from_pretrained(WAVLM).to(device()).eval()

    @torch.no_grad()
    def __call__(self, audio16k):
        x = torch.from_numpy(audio16k).to(device())[None]
        x = (x - x.mean()) / (x.std() + 1e-7)
        h = self.model(x, output_hidden_states=True).hidden_states[LAYER]
        return h[0].float().cpu().numpy().astype(np.float16)


def read_ctl(path):
    return np.fromfile(path, dtype="<f4").reshape(-1, FRAME)


def extract(split, augment_seed=None):
    """Concatenate WavLM features and aligned controls for a dataset directory."""
    split = Path(split)
    meta = [json.loads(line) for line in (split / "meta.jsonl").read_text().splitlines()]
    feats = Features()
    xs, ys, offsets = [], [], [0]
    rng = np.random.default_rng(augment_seed) if augment_seed is not None else None
    for i, row in enumerate(meta):
        audio, fs = sf.read(split / f"{row['id']}.wav", dtype="float32")
        x = to16k(audio, fs)
        if rng is not None:
            x = augment(x, rng)
        h = feats(x)
        c = read_ctl(split / f"{row['id']}.ctl")
        n = min(len(h), len(c) // UP)
        xs.append(h[:n])
        ys.append(c[:n * UP].reshape(n, UP * FRAME))
        offsets.append(offsets[-1] + n)
        if i % 500 == 0:
            print(f"{split.name}: {i}/{len(meta)}", flush=True)
    tag = "aug" if rng is not None else "clean"
    np.save(split / f"wavlm-{tag}.npy", np.concatenate(xs))
    np.save(split / "ctl.npy", np.concatenate(ys).astype(np.float32))
    np.save(split / "offsets.npy", np.array(offsets))


class Normalizer:
    """Per-key value standardisation (f0 on a log scale) and output decoding."""
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    @staticmethod
    def fit(ctl):
        c = ctl.reshape(-1, FRAME)
        values, weights = c[:, :N_CONTROL].copy(), c[:, N_CONTROL:2 * N_CONTROL]
        values[:, F0] = np.log(np.maximum(values[:, F0], 1))
        mean, std = np.zeros(N_CONTROL), np.ones(N_CONTROL)
        for k in range(N_CONTROL):
            active = values[weights[:, k] > 0.02, k]
            if len(active) > 100:
                mean[k], std[k] = active.mean(), max(active.std(), 1e-3)
        return Normalizer(mean, std)

    def targets(self, c):
        """(..., FRAME) controls -> normalised value, weight, log speed, flag."""
        v = c[..., :N_CONTROL].clone()
        v[..., F0] = torch.log(torch.clamp(v[..., F0], min=1))
        v = (v - torch.as_tensor(self.mean, device=c.device)) / torch.as_tensor(self.std, device=c.device)
        w = c[..., N_CONTROL:2 * N_CONTROL]
        s = torch.log(torch.clamp(c[..., 2 * N_CONTROL:3 * N_CONTROL], min=1e-3))
        return v, w, s, c[..., -1]

    def decode(self, out):
        """Network output (frames, FRAME) -> control track the engine replays."""
        v = out[:, :N_CONTROL] * self.std + self.mean
        v[:, F0] = np.exp(v[:, F0])
        w = np.expm1(np.logaddexp(0, out[:, N_CONTROL:2 * N_CONTROL]))  # inverse of log1p, via softplus
        s = np.exp(out[:, 2 * N_CONTROL:3 * N_CONTROL])
        flag = (out[:, -1] > 0).astype(np.float32)
        return np.concatenate([v, w, s, flag[:, None]], axis=1).astype(np.float32)


class InverseNet(nn.Module):
    def __init__(self, width=384, layers=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(768, width), nn.GELU(), nn.LayerNorm(width))
        self.conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(), nn.GroupNorm(1, width)) for _ in range(2)])
        self.rnn = nn.GRU(width, width // 2, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(width, UP * FRAME)

    def forward(self, x):  # (B, T, 768) -> (B, T*UP, FRAME)
        h = self.inp(x).transpose(1, 2)
        for c in self.conv:
            h = h + c(h)
        h, _ = self.rnn(h.transpose(1, 2))
        y = self.out(h)
        return y.reshape(y.shape[0], -1, FRAME)


def loss_terms(out, c, norm):
    v, w, s, flag = norm.targets(c)
    mask = torch.clamp(w, 0, 1)  # a key's value only matters while a gesture drives it
    value = (abs(out[..., :N_CONTROL] - v) * mask).sum() / mask.sum().clamp(min=1)
    # Weights span ~1 (vowels) to 40-80 (consonant closures): compare log(1 + w).
    pw = nn.functional.softplus(out[..., N_CONTROL:2 * N_CONTROL])
    weight = abs(pw - torch.log1p(w)).mean()
    speed = abs(out[..., 2 * N_CONTROL:3 * N_CONTROL] - s).mean()
    gate = nn.functional.binary_cross_entropy_with_logits(out[..., -1], flag)
    return value, weight, speed, gate


class Shards:
    """Random fixed-length crops across every train shard's features."""
    def __init__(self, dirs, tag, crop):
        self.parts = []
        for d in dirs:
            x = np.load(d / f"wavlm-{tag}.npy", mmap_mode="r")
            y = np.load(d / "ctl.npy", mmap_mode="r")
            off = np.load(d / "offsets.npy")
            self.parts += [(x, y, a, b) for a, b in zip(off[:-1], off[1:]) if b - a >= 8]
        self.crop = crop

    def batch(self, rng, size):
        xs, ys = [], []
        for _ in range(size):
            x, y, a, b = self.parts[rng.integers(len(self.parts))]
            n = min(self.crop, b - a)
            s = a + rng.integers(b - a - n + 1)
            xp = np.zeros((self.crop, x.shape[1]), np.float32)
            yp = np.zeros((self.crop, y.shape[1]), np.float32)
            xp[:n], yp[:n] = x[s:s + n], y[s:s + n]
            if n < self.crop:  # pad as silence at rest: zero weights, unit speed, no score
                yp[n:] = np.tile(np.r_[np.zeros(2 * N_CONTROL), np.ones(N_CONTROL), 0], UP)
            xs.append(xp)
            ys.append(yp)
        return np.stack(xs), np.stack(ys).reshape(size, -1, FRAME)


def evaluate_split(model, d, tag, norm, dev):
    x = np.load(d / f"wavlm-{tag}.npy")
    y = np.load(d / "ctl.npy")
    off = np.load(d / "offsets.npy")
    totals = np.zeros(4)
    model.eval()
    with torch.no_grad():
        for a, b in zip(off[:-1], off[1:]):
            xt = torch.from_numpy(x[a:b].astype(np.float32))[None].to(dev)
            yt = torch.from_numpy(y[a:b].reshape(1, -1, FRAME)).to(dev)
            totals += [float(t) for t in loss_terms(model(xt), yt, norm)]
    model.train()
    return totals / (len(off) - 1)


def train(args):
    data = Path(args.data)
    dirs = sorted(data.glob("train-*"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = device()
    norm = Normalizer.fit(np.concatenate([np.load(d / "ctl.npy", mmap_mode="r")[::50] for d in dirs]))
    np.savez(out / "norm.npz", mean=norm.mean, std=norm.std)
    sets = [Shards(dirs, "aug", args.crop), Shards(dirs, "clean", args.crop)]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = InverseNet(args.width, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.03)
    config = {k: v for k, v in vars(args).items() if k != "func"} | {
        "layer": LAYER, "wavlm": WAVLM, "train_utterances": sum(len(s.parts) for s in sets[:1]),
        "engine_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log = (out / "log.csv").open("w")
    log.write("step,value,weight,speed,gate,val_value,val_weight,val_speed,val_gate,elapsed\n")
    best, start, run = float("inf"), time.monotonic(), np.zeros(4)
    for step in range(1, args.steps + 1):
        x, y = sets[int(rng.uniform() < 0.25)].batch(rng, args.batch)  # 75% augmented
        terms = loss_terms(model(torch.from_numpy(x).to(dev)), torch.from_numpy(y).to(dev), norm)
        loss = terms[0] + terms[1] + 0.5 * terms[2] + 0.5 * terms[3]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += [float(t) for t in terms]
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate_split(model, data / "val", "clean", norm, dev)
            row = [step, *(run / args.eval_every), *val, time.monotonic() - start]
            log.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row) + "\n")
            log.flush()
            print(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row), flush=True)
            run[:] = 0
            score = val[0] + val[1] + 0.5 * val[2] + 0.5 * val[3]
            torch.save(model.state_dict(), out / "last.pt")
            if score < best:
                best = score
                torch.save(model.state_dict(), out / "best.pt")


def load(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    model = InverseNet(config["width"], config["layers"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    n = np.load(run / "norm.npz")
    return model.to(device()).eval(), Normalizer(n["mean"], n["std"])


def speak(args):
    """Any WAV (16 or 48 kHz) -> predicted control track -> the engine re-speaks it."""
    model, norm = load(args.run)
    feats = Features()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    binary = ROOT / "target/release/chimera-learn"
    for path in map(Path, args.wav):
        audio, fs = sf.read(path, dtype="float32", always_2d=True)
        h = feats(to16k(audio.mean(axis=1), fs))
        with torch.no_grad():
            y = model(torch.from_numpy(h.astype(np.float32))[None].to(device()))[0].cpu().numpy()
        ctl = norm.decode(y)
        ctl.astype("<f4").tofile(out / f"{path.stem}.ctl")
        subprocess.run([str(binary), "replay", str(out / f"{path.stem}.ctl"), str(out / f"{path.stem}.wav")], check=True)
        print(out / f"{path.stem}.wav")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("features")
    f.add_argument("split", type=Path)
    f.add_argument("--augment-seed", type=int)
    f.set_defaults(func=lambda a: extract(a.split, a.augment_seed))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--crop", type=int, default=200, help="WavLM frames (20 ms) per training crop")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--width", type=int, default=384)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=train)
    s = sub.add_parser("speak")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("wav", nargs="+")
    s.set_defaults(func=speak)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
