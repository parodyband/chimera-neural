"""Steady-vowel goal reaching through the real, non-differentiable simulator."""
from collections import deque
import json
from pathlib import Path
import subprocess

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_RATE = 48000
N_BANDS = 40
HISTORY = 4
HORIZON = 20


class Bridge:
    def __init__(self, binary=None):
        binary = Path(binary) if binary else ROOT / "target/release/chimera-learn"
        if not binary.is_file():
            raise FileNotFoundError(f"Build the renderer first: cargo build --release -p chimera-learn ({binary})")
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.schema = self.call(op="schema")
        if self.schema["version"] != 1 or self.schema["sample_rate"] != SAMPLE_RATE:
            self.close()
            raise RuntimeError("Unsupported renderer protocol")

    def call(self, **request):
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Renderer exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


class AcousticFeatures:
    """Level-normalized log-mel shape plus separate RMS, from 32 ms windows.

    This is a starting reward for stationary vowels, not a perceptual quality metric.
    No per-clip peak normalization: silence must remain distinguishable from speech.
    """
    def __init__(self):
        self.n_fft = 1536
        self.hop = 480
        self.window = np.hanning(self.n_fft)
        hz = np.fft.rfftfreq(self.n_fft, 1 / SAMPLE_RATE)
        mel_edges = np.linspace(2595 * np.log10(1 + 80 / 700),
                                2595 * np.log10(1 + 8000 / 700), N_BANDS + 2)
        edges = 700 * (10 ** (mel_edges / 2595) - 1)
        self.filters = np.maximum(0, np.minimum(
            (hz[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None],
            (edges[2:, None] - hz[None, :]) / np.diff(edges)[1:, None]))
        self.filters /= np.maximum(self.filters.sum(axis=1, keepdims=True), 1e-12)

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 1 or len(audio) < self.n_fft or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono audio of at least 32 ms")
        frames = np.lib.stride_tricks.sliding_window_view(audio, self.n_fft)[::self.hop]
        power = abs(np.fft.rfft(frames * self.window, axis=-1)) ** 2 / self.n_fft**2
        db = 10 * np.log10(np.maximum(power @ self.filters.T, 1e-12))
        shape = db.mean(axis=0)
        shape = (shape - shape.mean()) / 40.0
        rms_db = 20 * np.log10(max(float(np.sqrt(np.mean(audio**2))), 1e-6))
        return np.concatenate([shape, [rms_db / 60]]).astype(np.float32)


def acoustic_distance(achieved, desired):
    return float(np.mean(abs(achieved[:-1] - desired[:-1]))
                 + 0.25 * abs(achieved[-1] - desired[-1]))


class VowelEnv(gym.Env):
    """Goal-conditioned control: target postures are NEVER exposed to the policy.

    Fixed male body/source/f0 isolates articulation. Four frames provide approximate
    history, not the complete hidden acoustic state. Actions increment commanded
    targets by <=7.5% of each parameter range; the engine applies its own dynamics.
    """
    metadata = {"render_modes": []}

    def __init__(self, targets=None, f0=140.0, binary=None):
        super().__init__()
        if not np.isfinite(f0) or not 70 <= f0 <= 400:
            raise ValueError("f0 must be between 70 and 400 Hz")
        self.f0 = float(f0)
        self.features = AcousticFeatures()
        self.recordings = []
        if targets:
            files = sorted(Path(targets).glob("*.wav"))
            if not files:
                raise ValueError("Target directory must contain steady-vowel WAV files")
            for path in files:
                audio, fs = sf.read(path, always_2d=True)
                if fs != SAMPLE_RATE or audio.shape[1] != 1 or not 0.1 <= len(audio) / fs <= 5:
                    raise ValueError(f"{path}: use 48 kHz mono, trimmed 0.1–5 second steady vowels")
                if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 1e-5:
                    raise ValueError(f"{path}: silent or non-finite recording")
                self.recordings.append((self.features(audio[:, 0]), audio[:, 0].copy()))
        self.bridge = Bridge(binary)
        self.low = np.array(self.bridge.schema["low"])
        self.high = np.array(self.bridge.schema["high"])
        table = json.loads((ROOT / "crates/chimera/src/phonetics/vowelTargets.generated.json").read_text())["male"]
        # Non-rhotic vowels; the initial task doesn't control retroflexion.
        self.postures = np.array([[v["artic"][k] for k in self.bridge.schema["keys"]]
                                 for v in table.values() if v["artic"].get("ttCurl", 0) == 0])
        self.action_space = spaces.Box(-1, 1, (8,), np.float32)
        frame_size = 8 + 8 + N_BANDS + 1  # measured position, commanded target, acoustic feedback
        self.observation_space = spaces.Box(-np.inf, np.inf,
            (HISTORY * frame_size + N_BANDS + 2,), np.float32)
        self.history = deque(maxlen=HISTORY)
        self.goal = None
        self.steps = 0

    def _receive(self, response):
        self.audio = np.asarray(response["audio"], dtype=np.float32)
        self.achieved = self.features(self.audio)
        self.positions = np.asarray(response["positions"], dtype=np.float32)

    def _observation(self):
        frame = np.concatenate([self.positions, self.command, self.achieved])
        self.history.append(frame)
        while len(self.history) < HISTORY:
            self.history.append(frame.copy())
        return np.concatenate([*self.history, self.goal, [self.steps / HORIZON]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.history.clear()
        if self.recordings:
            features, audio = self.recordings[int(self.np_random.integers(len(self.recordings)))]
            self.goal = features.copy()
            self.target_audio = audio.copy()
        else:
            # Continuous goals from convex mixtures, not a tiny fixed lookup task.
            pair = self.postures[self.np_random.integers(len(self.postures), size=2)]
            mix = self.np_random.uniform()
            posture = mix * pair[0] + (1 - mix) * pair[1]
            target = np.clip(2 * (posture - self.low) / (self.high - self.low) - 1, -1, 1)
            response = self.bridge.call(op="reset", f0=self.f0, targets=target.tolist())
            self.target_audio = np.asarray(response["audio"], dtype=np.float32)
            self.goal = self.features(response["audio"])
        self.command = np.asarray(self.bridge.schema["neutral"], dtype=np.float32)
        self._receive(self.bridge.call(op="reset", f0=self.f0, targets=self.command.tolist()))
        return self._observation(), {"distance": acoustic_distance(self.achieved, self.goal)}

    def step(self, action):
        if self.goal is None or self.steps >= HORIZON:
            raise RuntimeError("reset before starting another episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Expected 8 finite actions")
        action = np.clip(action, -1, 1)
        old_command = self.command.copy()
        self.command = np.clip(self.command + 0.15 * action, -1, 1)
        self._receive(self.bridge.call(op="step", targets=self.command.tolist()))
        self.steps += 1
        distance = acoustic_distance(self.achieved, self.goal)
        movement = float(np.mean((self.command - old_command)**2))
        reward = -distance - 0.02 * movement
        return self._observation(), reward, False, self.steps >= HORIZON, {"distance": distance}

    def close(self):
        self.bridge.close()
