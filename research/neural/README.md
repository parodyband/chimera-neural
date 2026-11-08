# Learning to operate CHIMERA

Goal: learn transferable correlations between mouth movement and sound by trying actions
in the physical simulator. This is a neural reinforcement-learning experiment, not a
separate optimizer per recording. See [the research decision](../../docs/neural-learning.md).

## Setup

From the repository root (Rust, Python 3.11+, and Bun for the original app checks):

```sh
cargo build --release -p chimera-learn
python3 -m venv research/neural/.venv
research/neural/.venv/bin/python -m pip install -r research/neural/requirements-lock.txt
research/neural/.venv/bin/python -m pip install -e 'research/neural[test]'
research/neural/.venv/bin/python -m pytest research/neural/tests -q
research/neural/.venv/bin/chimera-neural smoke --out research/neural/runs/smoke
bun run test
bun run typecheck
```

The dependency snapshot was resolved on macOS arm64 / Python 3.14. Other platforms may
need to resolve from `pyproject.toml` instead. No GPU, external dataset, cloud credentials,
or pretrained network is needed. CPU is intentional for the small MLP. The bridge must
be rebuilt after Rust edits. Each checkout uses its own cargo target directory.

`smoke` runs the Gymnasium/SB3 checker, 64 simulator actions and 48 gradient updates,
then saves a checkpoint. It is only a setup check; it does not demonstrate learning.
Use a fresh output directory for each command; existing directories are never overwritten.

## First learning run (explicit budget)

```sh
research/neural/.venv/bin/chimera-neural train \
  --steps 20000 --seed 0 --out research/neural/runs/sac-seed0
research/neural/.venv/bin/chimera-neural evaluate \
  --run research/neural/runs/sac-seed0 --episodes 20 --seed 1000000 \
  --out research/neural/runs/sac-seed0-eval
```

20,000 is a proposed starting budget, not an established convergence point. This setup
does not launch that run automatically. Repeat with training seeds 1 and 2 before drawing
conclusions. For stronger baselines add budget-matched direct search; random actions alone
do not establish superiority to numerical optimization.

Training writes `policy.zip`, `replay.pkl`, `progress.csv`, and `config.json` (seed, step
budget, architecture, engine commit/dirty state, renderer hash, target hashes and elapsed
time). Evaluation writes per-episode distances to `evaluation.json`, the first episode's
audio for each policy and its target to WAV, and normalized movement trajectories to JSON.
Synthetic target audio is a 50 ms steady-state excerpt; generated attempts last one second.
It evaluates learned, neutral and random controls on the
same independently seeded goals. The metric averages the last five frames, rewarding
reaching and holding the sound. Inspect individual episodes and listen, not just averages.
Fresh synthetic seeds test interpolation within the same vowel-mixture distribution;
they do not establish unseen-speaker or real-speech generalization. Do not tune on final
test seeds or final test recordings.

## Real recordings

Put recordings you have rights to use under ignored `research/neural/data/`. Start with
one speaker and steady vowels near the same pitch. Supply **mono 48 kHz WAV**, trimmed to
the stable vowel, 0.1–5 seconds each. The loader deliberately rejects other formats rather
than silently changing them. Supply the measured fundamental frequency with `--f0`:

```sh
research/neural/.venv/bin/chimera-neural train \
  --targets research/neural/data/train --f0 140 --steps 20000 --seed 0 \
  --out research/neural/runs/real-seed0
research/neural/.venv/bin/chimera-neural evaluate \
  --run research/neural/runs/real-seed0 --targets research/neural/data/test \
  --out research/neural/runs/real-seed0-test
```

`140` is an example, not a pitch estimator. All files in a run currently share one f0;
the source and male anatomy are fixed. Evaluation inherits the run's f0. This means real
speaker differences can dominate the loss. There is no pitch extraction, speaker fitting,
resampling, room correction, phoneme alignment, or sentence imitation yet. The target
is the mean spectrum of each clip, so a phrase or glide is NOT an appropriate input.
Keep train/validation/test recordings distinct. Never put recordings or derived models in
git; follow the existing CSD restrictions if using that dataset.

## Environment contract

- Native process stays alive across steps. `reset` creates a fresh deterministic voice,
  with 250 ms settling; `step` preserves acoustic and articulator state.
- Eight controls: tongue-body x/y, jaw, lip aperture/protrusion/width, tongue root, larynx.
  Bounds follow the vowel calibrator; tongue tip follows the engine's free-tip dynamics.
- A 50 ms step increments normalized target positions by at most 0.15 (7.5% of physical
  range). Existing damped articulator dynamics smooth the resulting movement. Twenty
  steps make one second. Truncation marks the time limit, not successful imitation.
- Observation: four frames of measured positions, commanded positions and acoustic
  features, plus goal features and elapsed fraction. The hidden goal posture is never
  provided. History only approximates hidden acoustic state.
- Reward: negative mean absolute normalized log-mel shape error, plus RMS mismatch and
  a small squared target-movement penalty. Forty mel bands span 80–8000 Hz; 32 ms windows
  use a 10 ms hop. No neural decoder replaces the physical simulator.
- Synthetic targets are random convex mixtures of existing non-rhotic vowel postures,
  rendered by the same engine and then forgotten. No inverse posture labels train the
  policy. These are feasible targets for testing the learning loop before human audio.

See `chimera-neural --help`. The initial scaffold saves replay data but does not implement
resume, HER, a world model, or automatic hyperparameter sweeps.

## Setup verification

On 2026-11-08, macOS arm64 / Python 3.14, the initial setup passed:

- `bun run test`: 109 Rust tests passed, 2 pre-existing ignored; 30 Bun tests passed,
  724 assertions. This rebuilds WebAssembly before running its parity checks.
- `bun run typecheck` and `cargo fmt -p chimera-learn --check`.
- Python tests: 4 passed, including real-engine reset determinism, action effects,
  episode boundaries, acoustic features, and recording validation.
- SB3 environment checker; 64-action CPU smoke with 48 gradient updates; checkpoint
  save/load and two-episode-per-policy evaluation with WAV/trajectory exports.
- `pip check`: no broken requirements.

No longer learning run, human speech quality evaluation, GPU benchmark, browser UI test,
or evidence of convergence is included. The smoke's tiny evaluation is not a performance
claim. All generated artifacts remain local and ignored.
