"""Explicitly budgeted training and independent evaluation, all local."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from .env import ROOT, SAMPLE_RATE, VowelEnv


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def evaluate(model, env, episodes, seed, output):
    """Same goals for all policies; report tail distance, never training return alone."""
    rows = []
    for policy in ("neutral", "random", "learned"):
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            rng = np.random.default_rng(seed + episode)
            distances, audio, trajectory = [], [], []
            if episode == 0 and policy == "neutral":
                sf.write(output / "target.wav", env.target_audio, SAMPLE_RATE, subtype="FLOAT")
            done = False
            while not done:
                if policy == "neutral":
                    action = np.zeros(8, dtype=np.float32)
                elif policy == "random":
                    action = rng.uniform(-1, 1, size=8).astype(np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                distances.append(info["distance"])
                audio.append(env.audio.copy())
                trajectory.append({"positions": env.positions.tolist(),
                                   "command": env.command.tolist(), "distance": info["distance"]})
            rows.append({"policy": policy, "seed": seed + episode,
                         "tail_distance": float(np.mean(distances[-5:]))})
            if episode == 0:
                sf.write(output / f"{policy}.wav", np.concatenate(audio), SAMPLE_RATE, subtype="FLOAT")
                (output / f"{policy}-trajectory.json").write_text(json.dumps({
                    "keys": env.bridge.schema["keys"], "low": env.low.tolist(), "high": env.high.tolist(),
                    "step_seconds": 0.05, "normalized": True, "frames": trajectory}, indent=2) + "\n")
    report = {"episodes": rows, "mean_tail_distance": {
        policy: float(np.mean([r["tail_distance"] for r in rows if r["policy"] == policy]))
        for policy in ("neutral", "random", "learned")}}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["mean_tail_distance"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Check real renderer, Gym contract, and a few SAC updates")
    smoke.add_argument("--out", type=Path, default=ROOT / "research/neural/runs/smoke")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=positive, required=True, help="Explicit environment-step budget")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--targets", type=Path, help="Directory of steady-vowel recordings; default synthetic goals")
    train.add_argument("--f0", type=float, default=140)
    train.add_argument("--out", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--run", type=Path, required=True)
    evaluate_parser.add_argument("--out", type=Path, required=True)
    evaluate_parser.add_argument("--targets", type=Path, help="Separate held-out recordings; default synthetic")
    evaluate_parser.add_argument("--episodes", type=positive, default=20)
    evaluate_parser.add_argument("--seed", type=int, default=1000000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "evaluate":
        config = json.loads((args.run / "config.json").read_text())
        model = SAC.load(args.run / "policy", device="cpu")
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "config.json").write_text(json.dumps({
            "training": config, "seed": args.seed, "episodes": args.episodes,
            "targets": str(args.targets.resolve()) if args.targets else None,
            "target_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(args.targets.glob("*.wav"))} if args.targets else {},
            "renderer_sha256": hashlib.sha256((ROOT / "target/release/chimera-learn").read_bytes()).hexdigest(),
            "policy_sha256": hashlib.sha256((args.run / "policy.zip").read_bytes()).hexdigest()
        }, indent=2) + "\n")
        env = VowelEnv(targets=args.targets, f0=config["f0"])
        try:
            evaluate(model, env, args.episodes, args.seed, args.out)
        finally:
            env.close()
        return

    is_smoke = args.command == "smoke"
    steps = 64 if is_smoke else args.steps
    seed = 0 if is_smoke else args.seed
    f0 = 140 if is_smoke else args.f0
    targets = None if is_smoke else args.targets
    # Prevent accidental checkpoint replacement.
    args.out.mkdir(parents=True, exist_ok=False)
    config = {"algorithm": "SAC", "network": [128, 128], "steps": steps, "seed": seed,
              "f0": f0, "targets": str(targets.resolve()) if targets else None,
              "engine_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
              "purpose": "plumbing verification, not learning evidence" if is_smoke else "training"}
    config["renderer_sha256"] = hashlib.sha256((ROOT / "target/release/chimera-learn").read_bytes()).hexdigest()
    config["target_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(targets.glob("*.wav"))} if targets else {}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = VowelEnv(targets=targets, f0=f0)
    start = time.monotonic()
    try:
        if is_smoke:
            check_env(env, warn=True)
        model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]},
                    learning_starts=16 if is_smoke else 1000, batch_size=32 if is_smoke else 256,
                    buffer_size=100000, seed=seed, device="cpu", verbose=1)
        model.set_logger(configure(str(args.out), ["stdout", "csv"]))
        model.learn(total_timesteps=steps)
        model.save(args.out / "policy")
        model.save_replay_buffer(args.out / "replay.pkl")
        config["elapsed_seconds"] = time.monotonic() - start
        config["gradient_updates"] = model._n_updates
        (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(config, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
