import numpy as np
import pytest

from chimera_neural.env import AcousticFeatures, Bridge, HORIZON, VowelEnv, acoustic_distance


def test_features_distinguish_silence_pitch_and_gain():
    features = AcousticFeatures()
    t = np.arange(2400) / 48000
    a = features(0.1 * np.sin(2 * np.pi * 200 * t))
    louder = features(0.2 * np.sin(2 * np.pi * 200 * t))
    other = features(0.1 * np.sin(2 * np.pi * 2000 * t))
    silent = features(np.zeros(2400))
    assert acoustic_distance(a, a) == 0
    assert louder[-1] > a[-1] > silent[-1]
    assert acoustic_distance(a, other) > 0.1
    assert acoustic_distance(a, silent) > 0.1


def test_real_renderer_protocol_and_determinism():
    bridge = Bridge()
    try:
        with pytest.raises(RuntimeError, match="reset"):
            bridge.call(op="step", targets=[0] * 8)
        first = bridge.call(op="reset", f0=140)
        bridge.call(op="step", targets=[0] * 8)
        assert bridge.call(op="reset", f0=140) == first
        assert len(first["audio"]) == bridge.schema["frames"]
        with pytest.raises(RuntimeError, match="normalized"):
            bridge.call(op="step", targets=[2] * 8)
    finally:
        bridge.close()


def test_goal_replay_action_effect_and_episode_boundary():
    env = VowelEnv()
    try:
        a, _ = env.reset(seed=42)
        b, _ = env.reset(seed=42)
        np.testing.assert_array_equal(a, b)
        assert env.observation_space.contains(a)
        initial = env.positions.copy()
        for _ in range(HORIZON):
            observation, reward, terminated, truncated, info = env.step(np.ones(8))
            assert np.isfinite(observation).all() and np.isfinite(reward)
            assert info["distance"] >= 0 and not terminated
        assert truncated
        assert np.max(abs(env.positions - initial)) > 0.05
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(8))
        env.reset(seed=43)
        with pytest.raises(ValueError, match="finite"):
            env.step(np.full(8, np.nan))
    finally:
        env.close()


def test_real_recording_validation(tmp_path):
    import soundfile as sf
    sf.write(tmp_path / "silence.wav", np.zeros(4800), 48000)
    with pytest.raises(ValueError, match="silent"):
        VowelEnv(targets=tmp_path)
    sf.write(tmp_path / "silence.wav", np.sin(np.arange(4800) * 0.02) * 0.1, 48000)
    env = VowelEnv(targets=tmp_path)
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert len(env.recordings) == 1
    finally:
        env.close()
