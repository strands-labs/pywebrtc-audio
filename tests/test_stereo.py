"""Stereo / multi-channel tests for all classes."""

import numpy as np
import pytest

from pywebrtc_audio import AudioProcessor, EchoCanceller, GainController, NoiseSuppressor, VoiceDetector

RATE = 16000
FRAME_SIZE = 160  # samples per channel
STRIDE = FRAME_SIZE * 2  # interleaved stereo frame


def stereo_interleave(left, right):
    out = np.empty(len(left) * 2, dtype=left.dtype)
    out[0::2] = left
    out[1::2] = right
    return out


# --- NoiseSuppressor ---


def test_ns_stereo_basic():
    ns = NoiseSuppressor(sample_rate=RATE, num_channels=2)
    result = ns.process(np.zeros(STRIDE, dtype=np.int16))
    assert result.shape == (STRIDE,)
    assert result.dtype == np.int16


def test_ns_stereo_channel_independence():
    """Left=sine, right=silence. NS should preserve channel separation."""
    ns = NoiseSuppressor(sample_rate=RATE, num_channels=2)
    t = np.arange(FRAME_SIZE) / RATE
    left = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    right = np.zeros(FRAME_SIZE, dtype=np.int16)
    audio = stereo_interleave(left, right)

    for _ in range(100):
        result = ns.process(audio)

    assert np.sqrt(np.mean(result[0::2].astype(float) ** 2)) > 100
    assert np.sqrt(np.mean(result[1::2].astype(float) ** 2)) < 10


def test_ns_stereo_no_cross_frame_leakage():
    """Frame N output must not depend on frame N+1 data."""
    rng = np.random.default_rng(42)
    warmup = (rng.standard_normal(STRIDE) * 1000).astype(np.int16)

    ns_a = NoiseSuppressor(sample_rate=RATE, num_channels=2)
    ns_b = NoiseSuppressor(sample_rate=RATE, num_channels=2)
    for _ in range(50):
        ns_a.process(warmup)
        ns_b.process(warmup)

    data_a = np.zeros(STRIDE * 2, dtype=np.int16)
    data_a[:STRIDE] = 5000
    data_b = data_a.copy()
    data_b[STRIDE:] = 10000

    np.testing.assert_array_equal(
        ns_a.process(data_a)[:STRIDE],
        ns_b.process(data_b)[:STRIDE],
    )


def test_ns_stereo_arbitrary_length():
    ns = NoiseSuppressor(sample_rate=RATE, num_channels=2)
    result = ns.process(np.zeros(250, dtype=np.int16))
    assert result.shape == (250,)


def test_ns_stereo_float32():
    ns = NoiseSuppressor(sample_rate=RATE, num_channels=2)
    audio = np.zeros(STRIDE, dtype=np.float32)
    result = ns.process(audio)
    assert result.dtype == np.float32
    assert result.shape == (STRIDE,)


# --- EchoCanceller ---


def test_ec_stereo_basic():
    ec = EchoCanceller(sample_rate=RATE, num_channels=2)
    near = np.zeros(STRIDE, dtype=np.int16)
    far = np.zeros(STRIDE, dtype=np.int16)
    result = ec.process(near, far)
    assert result.shape == (STRIDE,)


def test_ec_stereo_echo_reduction():
    ec = EchoCanceller(sample_rate=RATE, num_channels=2)
    rng = np.random.default_rng(42)

    for _ in range(200):
        far = (rng.standard_normal(STRIDE) * 5000).astype(np.int16)
        near = (far * 0.3 + rng.standard_normal(STRIDE) * 100).astype(np.int16)
        result = ec.process(near, far)

    assert np.sum(result.astype(float) ** 2) < np.sum(near.astype(float) ** 2) * 0.5


def test_ec_stereo_float32():
    ec = EchoCanceller(sample_rate=RATE, num_channels=2)
    near = np.zeros(STRIDE, dtype=np.float32)
    far = np.zeros(STRIDE, dtype=np.float32)
    result = ec.process(near, far)
    assert result.dtype == np.float32


# --- VoiceDetector ---


def test_vd_stereo_basic():
    vd = VoiceDetector(sample_rate=RATE, num_channels=2)
    prob = vd.process(np.zeros(STRIDE, dtype=np.int16))
    assert 0.0 <= prob <= 1.0


def test_vd_stereo_arbitrary_length():
    vd = VoiceDetector(sample_rate=RATE, num_channels=2)
    prob = vd.process(np.zeros(250, dtype=np.int16))
    assert 0.0 <= prob <= 1.0


# --- GainController ---


def test_gc_stereo_basic():
    gc = GainController(sample_rate=RATE, num_channels=2)
    result = gc.process(np.zeros(STRIDE, dtype=np.int16))
    assert result.shape == (STRIDE,)


def test_gc_stereo_amplifies():
    gc = GainController(sample_rate=RATE, num_channels=2)
    rng = np.random.default_rng(42)
    quiet = (rng.standard_normal(RATE) * 100).astype(np.int16)
    loud = gc.process(quiet)
    assert np.abs(loud).mean() > np.abs(quiet).mean()


# --- AudioProcessor ---


def test_ap_stereo_ns():
    ap = AudioProcessor(sample_rate=RATE, num_channels=2, noise_suppression=True)
    result = ap.process(np.zeros(STRIDE, dtype=np.int16))
    assert result.shape == (STRIDE,)


def test_ap_stereo_aec():
    ap = AudioProcessor(sample_rate=RATE, num_channels=2, echo_cancellation=True)
    near = np.zeros(STRIDE, dtype=np.int16)
    far = np.zeros(STRIDE, dtype=np.int16)
    result = ap.process(near, far)
    assert result.shape == (STRIDE,)


def test_ap_stereo_all_features():
    ap = AudioProcessor(
        sample_rate=RATE, num_channels=2,
        echo_cancellation=True, noise_suppression=True, auto_gain_control=True,
    )
    rng = np.random.default_rng(42)

    for _ in range(50):
        far = (rng.standard_normal(STRIDE) * 5000).astype(np.int16)
        near = (far * 0.3 + rng.standard_normal(STRIDE) * 500).astype(np.int16)
        result = ap.process(near, far)

    assert result.shape == near.shape
    assert 0.0 <= ap.speech_probability <= 1.0
    assert isinstance(ap.gain_db, float)


def test_ap_stereo_channel_independence():
    """HP filter on stereo DC: both channels should be filtered independently."""
    ap = AudioProcessor(sample_rate=RATE, num_channels=2, high_pass_filter=True)
    left_dc = (np.ones(FRAME_SIZE) * 5000).astype(np.int16)
    right_dc = (np.ones(FRAME_SIZE) * -5000).astype(np.int16)
    audio = stereo_interleave(left_dc, right_dc)

    for _ in range(100):
        result = ap.process(audio)

    # HP filter removes DC - both channels should converge toward 0
    assert abs(result[0::2].mean()) < abs(left_dc.mean())
    assert abs(result[1::2].mean()) < abs(right_dc.mean())


@pytest.mark.parametrize("rate", [8000, 24000, 44100, 96000])
def test_ap_stereo_resampled_float32(rate):
    frame_stride = (rate // 100) * 2
    ap = AudioProcessor(
        sample_rate=rate,
        num_channels=2,
        echo_cancellation=True,
        noise_suppression=True,
        auto_gain_control=True,
    )
    near = np.zeros(frame_stride, dtype=np.float32)
    far = np.zeros(frame_stride, dtype=np.float32)

    result = ap.process(near, far)

    assert result.shape == (frame_stride,)
    assert result.dtype == np.float32


def test_ap_stereo_resampled_float32_channel_independence():
    rate = 24000
    frame_size = rate // 100
    t = np.arange(frame_size) / rate
    left = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    right = np.zeros(frame_size, dtype=np.float32)
    audio = stereo_interleave(left, right)
    ap = AudioProcessor(sample_rate=rate, num_channels=2)

    result = ap.process(audio)

    input_left_rms = np.sqrt(np.mean(left.astype(float) ** 2))
    output_left_rms = np.sqrt(np.mean(result[0::2].astype(float) ** 2))
    assert output_left_rms > input_left_rms * 0.5
    assert np.max(np.abs(result[1::2])) < 1e-6


@pytest.mark.parametrize("rate", [16000, 32000, 48000])
def test_stereo_all_sample_rates(rate):
    stride = (rate // 100) * 2
    ns = NoiseSuppressor(sample_rate=rate, num_channels=2)
    result = ns.process(np.zeros(stride, dtype=np.int16))
    assert result.shape == (stride,)
