import numpy as np
import pytest

from pywebrtc_audio import AudioProcessor


def test_create_default():
    assert AudioProcessor() is not None


def test_create_all_features():
    ap = AudioProcessor(echo_cancellation=True, noise_suppression=True, high_pass_filter=True)
    assert ap is not None


def test_invalid_ns_level():
    with pytest.raises(ValueError):
        AudioProcessor(ns_level=5)


def test_ns_only():
    ap = AudioProcessor(noise_suppression=True, ns_level=2)
    rng = np.random.default_rng(42)

    for _ in range(200):
        noise = (rng.standard_normal(160) * 1000).astype(np.int16)
        result = ap.process(noise)

    assert np.sum(result.astype(float) ** 2) < np.sum(noise.astype(float) ** 2)


def test_aec_plus_ns():
    ap = AudioProcessor(echo_cancellation=True, noise_suppression=True)
    rng = np.random.default_rng(42)

    for _ in range(200):
        far = (rng.standard_normal(160) * 5000).astype(np.int16)
        near = (far * 0.3 + rng.standard_normal(160) * 500).astype(np.int16)
        result = ap.process(near, far)

    input_energy = np.sum(near.astype(float) ** 2)
    output_energy = np.sum(result.astype(float) ** 2)
    assert output_energy < input_energy * 0.01


def test_hpf_only():
    ap = AudioProcessor(high_pass_filter=True)
    result = ap.process(np.zeros(160, dtype=np.int16))
    assert result.shape == (160,)
    assert result.dtype == np.int16


def test_aec_requires_far():
    ap = AudioProcessor(echo_cancellation=True)
    with pytest.raises(ValueError, match="far is required"):
        ap.process(np.zeros(160, dtype=np.int16))


def test_arbitrary_length():
    ap = AudioProcessor(noise_suppression=True)
    result = ap.process(np.zeros(100, dtype=np.int16))
    assert result.shape == (100,)
    assert result.dtype == np.int16


def test_wrong_far_frame_size():
    ap = AudioProcessor(echo_cancellation=True)
    with pytest.raises(ValueError):
        ap.process(np.zeros(160, dtype=np.int16), np.zeros(100, dtype=np.int16))


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
@pytest.mark.parametrize(
    "rate,frame_size",
    [
        (8000, 80),
        (10000, 100),
        (11025, 110),
        (16000, 160),
        (24000, 240),
        (32000, 320),
        (44100, 441),
        (48000, 480),
        (96000, 960),
        (384000, 3840),
    ],
)
def test_sample_rate_processing(rate, frame_size, dtype):
    ap = AudioProcessor(sample_rate=rate, echo_cancellation=True, noise_suppression=True)
    near = np.zeros(frame_size, dtype=dtype)
    far = np.zeros(frame_size, dtype=dtype)
    result = ap.process(near, far)
    assert result.shape == (frame_size,)
    assert result.dtype == dtype


@pytest.mark.parametrize("rate", [7999, 384001])
def test_sample_rate_outside_supported_range(rate):
    with pytest.raises(ValueError, match="between 8000 and 384000"):
        AudioProcessor(sample_rate=rate)


@pytest.mark.parametrize("rate", [10000, 11025, 24000, 44100, 96000])
def test_resampled_rate_aec_plus_ns_reduces_echo(rate):
    frame_size = rate // 100
    ap = AudioProcessor(sample_rate=rate, echo_cancellation=True, noise_suppression=True)
    rng = np.random.default_rng(42)

    for _ in range(200):
        far = (rng.standard_normal(frame_size) * 5000).astype(np.int16)
        near = (far * 0.3 + rng.standard_normal(frame_size) * 500).astype(np.int16)
        result = ap.process(near, far)

    input_energy = np.sum(near.astype(float) ** 2)
    output_energy = np.sum(result.astype(float) ** 2)
    assert output_energy < input_energy * 0.01


def test_no_features_passthrough():
    ap = AudioProcessor()
    frame = (np.ones(160) * 1000).astype(np.int16)
    result = ap.process(frame)
    assert result.shape == (160,)
    assert result.dtype == np.int16


@pytest.mark.parametrize(
    "rate,frame_size",
    [(16000, 160), (24000, 240), (32000, 320), (44100, 441), (48000, 480)],
)
def test_reset_clears_state(rate, frame_size):
    """After reset, output should match a freshly constructed instance."""
    rng = np.random.default_rng(42)
    far = (rng.standard_normal(frame_size) * 3000).astype(np.int16)
    near = (far * 0.5 + rng.standard_normal(frame_size) * 200).astype(np.int16)

    ap = AudioProcessor(
        sample_rate=rate,
        echo_cancellation=True,
        noise_suppression=True,
        ns_level=2,
    )
    for _ in range(200):
        ap.process(near, far)
    ap.reset()

    fresh = AudioProcessor(
        sample_rate=rate,
        echo_cancellation=True,
        noise_suppression=True,
        ns_level=2,
    )
    assert np.array_equal(ap.process(near, far), fresh.process(near, far))


def test_stream_delay_ms_default():
    ap = AudioProcessor(echo_cancellation=True)
    assert ap.stream_delay_ms == 0


def test_stream_delay_ms_set():
    ap = AudioProcessor(echo_cancellation=True)
    ap.stream_delay_ms = 75
    assert ap.stream_delay_ms == 75


def test_stream_delay_ms_negative_raises():
    ap = AudioProcessor(echo_cancellation=True)
    with pytest.raises(ValueError):
        ap.stream_delay_ms = -1


def test_speech_probability_with_ns():
    ap = AudioProcessor(noise_suppression=True)
    rng = np.random.default_rng(42)
    for _ in range(10):
        ap.process((rng.standard_normal(160) * 1000).astype(np.int16))
    prob = ap.speech_probability
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_speech_probability_with_agc_only():
    ap = AudioProcessor(auto_gain_control=True)
    rng = np.random.default_rng(42)
    ap.process((rng.standard_normal(16000) * 100).astype(np.int16))
    prob = ap.speech_probability
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_speech_probability_without_ns_or_agc():
    ap = AudioProcessor(high_pass_filter=True)
    ap.process(np.zeros(160, dtype=np.int16))
    prob = ap.speech_probability
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def _ap_long_delay_reduction(ap, delay_ms):
    """Reduction (dB) on synthetic delayed + doubly-reflected white-noise echo."""
    rng = np.random.default_rng(7)
    reference = np.clip(rng.normal(0, 2500, 16000 * 18), -15000, 15000).astype(np.int16)
    delay = delay_ms * 16
    mic = np.zeros_like(reference)
    mic[delay:] = (reference[:-delay] * 0.6).astype(np.int16)
    mic[delay + 192:] += (reference[:-delay - 192] * 0.2).astype(np.int16)
    cleaned = ap.process(mic, reference)
    raw = mic[-80000:].astype(np.float64)
    out = cleaned[-80000:].astype(np.float64)
    return 10 * np.log10((np.mean(raw ** 2) + 1) / (np.mean(out ** 2) + 1))


def test_audioprocessor_num_filters_long_delay():
    default = AudioProcessor(sample_rate=16000, echo_cancellation=True)
    assert _ap_long_delay_reduction(default, 950) < 15
    tuned = AudioProcessor(sample_rate=16000, echo_cancellation=True, num_filters=16)
    assert _ap_long_delay_reduction(tuned, 950) >= 15


def test_audioprocessor_num_filters_invalid_raises():
    with pytest.raises(ValueError):
        AudioProcessor(sample_rate=16000, echo_cancellation=True, num_filters=0)
    with pytest.raises(ValueError):
        AudioProcessor(sample_rate=16000, echo_cancellation=True, num_filters=5001)


def test_audioprocessor_reset_preserves_num_filters():
    ap = AudioProcessor(sample_rate=16000, echo_cancellation=True, num_filters=16)
    ap.reset()
    assert _ap_long_delay_reduction(ap, 950) >= 15
