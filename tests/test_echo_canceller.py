import numpy as np
import pytest

from pywebrtc_audio import EchoCanceller


def test_create_default():
    ec = EchoCanceller()
    assert ec is not None


@pytest.mark.parametrize("rate", [16000, 32000, 48000])
def test_create_with_sample_rate(rate):
    ec = EchoCanceller(sample_rate=rate)
    assert ec is not None


def test_invalid_sample_rate():
    with pytest.raises(ValueError):
        EchoCanceller(sample_rate=8000)


def test_process_silence():
    ec = EchoCanceller(sample_rate=16000)
    near = np.zeros(160, dtype=np.int16)
    far = np.zeros(160, dtype=np.int16)
    result = ec.process(near, far)
    assert result.shape == near.shape
    assert result.dtype == np.int16


def test_echo_reduction():
    ec = EchoCanceller(sample_rate=16000)
    frame_size = 160
    rng = np.random.default_rng(42)

    far = (rng.standard_normal(frame_size) * 3000).astype(np.int16)
    near = (far * 0.5 + rng.standard_normal(frame_size) * 100).astype(np.int16)

    for _ in range(200):
        result = ec.process(near, far)

    input_energy = np.sum(near.astype(np.float64) ** 2)
    output_energy = np.sum(result.astype(np.float64) ** 2)
    assert output_energy < input_energy


def test_process_arbitrary_length():
    ec = EchoCanceller(sample_rate=16000)
    near = np.zeros(320, dtype=np.int16)
    far = np.zeros(320, dtype=np.int16)
    result = ec.process(near, far)
    assert result.shape == (320,)
    assert result.dtype == np.int16


def test_process_mismatched_lengths():
    ec = EchoCanceller(sample_rate=16000)
    near = np.zeros(160, dtype=np.int16)
    far = np.zeros(320, dtype=np.int16)
    with pytest.raises(ValueError):
        ec.process(near, far)


def test_reset_clears_state():
    """After reset, output should match a freshly constructed instance."""
    rng = np.random.default_rng(42)
    far = (rng.standard_normal(160) * 3000).astype(np.int16)
    near = (far * 0.5).astype(np.int16)

    ec = EchoCanceller(sample_rate=16000)
    for _ in range(200):
        ec.process(near, far)
    ec.reset()

    fresh = EchoCanceller(sample_rate=16000)
    assert np.array_equal(ec.process(near, far), fresh.process(near, far))


def test_stream_delay_ms_default():
    ec = EchoCanceller()
    assert ec.stream_delay_ms == 0


def test_stream_delay_ms_init():
    ec = EchoCanceller(stream_delay_ms=50)
    assert ec.stream_delay_ms == 50


def test_stream_delay_ms_set():
    ec = EchoCanceller()
    ec.stream_delay_ms = 100
    assert ec.stream_delay_ms == 100


def test_stream_delay_ms_negative_raises():
    ec = EchoCanceller()
    with pytest.raises(ValueError):
        ec.stream_delay_ms = -1


def _long_delay_echo_reduction(ec, delay_ms):
    """Reduction (dB) on synthetic delayed + doubly-reflected white-noise echo."""
    rng = np.random.default_rng(7)
    reference = np.clip(rng.normal(0, 2500, 16000 * 18), -15000, 15000).astype(np.int16)
    delay = delay_ms * 16
    mic = np.zeros_like(reference)
    mic[delay:] = (reference[:-delay] * 0.6).astype(np.int16)
    mic[delay + 192:] += (reference[:-delay - 192] * 0.2).astype(np.int16)
    cleaned = ec.process(mic, reference)  # near=mic (with echo), far=reference
    raw = mic[-80000:].astype(np.float64)
    out = cleaned[-80000:].astype(np.float64)
    return 10 * np.log10((np.mean(raw ** 2) + 1) / (np.mean(out ** 2) + 1))


def test_num_filters_extends_long_delay():
    # Default (5 filters) cannot model a 950 ms echo path.
    assert _long_delay_echo_reduction(EchoCanceller(sample_rate=16000), 950) < 15
    # 16 filters restores strong cancellation at 950 ms.
    assert _long_delay_echo_reduction(
        EchoCanceller(sample_rate=16000, num_filters=16), 950) >= 15


def test_num_filters_invalid_raises():
    with pytest.raises(ValueError):
        EchoCanceller(sample_rate=16000, num_filters=0)


def test_num_filters_upper_bound():
    EchoCanceller(sample_rate=16000, num_filters=5000)  # in range: constructs
    with pytest.raises(ValueError):
        EchoCanceller(sample_rate=16000, num_filters=5001)


def test_reset_preserves_num_filters():
    # reset() must reuse the stored config, not fall back to the default 5 filters.
    ec = EchoCanceller(sample_rate=16000, num_filters=16)
    ec.reset()
    assert _long_delay_echo_reduction(ec, 950) >= 15
