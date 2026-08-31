from typing import overload

import numpy as np
import numpy.typing as npt


class EchoCanceller:
    """Acoustic echo canceller using WebRTC's AEC3 module.

    Not thread-safe. Use one instance per thread or synchronize externally.
    """

    stream_delay_ms: int

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1, stream_delay_ms: int = 0) -> None: ...
    @overload
    def process(
        self,
        near: npt.NDArray[np.int16],
        far: npt.NDArray[np.int16],
    ) -> npt.NDArray[np.int16]: ...
    @overload
    def process(
        self,
        near: npt.NDArray[np.float32],
        far: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]: ...
    def reset(self) -> None:
        """Reset internal state (AEC filter, high-pass filter) while keeping config."""
        ...


class NoiseSuppressor:
    """Noise suppressor using WebRTC NS.

    Not thread-safe. Use one instance per thread or synchronize externally.
    """

    speech_probability: float
    """Speech probability (0.0-1.0) from the most recent ``process()`` call.

    Based on spectral features (likelihood ratio, flatness, template difference),
    not simple energy thresholds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        level: int = 1,
    ) -> None: ...
    @overload
    def process(
        self,
        audio: npt.NDArray[np.int16],
    ) -> npt.NDArray[np.int16]: ...
    @overload
    def process(
        self,
        audio: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]: ...
    def reset(self) -> None:
        """Reset internal state (noise estimates) while keeping config."""
        ...


class VoiceDetector:
    """Lightweight voice activity detector using WebRTC's noise analysis.

    Runs the same spectral analysis as ``NoiseSuppressor`` to compute speech
    probability, but skips the Wiener filter - no noise suppression is applied.
    Use this when you only need VAD without modifying the audio.

    Not thread-safe. Use one instance per thread or synchronize externally.
    """

    speech_probability: float
    """Speech probability (0.0-1.0) from the most recent ``process()`` call."""

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1) -> None: ...
    @overload
    def process(self, audio: npt.NDArray[np.int16]) -> float: ...
    @overload
    def process(self, audio: npt.NDArray[np.float32]) -> float: ...
    def reset(self) -> None:
        """Reset internal state while keeping config."""
        ...


class GainController:
    """Automatic gain control using WebRTC's AGC2 algorithm.

    Combines speech/noise level estimation, adaptive digital gain, fixed digital
    gain, and a limiter. Uses an internal VAD (same spectral analysis as
    ``NoiseSuppressor``) unless ``speech_probability`` is provided to ``process()``.

    Not thread-safe. Use one instance per thread or synchronize externally.
    """

    gain_db: float
    """Current applied gain in dB from the most recent ``process()`` call."""

    speech_probability: float
    """Speech probability (0.0-1.0) from the most recent ``process()`` call.

    Uses an internal RNN VAD unless ``speech_probability`` is provided to ``process()``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        fixed_gain_db: float = 0.0,
        adaptive_digital: bool = True,
        max_gain_db: float = 50.0,
        headroom_db: float = 5.0,
        max_gain_change_db_per_second: float = 6.0,
        max_output_noise_level_dbfs: float = -50.0,
    ) -> None: ...
    @overload
    def process(
        self,
        audio: npt.NDArray[np.int16],
        speech_probability: float | None = None,
    ) -> npt.NDArray[np.int16]: ...
    @overload
    def process(
        self,
        audio: npt.NDArray[np.float32],
        speech_probability: float | None = None,
    ) -> npt.NDArray[np.float32]: ...
    def reset(self) -> None:
        """Reset internal state (gain estimates, noise/speech levels) while keeping config."""
        ...


class AudioProcessor:
    """Combined audio processing pipeline using WebRTC.

    Runs echo cancellation, noise suppression, automatic gain control, and
    high-pass filtering in a single optimized pass over shared audio buffers.

    Not thread-safe. Use one instance per thread or synchronize externally.

    Args:
        sample_rate: Audio sample rate in Hz from 8000 through 384000. Rates without an integer number
            of samples per 10ms frame are approximated.
        num_channels: Number of audio channels (1 for mono, 2 for stereo).
        echo_cancellation: Enable AEC3 echo cancellation.
        noise_suppression: Enable noise suppression.
        high_pass_filter: Enable high-pass filter (also enabled automatically with AEC).
        auto_gain_control: Enable AGC2 automatic gain control.
        ns_level: Noise suppression level 0-3 (6dB, 12dB, 18dB, 21dB).
        agc_gain_db: Fixed gain in dB applied after adaptive gain. Default 0.
        agc_max_gain_db: Maximum adaptive gain in dB. Default 50.
        stream_delay_ms: Audio buffer delay hint in milliseconds for AEC.
    """

    stream_delay_ms: int
    speech_probability: float
    """Speech probability (0.0-1.0) from the most recent ``process()`` call.

    Always available. Priority: noise suppressor's spectral estimate (when
    ``noise_suppression=True``), then AGC's internal RNN VAD estimate (when
    ``auto_gain_control=True``), then a lightweight spectral analysis (same
    as ``VoiceDetector``).
    """
    gain_db: float
    """Current applied gain in dB from the most recent ``process()`` call.

    Only available when ``auto_gain_control=True``. Raises ``RuntimeError`` otherwise.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        echo_cancellation: bool = False,
        noise_suppression: bool = False,
        high_pass_filter: bool = False,
        auto_gain_control: bool = False,
        ns_level: int = 1,
        agc_gain_db: float = 0.0,
        agc_max_gain_db: float = 50.0,
        stream_delay_ms: int = 0,
    ) -> None: ...
    @overload
    def process(
        self,
        near: npt.NDArray[np.int16],
        far: npt.NDArray[np.int16] | None = None,
    ) -> npt.NDArray[np.int16]: ...
    @overload
    def process(
        self,
        near: npt.NDArray[np.float32],
        far: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]: ...
    def reset(self) -> None:
        """Reset internal state (AEC filter, noise estimates, high-pass filter, AGC) while keeping config."""
        ...
