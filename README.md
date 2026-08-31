# pywebrtc-audio

Python bindings for the WebRTC audio processing module. Echo cancellation, noise suppression, automatic gain control, voice activity detection, and high-pass filtering - the same algorithms that run in Chrome, Edge, and every WebRTC-based application.

```python
from pywebrtc_audio import AudioProcessor

ap = AudioProcessor(
    sample_rate=16000,
    noise_suppression=True,
    echo_cancellation=True,
    auto_gain_control=True,
    stream_delay_ms=40,
)
ap.stream_delay_ms = 50  # adjustable at runtime

# near = what the mic picked up (speech + echo + noise)
# far  = what you played through the speaker (reference signal)

# accepts int16 or float32 numpy arrays, returns the same dtype
clean = ap.process(near, far)

# speech probability from the noise suppressor's spectral analysis
print(ap.speech_probability)  # 0.0-1.0
print(ap.gain_db)             # current AGC gain in dB
```

## Installation

```bash
pip install pywebrtc-audio
```

Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64). Python 3.10-3.14.

## Examples

See the [`examples/`](examples/) directory:

- [`basic.py`](examples/basic.py) - Minimal usage
- [`strands_agents_bidi.py`](examples/strands_agents_bidi.py) - [Strands](https://github.com/strands-agents) BidiAgent with live echo cancellation
- [`stereo.py`](examples/stereo.py) - Stereo (multi-channel) processing with interleaved layout
- [`agc.py`](examples/agc.py) - Automatic gain control
- [`vad_realtime.py`](examples/vad_realtime.py) - Real-time voice activity detection from the mic
- [`wav_file.py`](examples/wav_file.py) - Process wav files offline
- [`pyaudio_realtime.py`](examples/pyaudio_realtime.py) - Real-time echo cancellation with PyAudio
- [`e2e_verify.py`](examples/e2e_verify.py) - Record from mic + speakers, compare raw vs AEC output
- [`e2e_speech.py`](examples/e2e_speech.py) - Talk while a tone plays, verify speech is preserved

## Use cases

- **Voice agents and assistants** - When an AI agent speaks through a speaker and listens through a mic on the same device, it hears its own output as echo. AEC removes the agent's voice from the mic capture so it only hears the user. See [`examples/strands_agents_bidi.py`](examples/strands_agents_bidi.py) for a working [Strands](https://github.com/strands-agents) BidiAgent integration.

- **Speech-to-text preprocessing** - Clean up mic audio before sending it to a transcription service. Noise suppression removes background noise (fans, traffic, keyboard), AGC normalizes volume across speakers, and the high-pass filter removes low-frequency rumble. Reduces word error rates without any model changes.

- **Telephony and VoIP** - The same processing pipeline that runs in Chrome for WebRTC calls, available as a Python library. Process audio from SIP trunks, WebSocket streams, or any other audio source that needs echo cancellation and noise reduction.

- **Voice activity detection** - Use `VoiceDetector` or `speech_probability` to detect when someone is speaking. Useful for turn-taking in conversational AI, silence trimming in recordings, or triggering wake-word pipelines only when speech is present. Runs in ~2µs per 10ms frame.

- **Robotics** - Robots with speakers and microphones face the same echo problem as voice assistants, often worse due to motor noise and reverberant environments. The full pipeline (AEC + NS + AGC) handles all of this in a single `process()` call.

- **Audio recording and podcasting** - Clean up recordings after the fact with `examples/wav_file.py`. Remove background noise from interview recordings, normalize volume levels across multiple speakers, or batch-process audio files through the pipeline.

- **Real-time audio monitoring** - Build live audio meters, speech detectors, or noise level monitors. All processing runs in C++ with the GIL released, so it won't block your Python event loop or UI thread.

## Performance

All processing runs in C++ with the GIL released. At 16kHz mono (the most common voice configuration), processing 100ms of int16 audio on an Apple M3 Pro:

| Pipeline | Time | Realtime factor |
|----------|-----:|----------------:|
| VoiceDetector | 21 µs | 4,665x |
| NoiseSuppressor | 32 µs | 3,089x |
| GainController | 103 µs | 970x |
| EchoCanceller | 622 µs | 161x |
| AudioProcessor (AEC+NS+AGC) | 649 µs | 154x |
| AudioProcessor (all features) | 686 µs | 146x |

The full pipeline processes 1 second of audio in ~7ms. Even at 48kHz stereo with all features, it runs at 82x real-time. int16 and float32 perform nearly identically.

See [benchmarks/BENCHMARK.md](benchmarks/BENCHMARK.md) for detailed results across all sample rates, dtypes, chunk sizes, and stereo.

## API

Five classes, each with a `process()` method that accepts `int16` or `float32` numpy arrays of any length. Internally splits into 10ms frames in a single GIL-released loop. The last frame is zero-padded if the input isn't a multiple of the frame size, and the output is truncated to match the original input length. `VoiceDetector` returns speech probability instead of audio. `AudioProcessor` combines them into a single pipeline.

Multi-channel audio uses interleaved layout: `[L0, R0, L1, R1, ...]`. A 10ms stereo frame at 16kHz is 320 samples (160 per channel × 2 channels). Mono is the default and most common for voice processing.

Instances are not thread-safe. Use one per thread or synchronize externally.

### AudioProcessor

```python
AudioProcessor(
    sample_rate=16000,
    num_channels=1,
    echo_cancellation=False,
    noise_suppression=False,
    high_pass_filter=False,
    auto_gain_control=False,
    ns_level=1,
    agc_gain_db=0.0,
    agc_max_gain_db=50.0,
    stream_delay_ms=0,
)
```

Combined audio processing pipeline. Runs echo cancellation, noise suppression, automatic gain control, and high-pass filtering in a single optimized pass over shared audio buffers - avoids the overhead of copying frames between separate processors. Processing order: HP filter -> AEC -> NS -> AGC.

- `sample_rate`: Audio sample rate in Hz from 8000 through 384000. WebRTC resamples to an internal processing rate of 16000, 32000, or 48000 Hz. Rates that do not contain an integer number of samples in 10ms may reduce echo-cancellation quality.
- `echo_cancellation`: Enable AEC3 echo cancellation.
- `noise_suppression`: Enable noise suppression.
- `high_pass_filter`: Enable high-pass filter (also enabled automatically with AEC).
- `auto_gain_control`: Enable AGC2 automatic gain control. Uses speech probability from NS if enabled, otherwise runs its own internal RNN VAD.
- `ns_level`: Noise suppression level 0-3 (6dB, 12dB, 18dB, 21dB).
- `agc_gain_db`: Fixed gain in dB applied after adaptive gain. Default 0.
- `agc_max_gain_db`: Maximum adaptive gain in dB. Default 50.
- `stream_delay_ms`: Audio buffer delay hint in milliseconds for AEC. Also available as a read/write property. This is the delay between writing audio to the speaker buffer and the corresponding echo appearing in the mic capture. Most audio APIs report their buffer size - for PyAudio it's `frames_per_buffer / sample_rate * 1000`. Default 0 lets AEC3's internal delay estimator figure it out, but providing a hint helps it converge faster.

Note: When `echo_cancellation` is enabled, a high-pass filter is always applied to the capture signal before echo cancellation, regardless of the `high_pass_filter` setting. This matches Chrome's behavior - the HP filter removes DC offset that would otherwise degrade AEC performance.

```python
AudioProcessor.process(near, far=None) -> np.ndarray
```

Process audio of any length.

- `near`: Microphone capture signal (`int16` or `float32` numpy array, any length).
- `far`: Speaker reference signal (required when `echo_cancellation=True`, same length as `near`).
- Returns: Processed audio (same dtype and length as input).

```python
AudioProcessor.reset()
```

Reset all internal DSP state (AEC filter coefficients, noise estimates, high-pass filter, AGC gain state) while keeping the original configuration. Useful between conversations or after interruptions to avoid stale state affecting the next audio stream.

```python
AudioProcessor.speech_probability
```

Read-only property. Speech probability (0.0-1.0) from the most recent `process()` call. Always available. Priority: noise suppressor's spectral estimate (when `noise_suppression=True`), then AGC's internal RNN VAD estimate (when `auto_gain_control=True`), then a lightweight spectral analysis (same as `VoiceDetector`).

```python
AudioProcessor.gain_db
```

Read-only property. Current applied gain in dB from the most recent `process()` call. Only available when `auto_gain_control=True`; raises `RuntimeError` otherwise.


### GainController

```python
GainController(
    sample_rate=16000,
    num_channels=1,
    fixed_gain_db=0.0,
    adaptive_digital=True,
    max_gain_db=50.0,
    headroom_db=5.0,
    max_gain_change_db_per_second=6.0,
    max_output_noise_level_dbfs=-50.0,
)
```

Standalone automatic gain control using the AGC2 algorithm. Combines adaptive digital gain, fixed digital gain, and a limiter. Uses an internal VAD (same spectral analysis as `NoiseSuppressor`) unless `speech_probability` is provided to `process()`.

- `sample_rate`: Audio sample rate in Hz. Supported: 16000, 32000, 48000.
- `num_channels`: Number of audio channels (1 for mono, 2 for stereo).
- `fixed_gain_db`: Constant gain in dB applied after adaptive gain. Default 0.
- `adaptive_digital`: Enable adaptive digital gain. Default True.
- `max_gain_db`: Maximum adaptive gain in dB. Default 50.
- `headroom_db`: Safety margin below 0 dBFS. Default 5.
- `max_gain_change_db_per_second`: Gain slew rate. Default 6.
- `max_output_noise_level_dbfs`: Limits gain to avoid amplifying noise. Default -50.

```python
GainController.process(audio, speech_probability=None) -> np.ndarray
```

Process audio of any length.

- `audio`: Input audio signal (`int16` or `float32` numpy array, any length).
- `speech_probability`: Float 0.0-1.0, optional. If not provided, uses internal VAD.
- Returns: Gained audio (same dtype and length as input).

```python
GainController.reset()
```

Reset internal state (gain estimates, noise/speech levels) while keeping the original configuration.

```python
GainController.gain_db
```

Read-only property. Current applied gain in dB from the most recent `process()` call.

### EchoCanceller

```python
EchoCanceller(
    sample_rate=16000,
    num_channels=1,
    stream_delay_ms=0,
)
```

Create an echo canceller. A high-pass filter is always applied to the capture signal before echo cancellation to remove DC offset (matching Chrome's behavior).

- `sample_rate`: Audio sample rate in Hz. Supported: 16000, 32000, 48000.
- `num_channels`: Number of audio channels (1 for mono, 2 for stereo).
- `stream_delay_ms`: Audio buffer delay hint (see `AudioProcessor` above). Also available as a read/write property.

```python
EchoCanceller.process(near, far) -> np.ndarray
```

Process audio of any length.

- `near`: Microphone capture signal (`int16` or `float32` numpy array, any length).
- `far`: Speaker reference signal (`int16` or `float32` numpy array, same length as near).
- Returns: Cleaned audio with echo removed (same dtype and length as input).

```python
EchoCanceller.reset()
```

Reset internal AEC state while keeping the original configuration.

### NoiseSuppressor

```python
NoiseSuppressor(
    sample_rate=16000,
    num_channels=1,
    level=1,
)
```

Create a noise suppressor.

- `sample_rate`: Audio sample rate in Hz. Supported: 16000, 32000, 48000.
- `num_channels`: Number of audio channels (1 for mono, 2 for stereo).
- `level`: Suppression level 0-3 (6dB, 12dB, 18dB, 21dB). Default: 1 (12dB).

```python
NoiseSuppressor.process(audio) -> np.ndarray
```

Process audio of any length.

- `audio`: Input audio signal (`int16` or `float32` numpy array, any length).
- Returns: Audio with noise suppressed (same dtype and length as input).

```python
NoiseSuppressor.reset()
```

Reset internal noise suppression state while keeping the original configuration.

```python
NoiseSuppressor.speech_probability
```

Read-only property. Speech probability (0.0-1.0) from the most recent `process()` call.

### VoiceDetector

```python
VoiceDetector(
    sample_rate=16000,
    num_channels=1,
)
```

Lightweight voice activity detector. Runs the same spectral analysis as `NoiseSuppressor` to compute speech probability, but skips the Wiener filter - no noise suppression is applied to the audio. Use this when you only need VAD.

- `sample_rate`: Audio sample rate in Hz. Supported: 16000, 32000, 48000.
- `num_channels`: Number of audio channels (1 for mono, 2 for stereo).

```python
VoiceDetector.process(audio) -> float
```

Analyze audio and return speech probability.

- `audio`: Input audio signal (`int16` or `float32` numpy array, any length).
- Returns: Speech probability (0.0-1.0).

```python
VoiceDetector.reset()
```

Reset internal state while keeping the original configuration.
