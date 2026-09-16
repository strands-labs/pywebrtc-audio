#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
using ssize_t = std::ptrdiff_t;
#endif

#include "api/echo_canceller3_factory.h"
#include "api/echo_canceller3_config.h"
#include "api/echo_control.h"
#include "api/audio/audio_processing.h"
#include "api/audio/audio_view.h"
#include "api/field_trials_view.h"
#include "audio_processing/audio_buffer.h"
#include "audio_processing/channel_buffer.h"
#include "audio_processing/audio_frame.h"
#include "audio_processing/high_pass_filter.h"
#include "audio_processing/include/audio_processing.h"
#include "audio_processing/include/audio_util.h"
#include "audio_processing/agc2/adaptive_digital_gain_controller.h"
#include "audio_processing/agc2/agc2_common.h"
#include "audio_processing/agc2/cpu_features.h"
#include "audio_processing/agc2/gain_applier.h"
#include "audio_processing/agc2/limiter.h"
#include "audio_processing/agc2/noise_level_estimator.h"
#include "audio_processing/agc2/saturation_protector.h"
#include "audio_processing/agc2/speech_level_estimator.h"
#include "audio_processing/agc2/vad_wrapper.h"
#include "audio_processing/logging/apm_data_dumper.h"
#include "audio_processing/ns/noise_suppressor.h"
#include "audio_processing/ns/ns_config.h"

namespace py = pybind11;

namespace {

constexpr float kInt16Max = 32767.0f;
constexpr float kClampMin = -32768.0f;
constexpr float kClampMax = 32767.0f;
constexpr int kMinSampleRate = 8000;

/// Selects the native WebRTC processing rate for an external sample rate.
int select_processing_rate(int sample_rate) {
    if (sample_rate <= 16000)
        return 16000;
    if (sample_rate <= 32000)
        return 32000;
    return 48000;
}

// Validate dtype is float32 or int16; returns true if float32
bool validate_audio_dtype(const py::buffer_info& info) {
    if (info.format == py::format_descriptor<float>::format()) return true;
    if (info.format == py::format_descriptor<int16_t>::format()) return false;
    throw std::invalid_argument(
        "audio dtype must be float32 or int16, got format '" + info.format + "'");
}

// Copy interleaved float32 [-1,1] into AudioBuffer channels as floatS16
void float_to_buffer(const float* interleaved, int frame_size, int num_channels,
                     webrtc::AudioBuffer* buf) {
    for (int ch = 0; ch < num_channels; ch++) {
        float* dst = buf->channels()[ch];
        for (int s = 0; s < frame_size; s++)
            dst[s] = std::clamp(interleaved[s * num_channels + ch] * kInt16Max,
                                kClampMin, kClampMax);
    }
}

// Copy AudioBuffer channels (floatS16) back to interleaved float32 [-1,1]
void buffer_to_float(const webrtc::AudioBuffer* buf, int frame_size,
                     int num_channels, float* interleaved) {
    for (int ch = 0; ch < num_channels; ch++) {
        const float* src = buf->channels_const()[ch];
        for (int s = 0; s < frame_size; s++)
            interleaved[s * num_channels + ch] = src[s] / kInt16Max;
    }
}

// Build a validated AEC3 config, overriding delay.num_filters when provided.
// A wider filter bank cancels longer echo paths than the default (5) can model.
webrtc::EchoCanceller3Config build_aec3_config(std::optional<int> num_filters) {
    webrtc::EchoCanceller3Config config;
    if (num_filters.has_value()) {
        if (*num_filters < 1 || *num_filters > 5000)
            throw std::invalid_argument("num_filters must be between 1 and 5000");
        config.delay.num_filters = static_cast<size_t>(*num_filters);
    }
    if (!webrtc::EchoCanceller3Config::Validate(&config))
        throw std::invalid_argument("Invalid AEC3 configuration");
    return config;
}

// AGC2 constants
constexpr int kAgcAdjacentSpeechFramesThreshold = 12;

struct Agc2State {
    std::unique_ptr<webrtc::ApmDataDumper> dumper;
    std::unique_ptr<webrtc::VoiceActivityDetectorWrapper> vad;
    std::unique_ptr<webrtc::NoiseLevelEstimator> noise_estimator;
    std::unique_ptr<webrtc::SpeechLevelEstimator> speech_estimator;
    std::unique_ptr<webrtc::SaturationProtector> saturation_protector;
    std::unique_ptr<webrtc::AdaptiveDigitalGainController> adaptive_ctrl;
    std::unique_ptr<webrtc::Limiter> limiter;
    webrtc::GainApplier fixed_gain_applier;

    webrtc::AudioProcessing::Config::GainController2::AdaptiveDigital ad_config;
    bool adaptive_enabled;
    int sample_rate;
    float last_gain_db = 0.0f;
    float last_speech_probability = 0.0f;

    Agc2State(float fixed_gain_db, bool adaptive, float max_gain_db,
              float headroom_db, float max_change_per_sec,
              float max_noise_dbfs, int sample_rate_, int frame_size)
        : fixed_gain_applier(true, webrtc::DbToRatio(fixed_gain_db)),
          adaptive_enabled(adaptive), sample_rate(sample_rate_) {

        dumper = std::make_unique<webrtc::ApmDataDumper>(0);

        ad_config.enabled = adaptive;
        ad_config.headroom_db = headroom_db;
        ad_config.max_gain_db = max_gain_db;
        ad_config.initial_gain_db = 15.0f;
        ad_config.max_gain_change_db_per_second = max_change_per_sec;
        ad_config.max_output_noise_level_dbfs = max_noise_dbfs;

        webrtc::FieldTrialsView field_trials;
        auto cpu = webrtc::GetAvailableCpuFeatures();

        vad = std::make_unique<webrtc::VoiceActivityDetectorWrapper>(
            cpu, sample_rate_);
        noise_estimator = webrtc::CreateNoiseFloorEstimator(dumper.get());
        speech_estimator = webrtc::SpeechLevelEstimator::Create(
            field_trials, dumper.get(), ad_config,
            kAgcAdjacentSpeechFramesThreshold);
        saturation_protector = webrtc::CreateSaturationProtector(
            ad_config.headroom_db, kAgcAdjacentSpeechFramesThreshold,
            dumper.get());

        if (adaptive) {
            adaptive_ctrl = std::make_unique<webrtc::AdaptiveDigitalGainController>(
                dumper.get(), ad_config, kAgcAdjacentSpeechFramesThreshold);
        }

        limiter = std::make_unique<webrtc::Limiter>(
            dumper.get(), frame_size, "agc2");
    }

    void reset(int frame_size) {
        webrtc::FieldTrialsView field_trials;
        auto cpu = webrtc::GetAvailableCpuFeatures();
        vad = std::make_unique<webrtc::VoiceActivityDetectorWrapper>(
            cpu, sample_rate);
        noise_estimator = webrtc::CreateNoiseFloorEstimator(dumper.get());
        speech_estimator = webrtc::SpeechLevelEstimator::Create(
            field_trials, dumper.get(), ad_config,
            kAgcAdjacentSpeechFramesThreshold);
        saturation_protector = webrtc::CreateSaturationProtector(
            ad_config.headroom_db, kAgcAdjacentSpeechFramesThreshold,
            dumper.get());
        if (adaptive_enabled) {
            adaptive_ctrl = std::make_unique<webrtc::AdaptiveDigitalGainController>(
                dumper.get(), ad_config, kAgcAdjacentSpeechFramesThreshold);
        }
        limiter = std::make_unique<webrtc::Limiter>(
            dumper.get(), frame_size, "agc2");
        last_gain_db = 0.0f;
        last_speech_probability = 0.0f;
    }

    void process(float* data, int len, int num_channels, float external_speech_prob) {
        // Analyze on channel 0 (mono view of first channel)
        webrtc::DeinterleavedView<const float> const_frame(data, len, 1);

        float speech_probability;
        if (external_speech_prob >= 0.0f && external_speech_prob <= 1.0f) {
            speech_probability = external_speech_prob;
        } else {
            speech_probability = vad->Analyze(const_frame);
        }

        float peak = 0.0f;
        for (int i = 0; i < len; i++)
            peak = std::max(peak, std::abs(data[i]));
        float peak_dbfs = webrtc::FloatS16ToDbfs(std::max(peak, 1.0f));

        last_speech_probability = speech_probability;

        float sum_sq = 0.0f;
        for (int i = 0; i < len; i++)
            sum_sq += data[i] * data[i];
        float rms = std::sqrt(sum_sq / len);
        float rms_dbfs = webrtc::FloatS16ToDbfs(std::max(rms, 1.0f));

        float noise_dbfs = noise_estimator->Analyze(const_frame);
        speech_estimator->Update(rms_dbfs, speech_probability);
        saturation_protector->Analyze(speech_probability, peak_dbfs,
                                      speech_estimator->GetLevelDbfs());

        // Apply gain to all channels
        int total = len * num_channels;
        webrtc::DeinterleavedView<float> frame(data, len, num_channels);

        if (adaptive_enabled) {
            webrtc::AdaptiveDigitalGainController::FrameInfo info;
            info.speech_probability = speech_probability;
            info.speech_level_dbfs = speech_estimator->GetLevelDbfs();
            info.speech_level_reliable = speech_estimator->IsConfident();
            info.noise_rms_dbfs = noise_dbfs;
            info.headroom_db = saturation_protector->HeadroomDb();
            info.limiter_envelope_dbfs =
                webrtc::FloatS16ToDbfs(std::max(limiter->LastAudioLevel(), 1.0f));

            adaptive_ctrl->Process(info, frame);
        }

        fixed_gain_applier.ApplyGain(frame);
        limiter->Process(frame);

        float out_peak = 0.0f;
        for (int i = 0; i < total; i++)
            out_peak = std::max(out_peak, std::abs(data[i]));
        if (peak > 1.0f && out_peak > 1.0f)
            last_gain_db = webrtc::FloatS16ToDbfs(out_peak) -
                           webrtc::FloatS16ToDbfs(peak);
    }
};

}  // namespace

class GainController {
    Agc2State agc_;
    int sample_rate_;
    int num_channels_;
    int frame_size_;      // samples per channel per 10ms
    int frame_stride_;    // total interleaved samples per frame

public:
    GainController(int sample_rate = 16000, int num_channels = 1,
                   float fixed_gain_db = 0.0f, bool adaptive_digital = true,
                   float max_gain_db = 50.0f, float headroom_db = 5.0f,
                   float max_gain_change_db_per_second = 6.0f,
                   float max_output_noise_level_dbfs = -50.0f)
        : agc_(fixed_gain_db, adaptive_digital, max_gain_db, headroom_db,
               max_gain_change_db_per_second, max_output_noise_level_dbfs,
               sample_rate, sample_rate / 100),
          sample_rate_(sample_rate), num_channels_(num_channels) {

        if (sample_rate != 16000 && sample_rate != 32000 && sample_rate != 48000)
            throw std::invalid_argument("sample_rate must be 16000, 32000, or 48000");
        if (num_channels < 1)
            throw std::invalid_argument("num_channels must be >= 1");

        frame_size_ = sample_rate / 100;
        frame_stride_ = frame_size_ * num_channels;
    }

    py::object process(py::array audio_arr,
                       std::optional<float> speech_probability) {
        auto info = audio_arr.request();
        if (info.ndim != 1)
            throw std::invalid_argument("audio must be a 1-D array");

        auto original_len = info.shape[0];
        if (original_len == 0)
            throw std::invalid_argument("audio must not be empty");

        bool input_float = validate_audio_dtype(info);

        if (speech_probability.has_value()) {
            float sp_val = speech_probability.value();
            if (sp_val < 0.0f || sp_val > 1.0f)
                throw std::invalid_argument(
                    "speech_probability must be between 0.0 and 1.0");
        }
        float sp = speech_probability.has_value() ? speech_probability.value() : -1.0f;

        auto padded_len = ((original_len + frame_stride_ - 1) / frame_stride_) * frame_stride_;

        if (input_float) {
            std::vector<float> padded(padded_len, 0.0f);
            const float* src = static_cast<float*>(info.ptr);
            for (ssize_t i = 0; i < original_len; i++)
                padded[i] = std::clamp(src[i] * kInt16Max, kClampMin, kClampMax);

            auto result = py::array_t<float>(original_len);
            auto* out_ptr = static_cast<float*>(result.request().ptr);

            {
                py::gil_scoped_release release;
                std::vector<float> deinterleaved(frame_size_ * num_channels_);
                for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                    // Deinterleave: [L0,R0,L1,R1,...] -> [L0,L1,...,R0,R1,...]
                    for (int ch = 0; ch < num_channels_; ch++)
                        for (int s = 0; s < frame_size_; s++)
                            deinterleaved[ch * frame_size_ + s] = padded[offset + s * num_channels_ + ch];

                    agc_.process(deinterleaved.data(), frame_size_, num_channels_, sp);

                    // Re-interleave
                    for (int ch = 0; ch < num_channels_; ch++)
                        for (int s = 0; s < frame_size_; s++)
                            padded[offset + s * num_channels_ + ch] = deinterleaved[ch * frame_size_ + s];

                    auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                             original_len - offset);
                    if (copy_len > 0)
                        for (ssize_t i = 0; i < copy_len; i++)
                            out_ptr[offset + i] = padded[offset + i] / kInt16Max;
                }
            }
            return std::move(result);
        }

        std::vector<int16_t> padded(padded_len, 0);
        std::memcpy(padded.data(), info.ptr, original_len * sizeof(int16_t));

        auto result = py::array_t<int16_t>(original_len);
        auto* out_ptr = static_cast<int16_t*>(result.request().ptr);

        {
            py::gil_scoped_release release;

            std::vector<float> deinterleaved(frame_size_ * num_channels_);
            for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                // Deinterleave int16 -> floatS16
                for (int ch = 0; ch < num_channels_; ch++)
                    for (int s = 0; s < frame_size_; s++)
                        deinterleaved[ch * frame_size_ + s] =
                            static_cast<float>(padded[offset + s * num_channels_ + ch]);

                agc_.process(deinterleaved.data(), frame_size_, num_channels_, sp);

                // Re-interleave floatS16 -> int16
                for (int ch = 0; ch < num_channels_; ch++)
                    for (int s = 0; s < frame_size_; s++)
                        padded[offset + s * num_channels_ + ch] =
                            webrtc::FloatS16ToS16(deinterleaved[ch * frame_size_ + s]);

                auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                         original_len - offset);
                if (copy_len > 0)
                    std::memcpy(out_ptr + offset, padded.data() + offset,
                                copy_len * sizeof(int16_t));
            }
        }
        return std::move(result);
    }

    void reset() {
        agc_.reset(frame_size_);
    }

    float get_gain_db() const { return agc_.last_gain_db; }
    float get_speech_probability() const { return agc_.last_speech_probability; }
};

class EchoCanceller {
    std::unique_ptr<webrtc::EchoControl> aec_;
    std::unique_ptr<webrtc::AudioBuffer> near_buf_;
    std::unique_ptr<webrtc::AudioBuffer> far_buf_;
    std::unique_ptr<webrtc::HighPassFilter> hp_filter_;
    webrtc::EchoCanceller3Config config_;
    int sample_rate_;
    int num_channels_;
    int frame_size_;      // samples per channel per 10ms frame
    int frame_stride_;    // total interleaved samples per frame (frame_size_ * num_channels_)

    void process_frame(const int16_t* near_ptr, const int16_t* far_ptr,
                       int16_t* out_ptr) {
        webrtc::AudioFrame far_frame, near_frame;
        far_frame.UpdateFrame(0, far_ptr, frame_size_, sample_rate_,
                              webrtc::AudioFrame::kNormalSpeech,
                              webrtc::AudioFrame::kVadActive, num_channels_);
        near_frame.UpdateFrame(0, near_ptr, frame_size_, sample_rate_,
                               webrtc::AudioFrame::kNormalSpeech,
                               webrtc::AudioFrame::kVadActive, num_channels_);

        far_buf_->CopyFrom(&far_frame);
        near_buf_->CopyFrom(&near_frame);

        process_buffers();

        near_buf_->CopyTo(&near_frame);
        std::memcpy(out_ptr, near_frame.data(), frame_stride_ * sizeof(int16_t));
    }

    void process_frame_float(const float* near_ptr, const float* far_ptr,
                             float* out_ptr) {
        float_to_buffer(far_ptr, frame_size_, num_channels_, far_buf_.get());
        float_to_buffer(near_ptr, frame_size_, num_channels_, near_buf_.get());

        process_buffers();

        buffer_to_float(near_buf_.get(), frame_size_, num_channels_, out_ptr);
    }

    void process_buffers() {
        far_buf_->SplitIntoFrequencyBands();
        aec_->AnalyzeRender(far_buf_.get());
        far_buf_->MergeFrequencyBands();

        aec_->AnalyzeCapture(near_buf_.get());
        near_buf_->SplitIntoFrequencyBands();
        hp_filter_->Process(near_buf_.get(), true);
        aec_->SetAudioBufferDelay(stream_delay_ms_);
        aec_->ProcessCapture(near_buf_.get(), nullptr, false);
        near_buf_->MergeFrequencyBands();
    }

public:
    int stream_delay_ms_ = 0;

    EchoCanceller(int sample_rate = 16000, int num_channels = 1,
                  int stream_delay_ms = 0,
                  std::optional<int> num_filters = std::nullopt)
        : sample_rate_(sample_rate), num_channels_(num_channels),
          stream_delay_ms_(stream_delay_ms) {

        if (sample_rate != 16000 && sample_rate != 32000 && sample_rate != 48000) {
            throw std::invalid_argument(
                "sample_rate must be 16000, 32000, or 48000");
        }
        if (num_channels < 1) {
            throw std::invalid_argument("num_channels must be >= 1");
        }

        frame_size_ = sample_rate / 100;
        frame_stride_ = frame_size_ * num_channels;

        config_ = build_aec3_config(num_filters);
        webrtc::EchoCanceller3Factory factory(config_);
        aec_ = factory.Create(sample_rate, num_channels, num_channels);

        hp_filter_ = std::make_unique<webrtc::HighPassFilter>(
            sample_rate, num_channels);

        near_buf_ = std::make_unique<webrtc::AudioBuffer>(
            sample_rate, num_channels,
            sample_rate, num_channels,
            sample_rate, num_channels);

        far_buf_ = std::make_unique<webrtc::AudioBuffer>(
            sample_rate, num_channels,
            sample_rate, num_channels,
            sample_rate, num_channels);
    }

    py::object process(py::array near_arr, py::array far_arr) {
        auto near_info = near_arr.request();
        auto far_info = far_arr.request();

        if (near_info.ndim != 1 || far_info.ndim != 1)
            throw std::invalid_argument("near and far must be 1-D arrays");

        auto original_len = near_info.shape[0];
        if (original_len == 0)
            throw std::invalid_argument("near must not be empty");
        if (far_info.shape[0] != original_len)
            throw std::invalid_argument("near and far must have the same length");

        bool input_float = validate_audio_dtype(near_info);
        bool far_float = validate_audio_dtype(far_info);
        if (input_float != far_float)
            throw std::invalid_argument("near and far must have the same dtype");
        auto padded_len = ((original_len + frame_stride_ - 1) / frame_stride_) * frame_stride_;

        if (input_float) {
            std::vector<float> near_padded(padded_len, 0.0f);
            std::vector<float> far_padded(padded_len, 0.0f);
            std::memcpy(near_padded.data(), near_info.ptr, original_len * sizeof(float));
            std::memcpy(far_padded.data(), far_info.ptr, original_len * sizeof(float));

            auto result = py::array_t<float>(original_len);
            auto* out_ptr = static_cast<float*>(result.request().ptr);

            {
                py::gil_scoped_release release;
                std::vector<float> frame_out(frame_stride_);
                for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                    process_frame_float(near_padded.data() + offset,
                                        far_padded.data() + offset,
                                        frame_out.data());
                    auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                             original_len - offset);
                    if (copy_len > 0)
                        std::memcpy(out_ptr + offset, frame_out.data(),
                                    copy_len * sizeof(float));
                }
            }
            return std::move(result);
        }

        std::vector<int16_t> near_padded(padded_len, 0);
        std::vector<int16_t> far_padded(padded_len, 0);
        std::memcpy(near_padded.data(), near_info.ptr, original_len * sizeof(int16_t));
        std::memcpy(far_padded.data(), far_info.ptr, original_len * sizeof(int16_t));

        auto result = py::array_t<int16_t>(original_len);
        auto* out_ptr = static_cast<int16_t*>(result.request().ptr);

        {
            py::gil_scoped_release release;
            std::vector<int16_t> frame_out(frame_stride_);
            for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                process_frame(near_padded.data() + offset,
                              far_padded.data() + offset,
                              frame_out.data());
                auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                         original_len - offset);
                if (copy_len > 0)
                    std::memcpy(out_ptr + offset, frame_out.data(),
                                copy_len * sizeof(int16_t));
            }
        }
        return std::move(result);
    }

    void reset() {
        webrtc::EchoCanceller3Factory factory(config_);
        aec_ = factory.Create(sample_rate_, num_channels_, num_channels_);
        hp_filter_ = std::make_unique<webrtc::HighPassFilter>(
            sample_rate_, num_channels_);
    }

    int get_stream_delay_ms() const { return stream_delay_ms_; }

    void set_stream_delay_ms(int delay_ms) {
        if (delay_ms < 0)
            throw std::invalid_argument("stream_delay_ms must be >= 0");
        stream_delay_ms_ = delay_ms;
    }
};

class NoiseSuppressor {
    std::unique_ptr<webrtc::NoiseSuppressor> ns_;
    std::unique_ptr<webrtc::AudioBuffer> buf_;
    int sample_rate_;
    int num_channels_;
    int frame_size_;
    int frame_stride_;
    int level_;

    void process_frame(const int16_t* in_ptr, int16_t* out_ptr) {
        webrtc::AudioFrame frame;
        frame.UpdateFrame(0, in_ptr, frame_size_, sample_rate_,
                          webrtc::AudioFrame::kNormalSpeech,
                          webrtc::AudioFrame::kVadActive, num_channels_);

        buf_->CopyFrom(&frame);
        process_buffer();
        buf_->CopyTo(&frame);
        std::memcpy(out_ptr, frame.data(), frame_stride_ * sizeof(int16_t));
    }

    void process_frame_float(const float* in_ptr, float* out_ptr) {
        float_to_buffer(in_ptr, frame_size_, num_channels_, buf_.get());
        process_buffer();
        buffer_to_float(buf_.get(), frame_size_, num_channels_, out_ptr);
    }

    void process_buffer() {
        buf_->SplitIntoFrequencyBands();
        ns_->Analyze(*buf_);
        ns_->Process(buf_.get());
        buf_->MergeFrequencyBands();
    }

public:
    NoiseSuppressor(int sample_rate = 16000, int num_channels = 1, int level = 1)
        : sample_rate_(sample_rate), num_channels_(num_channels), level_(level) {

        if (sample_rate != 16000 && sample_rate != 32000 && sample_rate != 48000) {
            throw std::invalid_argument(
                "sample_rate must be 16000, 32000, or 48000");
        }
        if (num_channels < 1) {
            throw std::invalid_argument("num_channels must be >= 1");
        }
        if (level < 0 || level > 3) {
            throw std::invalid_argument("level must be 0-3 (6/12/18/21 dB)");
        }

        frame_size_ = sample_rate / 100;
        frame_stride_ = frame_size_ * num_channels;

        webrtc::NsConfig config;
        config.target_level = static_cast<webrtc::NsConfig::SuppressionLevel>(level);

        ns_ = std::make_unique<webrtc::NoiseSuppressor>(
            config, sample_rate, num_channels);

        buf_ = std::make_unique<webrtc::AudioBuffer>(
            sample_rate, num_channels,
            sample_rate, num_channels,
            sample_rate, num_channels);
    }

    py::object process(py::array audio_arr) {
        auto info = audio_arr.request();

        if (info.ndim != 1)
            throw std::invalid_argument("audio must be a 1-D array");

        auto original_len = info.shape[0];
        if (original_len == 0)
            throw std::invalid_argument("audio must not be empty");

        bool input_float = validate_audio_dtype(info);
        auto padded_len = ((original_len + frame_stride_ - 1) / frame_stride_) * frame_stride_;

        if (input_float) {
            std::vector<float> padded(padded_len, 0.0f);
            std::memcpy(padded.data(), info.ptr, original_len * sizeof(float));

            auto result = py::array_t<float>(original_len);
            auto* out_ptr = static_cast<float*>(result.request().ptr);

            {
                py::gil_scoped_release release;
                std::vector<float> frame_out(frame_stride_);
                for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                    process_frame_float(padded.data() + offset, frame_out.data());
                    auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                             original_len - offset);
                    if (copy_len > 0)
                        std::memcpy(out_ptr + offset, frame_out.data(),
                                    copy_len * sizeof(float));
                }
            }
            return std::move(result);
        }

        std::vector<int16_t> padded(padded_len, 0);
        std::memcpy(padded.data(), info.ptr, original_len * sizeof(int16_t));

        auto result = py::array_t<int16_t>(original_len);
        auto* out_ptr = static_cast<int16_t*>(result.request().ptr);

        {
            py::gil_scoped_release release;
            std::vector<int16_t> frame_out(frame_stride_);
            for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                process_frame(padded.data() + offset, frame_out.data());
                auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                         original_len - offset);
                if (copy_len > 0)
                    std::memcpy(out_ptr + offset, frame_out.data(),
                                copy_len * sizeof(int16_t));
            }
        }
        return std::move(result);
    }

    void reset() {
        webrtc::NsConfig config;
        config.target_level =
            static_cast<webrtc::NsConfig::SuppressionLevel>(level_);
        ns_ = std::make_unique<webrtc::NoiseSuppressor>(
            config, sample_rate_, num_channels_);
    }

    float get_speech_probability() const {
        return ns_->GetSpeechProbability();
    }
};

class VoiceDetector {
    std::unique_ptr<webrtc::NoiseSuppressor> ns_;
    std::unique_ptr<webrtc::AudioBuffer> buf_;
    int sample_rate_;
    int num_channels_;
    int frame_size_;
    int frame_stride_;
    float last_speech_probability_ = 0.0f;

    void analyze_frame(const int16_t* in_ptr) {
        webrtc::AudioFrame frame;
        frame.UpdateFrame(0, in_ptr, frame_size_, sample_rate_,
                          webrtc::AudioFrame::kNormalSpeech,
                          webrtc::AudioFrame::kVadActive, num_channels_);

        buf_->CopyFrom(&frame);
        buf_->SplitIntoFrequencyBands();
        ns_->Analyze(*buf_);
    }

    void analyze_frame_float(const float* in_ptr) {
        float_to_buffer(in_ptr, frame_size_, num_channels_, buf_.get());
        buf_->SplitIntoFrequencyBands();
        ns_->Analyze(*buf_);
    }

public:
    VoiceDetector(int sample_rate = 16000, int num_channels = 1)
        : sample_rate_(sample_rate), num_channels_(num_channels) {

        if (sample_rate != 16000 && sample_rate != 32000 && sample_rate != 48000)
            throw std::invalid_argument(
                "sample_rate must be 16000, 32000, or 48000");
        if (num_channels < 1)
            throw std::invalid_argument("num_channels must be >= 1");

        frame_size_ = sample_rate / 100;
        frame_stride_ = frame_size_ * num_channels;

        webrtc::NsConfig config;
        ns_ = std::make_unique<webrtc::NoiseSuppressor>(
            config, sample_rate, num_channels);

        buf_ = std::make_unique<webrtc::AudioBuffer>(
            sample_rate, num_channels,
            sample_rate, num_channels,
            sample_rate, num_channels);
    }

    float process(py::array audio_arr) {
        auto info = audio_arr.request();

        if (info.ndim != 1)
            throw std::invalid_argument("audio must be a 1-D array");

        auto original_len = info.shape[0];
        if (original_len == 0)
            throw std::invalid_argument("audio must not be empty");

        bool input_float = validate_audio_dtype(info);
        auto padded_len = ((original_len + frame_stride_ - 1) / frame_stride_) * frame_stride_;

        if (input_float) {
            std::vector<float> padded(padded_len, 0.0f);
            std::memcpy(padded.data(), info.ptr, original_len * sizeof(float));

            {
                py::gil_scoped_release release;
                for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_)
                    analyze_frame_float(padded.data() + offset);
            }
        } else {
            std::vector<int16_t> padded(padded_len, 0);
            std::memcpy(padded.data(), info.ptr, original_len * sizeof(int16_t));

            {
                py::gil_scoped_release release;
                for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_)
                    analyze_frame(padded.data() + offset);
            }
        }

        last_speech_probability_ = ns_->GetSpeechProbability();
        return last_speech_probability_;
    }

    void reset() {
        webrtc::NsConfig config;
        ns_ = std::make_unique<webrtc::NoiseSuppressor>(
            config, sample_rate_, num_channels_);
        last_speech_probability_ = 0.0f;
    }

    float get_speech_probability() const { return last_speech_probability_; }
};

class AudioProcessor {
    std::unique_ptr<webrtc::EchoControl> aec_;
    std::unique_ptr<webrtc::NoiseSuppressor> ns_;
    std::unique_ptr<webrtc::HighPassFilter> hp_filter_;
    std::unique_ptr<webrtc::AudioBuffer> near_buf_;
    std::unique_ptr<webrtc::AudioBuffer> far_buf_;
    int sample_rate_;
    // WebRTC DSP runs at a native rate while AudioBuffer resamples external audio.
    int processing_rate_;
    int num_channels_;
    int frame_size_;
    int processing_frame_size_;
    int frame_stride_;
    bool aec_enabled_;
    bool ns_enabled_;
    bool hp_enabled_;
    bool agc_enabled_;
    int ns_level_;
    int stream_delay_ms_ = 0;
    webrtc::EchoCanceller3Config aec3_config_;
    std::unique_ptr<Agc2State> agc_;
    std::unique_ptr<webrtc::NoiseSuppressor> vad_;
    // Reused to bridge interleaved multichannel audio and WebRTC's planar API.
    std::unique_ptr<webrtc::ChannelBuffer<float>> resampling_scratch_;

    /// Creates a buffer that resamples between the external and processing rates.
    std::unique_ptr<webrtc::AudioBuffer> create_audio_buffer() const {
        return std::make_unique<webrtc::AudioBuffer>(
            sample_rate_, num_channels_,
            processing_rate_, num_channels_,
            sample_rate_, num_channels_);
    }

    void process_frame(const int16_t* near_ptr, const int16_t* far_ptr,
                       int16_t* out_ptr) {
        webrtc::AudioFrame near_frame;
        near_frame.UpdateFrame(0, near_ptr, frame_size_, sample_rate_,
                               webrtc::AudioFrame::kNormalSpeech,
                               webrtc::AudioFrame::kVadActive, num_channels_);
        near_buf_->CopyFrom(&near_frame);

        if (aec_enabled_) {
            webrtc::AudioFrame far_frame;
            far_frame.UpdateFrame(0, far_ptr, frame_size_, sample_rate_,
                                  webrtc::AudioFrame::kNormalSpeech,
                                  webrtc::AudioFrame::kVadActive, num_channels_);
            far_buf_->CopyFrom(&far_frame);
        }

        process_buffers();

        near_buf_->CopyTo(&near_frame);
        std::memcpy(out_ptr, near_frame.data(), frame_stride_ * sizeof(int16_t));
    }

    void process_frame_float(const float* near_ptr, const float* far_ptr,
                             float* out_ptr) {
        if (sample_rate_ == processing_rate_) {
            float_to_buffer(
                near_ptr, frame_size_, num_channels_, near_buf_.get());
            if (aec_enabled_)
                float_to_buffer(
                    far_ptr, frame_size_, num_channels_, far_buf_.get());

            process_buffers();

            buffer_to_float(
                near_buf_.get(), frame_size_, num_channels_, out_ptr);
            return;
        }

        resample_float_to_buffer(near_ptr, near_buf_.get());
        if (aec_enabled_)
            resample_float_to_buffer(far_ptr, far_buf_.get());

        process_buffers();

        resample_buffer_to_float(near_buf_.get(), out_ptr);
    }

    /// Copies interleaved float input into an AudioBuffer at the processing rate.
    void resample_float_to_buffer(const float* interleaved,
                                  webrtc::AudioBuffer* buffer) {
        webrtc::StreamConfig config(sample_rate_, num_channels_);
        if (num_channels_ == 1) {
            const float* channels[] = {interleaved};
            buffer->CopyFrom(channels, config);
            return;
        }

        webrtc::Deinterleave(
            interleaved, frame_size_, num_channels_, resampling_scratch_->channels());
        const auto& scratch = *resampling_scratch_;
        buffer->CopyFrom(scratch.channels(), config);
    }

    /// Copies processed audio to interleaved float output at the external rate.
    void resample_buffer_to_float(webrtc::AudioBuffer* buffer,
                                  float* interleaved) {
        webrtc::StreamConfig config(sample_rate_, num_channels_);
        if (num_channels_ == 1) {
            float* channels[] = {interleaved};
            buffer->CopyTo(config, channels);
            return;
        }

        buffer->CopyTo(config, resampling_scratch_->channels());
        const auto& scratch = *resampling_scratch_;
        webrtc::Interleave(
            scratch.channels(), frame_size_, num_channels_, interleaved);
    }

    void process_buffers() {
        if (aec_enabled_) {
            far_buf_->SplitIntoFrequencyBands();
            aec_->AnalyzeRender(far_buf_.get());
            aec_->AnalyzeCapture(near_buf_.get());
        }

        if (ns_enabled_)
            ns_->Analyze(*near_buf_);

        if (vad_)
            vad_->Analyze(*near_buf_);

        near_buf_->SplitIntoFrequencyBands();

        if (hp_filter_)
            hp_filter_->Process(near_buf_.get(), true);

        if (aec_enabled_) {
            aec_->SetAudioBufferDelay(stream_delay_ms_);
            aec_->ProcessCapture(near_buf_.get(), nullptr, false);
        }

        if (ns_enabled_)
            ns_->Process(near_buf_.get());

        near_buf_->MergeFrequencyBands();

        if (agc_enabled_) {
            float sp = ns_enabled_ ? ns_->GetSpeechProbability() : -1.0f;
            if (num_channels_ == 1) {
                agc_->process(
                    near_buf_->channels()[0], processing_frame_size_, 1, sp);
            } else {
                std::vector<float> contiguous(
                    processing_frame_size_ * num_channels_);
                for (int ch = 0; ch < num_channels_; ch++)
                    std::memcpy(
                        contiguous.data() + ch * processing_frame_size_,
                        near_buf_->channels()[ch],
                        processing_frame_size_ * sizeof(float));
                agc_->process(contiguous.data(), processing_frame_size_,
                              num_channels_, sp);
                for (int ch = 0; ch < num_channels_; ch++)
                    std::memcpy(near_buf_->channels()[ch],
                                contiguous.data() + ch * processing_frame_size_,
                                processing_frame_size_ * sizeof(float));
            }
        }
    }

public:
    AudioProcessor(int sample_rate = 16000, int num_channels = 1,
                   bool echo_cancellation = false,
                   bool noise_suppression = false,
                   bool high_pass_filter = false,
                   bool auto_gain_control = false,
                   int ns_level = 1,
                   float agc_gain_db = 0.0f,
                   float agc_max_gain_db = 50.0f,
                   int stream_delay_ms = 0,
                   std::optional<int> num_filters = std::nullopt)
        : sample_rate_(sample_rate),
          processing_rate_(select_processing_rate(sample_rate)),
          num_channels_(num_channels),
          aec_enabled_(echo_cancellation), ns_enabled_(noise_suppression),
          hp_enabled_(high_pass_filter), agc_enabled_(auto_gain_control),
          ns_level_(ns_level), stream_delay_ms_(stream_delay_ms) {

        if (sample_rate < kMinSampleRate ||
            sample_rate > static_cast<int>(webrtc::AudioBuffer::kMaxSampleRate))
            throw std::invalid_argument(
                "sample_rate must be between 8000 and 384000");
        if (num_channels < 1)
            throw std::invalid_argument("num_channels must be >= 1");
        if (ns_level < 0 || ns_level > 3)
            throw std::invalid_argument("ns_level must be 0-3");
        if (stream_delay_ms < 0)
            throw std::invalid_argument("stream_delay_ms must be >= 0");
        aec3_config_ = build_aec3_config(num_filters);

        frame_size_ = sample_rate / 100;
        processing_frame_size_ = processing_rate_ / 100;
        frame_stride_ = frame_size_ * num_channels;

        if (sample_rate_ != processing_rate_ && num_channels_ > 1) {
            resampling_scratch_ = std::make_unique<webrtc::ChannelBuffer<float>>(
                frame_size_, num_channels_);
        }

        near_buf_ = create_audio_buffer();

        if (aec_enabled_) {
            webrtc::EchoCanceller3Factory factory(aec3_config_);
            aec_ = factory.Create(
                processing_rate_, num_channels, num_channels);
            far_buf_ = create_audio_buffer();
        }

        if (hp_enabled_ || aec_enabled_) {
            hp_filter_ = std::make_unique<webrtc::HighPassFilter>(
                processing_rate_, num_channels);
        }

        if (ns_enabled_) {
            webrtc::NsConfig ns_config;
            ns_config.target_level =
                static_cast<webrtc::NsConfig::SuppressionLevel>(ns_level);
            ns_ = std::make_unique<webrtc::NoiseSuppressor>(
                ns_config, processing_rate_, num_channels);
        }

        if (agc_enabled_) {
            agc_ = std::make_unique<Agc2State>(
                agc_gain_db, true, agc_max_gain_db, 5.0f, 6.0f, -50.0f,
                processing_rate_, processing_frame_size_);
        }

        if (!ns_enabled_ && !agc_enabled_) {
            webrtc::NsConfig vad_config;
            vad_ = std::make_unique<webrtc::NoiseSuppressor>(
                vad_config, processing_rate_, num_channels);
        }
    }

    py::object process(py::array near_arr,
                       std::optional<py::array> far_arr) {
        auto near_info = near_arr.request();
        if (near_info.ndim != 1)
            throw std::invalid_argument("near must be a 1-D array");

        auto original_len = near_info.shape[0];
        if (original_len == 0)
            throw std::invalid_argument("near must not be empty");

        if (aec_enabled_ && !far_arr.has_value())
            throw std::invalid_argument(
                "far is required when echo_cancellation is enabled");

        bool input_float = validate_audio_dtype(near_info);

        if (far_arr.has_value()) {
            auto far_info = far_arr->request();
            if (far_info.ndim != 1)
                throw std::invalid_argument("far must be a 1-D array");
            if (far_info.shape[0] != original_len)
                throw std::invalid_argument("near and far must have the same length");
            bool far_float = validate_audio_dtype(far_info);
            if (input_float != far_float)
                throw std::invalid_argument("near and far must have the same dtype");
        }

        auto padded_len = ((original_len + frame_stride_ - 1) / frame_stride_) * frame_stride_;

        if (input_float) {
            std::vector<float> near_padded(padded_len, 0.0f);
            std::memcpy(near_padded.data(), near_info.ptr, original_len * sizeof(float));

            std::vector<float> far_padded;
            if (aec_enabled_) {
                far_padded.resize(padded_len, 0.0f);
                auto far_info = far_arr->request();
                std::memcpy(far_padded.data(), far_info.ptr, original_len * sizeof(float));
            }

            auto result = py::array_t<float>(original_len);
            auto* out_ptr = static_cast<float*>(result.request().ptr);

            {
                py::gil_scoped_release release;
                std::vector<float> frame_out(frame_stride_);
                for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                    process_frame_float(
                        near_padded.data() + offset,
                        aec_enabled_ ? far_padded.data() + offset : nullptr,
                        frame_out.data());
                    auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                             original_len - offset);
                    if (copy_len > 0)
                        std::memcpy(out_ptr + offset, frame_out.data(),
                                    copy_len * sizeof(float));
                }
            }
            return std::move(result);
        }

        std::vector<int16_t> near_padded(padded_len, 0);
        std::memcpy(near_padded.data(), near_info.ptr, original_len * sizeof(int16_t));

        std::vector<int16_t> far_padded;
        if (aec_enabled_) {
            far_padded.resize(padded_len, 0);
            auto far_info = far_arr->request();
            std::memcpy(far_padded.data(), far_info.ptr, original_len * sizeof(int16_t));
        }

        auto result = py::array_t<int16_t>(original_len);
        auto* out_ptr = static_cast<int16_t*>(result.request().ptr);

        {
            py::gil_scoped_release release;
            std::vector<int16_t> frame_out(frame_stride_);
            for (ssize_t offset = 0; offset < padded_len; offset += frame_stride_) {
                process_frame(near_padded.data() + offset,
                              aec_enabled_ ? far_padded.data() + offset : nullptr,
                              frame_out.data());
                auto copy_len = std::min(static_cast<ssize_t>(frame_stride_),
                                         original_len - offset);
                if (copy_len > 0)
                    std::memcpy(out_ptr + offset, frame_out.data(),
                                copy_len * sizeof(int16_t));
            }
        }
        return std::move(result);
    }

    int get_stream_delay_ms() const { return stream_delay_ms_; }

    void set_stream_delay_ms(int delay_ms) {
        if (delay_ms < 0)
            throw std::invalid_argument("stream_delay_ms must be >= 0");
        stream_delay_ms_ = delay_ms;
    }

    void reset() {
        near_buf_ = create_audio_buffer();

        if (aec_enabled_) {
            far_buf_ = create_audio_buffer();
            webrtc::EchoCanceller3Factory factory(aec3_config_);
            aec_ = factory.Create(
                processing_rate_, num_channels_, num_channels_);
        }

        if (hp_enabled_ || aec_enabled_) {
            hp_filter_ = std::make_unique<webrtc::HighPassFilter>(
                processing_rate_, num_channels_);
        }

        if (ns_enabled_) {
            webrtc::NsConfig ns_config;
            ns_config.target_level =
                static_cast<webrtc::NsConfig::SuppressionLevel>(ns_level_);
            ns_ = std::make_unique<webrtc::NoiseSuppressor>(
                ns_config, processing_rate_, num_channels_);
        }

        if (agc_enabled_) {
            agc_->reset(processing_frame_size_);
        }

        if (vad_) {
            webrtc::NsConfig vad_config;
            vad_ = std::make_unique<webrtc::NoiseSuppressor>(
                vad_config, processing_rate_, num_channels_);
        }
    }

    float get_speech_probability() const {
        if (ns_enabled_)
            return ns_->GetSpeechProbability();
        if (agc_enabled_)
            return agc_->last_speech_probability;
        return vad_->GetSpeechProbability();
    }

    float get_gain_db() const {
        if (!agc_enabled_)
            throw std::runtime_error(
                "gain_db requires auto_gain_control=True");
        return agc_->last_gain_db;
    }
};

PYBIND11_MODULE(_webrtc_audio, m) {
    m.doc() = "WebRTC audio processing: echo cancellation, noise suppression, and combined pipeline";

    py::class_<EchoCanceller>(m, "EchoCanceller")
        .def(py::init<int, int, int, std::optional<int>>(),
             py::arg("sample_rate") = 16000,
             py::arg("num_channels") = 1,
             py::arg("stream_delay_ms") = 0,
             py::arg("num_filters") = py::none())
        .def("process", &EchoCanceller::process,
             py::arg("near"), py::arg("far"))
        .def("reset", &EchoCanceller::reset)
        .def_property("stream_delay_ms",
                       &EchoCanceller::get_stream_delay_ms,
                       &EchoCanceller::set_stream_delay_ms);

    py::class_<NoiseSuppressor>(m, "NoiseSuppressor")
        .def(py::init<int, int, int>(),
             py::arg("sample_rate") = 16000,
             py::arg("num_channels") = 1,
             py::arg("level") = 1)
        .def("process", &NoiseSuppressor::process,
             py::arg("audio"))
        .def("reset", &NoiseSuppressor::reset)
        .def_property_readonly("speech_probability",
                               &NoiseSuppressor::get_speech_probability);

    py::class_<VoiceDetector>(m, "VoiceDetector")
        .def(py::init<int, int>(),
             py::arg("sample_rate") = 16000,
             py::arg("num_channels") = 1)
        .def("process", &VoiceDetector::process,
             py::arg("audio"))
        .def("reset", &VoiceDetector::reset)
        .def_property_readonly("speech_probability",
                               &VoiceDetector::get_speech_probability);

    py::class_<GainController>(m, "GainController")
        .def(py::init<int, int, float, bool, float, float, float, float>(),
             py::arg("sample_rate") = 16000,
             py::arg("num_channels") = 1,
             py::arg("fixed_gain_db") = 0.0f,
             py::arg("adaptive_digital") = true,
             py::arg("max_gain_db") = 50.0f,
             py::arg("headroom_db") = 5.0f,
             py::arg("max_gain_change_db_per_second") = 6.0f,
             py::arg("max_output_noise_level_dbfs") = -50.0f)
        .def("process", &GainController::process,
             py::arg("audio"), py::arg("speech_probability") = py::none())
        .def("reset", &GainController::reset)
        .def_property_readonly("gain_db", &GainController::get_gain_db)
        .def_property_readonly("speech_probability",
                               &GainController::get_speech_probability);

    py::class_<AudioProcessor>(m, "AudioProcessor")
        .def(py::init<int, int, bool, bool, bool, bool, int, float, float, int,
                      std::optional<int>>(),
             py::arg("sample_rate") = 16000,
             py::arg("num_channels") = 1,
             py::arg("echo_cancellation") = false,
             py::arg("noise_suppression") = false,
             py::arg("high_pass_filter") = false,
             py::arg("auto_gain_control") = false,
             py::arg("ns_level") = 1,
             py::arg("agc_gain_db") = 0.0f,
             py::arg("agc_max_gain_db") = 50.0f,
             py::arg("stream_delay_ms") = 0,
             py::arg("num_filters") = py::none())
        .def("process", &AudioProcessor::process,
             py::arg("near"), py::arg("far") = py::none())
        .def("reset", &AudioProcessor::reset)
        .def_property("stream_delay_ms",
                       &AudioProcessor::get_stream_delay_ms,
                       &AudioProcessor::set_stream_delay_ms)
        .def_property_readonly("speech_probability",
                               &AudioProcessor::get_speech_probability)
        .def_property_readonly("gain_db",
                               &AudioProcessor::get_gain_db);
}
