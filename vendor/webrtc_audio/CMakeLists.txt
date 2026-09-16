cmake_minimum_required(VERSION 3.15)

set(WEBRTC_AUDIO_SOURCES
    api/echo_canceller3_config.cc
    api/echo_canceller3_factory.cc
    audio_processing/aec3/adaptive_fir_filter_erl.cc
    audio_processing/aec3/adaptive_fir_filter.cc
    audio_processing/aec3/aec_state.cc
    audio_processing/aec3/aec3_common.cc
    audio_processing/aec3/aec3_fft.cc
    audio_processing/aec3/alignment_mixer.cc
    audio_processing/aec3/api_call_jitter_metrics.cc
    audio_processing/aec3/block_buffer.cc
    audio_processing/aec3/block_delay_buffer.cc
    audio_processing/aec3/block_framer.cc
    audio_processing/aec3/block_processor_metrics.cc
    audio_processing/aec3/block_processor.cc
    audio_processing/aec3/clockdrift_detector.cc
    audio_processing/aec3/comfort_noise_generator.cc
    audio_processing/aec3/decimator.cc
    audio_processing/aec3/dominant_nearend_detector.cc
    audio_processing/aec3/downsampled_render_buffer.cc
    audio_processing/aec3/echo_audibility.cc
    audio_processing/aec3/echo_canceller3.cc
    audio_processing/aec3/echo_path_delay_estimator.cc
    audio_processing/aec3/echo_path_variability.cc
    audio_processing/aec3/echo_remover_metrics.cc
    audio_processing/aec3/echo_remover.cc
    audio_processing/aec3/erl_estimator.cc
    audio_processing/aec3/erle_estimator.cc
    audio_processing/aec3/fft_buffer.cc
    audio_processing/aec3/filter_analyzer.cc
    audio_processing/aec3/frame_blocker.cc
    audio_processing/aec3/fullband_erle_estimator.cc
    audio_processing/aec3/main_filter_update_gain.cc
    audio_processing/aec3/matched_filter_lag_aggregator.cc
    audio_processing/aec3/matched_filter.cc
    audio_processing/aec3/moving_average.cc
    audio_processing/aec3/render_buffer.cc
    audio_processing/aec3/render_delay_buffer.cc
    audio_processing/aec3/render_delay_controller_metrics.cc
    audio_processing/aec3/render_delay_controller.cc
    audio_processing/aec3/render_signal_analyzer.cc
    audio_processing/aec3/residual_echo_estimator.cc
    audio_processing/aec3/reverb_decay_estimator.cc
    audio_processing/aec3/reverb_frequency_response.cc
    audio_processing/aec3/reverb_model_estimator.cc
    audio_processing/aec3/reverb_model.cc
    audio_processing/aec3/shadow_filter_update_gain.cc
    audio_processing/aec3/signal_dependent_erle_estimator.cc
    audio_processing/aec3/spectrum_buffer.cc
    audio_processing/aec3/stationarity_estimator.cc
    audio_processing/aec3/subband_erle_estimator.cc
    audio_processing/aec3/subband_nearend_detector.cc
    audio_processing/aec3/subtractor_output_analyzer.cc
    audio_processing/aec3/subtractor_output.cc
    audio_processing/aec3/subtractor.cc
    audio_processing/aec3/suppression_filter.cc
    audio_processing/aec3/suppression_gain.cc
    audio_processing/audio_buffer.cc
    audio_processing/audio_frame.cc
    audio_processing/channel_buffer.cc
    audio_processing/channel_layout.cc
    audio_processing/high_pass_filter.cc
    audio_processing/logging/apm_data_dumper.cc
    audio_processing/logging/wav_file.cc
    audio_processing/logging/wav_header.cc
    audio_processing/resampler/push_sinc_resampler.cc
    audio_processing/resampler/sinc_resampler.cc
    audio_processing/sparse_fir_filter.cc
    audio_processing/splitting_filter_c.c
    audio_processing/splitting_filter.cc
    audio_processing/three_band_filter_bank.cc
    audio_processing/utility/cascaded_biquad_filter.cc
    audio_processing/utility/delay_estimator_wrapper.cc
    audio_processing/utility/delay_estimator.cc
    audio_processing/utility/ooura_fft.cc
    audio_processing/ns/fast_math.cc
    audio_processing/ns/histograms.cc
    audio_processing/ns/noise_estimator.cc
    audio_processing/ns/noise_suppressor.cc
    audio_processing/ns/ns_fft.cc
    audio_processing/ns/prior_signal_model_estimator.cc
    audio_processing/ns/prior_signal_model.cc
    audio_processing/ns/quantile_noise_estimator.cc
    audio_processing/ns/signal_model_estimator.cc
    audio_processing/ns/signal_model.cc
    audio_processing/ns/speech_probability_estimator.cc
    audio_processing/ns/suppression_params.cc
    audio_processing/ns/wiener_filter.cc
    audio_processing/agc2/adaptive_digital_gain_controller.cc
    audio_processing/agc2/biquad_filter.cc
    audio_processing/agc2/fixed_digital_level_estimator.cc
    audio_processing/agc2/gain_applier.cc
    audio_processing/agc2/interpolated_gain_curve.cc
    audio_processing/agc2/limiter.cc
    audio_processing/agc2/limiter_db_gain_curve.cc
    audio_processing/agc2/noise_level_estimator.cc
    audio_processing/agc2/saturation_protector.cc
    audio_processing/agc2/saturation_protector_buffer.cc
    audio_processing/agc2/speech_level_estimator.cc
    audio_processing/agc2/speech_level_estimator_impl.cc
    audio_processing/agc2/speech_level_estimator_experimental_impl.cc
    audio_processing/agc2/speech_probability_buffer.cc
    audio_processing/agc2/vad_wrapper.cc
    audio_processing/agc2/cpu_features.cc
    audio_processing/agc2/rnn_vad/auto_correlation.cc
    audio_processing/agc2/rnn_vad/features_extraction.cc
    audio_processing/agc2/rnn_vad/lp_residual.cc
    audio_processing/agc2/rnn_vad/pitch_search.cc
    audio_processing/agc2/rnn_vad/pitch_search_internal.cc
    audio_processing/agc2/rnn_vad/rnn.cc
    audio_processing/agc2/rnn_vad/rnn_fc.cc
    audio_processing/agc2/rnn_vad/rnn_gru.cc
    audio_processing/agc2/rnn_vad/spectral_features.cc
    audio_processing/agc2/rnn_vad/spectral_features_internal.cc
    audio_processing/resampler/push_resampler.cc
    third_party/rnnoise/src/rnn_vad_weights.cc
    rtc_base/cpu_info.cc
    audio_processing/utility/pffft_wrapper.cc
    third_party/pffft/src/pffft.c
    common_audio/third_party/ooura/fft_size_256/fft4g.cc
    rtc_base/checks.cc
    rtc_base/critical_section.cc
    rtc_base/logging.cc
    rtc_base/memory/aligned_malloc.cc
    rtc_base/platform_thread_types.cc
    rtc_base/race_checker.cc
    rtc_base/string_encode.cc
    rtc_base/string_to_number.cc
    rtc_base/string_utils.cc
    rtc_base/strings/string_builder.cc
    rtc_base/system/file_wrapper.cc
    rtc_base/time_utils.cc
    system_wrappers/source/cpu_features.cc
    system_wrappers/source/field_trial.cc
    system_wrappers/source/metrics.cc
    abseil/absl/types/bad_optional_access.cc
)

# Determine target architecture (CMAKE_OSX_ARCHITECTURES overrides on macOS cross-compile)
if(CMAKE_OSX_ARCHITECTURES)
    set(_target_arch "${CMAKE_OSX_ARCHITECTURES}")
else()
    set(_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Platform-specific SIMD sources
if(_target_arch MATCHES "x86_64|AMD64|amd64")
    list(APPEND WEBRTC_AUDIO_SOURCES
        audio_processing/resampler/sinc_resampler_sse.cc
        audio_processing/utility/ooura_fft_sse2.cc
        audio_processing/agc2/rnn_vad/vector_math_avx2.cc
    )
elseif(_target_arch MATCHES "aarch64|arm64|ARM64")
    list(APPEND WEBRTC_AUDIO_SOURCES
        audio_processing/resampler/sinc_resampler_neon.cc
        audio_processing/utility/ooura_fft_neon.cc
    )
endif()

# Linux-specific cpu features
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND WEBRTC_AUDIO_SOURCES system_wrappers/source/cpu_features_linux.c)
endif()

add_library(webrtc_audio_static STATIC ${WEBRTC_AUDIO_SOURCES})

target_include_directories(webrtc_audio_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/audio_processing
    ${CMAKE_CURRENT_SOURCE_DIR}/audio_processing/logging
    ${CMAKE_CURRENT_SOURCE_DIR}/abseil
)

target_compile_definitions(webrtc_audio_static PRIVATE
    WEBRTC_APM_DEBUG_DUMP=0
    RTC_DISABLE_CHECK_MSG
)

if(WIN32)
    target_link_libraries(webrtc_audio_static PUBLIC winmm)
    target_compile_definitions(webrtc_audio_static PUBLIC WEBRTC_WIN NOMINMAX)
else()
    target_compile_definitions(webrtc_audio_static PUBLIC WEBRTC_POSIX)
    if(APPLE)
        target_compile_definitions(webrtc_audio_static PUBLIC WEBRTC_MAC)
        target_link_libraries(webrtc_audio_static PUBLIC "-framework CoreFoundation")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(webrtc_audio_static PUBLIC WEBRTC_LINUX WEBRTC_THREAD_RR)
    endif()
endif()

if(_target_arch MATCHES "x86_64|AMD64|amd64")
    target_compile_definitions(webrtc_audio_static PRIVATE WEBRTC_ARCH_X86_FAMILY)
    if(NOT MSVC)
        set_source_files_properties(
            audio_processing/resampler/sinc_resampler_sse.cc
            audio_processing/utility/ooura_fft_sse2.cc
            PROPERTIES COMPILE_FLAGS "-msse2"
        )
        set_source_files_properties(
            audio_processing/agc2/rnn_vad/vector_math_avx2.cc
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
        )
    endif()
elseif(_target_arch MATCHES "aarch64|arm64|ARM64")
    target_compile_definitions(webrtc_audio_static PRIVATE WEBRTC_ARCH_ARM64 WEBRTC_ARCH_ARM_FAMILY WEBRTC_HAS_NEON)
endif()

set_target_properties(webrtc_audio_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_features(webrtc_audio_static PRIVATE cxx_std_20)
    target_compile_options(webrtc_audio_static PRIVATE /FI"${CMAKE_CURRENT_SOURCE_DIR}/compat_includes.h")
else()
    target_compile_features(webrtc_audio_static PRIVATE cxx_std_17)
    target_compile_options(webrtc_audio_static PRIVATE
        "-include" "${CMAKE_CURRENT_SOURCE_DIR}/compat_includes.h"
        -Wno-deprecated-declarations -Wno-unused-variable)
endif()
