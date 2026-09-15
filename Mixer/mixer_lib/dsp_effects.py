import os
import random
import tempfile

import librosa
import numpy as np
from scipy import signal as scipy_signal

try:
    from moviepy.audio.AudioClip import AudioArrayClip
except ImportError:
    from moviepy import AudioArrayClip

try:
    from moviepy.audio.compositing.concatenate import concatenate_audioclips
except ImportError:
    from moviepy import concatenate_audioclips

from .config import (
    SIREN_FREQ_LOW_HZ,
    SIREN_FREQ_HIGH_HZ,
    SIREN_SWEEP_RATE_HZ,
    AIR_HORN_CHORD_HZ,
    SCRATCH_RATE_HZ,
    SCRATCH_WINDOW_SECONDS,
    RISER_BAND_LOW_HZ,
    RISER_BAND_HIGH_HZ,
    IMPACT_START_FREQ_HZ,
    IMPACT_END_FREQ_HZ,
    DJ_EFFECT_FADE_IN_SECONDS,
    DJ_EFFECT_FADE_OUT_SECONDS,
    SIREN_PROBABILITY,
    SCRATCH_PROBABILITY,
    AIR_HORN_PROBABILITY,
    RISER_PROBABILITY,
    IMPACT_PROBABILITY,
    MIN_DURATION_FOR_SMART_BLEND,
    BASS_CUTOFF_HZ,
    BASS_SWAP_OUTGOING_END_FRACTION,
    BASS_SWAP_INCOMING_START_FRACTION,
    BASS_SWAP_INCOMING_FULL_FRACTION,
    MAX_TEMPO_STRETCH,
    DJ_EFFECT_MAX_SECONDS,
    DJ_EFFECT_MIX_GAIN,
)
from .utils import clamp


def synthesize_siren(duration: float, sr: int) -> np.ndarray:
    n_samples = max(1, int(round(duration * sr)))

    if n_samples <= 1:
        return np.zeros(n_samples, dtype=float)

    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    sweep_period = 1.0 / max(SIREN_SWEEP_RATE_HZ, 1e-6)
    ramp_phase = (t % sweep_period) / sweep_period

    freq_range = SIREN_FREQ_HIGH_HZ - SIREN_FREQ_LOW_HZ
    base_freq = SIREN_FREQ_LOW_HZ + freq_range * ramp_phase

    build_progress = t / max(duration, 1e-6)
    instantaneous_freq = base_freq + freq_range * 0.15 * build_progress

    instantaneous_freq = np.clip(instantaneous_freq, 20.0, sr / 2.0 - 100.0)

    phase = 2.0 * np.pi * np.cumsum(instantaneous_freq) / sr
    phase_wrapped = np.mod(phase, 2.0 * np.pi)
    sawtooth = (phase_wrapped / np.pi) - 1.0

    tone = np.tanh(sawtooth * 1.8)

    peak = float(np.max(np.abs(tone)))
    if peak > 0:
        tone = tone / peak

    return tone.astype(float)


def synthesize_air_horn(duration: float, sr: int) -> np.ndarray:
    """
    Synthesize a stacked-tone air horn chord from oscillators. Nothing
    here is sampled from any external recording. Fade in/out is applied
    later by apply_effect_envelope, shared across every effect type.
    """
    n_samples = max(1, int(round(duration * sr)))
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    tone = np.zeros(n_samples)
    for freq in AIR_HORN_CHORD_HZ:
        tone += np.sin(2.0 * np.pi * freq * t)

    tone /= max(1, len(AIR_HORN_CHORD_HZ))
    tone = np.tanh(tone * 2.5)

    return tone.astype(float)


def synthesize_scratch(source_mono: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """
    Simulate a turntable scratch by reading back and forth across a short
    window of the transition's own already-present audio, the way a DJ
    moves a record under the needle. This reads audio that is already
    part of the mix; it does not add any new recording. Fade in/out is
    applied later by apply_effect_envelope, shared across every effect
    type.
    """
    n_samples = max(1, int(round(duration * sr)))

    if source_mono is None or len(source_mono) < 8:
        return np.zeros(n_samples)

    window_len = int(min(len(source_mono), sr * SCRATCH_WINDOW_SECONDS))
    window_len = max(window_len, 8)
    start_max = max(0, len(source_mono) - window_len)
    start = random.randint(0, start_max) if start_max > 0 else 0
    window = source_mono[start:start + window_len]

    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    position = (window_len / 2.0) + (window_len / 2.0) * np.sin(
        2.0 * np.pi * SCRATCH_RATE_HZ * t
    )
    indices = np.clip(position.astype(int), 0, window_len - 1)
    scratched = window[indices]

    return scratched.astype(float)


def butter_bandpass_sos(low_hz: float, high_hz: float, sr: int, order: int = 4):
    nyquist = sr / 2.0
    low = clamp(low_hz / nyquist, 0.001, 0.98)
    high = clamp(high_hz / nyquist, low + 0.001, 0.99)
    return scipy_signal.butter(order, [low, high], btype="band", output="sos")


def synthesize_riser(duration: float, sr: int) -> np.ndarray:
    """
    Synthesize a filtered-noise riser that builds in intensity, the kind
    used to lead into a drop. Built from random noise passed through a
    bandpass filter; nothing here is sampled from any external recording.
    """
    n_samples = max(1, int(round(duration * sr)))
    noise = np.random.default_rng().normal(0.0, 1.0, n_samples)

    sos = butter_bandpass_sos(RISER_BAND_LOW_HZ, RISER_BAND_HIGH_HZ, sr)
    filtered = scipy_signal.sosfiltfilt(sos, noise)

    peak = float(np.max(np.abs(filtered))) if n_samples else 0.0
    if peak > 0:
        filtered = filtered / peak

    t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    build_curve = t ** 2

    return (filtered * build_curve).astype(float)


def synthesize_impact(duration: float, sr: int) -> np.ndarray:
    """
    Synthesize a falling-pitch impact hit, the kind used to punctuate a
    transition. A tone sweeps from a high starting pitch down to a low
    sub frequency with a fast attack and natural decay. Nothing here is
    sampled from any external recording.
    """
    n_samples = max(1, int(round(duration * sr)))
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    decay_rate = 6.0 / max(duration, 0.01)
    freq_curve = IMPACT_END_FREQ_HZ + (
        IMPACT_START_FREQ_HZ - IMPACT_END_FREQ_HZ
    ) * np.exp(-decay_rate * t)

    phase = 2.0 * np.pi * np.cumsum(freq_curve) / sr
    tone = np.sin(phase)

    amplitude_curve = np.exp(-decay_rate * t)

    return (tone * amplitude_curve).astype(float)


def apply_effect_envelope(
    audio: np.ndarray, sr: int, fade_in_seconds: float, fade_out_seconds: float,
) -> np.ndarray:
    """
    Apply one shared fade-in/fade-out envelope to a synthesized effect so
    it always eases in and tapers off, rather than starting or ending
    abruptly at the transition boundary.
    """
    n_samples = len(audio)

    if n_samples == 0:
        return audio

    envelope = np.ones(n_samples)
    half = max(1, n_samples // 2) if n_samples > 1 else n_samples

    fade_in_samples = min(half, max(1, int(round(fade_in_seconds * sr))))
    fade_out_samples = min(
        n_samples - fade_in_samples,
        max(1, int(round(fade_out_seconds * sr))),
    )

    if fade_in_samples > 0:
        envelope[:fade_in_samples] = np.linspace(0.0, 1.0, fade_in_samples)

    if fade_out_samples > 0:
        envelope[-fade_out_samples:] = np.linspace(1.0, 0.0, fade_out_samples)

    return audio * envelope


def choose_dj_effect(applied_duration: float, enabled: bool) -> str | None:
    """
    Roll for whether a transition gets a synthesized DJ effect layered
    under it, and which one. Returns None most of the time, and always
    when the transition is too short for the effect to have been applied
    anyway.
    """
    if not enabled:
        return None

    if applied_duration < MIN_DURATION_FOR_SMART_BLEND:
        return None

    roll = random.random()

    if roll < SIREN_PROBABILITY:
        return "siren"

    roll -= SIREN_PROBABILITY
    if roll < SCRATCH_PROBABILITY:
        return "scratch"

    roll -= SCRATCH_PROBABILITY
    if roll < AIR_HORN_PROBABILITY:
        return "air_horn"

    roll -= AIR_HORN_PROBABILITY
    if roll < RISER_PROBABILITY:
        return "riser"

    roll -= RISER_PROBABILITY
    if roll < IMPACT_PROBABILITY:
        return "impact"

    return None


def butter_lowpass_sos(cutoff_hz: float, sr: int, order: int = 4):
    nyquist = sr / 2.0
    normalized_cutoff = clamp(cutoff_hz / nyquist, 0.001, 0.99)
    return scipy_signal.butter(order, normalized_cutoff, btype="low", output="sos")


def split_bass(y: np.ndarray, sr: int) -> tuple:
    """
    Split a mono or multichannel array into a bass band and the remainder,
    using a zero-phase low-pass filter so the two bands sum back to the
    original signal exactly (no crossover phase artifacts).
    """
    sos = butter_lowpass_sos(BASS_CUTOFF_HZ, sr)

    if y.ndim == 1:
        bass = scipy_signal.sosfiltfilt(sos, y)
        return bass, y - bass

    bass = np.zeros_like(y)
    for channel in range(y.shape[0]):
        bass[channel] = scipy_signal.sosfiltfilt(sos, y[channel])

    return bass, y - bass


def equal_power_curve(n_samples: int) -> tuple:
    """
    Equal-power fade curves. Unlike a linear crossfade, the combined
    loudness of the outgoing and incoming signal stays roughly constant
    through the middle of the transition instead of dipping.
    """
    if n_samples <= 0:
        return np.array([]), np.array([])

    t = np.linspace(0.0, 1.0, n_samples)
    fade_out = np.cos(t * np.pi / 2.0)
    fade_in = np.sin(t * np.pi / 2.0)

    return fade_out, fade_in


def bass_swap_envelope(n_samples: int, outgoing: bool) -> np.ndarray:
    """
    Envelope for the bass band during a transition.

    The outgoing track's bass is cleared out well before the transition
    ends, and the incoming track's bass is brought in early, so the two
    basslines are never both at full strength at once. This mirrors the
    bass-kill technique used on a real DJ mixer's EQ.
    """
    if n_samples <= 0:
        return np.array([])

    t = np.linspace(0.0, 1.0, n_samples)

    if outgoing:
        envelope = 1.0 - np.clip(t / BASS_SWAP_OUTGOING_END_FRACTION, 0.0, 1.0)
    else:
        start = BASS_SWAP_INCOMING_START_FRACTION
        end = BASS_SWAP_INCOMING_FULL_FRACTION
        envelope = np.clip((t - start) / max(end - start, 1e-6), 0.0, 1.0)

    return envelope


def time_stretch_array(y: np.ndarray, rate: float, target_length: int) -> np.ndarray:
    """
    Time-stretch audio toward a target tempo ratio, then trim or pad it back
    to an exact sample count so it still fits its slot on the timeline.
    """
    rate = clamp(rate, 1.0 - MAX_TEMPO_STRETCH, 1.0 + MAX_TEMPO_STRETCH)

    if abs(rate - 1.0) < 0.005:
        stretched = y
    elif y.ndim == 1:
        stretched = librosa.effects.time_stretch(y, rate=rate)
    else:
        channels = [
            librosa.effects.time_stretch(y[ch], rate=rate)
            for ch in range(y.shape[0])
        ]
        max_len = max(len(c) for c in channels)
        channels = [np.pad(c, (0, max_len - len(c))) for c in channels]
        stretched = np.stack(channels)

    current_length = stretched.shape[-1]

    if current_length == target_length:
        return stretched

    if current_length > target_length:
        if stretched.ndim == 1:
            return stretched[:target_length]
        return stretched[:, :target_length]

    pad_width = target_length - current_length

    if stretched.ndim == 1:
        return np.pad(stretched, (0, pad_width))

    return np.pad(stretched, ((0, 0), (0, pad_width)))


def extract_clip_audio_array(clip, start: float, end: float, sr: int) -> np.ndarray:
    """
    Render a precise time range of a clip's audio to a numpy array via a
    temporary WAV file, reusing the same extraction path as the analysis
    stage for consistency. Always returns a 2D (channels, samples) array.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_path = tmp_audio.name

    try:
        subclip_audio = clip.audio.subclipped(start, end)
        subclip_audio.write_audiofile(
            tmp_path,
            fps=sr,
            logger=None,
            codec="pcm_s16le",
        )

        y, _ = librosa.load(tmp_path, sr=sr, mono=False)

        if y.ndim == 1:
            y = y[np.newaxis, :]

        return y

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_processed_edge_audio(
    clip,
    edge_start: float,
    edge_end: float,
    sr: int,
    outgoing: bool,
    stretch_rate: float,
    effect_type: str | None = None,
):
    """
    Build the processed head or tail audio for one side of a transition:
    tempo-matched toward the shared target, band-split into bass/rest, and
    recombined with a bass-swap envelope on the bass band and an
    equal-power fade on the rest. Optionally layers a synthesized DJ
    effect under the portion closest to the cut. Returns an
    AudioArrayClip of exactly edge_end - edge_start seconds.
    """
    duration = edge_end - edge_start
    target_samples = max(1, int(round(duration * sr)))

    y = extract_clip_audio_array(clip, edge_start, edge_end, sr)
    y = time_stretch_array(y, stretch_rate, target_samples)

    bass, rest = split_bass(y, sr)

    bass_envelope = bass_swap_envelope(target_samples, outgoing)
    power_fade_out, power_fade_in = equal_power_curve(target_samples)
    rest_envelope = power_fade_out if outgoing else power_fade_in

    processed = bass * bass_envelope + rest * rest_envelope

    if effect_type:
        effect_duration = min(DJ_EFFECT_MAX_SECONDS, duration)
        effect_samples = max(1, int(round(effect_duration * sr)))
        effect_samples = min(effect_samples, target_samples)

        if effect_type == "siren":
            effect_audio = synthesize_siren(effect_samples / sr, sr)
        elif effect_type == "air_horn":
            effect_audio = synthesize_air_horn(effect_samples / sr, sr)
        elif effect_type == "scratch":
            source_mono = np.mean(y, axis=0) if y.ndim > 1 else y
            effect_audio = synthesize_scratch(
                source_mono, sr, effect_samples / sr
            )
        elif effect_type == "riser":
            effect_audio = synthesize_riser(effect_samples / sr, sr)
        elif effect_type == "impact":
            effect_audio = synthesize_impact(effect_samples / sr, sr)
        else:
            effect_audio = None

        if effect_audio is not None and len(effect_audio) > 0:
            effect_audio = effect_audio[:effect_samples]
            effect_audio = apply_effect_envelope(
                effect_audio, sr,
                DJ_EFFECT_FADE_IN_SECONDS, DJ_EFFECT_FADE_OUT_SECONDS,
            )
            effect_audio = effect_audio * DJ_EFFECT_MIX_GAIN

            if outgoing:
                processed[:, -effect_samples:] += effect_audio
            else:
                processed[:, :effect_samples] += effect_audio

    processed = np.clip(processed, -1.0, 1.0)

    array_for_clip = processed.T

    return AudioArrayClip(array_for_clip, fps=sr)


def attach_smart_audio(clip, transition_in: dict, transition_out: dict, native_bpm: float):
    """
    Replace a clip's audio with a version whose head and/or tail have been
    processed for a real-mixer style blend (bass swap, tempo match,
    equal-power fade), leaving the untouched middle of the clip exactly as
    it was. transition_in/transition_out may be None when that side isn't
    being smart-blended (e.g. it's a fade_black transition or too short).
    """
    if clip.audio is None:
        return clip

    sr = clip.audio.fps
    total_duration = clip.duration

    head_duration = 0.0
    tail_duration = 0.0

    if transition_in and transition_in["applied_duration"] >= MIN_DURATION_FOR_SMART_BLEND:
        head_duration = transition_in["applied_duration"]

    if transition_out and transition_out["applied_duration"] >= MIN_DURATION_FOR_SMART_BLEND:
        tail_duration = transition_out["applied_duration"]

    if head_duration <= 0 and tail_duration <= 0:
        return clip

    if head_duration + tail_duration >= total_duration:
        return clip

    segments = []

    if head_duration > 0:
        stretch_rate = (
            transition_in["target_bpm"] / native_bpm
            if native_bpm > 0
            else 1.0
        )
        segments.append(
            build_processed_edge_audio(
                clip, 0.0, head_duration, sr,
                outgoing=False, stretch_rate=stretch_rate,
                effect_type=transition_in.get("effect"),
            )
        )

    middle_start = head_duration
    middle_end = total_duration - tail_duration

    if middle_end > middle_start:
        segments.append(clip.audio.subclipped(middle_start, middle_end))

    if tail_duration > 0:
        stretch_rate = (
            transition_out["target_bpm"] / native_bpm
            if native_bpm > 0
            else 1.0
        )
        tail_start = total_duration - tail_duration
        segments.append(
            build_processed_edge_audio(
                clip, tail_start, total_duration, sr,
                outgoing=True, stretch_rate=stretch_rate,
                effect_type=transition_out.get("effect"),
            )
        )

    new_audio = concatenate_audioclips(segments)
    return clip.with_audio(new_audio)