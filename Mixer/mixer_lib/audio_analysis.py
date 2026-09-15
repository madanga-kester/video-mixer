import os
import tempfile

import librosa
import numpy as np
from moviepy import VideoFileClip

from .config import (
    ENERGY_THRESHOLD_RATIO,
    MAX_INTRO_SKIP_SECONDS,
    VOCAL_BAND_LOW_HZ,
    VOCAL_BAND_HIGH_HZ,
    BEATS_PER_PHRASE,
    MIN_ONSET_DENSITY_FOR_BEAT,
    MIN_BEAT_COUNT_FOR_LOCK,
)
from .utils import clamp, safe_float


def normalize_energy(values: np.ndarray) -> float:
    if values is None or len(values) == 0:
        return 0.0

    values = np.asarray(values, dtype=float)
    if np.max(values) <= 0:
        return 0.0

    return float(np.clip(np.mean(values) / (np.max(values) + 1e-12), 0.0, 1.0))


def detect_intro(rms: np.ndarray, times: np.ndarray) -> float:
    """
    Estimate where the real song begins.

    Unlike simply taking the first frame above a threshold, this looks for
    sustained activity. This reduces false intro skips caused by one isolated
    kick, click, or sound effect.
    """
    if len(rms) == 0 or len(times) == 0:
        return 0.0

    peak = float(np.max(rms))
    if peak <= 0:
        return 0.0

    threshold = peak * ENERGY_THRESHOLD_RATIO

    active = rms >= threshold

    run_length = max(3, int(round(0.35 / max(times[1] - times[0], 0.01))))
    run_length = min(run_length, len(active))

    for i in range(0, len(active) - run_length + 1):
        if np.all(active[i:i + run_length]):
            candidate = float(times[i])
            return round(min(candidate, MAX_INTRO_SKIP_SECONDS), 1)

    return 0.0


def compute_vocal_activity(y: np.ndarray, sr: int, hop_length: int = 512) -> tuple:
    """
    Estimate where vocal energy is concentrated over time.

    This is a proxy, not a true vocal detector. It isolates the harmonic
    component of the signal and measures how much of that harmonic energy
    falls inside the typical vocal frequency range. High values suggest a
    vocal line is likely present; low values suggest an instrumental
    passage, which is where transitions blend most cleanly.
    """
    if y is None or len(y) < sr * 2:
        return np.array([]), np.array([])

    try:
        y_harmonic, _ = librosa.effects.hpss(y)

        stft = np.abs(librosa.stft(y_harmonic, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=(stft.shape[0] - 1) * 2)

        vocal_band = (freqs >= VOCAL_BAND_LOW_HZ) & (freqs <= VOCAL_BAND_HIGH_HZ)

        vocal_energy = np.sum(stft[vocal_band, :], axis=0)
        total_energy = np.sum(stft, axis=0) + 1e-12

        activity = vocal_energy / total_energy

        kernel_size = max(1, int(round(sr / hop_length * 0.5)))
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            activity = np.convolve(activity, kernel, mode="same")

        times = librosa.frames_to_time(
            np.arange(len(activity)),
            sr=sr,
            hop_length=hop_length,
        )

        peak = float(np.max(activity)) if len(activity) else 0.0
        if peak > 0:
            activity = activity / peak

        return times, activity

    except Exception:
        return np.array([]), np.array([])


def detect_sections(y: np.ndarray, sr: int, chroma: np.ndarray = None) -> list:
    """
    Estimate structural boundaries.

    This is intentionally conservative: it does not claim to know exact
    verse/chorus labels. Instead it identifies meaningful musical change
    points using chroma similarity combined with onset novelty.
    """
    if len(y) < sr * 8:
        return []

    try:
        hop_length = 512

        if chroma is None:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
            chroma = librosa.util.normalize(chroma, axis=0)

        chroma_novelty = np.zeros(chroma.shape[1])

        if chroma.shape[1] > 1:
            frame_similarity = np.sum(
                chroma[:, :-1] * chroma[:, 1:],
                axis=0,
            )
            chroma_novelty[1:] = 1.0 - frame_similarity

        onset_novelty = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        frame_count = min(len(chroma_novelty), len(onset_novelty))
        chroma_novelty = chroma_novelty[:frame_count]
        onset_novelty = onset_novelty[:frame_count]

        chroma_max = float(np.max(chroma_novelty)) if frame_count else 0.0
        onset_max = float(np.max(onset_novelty)) if frame_count else 0.0

        chroma_normalized = (
            chroma_novelty / chroma_max if chroma_max > 0 else chroma_novelty
        )
        onset_normalized = (
            onset_novelty / onset_max if onset_max > 0 else onset_novelty
        )

        combined_novelty = (
            chroma_normalized * 0.5
            + onset_normalized * 0.5
        )

        peaks = librosa.util.peak_pick(
            combined_novelty,
            pre_max=max(1, int(sr / hop_length)),
            post_max=max(1, int(sr / hop_length)),
            pre_avg=max(1, int(2 * sr / hop_length)),
            post_avg=max(1, int(2 * sr / hop_length)),
            delta=max(0.05, float(np.std(combined_novelty) * 0.35)),
            wait=max(1, int(4 * sr / hop_length)),
        )

        times = librosa.frames_to_time(
            peaks,
            sr=sr,
            hop_length=hop_length,
        )

        audio_duration = len(y) / sr
        window_end = max(8.0, audio_duration - 8.0)

        result = []
        last = -999.0

        for t in times:
            t = float(t)
            if 8.0 <= t <= window_end and t - last >= 8.0:
                result.append(t)
                last = t

        return result[:12]

    except Exception:
        return []


def analyze_track(video_path: str) -> dict:
    """
    Extract and analyze audio once.

    Returns:
      bpm
      beat_times
      phrase_times
      intro_skip
      energy
      onset_density
      spectral_centroid
      section_times
      chroma_profile
      mean_rms
      duration
      vocal_times
      vocal_activity
    """
    default_chroma_profile = [1.0 / 12.0] * 12

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            duration = safe_float(clip.duration, 0.0)
            clip.close()

            return {
                "bpm": 120.0,
                "intro_skip": 0.0,
                "beat_times": [],
                "phrase_times": [],
                "section_times": [],
                "energy": 0.5,
                "onset_density": 0.0,
                "spectral_centroid": 0.0,
                "chroma_profile": default_chroma_profile,
                "mean_rms": 0.0,
                "duration": duration,
                "vocal_times": [],
                "vocal_activity": [],
            }

        duration = safe_float(clip.duration, 0.0)

        clip.audio.write_audiofile(
            tmp_audio_path,
            logger=None,
            codec="pcm_s16le",
        )
        clip.close()

        y, sr = librosa.load(
            tmp_audio_path,
            sr=None,
            mono=True,
        )

        if len(y) == 0:
            return {
                "bpm": 120.0,
                "intro_skip": 0.0,
                "beat_times": [],
                "phrase_times": [],
                "section_times": [],
                "energy": 0.5,
                "onset_density": 0.0,
                "spectral_centroid": 0.0,
                "chroma_profile": default_chroma_profile,
                "mean_rms": 0.0,
                "duration": duration,
                "vocal_times": [],
                "vocal_activity": [],
            }

        tempo_curve = librosa.feature.tempo(
            y=y,
            sr=sr,
            aggregate=None,
        )

        if tempo_curve is not None and len(tempo_curve) > 0:
            bpm_raw = float(np.median(tempo_curve))
        else:
            fallback_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm_raw = safe_float(
                fallback_tempo[0] if not np.isscalar(fallback_tempo) else fallback_tempo,
                120.0,
            )

        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        bpm = bpm_raw

        if bpm < 60:
            bpm *= 2
        elif bpm > 200:
            bpm /= 2

        bpm = round(clamp(bpm, 60.0, 200.0), 1)

        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=sr,
        ).tolist()

        phrase_times = []

        if beat_times:
            for i in range(0, len(beat_times), BEATS_PER_PHRASE):
                phrase_times.append(float(beat_times[i]))

        hop_length = 512
        rms = librosa.feature.rms(
            y=y,
            hop_length=hop_length,
        )[0]

        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=hop_length,
        )

        peak_energy = float(np.max(rms)) if len(rms) else 0.0
        mean_energy = float(np.mean(rms)) if len(rms) else 0.0

        relative_energy = (
            mean_energy / (peak_energy + 1e-12)
            if peak_energy > 0
            else 0.5
        )

        energy = float(np.clip(relative_energy, 0.0, 1.0))

        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            units="frames",
        )

        analyzed_duration = max(len(y) / sr, 1.0)
        onset_density = len(onset_frames) / analyzed_duration

        onset_density_normalized = float(
            np.clip(onset_density / 6.0, 0.0, 1.0)
        )

        if (
            onset_density_normalized < MIN_ONSET_DENSITY_FOR_BEAT
            or len(beat_times) < MIN_BEAT_COUNT_FOR_LOCK
        ):
            beat_times = []
            phrase_times = []

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )[0]

        spectral_centroid = (
            float(np.mean(centroid))
            if len(centroid)
            else 0.0
        )

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        chroma = librosa.util.normalize(chroma, axis=0)

        chroma_mean = np.mean(chroma, axis=1)
        chroma_sum = float(np.sum(chroma_mean))

        if chroma_sum > 0:
            chroma_profile = (chroma_mean / chroma_sum).tolist()
        else:
            chroma_profile = default_chroma_profile

        intro_skip = detect_intro(rms, times)
        section_times = detect_sections(y, sr, chroma=chroma)
        vocal_times, vocal_activity = compute_vocal_activity(y, sr, hop_length)

        return {
            "bpm": bpm,
            "intro_skip": intro_skip,
            "beat_times": beat_times,
            "phrase_times": phrase_times,
            "section_times": section_times,
            "energy": energy,
            "onset_density": onset_density_normalized,
            "spectral_centroid": spectral_centroid,
            "chroma_profile": chroma_profile,
            "mean_rms": mean_energy,
            "duration": duration,
            "vocal_times": vocal_times.tolist(),
            "vocal_activity": vocal_activity.tolist(),
        }

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def load_track_audio(video_path: str):
    """
    Load the full mono audio waveform for a track, independent of
    analyze_track (which discards its own y/sr after computing stats
    to keep memory bounded across many tracks). Returns (y, sr), or
    (None, None) if the clip has no audio track.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            clip.close()
            return None, None

        clip.audio.write_audiofile(
            tmp_audio_path,
            logger=None,
            codec="pcm_s16le",
        )
        clip.close()

        y, sr = librosa.load(tmp_audio_path, sr=None, mono=True)
        return y, sr

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)