import colorsys
import os

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, ImageClip, CompositeVideoClip
from moviepy.video.fx import FadeIn, FadeOut

from .config import (
    DEFAULT_VISUALIZER_BARS,
    VISUALIZER_BACKGROUND_TOP_COLOR,
    VISUALIZER_BACKGROUND_BOTTOM_COLOR,
    VISUALIZER_BAR_COLOR_LOW,
    VISUALIZER_BAR_COLOR_HIGH,
    VISUALIZER_BAR_CORNER_RADIUS_RATIO,
    VISUALIZER_REFLECTION_HEIGHT_RATIO,
    VISUALIZER_REFLECTION_OPACITY,
    VISUALIZER_FFT_HOP_LENGTH,
    VISUALIZER_BAR_FMIN_HZ,
    VISUALIZER_BAR_ATTACK_SECONDS,
    VISUALIZER_BAR_DECAY_SECONDS,
    VISUALIZER_BEAT_PULSE_DECAY_SECONDS,
    VISUALIZER_BEAT_PULSE_BRIGHTEN,
    VISUALIZER_BEAT_PULSE_ATTACK_SECONDS,
    VISUALIZER_BACKGROUND_HUE_DRIFT,
    VISUALIZER_PROGRESS_BAR_HEIGHT_RATIO,
    VISUALIZER_PROGRESS_BAR_TRACK_COLOR,
    VISUALIZER_PROGRESS_BAR_TRACK_OPACITY,
    TITLE_FONT_SIZE_RATIO,
    TITLE_PILL_COLOR,
    TITLE_PILL_OPACITY,
    TITLE_ACCENT_COLOR,
    SPEAKER_ENABLED,
    SPEAKER_POSITION_X_RATIO_LEFT,
    SPEAKER_POSITION_X_RATIO_RIGHT,
    SPEAKER_POSITION_Y_RATIO,
    SPEAKER_BASE_RADIUS_RATIO,
    SPEAKER_MAX_EXTRA_RADIUS_RATIO,
    SPEAKER_RIM_WIDTH_RATIO,
    SPEAKER_RIM_COLOR,
    SPEAKER_ATTACK_SECONDS,
    SPEAKER_DECAY_SECONDS,
)
from .utils import clamp, lerp_color
from .dsp_effects import extract_clip_audio_array


def compute_spectrum_bars(
    y: np.ndarray,
    sr: int,
    n_bars: int = DEFAULT_VISUALIZER_BARS,
    hop_length: int = VISUALIZER_FFT_HOP_LENGTH,
) -> tuple:
    """
    Precompute log-spaced frequency-band energy over the entire track in
    one pass, so per-frame rendering later is a cheap lookup instead of
    an FFT per video frame.

    Returns (times, bars) where bars has shape (n_bars, n_frames) with
    values normalized to roughly 0..1.
    """
    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    n_fft = (stft.shape[0] - 1) * 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    fmax = sr / 2.0
    band_edges = np.logspace(
        np.log10(VISUALIZER_BAR_FMIN_HZ),
        np.log10(fmax),
        n_bars + 1,
    )

    bars = np.zeros((n_bars, stft.shape[1]), dtype=float)

    for i in range(n_bars):
        band_mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])

        if np.any(band_mask):
            bars[i] = np.mean(stft[band_mask, :], axis=0)

    bars = np.log1p(bars)
    peak = float(np.max(bars)) if bars.size else 0.0

    if peak > 0:
        bars = bars / peak

    times = librosa.frames_to_time(
        np.arange(stft.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return times, bars


def compute_loudness_envelope(
    y: np.ndarray,
    sr: int,
    hop_length: int = VISUALIZER_FFT_HOP_LENGTH,
) -> tuple:
    if y is None or len(y) == 0:
        return np.array([0.0]), np.array([0.0])

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    if len(rms) == 0:
        return np.array([0.0]), np.array([0.0])

    rms = np.log1p(rms)

    peak = float(np.max(rms))
    if peak > 0:
        rms = rms / peak

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length,
    )

    return times, rms


def track_color_palette(entry: dict) -> tuple:
    """
    Derive a bar color pair from the track's own dominant chroma pitch
    and energy, instead of using the same fixed global colors for every
    track. Falls back to the global defaults if a chroma profile is not
    available.
    """
    chroma_profile = entry.get("chroma_profile")

    if not chroma_profile:
        return VISUALIZER_BAR_COLOR_LOW, VISUALIZER_BAR_COLOR_HIGH

    chroma_array = np.asarray(chroma_profile, dtype=float)

    if chroma_array.size == 0:
        return VISUALIZER_BAR_COLOR_LOW, VISUALIZER_BAR_COLOR_HIGH

    dominant_pitch = int(np.argmax(chroma_array))
    base_hue = dominant_pitch / 12.0
    energy = clamp(float(entry.get("energy", 0.5)), 0.0, 1.0)

    low_hue = base_hue
    high_hue = (base_hue + 0.5) % 1.0

    low_rgb = colorsys.hls_to_rgb(low_hue, 0.45 + energy * 0.1, 0.65)
    high_rgb = colorsys.hls_to_rgb(high_hue, 0.55 + energy * 0.1, 0.75)

    color_low = tuple(int(round(c * 255)) for c in low_rgb)
    color_high = tuple(int(round(c * 255)) for c in high_rgb)

    return color_low, color_high


def beat_pulse_at(
    time_value: float,
    beat_times_local: list,
    decay_seconds: float,
    attack_seconds: float = VISUALIZER_BEAT_PULSE_ATTACK_SECONDS,
) -> float:
    if not beat_times_local:
        return 0.0

    beat_array = np.asarray(beat_times_local, dtype=float)
    past_beats = beat_array[beat_array <= time_value]

    if len(past_beats) == 0:
        return 0.0

    elapsed = time_value - float(past_beats[-1])

    if attack_seconds > 0.0 and elapsed < attack_seconds:
        return float(np.clip(elapsed / attack_seconds, 0.0, 1.0))

    decay_elapsed = elapsed - max(attack_seconds, 0.0)
    return float(np.exp(-decay_elapsed / max(decay_seconds, 1e-6)))

def shift_gradient_hue(top_color: tuple, bottom_color: tuple, hue_shift: float) -> tuple:
    """
    Rotate the hue of both gradient endpoint colors by hue_shift,
    wrapping around the hue circle. Used to slowly drift the visualizer
    background color across a segment instead of holding one static
    gradient for the whole duration.
    """
    top_h, top_l, top_s = colorsys.rgb_to_hls(*[c / 255.0 for c in top_color])
    bottom_h, bottom_l, bottom_s = colorsys.rgb_to_hls(*[c / 255.0 for c in bottom_color])

    top_h = (top_h + hue_shift) % 1.0
    bottom_h = (bottom_h + hue_shift) % 1.0

    top_rgb = tuple(int(round(c * 255)) for c in colorsys.hls_to_rgb(top_h, top_l, top_s))
    bottom_rgb = tuple(int(round(c * 255)) for c in colorsys.hls_to_rgb(bottom_h, bottom_l, bottom_s))

    return top_rgb, bottom_rgb


def build_gradient_background(
    width: int, height: int, top_color: tuple, bottom_color: tuple,
) -> np.ndarray:
    """
    Build a vertical gradient background with vectorized numpy
    interpolation instead of a per-row PIL draw loop. This now runs
    once per rendered frame rather than once per segment, since the
    hue drifts over time, so it needs to stay cheap.
    """
    rows = np.linspace(0.0, 1.0, height).reshape(height, 1)
    top = np.array(top_color, dtype=float).reshape(1, 3)
    bottom = np.array(bottom_color, dtype=float).reshape(1, 3)

    gradient_rows = top + (bottom - top) * rows
    gradient = np.repeat(gradient_rows[:, np.newaxis, :], width, axis=1)

    return gradient.astype(np.uint8)


def draw_speaker(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    rim_radius: float,
    cone_radius: float,
    rim_width: float,
    rim_color: tuple,
    cone_color: tuple,
) -> None:
    rim_box = [
        center_x - rim_radius,
        center_y - rim_radius,
        center_x + rim_radius,
        center_y + rim_radius,
    ]
    draw.ellipse(rim_box, outline=rim_color, width=max(1, int(round(rim_width))))

    cone_radius = max(1.0, cone_radius)
    cone_box = [
        center_x - cone_radius,
        center_y - cone_radius,
        center_x + cone_radius,
        center_y + cone_radius,
    ]
    draw.ellipse(cone_box, fill=cone_color)


def draw_bars_frame(
    background: np.ndarray,
    width: int,
    height: int,
    bar_values: np.ndarray,
    color_low: tuple,
    color_high: tuple,
    pulse: float = 0.0,
    progress: float = 0.0,
    speaker_level: float = 0.0,
) -> np.ndarray:
    """
    Render one visualizer frame: bars colored on a per-track low-to-high
    gradient, rounded tops where the installed Pillow supports it, and a
    faint reflection under the baseline. The passed-in background is
    briefly brightened on each beat via pulse, and a thin progress bar
    at the bottom edge shows how far through the segment playback is.
    """
    pulse = clamp(float(pulse), 0.0, 1.0)

    if pulse > 0.0:
        brighten = pulse * VISUALIZER_BEAT_PULSE_BRIGHTEN
        frame_array = np.clip(
            background.astype(np.int16) + brighten, 0, 255
        ).astype(np.uint8)
    else:
        frame_array = background

    image = Image.fromarray(frame_array)
    draw = ImageDraw.Draw(image)

    n_bars = len(bar_values)
    margin = int(width * 0.05)
    usable_width = width - margin * 2
    gap = usable_width / n_bars * 0.25
    bar_width = max(1.0, usable_width / n_bars - gap)

    baseline = height * 0.85
    max_bar_height = height * 0.70
    reflection_height = max_bar_height * VISUALIZER_REFLECTION_HEIGHT_RATIO
    radius = max(1.0, bar_width * VISUALIZER_BAR_CORNER_RADIUS_RATIO / 2.0)

    round_rect = getattr(draw, "rounded_rectangle", None)

    x = float(margin)

    for value in bar_values:
        value = clamp(float(value), 0.0, 1.0)
        bar_height = max(2.0, value * max_bar_height)
        color = lerp_color(color_low, color_high, value)

        bar_box = [x, baseline - bar_height, x + bar_width, baseline]

        if round_rect:
            round_rect(bar_box, radius=radius, fill=color)
        else:
            draw.rectangle(bar_box, fill=color)

        reflect_height = min(reflection_height, bar_height)
        reflect_color = lerp_color(
            color, VISUALIZER_BACKGROUND_BOTTOM_COLOR,
            1.0 - VISUALIZER_REFLECTION_OPACITY,
        )
        draw.rectangle(
            [x, baseline, x + bar_width, baseline + reflect_height],
            fill=reflect_color,
        )

        x += bar_width + gap

    bar_area_height = max(2, int(height * VISUALIZER_PROGRESS_BAR_HEIGHT_RATIO))
    bar_y = height - bar_area_height

    track_color = (
        VISUALIZER_PROGRESS_BAR_TRACK_COLOR
        + (VISUALIZER_PROGRESS_BAR_TRACK_OPACITY,)
    )
    track_overlay = Image.new("RGBA", (width, bar_area_height), track_color)
    image.paste(track_overlay, (0, bar_y), track_overlay)

    filled_width = int(width * clamp(progress, 0.0, 1.0))
    if filled_width > 0:
        draw.rectangle(
            [0, bar_y, filled_width, height],
            fill=TITLE_ACCENT_COLOR,
        )

    if SPEAKER_ENABLED:
        speaker_level_clamped = clamp(float(speaker_level), 0.0, 1.0)

        base_radius = height * SPEAKER_BASE_RADIUS_RATIO
        extra_radius = height * SPEAKER_MAX_EXTRA_RADIUS_RATIO * speaker_level_clamped
        rim_radius = base_radius + height * SPEAKER_MAX_EXTRA_RADIUS_RATIO
        cone_radius = base_radius + extra_radius
        rim_width = max(1.0, height * SPEAKER_RIM_WIDTH_RATIO)

        cone_color = lerp_color(color_low, color_high, speaker_level_clamped)

        speaker_y = height * SPEAKER_POSITION_Y_RATIO

        draw_speaker(
            draw,
            width * SPEAKER_POSITION_X_RATIO_LEFT,
            speaker_y,
            rim_radius,
            cone_radius,
            rim_width,
            SPEAKER_RIM_COLOR,
            cone_color,
        )

        draw_speaker(
            draw,
            width * SPEAKER_POSITION_X_RATIO_RIGHT,
            speaker_y,
            rim_radius,
            cone_radius,
            rim_width,
            SPEAKER_RIM_COLOR,
            cone_color,
        )

    return np.array(image)


def make_visualizer_clip(
    times: np.ndarray,
    bars: np.ndarray,
    duration: float,
    size: tuple,
    fps: float,
    beat_times_local: list,
    color_low: tuple,
    color_high: tuple,
    loudness_times: np.ndarray,
    loudness_values: np.ndarray,
    track_offset: float = 0.0,
):
    width, height = size
    last_index = bars.shape[1] - 1
    last_loudness_index = len(loudness_values) - 1

    smoothed_values = np.zeros(bars.shape[0], dtype=float)
    speaker_state = {"value": 0.0}
    frame_state = {"last_t": None}

    def make_frame(t):
        target_time = track_offset + t
        index = int(np.searchsorted(times, target_time))
        index = int(clamp(index, 0, last_index))
        target_values = bars[:, index]

        last_t = frame_state["last_t"]
        dt = (1.0 / max(fps, 1.0)) if last_t is None else max(0.0, t - last_t)
        frame_state["last_t"] = t

        rising_mask = target_values > smoothed_values
        tau = np.where(
            rising_mask,
            VISUALIZER_BAR_ATTACK_SECONDS,
            VISUALIZER_BAR_DECAY_SECONDS,
        )
        alpha = 1.0 - np.exp(-dt / np.maximum(tau, 1e-6))
        smoothed_values[:] = smoothed_values + (target_values - smoothed_values) * alpha

        loudness_index = int(np.searchsorted(loudness_times, target_time))
        loudness_index = int(clamp(loudness_index, 0, max(last_loudness_index, 0)))
        target_loudness = float(loudness_values[loudness_index])

        current_speaker_value = speaker_state["value"]
        speaker_tau = (
            SPEAKER_ATTACK_SECONDS
            if target_loudness > current_speaker_value
            else SPEAKER_DECAY_SECONDS
        )
        speaker_alpha = 1.0 - np.exp(-dt / max(speaker_tau, 1e-6))
        speaker_state["value"] = current_speaker_value + (target_loudness - current_speaker_value) * speaker_alpha

        progress = clamp(t / max(duration, 1e-6), 0.0, 1.0)
        top_color, bottom_color = shift_gradient_hue(
            VISUALIZER_BACKGROUND_TOP_COLOR,
            VISUALIZER_BACKGROUND_BOTTOM_COLOR,
            progress * VISUALIZER_BACKGROUND_HUE_DRIFT,
        )
        background = build_gradient_background(width, height, top_color, bottom_color)

        pulse = beat_pulse_at(
            t,
            beat_times_local,
            VISUALIZER_BEAT_PULSE_DECAY_SECONDS,
            VISUALIZER_BEAT_PULSE_ATTACK_SECONDS,
        )

        return draw_bars_frame(
            background, width, height, smoothed_values,
            color_low, color_high, pulse, progress,
            speaker_state["value"],
        )

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def make_title_card_clip(
    text: str,
    size: tuple,
    duration: float,
    font_path: str = None,
):
    """
    Build a title-card overlay clip: a semi-transparent rounded pill
    behind uppercase text, with a thin accent line underneath, held at
    the top of the frame for the given duration so it stays visible
    and consistent for the whole song segment.
    """
    width, height = size
    font_size = max(24, int(height * TITLE_FONT_SIZE_RATIO))

    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default(size=font_size)
    except Exception:
        font = ImageFont.load_default()

    display_text = text.upper()

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), display_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    pad_x = int(text_height * 1.2)
    pad_y = int(text_height * 0.6)

    pill_width = text_width + pad_x * 2
    pill_height = text_height + pad_y * 2
    pill_x = (width - pill_width) / 2.0
    pill_y = height * 0.04

    pill_box = [pill_x, pill_y, pill_x + pill_width, pill_y + pill_height]
    pill_fill = TITLE_PILL_COLOR + (TITLE_PILL_OPACITY,)

    round_rect = getattr(draw, "rounded_rectangle", None)

    if round_rect:
        round_rect(pill_box, radius=pill_height / 2.0, fill=pill_fill)
    else:
        draw.rectangle(pill_box, fill=pill_fill)

    text_x = pill_x + pad_x - bbox[0]
    text_y = pill_y + pad_y - bbox[1]

    draw.text((text_x, text_y), display_text, font=font, fill=(255, 255, 255, 255))

    accent_height = max(2, int(text_height * 0.08))
    accent_width = min(pill_width * 0.4, text_width * 0.5)
    accent_x = (width - accent_width) / 2.0
    accent_y = pill_y + pill_height + max(4, int(text_height * 0.15))

    draw.rectangle(
        [accent_x, accent_y, accent_x + accent_width, accent_y + accent_height],
        fill=TITLE_ACCENT_COLOR + (255,),
    )

    array = np.array(image)
    rgb = array[:, :, :3]
    alpha = array[:, :, 3] / 255.0

    clip = ImageClip(rgb).with_duration(duration)
    mask_clip = ImageClip(alpha, is_mask=True).with_duration(duration)
    clip = clip.with_mask(mask_clip)

    fade_time = min(0.6, duration / 3.0)
    clip = clip.with_effects([FadeIn(fade_time)])

    return clip


def make_tracklist_clip(
    resolved_titles: list,
    size: tuple,
    duration: float,
    font_path: str = None,
):
    """
    Build a static tracklist card listing every song in play order,
    shown once before the mix begins. Takes titles already resolved
    earlier in build_mix, rather than resolving them again, so any
    titles.json warning for a missing entry only ever prints once per
    track.
    """
    width, height = size
    font_size = max(20, int(height * TITLE_FONT_SIZE_RATIO * 0.6))
    header_font_size = int(font_size * 1.4)

    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
            header_font = ImageFont.truetype(font_path, header_font_size)
        else:
            font = ImageFont.load_default(size=font_size)
            header_font = ImageFont.load_default(size=header_font_size)
    except Exception:
        font = ImageFont.load_default()
        header_font = font

    image = Image.new("RGB", (width, height), VISUALIZER_BACKGROUND_BOTTOM_COLOR)
    draw = ImageDraw.Draw(image)

    header_text = "TRACKLIST"
    header_bbox = draw.textbbox((0, 0), header_text, font=header_font)
    header_height = header_bbox[3] - header_bbox[1]
    header_x = (width - (header_bbox[2] - header_bbox[0])) / 2.0
    header_y = height * 0.12

    draw.text((header_x, header_y), header_text, font=header_font, fill=(255, 255, 255))

    line_y = header_y + header_height * 2.0
    line_spacing = header_height * 1.8

    for index, title_text in enumerate(resolved_titles, start=1):
        line_text = f"{index}. {title_text}"
        line_x = width * 0.12
        draw.text((line_x, line_y), line_text, font=font, fill=(230, 230, 230))
        line_y += line_spacing

        if line_y > height * 0.92:
            break

    array = np.array(image)
    clip = ImageClip(array).with_duration(duration)

    fade_time = min(1.0, duration / 4.0)
    clip = clip.with_effects([FadeIn(fade_time), FadeOut(fade_time)])

    return clip


def build_visualizer_segment(
    clip,
    entry: dict,
    start_time: float,
    end_time: float,
    title_text: str,
    size: tuple,
    fps: float,
    n_bars: int,
    font_path: str = None,
):
    """
    Build the segment clip for one track using the generated spectrum
    visualizer, with the song title held on screen at the top for the
    whole duration. The spectrum is computed only from the trimmed
    segment's own audio range, not the whole original track, and bar
    colors are derived from this track's own key and energy instead of
    a fixed global palette. Falls back to the plain raw footage if the
    track has no audio to visualize.
    """
    duration = end_time - start_time

    if clip.audio is None:
        print(
            f"Warning: {os.path.basename(entry['path'])} has no audio "
            "to visualize. Using raw footage for this segment instead."
        )
        return clip.subclipped(start_time, end_time)

    sr = clip.audio.fps
    segment_audio = extract_clip_audio_array(clip, start_time, end_time, sr)
    segment_mono = (
        np.mean(segment_audio, axis=0) if segment_audio.ndim > 1 else segment_audio
    )

    times, bars = compute_spectrum_bars(segment_mono, sr, n_bars=n_bars)
    loudness_times, loudness_values = compute_loudness_envelope(segment_mono, sr)

    beat_times_local = [
        beat_time - start_time
        for beat_time in entry.get("beat_times", [])
        if start_time <= beat_time <= end_time
    ]

    color_low, color_high = track_color_palette(entry)

    segment_clip = make_visualizer_clip(
        times, bars, duration, size, fps,
        beat_times_local, color_low, color_high,
        loudness_times, loudness_values,
    )

    segment_clip = segment_clip.with_audio(
        clip.audio.subclipped(start_time, end_time)
    )

    title_clip = make_title_card_clip(
        title_text, size, duration, font_path,
    ).with_start(0)

    composed = CompositeVideoClip([segment_clip, title_clip], size=size)
    return composed.with_duration(duration)