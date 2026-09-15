import shutil

from .config import ZOOM_PUNCH_MAX_SCALE


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp_color(color_a: tuple, color_b: tuple, t: float) -> tuple:
    if len(color_a) != len(color_b):
        raise ValueError("lerp_color: color_a and color_b must have the same number of channels")

    t = clamp(float(t), 0.0, 1.0)

    result = []
    for channel_a, channel_b in zip(color_a, color_b):
        interpolated = channel_a + (channel_b - channel_a) * t
        result.append(int(round(clamp(interpolated, 0, 255))))

    return tuple(result)


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_zoom_punch(clip, duration: float, punching_in: bool):
    """
    Apply a time-varying zoom to the start or end of a clip.

    punching_in True: the clip starts zoomed in and settles to normal
    size over the given duration. Used on the incoming clip.

    punching_in False: the clip zooms in toward the given duration before
    it ends. Used on the outgoing clip, right before the cut.
    """
    clip_duration = safe_float(clip.duration, 0.0)

    if duration <= 0 or clip_duration <= 0:
        return clip

    def scale_factor(t):
        if punching_in:
            progress = clamp(t / duration, 0.0, 1.0)
            return ZOOM_PUNCH_MAX_SCALE - (ZOOM_PUNCH_MAX_SCALE - 1.0) * progress
        else:
            time_remaining = clip_duration - t
            progress = clamp(time_remaining / duration, 0.0, 1.0)
            return ZOOM_PUNCH_MAX_SCALE - (ZOOM_PUNCH_MAX_SCALE - 1.0) * progress

    return clip.resized(scale_factor)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None