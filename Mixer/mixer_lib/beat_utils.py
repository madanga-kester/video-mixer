import random

import numpy as np

from .config import (
    MAX_BEAT_SNAP_DISTANCE,
    BEATS_PER_PHRASE,
    VOCAL_ACTIVITY_SEARCH_PHRASES,
    MIN_SEGMENT_SECONDS,
    MAX_SEGMENT_SECONDS,
    HIGH_ENERGY,
    LOW_ENERGY,
)
from .utils import clamp


def nearest_beat(
    target_time: float,
    beat_times: list,
    max_distance: float = MAX_BEAT_SNAP_DISTANCE,
) -> float:
    if not beat_times:
        return target_time

    beat_array = np.asarray(beat_times, dtype=float)

    closest_idx = int(
        np.argmin(np.abs(beat_array - target_time))
    )

    closest_beat = float(beat_array[closest_idx])

    if abs(closest_beat - target_time) <= max_distance:
        return closest_beat

    return target_time


def nearest_phrase(
    target_time: float,
    beat_times: list,
    beats_per_phrase: int = BEATS_PER_PHRASE,
) -> float:
    """
    Snap to a phrase boundary rather than blindly snapping to the nearest
    individual beat.
    """
    if not beat_times:
        return target_time

    beat_array = np.asarray(beat_times, dtype=float)

    closest_beat_idx = int(
        np.argmin(np.abs(beat_array - target_time))
    )

    phrase_idx = round(
        closest_beat_idx / beats_per_phrase
    ) * beats_per_phrase

    phrase_idx = max(
        0,
        min(phrase_idx, len(beat_array) - 1),
    )

    phrase_time = float(beat_array[phrase_idx])

    if abs(phrase_time - target_time) <= MAX_BEAT_SNAP_DISTANCE * 2:
        return phrase_time

    return nearest_beat(
        target_time,
        beat_times,
        MAX_BEAT_SNAP_DISTANCE,
    )


def select_low_vocal_time(
    candidate_time: float,
    beat_times: list,
    vocal_times: np.ndarray,
    vocal_activity: np.ndarray,
    beats_per_phrase: int = BEATS_PER_PHRASE,
    search_phrases: int = VOCAL_ACTIVITY_SEARCH_PHRASES,
) -> float:
    """
    Nudge a phrase-locked cut point toward a nearby phrase boundary with
    lower vocal activity, so transitions land on instrumental passages
    instead of cutting across a vocal line. Falls back to the original
    candidate whenever vocal data isn't available.
    """
    if (
        not beat_times
        or vocal_times is None
        or len(vocal_times) == 0
        or vocal_activity is None
        or len(vocal_activity) == 0
    ):
        return candidate_time

    beat_array = np.asarray(beat_times, dtype=float)
    vocal_times = np.asarray(vocal_times, dtype=float)
    vocal_activity = np.asarray(vocal_activity, dtype=float)

    closest_beat_idx = int(np.argmin(np.abs(beat_array - candidate_time)))
    phrase_idx = round(closest_beat_idx / beats_per_phrase)

    best_time = candidate_time
    best_score = None
    max_jump = MAX_BEAT_SNAP_DISTANCE * (search_phrases * 2 + 1)

    for offset in range(-search_phrases, search_phrases + 1):
        beat_idx = int(clamp(
            (phrase_idx + offset) * beats_per_phrase,
            0,
            len(beat_array) - 1,
        ))
        phrase_time = float(beat_array[beat_idx])

        if abs(phrase_time - candidate_time) > max_jump:
            continue

        vocal_idx = int(np.argmin(np.abs(vocal_times - phrase_time)))
        score = float(vocal_activity[vocal_idx])

        if best_score is None or score < best_score:
            best_score = score
            best_time = phrase_time

    return best_time


def choose_segment_length(
    analysis: dict,
    requested_length: float,
) -> float:
    """
    Choose a natural-looking clip duration.

    Strong/high-energy sections tend to work better with shorter rotations,
    while calmer material gets more breathing room.
    """
    energy = analysis.get("energy", 0.5)
    onset_density = analysis.get("onset_density", 0.5)

    if energy >= HIGH_ENERGY or onset_density >= 0.75:
        low = max(MIN_SEGMENT_SECONDS, requested_length * 0.65)
        high = min(MAX_SEGMENT_SECONDS, requested_length * 0.90)

    elif energy <= LOW_ENERGY:
        low = max(MIN_SEGMENT_SECONDS, requested_length * 0.95)
        high = min(MAX_SEGMENT_SECONDS, requested_length * 1.35)

    else:
        low = max(MIN_SEGMENT_SECONDS, requested_length * 0.80)
        high = min(MAX_SEGMENT_SECONDS, requested_length * 1.15)

    if high < low:
        high = low

    return float(random.uniform(low, high))