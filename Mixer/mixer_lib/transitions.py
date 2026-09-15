import random

from .config import (
    ENABLE_SLIDE_TRANSITIONS,
    ENABLE_FADE_BLACK_TRANSITIONS,
    HIGH_ENERGY,
    LOW_ENERGY,
    CROSSFADE_BARS_HIGH_ENERGY,
    CROSSFADE_BARS_MID_ENERGY,
    CROSSFADE_BARS_LOW_ENERGY,
    MIN_CROSSFADE_SECONDS,
    MAX_CROSSFADE_SECONDS,
)
from .utils import clamp


def choose_transition(
    current_analysis: dict,
    next_analysis: dict,
    previous_transition: str | None = None,
) -> str:
    bpm_difference = abs(
        current_analysis["bpm"] - next_analysis["bpm"]
    )

    energy_difference = abs(
        current_analysis["energy"] - next_analysis["energy"]
    )

    if energy_difference > 0.35:
        choices = ["zoom_punch", "fade_white", "crossfade"]
        weights = [0.40, 0.30, 0.30]

    elif bpm_difference <= 5:
        if ENABLE_SLIDE_TRANSITIONS:
            choices = ["crossfade", "crossfade", "slide"]
            weights = [0.60, 0.25, 0.15]
        else:
            choices = ["crossfade"]
            weights = [1.0]

    else:
        if ENABLE_SLIDE_TRANSITIONS:
            choices = ["crossfade", "slide", "fade_white"]
            weights = [0.55, 0.25, 0.20]
        else:
            choices = ["crossfade", "fade_white"]
            weights = [0.70, 0.30]

    if ENABLE_FADE_BLACK_TRANSITIONS:
        choices = choices + ["fade_black"]
        weights = weights + [0.15]

    if previous_transition in choices:
        alternatives = [
            (choice, weight)
            for choice, weight in zip(choices, weights)
            if choice != previous_transition
        ]

        if alternatives:
            choices = [item[0] for item in alternatives]
            weights = [item[1] for item in alternatives]

    return random.choices(
        choices,
        weights=weights,
        k=1,
    )[0]


def choose_crossfade_duration(
    current_analysis: dict,
    next_analysis: dict,
) -> float:
    """
    Pick a transition length in musical bars rather than a fixed number of
    seconds, so the blend always lines up with the beat grid and lasts long
    enough for a real EQ-swap style blend to actually be audible. Higher
    energy material gets a shorter, punchier blend; calmer material gets a
    longer, more gradual one.
    """
    avg_bpm = (current_analysis["bpm"] + next_analysis["bpm"]) / 2.0
    avg_energy = (current_analysis["energy"] + next_analysis["energy"]) / 2.0
    seconds_per_bar = (60.0 / max(avg_bpm, 1.0)) * 4.0

    if avg_energy >= HIGH_ENERGY:
        bars = random.uniform(*CROSSFADE_BARS_HIGH_ENERGY)
    elif avg_energy <= LOW_ENERGY:
        bars = random.uniform(*CROSSFADE_BARS_LOW_ENERGY)
    else:
        bars = random.uniform(*CROSSFADE_BARS_MID_ENERGY)

    duration = bars * seconds_per_bar

    return clamp(duration, MIN_CROSSFADE_SECONDS, MAX_CROSSFADE_SECONDS)