import random

import numpy as np

from .config import KEY_DIFFERENCE_WEIGHT, ARC_BPM_WEIGHT
from .utils import clamp


def track_distance(current: dict, candidate: dict) -> float:
    """
    Lower score = better musical compatibility.
    """
    bpm_difference = abs(
        current["bpm"] - candidate["bpm"]
    )

    energy_difference = abs(
        current["energy"] - candidate["energy"]
    )

    onset_difference = abs(
        current["onset_density"] - candidate["onset_density"]
    )

    centroid_a = current.get("spectral_centroid", 0.0)
    centroid_b = candidate.get("spectral_centroid", 0.0)

    centroid_difference = (
        abs(centroid_a - centroid_b)
        / max(centroid_a, centroid_b, 1.0)
    )

    profile_a = np.asarray(
        current.get("chroma_profile", [1.0 / 12.0] * 12),
        dtype=float,
    )
    profile_b = np.asarray(
        candidate.get("chroma_profile", [1.0 / 12.0] * 12),
        dtype=float,
    )

    norm_a = float(np.linalg.norm(profile_a))
    norm_b = float(np.linalg.norm(profile_b))

    if norm_a > 0 and norm_b > 0:
        chroma_similarity = float(
            np.dot(profile_a, profile_b) / (norm_a * norm_b)
        )
    else:
        chroma_similarity = 1.0

    key_difference = 1.0 - clamp(chroma_similarity, -1.0, 1.0)

    return (
        bpm_difference * 0.50
        + energy_difference * 25.0
        + onset_difference * 8.0
        + centroid_difference * 3.0
        + key_difference * KEY_DIFFERENCE_WEIGHT
    )


def order_tracks(entries: list, arc_bpms: np.ndarray = None) -> list:
    """
    Build a path through the tracks.

    This avoids the robotic "sort BPM ascending" behavior. The mix opens
    with the track that has the strongest immediate energy and onset
    activity, then chooses the next compatible track while adding a small
    amount of controlled variation. When arc_bpms is provided, each pick is
    also nudged toward that position's target tempo so the whole set moves
    through a gradual tempo arc instead of only satisfying pairwise
    compatibility one neighbor at a time.
    """
    if len(entries) <= 1:
        return entries

    remaining = entries.copy()

    remaining.sort(
        key=lambda e: (
            -(e["energy"] * 0.5 + e["onset_density"] * 0.5),
            e.get("intro_skip", 0.0),
        )
    )

    ordered = [remaining.pop(0)]

    while remaining:
        current = ordered[-1]
        position = len(ordered)

        scored = []
        for candidate in remaining:
            score = track_distance(current, candidate)

            if arc_bpms is not None and position < len(arc_bpms):
                arc_target = float(arc_bpms[position])
                score += abs(candidate["bpm"] - arc_target) * ARC_BPM_WEIGHT

            scored.append((score, candidate))

        scored.sort(key=lambda item: item[0])

        pool_size = min(3, len(scored))
        pool = scored[:pool_size]

        weights = []
        for index, (score, _) in enumerate(pool):
            weights.append(1.0 / (1.0 + score + index * 0.25))

        chosen_index = random.choices(
            range(len(pool)),
            weights=weights,
            k=1,
        )[0]

        chosen = pool[chosen_index][1]
        ordered.append(chosen)

        remaining.remove(chosen)

    return ordered