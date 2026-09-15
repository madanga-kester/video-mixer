import os
import random
from pathlib import Path

import numpy as np
from moviepy import VideoFileClip, CompositeVideoClip
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut, SlideIn, SlideOut
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut, MultiplyVolume

from .config import (
    DEFAULT_SEGMENT_SECONDS,
    DEFAULT_VISUALIZER_BARS,
    MIN_DURATION_FOR_SMART_BLEND,
    SLIDE_SIDES,
    ARC_BPM_PERCENTILE_LOW,
    ARC_BPM_PERCENTILE_HIGH,
    VOCAL_START_WEIGHT,
    TIMING_JITTER_SECONDS,
    TRACKLIST_DURATION_SECONDS,
    TRACKLIST_CROSSFADE_SECONDS,
    TARGET_LUFS,
)
from .utils import clamp, safe_float, ffmpeg_available, apply_zoom_punch
from .audio_analysis import analyze_track
from .beat_utils import nearest_phrase, select_low_vocal_time, choose_segment_length
from .track_ordering import order_tracks
from .transitions import choose_transition, choose_crossfade_duration
from .dsp_effects import choose_dj_effect, attach_smart_audio
from .file_io import collect_clips, load_titles_map, resolve_title, normalize_audio_file
from .visualizer import build_visualizer_segment, make_tracklist_clip


def build_mix(
    input_dir: str,
    output_path: str,
    segment_length: float = DEFAULT_SEGMENT_SECONDS,
    crossfade_duration: float = 3.0,
    order_by_bpm: bool = True,
    skip_intro: bool = True,
    beat_lock: bool = True,
    resolution: tuple = None,
    normalize_loudness: bool = True,
    smart_audio: bool = True,
    use_visualizer: bool = False,
    visualizer_bars: int = DEFAULT_VISUALIZER_BARS,
    title_font: str = None,
    dj_effects: bool = True,
    show_tracklist: bool = True,
):
    files = collect_clips(input_dir)

    titles_map = load_titles_map(input_dir) if use_visualizer else {}

    if normalize_loudness and not ffmpeg_available():
        print(
            "Warning: FFmpeg not found on PATH. Final loudness "
            "normalization will be skipped. Install FFmpeg or pass "
            "--no-normalize to silence this warning."
        )

    print(f"Found {len(files)} clip(s). Analyzing tracks...")

    entries = []

    for file_path in files:
        try:
            analysis = analyze_track(file_path)
        except Exception as error:
            print(
                f"  Skipping {os.path.basename(file_path)}: "
                f"analysis failed ({error})"
            )
            continue

        print(
            f"  {os.path.basename(file_path)}: "
            f"{analysis['bpm']} BPM | "
            f"energy {analysis['energy']:.2f} | "
            f"onsets {analysis['onset_density']:.2f} | "
            f"{len(analysis['beat_times'])} beats | "
            f"{len(analysis['section_times'])} structure points"
        )

        if skip_intro and analysis["intro_skip"] > 0:
            print(
                f"      intro skip: "
                f"{analysis['intro_skip']}s"
            )

        entries.append({
            "path": file_path,
            **analysis,
        })

    if not entries:
        raise RuntimeError("No clips could be analyzed successfully.")

    bpm_values = np.array([entry["bpm"] for entry in entries], dtype=float)

    if len(entries) > 1:
        low_arc_bpm = float(np.percentile(bpm_values, ARC_BPM_PERCENTILE_LOW))
        high_arc_bpm = float(np.percentile(bpm_values, ARC_BPM_PERCENTILE_HIGH))

        if high_arc_bpm < low_arc_bpm:
            low_arc_bpm, high_arc_bpm = high_arc_bpm, low_arc_bpm

        arc_bpms = np.linspace(low_arc_bpm, high_arc_bpm, len(entries))
    else:
        arc_bpms = bpm_values

    if order_by_bpm:
        entries = order_tracks(entries, arc_bpms)

        print("\nIntelligent mix order:")
        for index, entry in enumerate(entries, start=1):
            print(
                f"  {index}. "
                f"{os.path.basename(entry['path'])} "
                f"({entry['bpm']} BPM, "
                f"energy {entry['energy']:.2f})"
            )
    else:
        print("\nKeeping filename order.")

    resolved_titles = []

    if use_visualizer:
        for entry in entries:
            resolved_titles.append(resolve_title(entry["path"], titles_map))
    else:
        resolved_titles = [None] * len(entries)

    target_res = resolution
    target_fps = None
    base_clips = []
    metadata = []

    positive_levels = [
        entry["mean_rms"]
        for entry in entries
        if entry.get("mean_rms", 0.0) > 0.0
    ]

    target_level = (
        float(np.mean(positive_levels))
        if positive_levels
        else 0.0
    )

    print("\nPreparing musical segments...")

    for track_index, entry in enumerate(entries):
        clip = VideoFileClip(entry["path"])

        if target_fps is None:
            target_fps = safe_float(clip.fps, 30.0)

        if target_res is None:
            target_res = (clip.w, clip.h)
        else:
            source_aspect = clip.w / clip.h
            target_aspect = target_res[0] / target_res[1]

            if abs(source_aspect - target_aspect) > 0.01:
                if source_aspect > target_aspect:
                    scaled_height = target_res[1]
                    scaled_width = int(round(scaled_height * source_aspect))
                else:
                    scaled_width = target_res[0]
                    scaled_height = int(round(scaled_width / source_aspect))

                clip = clip.resized(new_size=(scaled_width, scaled_height))
                clip = clip.cropped(
                    x_center=scaled_width / 2,
                    y_center=scaled_height / 2,
                    width=target_res[0],
                    height=target_res[1],
                )
            else:
                clip = clip.resized(new_size=target_res)

        chosen_length = choose_segment_length(
            entry,
            segment_length,
        )

        is_opening_track = (track_index == 0)

        if is_opening_track:
            raw_start = 0.0
        else:
            raw_start = (
                entry["intro_skip"]
                if skip_intro
                else 0.0
            )

        if not is_opening_track and entry["section_times"]:
            valid_sections = [
                t for t in entry["section_times"]
                if t >= raw_start
                and t <= raw_start + 20.0
            ]

            if valid_sections:
                vocal_times_arr = np.asarray(entry.get("vocal_times", []))
                vocal_activity_arr = np.asarray(entry.get("vocal_activity", []))

                def section_score(t, _raw_start=raw_start,
                                   _vt=vocal_times_arr, _va=vocal_activity_arr):
                    proximity = abs(t - _raw_start)
                    if len(_vt) == 0:
                        return proximity
                    idx = int(np.argmin(np.abs(_vt - t)))
                    return proximity + float(_va[idx]) * VOCAL_START_WEIGHT

                raw_start = min(valid_sections, key=section_score)

        raw_end = min(
            raw_start + chosen_length,
            clip.duration,
        )

        if beat_lock and entry["beat_times"]:
            start_time = nearest_phrase(
                raw_start,
                entry["beat_times"],
            )

            end_time = nearest_phrase(
                raw_end,
                entry["beat_times"],
            )

            end_time = select_low_vocal_time(
                end_time,
                entry["beat_times"],
                entry.get("vocal_times", []),
                entry.get("vocal_activity", []),
            )
        else:
            start_time = raw_start
            end_time = raw_end

        if beat_lock and entry["beat_times"]:
            start_time += random.uniform(
                -TIMING_JITTER_SECONDS,
                TIMING_JITTER_SECONDS,
            )

        start_time = clamp(
            start_time,
            0.0,
            max(0.0, clip.duration - 2.0),
        )

        end_time = clamp(
            end_time,
            start_time + 5.0,
            clip.duration,
        )

        if end_time <= start_time + 5.0:
            start_time = max(
                0.0,
                min(
                    raw_start,
                    clip.duration - 5.0,
                ),
            )

            end_time = min(
                start_time + chosen_length,
                clip.duration,
            )

        if use_visualizer:
            title_text = resolved_titles[track_index]
            segment = build_visualizer_segment(
                clip,
                entry,
                start_time,
                end_time,
                title_text,
                target_res,
                target_fps if target_fps else 30,
                visualizer_bars,
                title_font,
            )
        else:
            segment = clip.subclipped(
                start_time,
                end_time,
            )

        entry_level = entry.get("mean_rms", 0.0)

        if target_level > 0.0 and entry_level > 0.0 and segment.audio is not None:
            gain = clamp(target_level / entry_level, 0.5, 2.0)
            segment = segment.with_effects([MultiplyVolume(gain)])

        base_clips.append(segment)

        metadata.append({
            "entry": entry,
            "start": start_time,
            "end": end_time,
            "duration": segment.duration,
        })

        print(
            f"  {os.path.basename(entry['path'])}: "
            f"{start_time:.2f}s -> {end_time:.2f}s "
            f"({segment.duration:.1f}s)"
        )

    if not base_clips:
        raise RuntimeError("No usable video clips were created.")

    transitions = []

    previous_transition = None

    for i in range(len(base_clips) - 1):
        current_analysis = metadata[i]["entry"]
        next_analysis = metadata[i + 1]["entry"]

        transition = choose_transition(
            current_analysis,
            next_analysis,
            previous_transition,
        )

        duration = choose_crossfade_duration(
            current_analysis,
            next_analysis,
        )

        applied_duration = min(
            duration,
            base_clips[i].duration / 2.0,
            base_clips[i + 1].duration / 2.0,
        )

        if i + 1 < len(arc_bpms):
            target_bpm = (float(arc_bpms[i]) + float(arc_bpms[i + 1])) / 2.0
        else:
            target_bpm = (current_analysis["bpm"] + next_analysis["bpm"]) / 2.0

        transition_entry = {
            "type": transition,
            "duration": duration,
            "applied_duration": applied_duration,
            "target_bpm": target_bpm,
            "effect": choose_dj_effect(applied_duration, dj_effects),
        }

        if transition == "slide":
            transition_entry["side"] = random.choice(SLIDE_SIDES)

        transitions.append(transition_entry)

        previous_transition = transition

    print("\nTransition plan:")

    for i, transition in enumerate(transitions, start=1):
        side_note = (
            f" [{transition['side']}]"
            if transition["type"] == "slide"
            else ""
        )

        effect_note = (
            f" + {transition['effect']}"
            if transition.get("effect")
            else ""
        )

        print(
            f"  {i}: "
            f"{transition['type']}{side_note}{effect_note} "
            f"({transition['duration']:.2f}s)"
        )

    positioned_clips = []
    smart_audio_failures = 0

    tracklist_duration = (
        TRACKLIST_DURATION_SECONDS
        if use_visualizer and show_tracklist
        else 0.0
    )

    tracklist_overlap = 0.0

    if tracklist_duration > 0.0:
        tracklist_overlap = min(
            TRACKLIST_CROSSFADE_SECONDS,
            tracklist_duration / 2.0,
            base_clips[0].duration / 2.0,
        )

        tracklist_clip = make_tracklist_clip(
            resolved_titles, target_res, tracklist_duration, title_font,
        ).with_start(0.0)

        if tracklist_overlap > 0.0:
            tracklist_clip = tracklist_clip.with_effects(
                [CrossFadeOut(tracklist_overlap)]
            )

        positioned_clips.append(tracklist_clip)

    current_start = tracklist_duration - tracklist_overlap

    for i, clip in enumerate(base_clips):
        transition_in = (
            transitions[i - 1]
            if i > 0
            else None
        )

        transition_out = (
            transitions[i]
            if i < len(transitions)
            else None
        )

        video_effects = []
        simple_audio_in = False
        simple_audio_out = False

        if i == 0 and tracklist_overlap > 0.0:
            video_effects.append(CrossFadeIn(tracklist_overlap))

        if transition_in:
            duration = transition_in["applied_duration"]
            transition_in_type = transition_in["type"]

            if transition_in_type == "crossfade":
                video_effects.append(CrossFadeIn(duration))

            elif transition_in_type == "fade_black":
                video_effects.append(FadeIn(duration))
                simple_audio_in = True

            elif transition_in_type == "fade_white":
                video_effects.append(
                    FadeIn(duration, initial_color=(255, 255, 255))
                )

            elif transition_in_type == "slide":
                video_effects.append(
                    SlideIn(duration, transition_in.get("side", "left"))
                )

            elif transition_in_type == "zoom_punch":
                clip = apply_zoom_punch(clip, duration, punching_in=True)

            if simple_audio_in:
                video_effects.append(AudioFadeIn(duration))

        if transition_out:
            duration = transition_out["applied_duration"]
            transition_out_type = transition_out["type"]

            if transition_out_type == "crossfade":
                video_effects.append(CrossFadeOut(duration))

            elif transition_out_type == "fade_black":
                video_effects.append(FadeOut(duration))
                simple_audio_out = True

            elif transition_out_type == "fade_white":
                video_effects.append(
                    FadeOut(duration, final_color=(255, 255, 255))
                )

            elif transition_out_type == "slide":
                video_effects.append(
                    SlideOut(duration, transition_out.get("side", "left"))
                )

            elif transition_out_type == "zoom_punch":
                clip = apply_zoom_punch(clip, duration, punching_in=False)

            if simple_audio_out:
                video_effects.append(AudioFadeOut(duration))

        if video_effects:
            clip = clip.with_effects(video_effects)

        if smart_audio and clip.audio is not None:
            try:
                native_bpm = metadata[i]["entry"]["bpm"]
                clip = attach_smart_audio(
                    clip,
                    transition_in if not simple_audio_in else None,
                    transition_out if not simple_audio_out else None,
                    native_bpm,
                )
            except Exception:
                smart_audio_failures += 1
                fallback_effects = []

                if (
                    not simple_audio_in
                    and transition_in
                    and transition_in["applied_duration"] >= MIN_DURATION_FOR_SMART_BLEND
                ):
                    fallback_effects.append(
                        AudioFadeIn(transition_in["applied_duration"])
                    )

                if (
                    not simple_audio_out
                    and transition_out
                    and transition_out["applied_duration"] >= MIN_DURATION_FOR_SMART_BLEND
                ):
                    fallback_effects.append(
                        AudioFadeOut(transition_out["applied_duration"])
                    )

                if fallback_effects:
                    clip = clip.with_effects(fallback_effects)

        clip = clip.with_start(current_start)
        positioned_clips.append(clip)

        if transition_out:
            overlap = (
                transition_out["applied_duration"]
                if transition_out["type"] == "crossfade"
                else 0.0
            )
        else:
            overlap = 0.0

        current_start = (
            current_start
            + clip.duration
            - overlap
        )

    if smart_audio_failures > 0:
        print(
            f"\n{smart_audio_failures} transition(s) fell back to a "
            "simple audio fade after the real-mixer blend failed to "
            "process."
        )

    total_duration = current_start

    print(
        f"\nFinal duration: "
        f"{total_duration / 60:.2f} minutes"
    )

    print(
        "\nRendering final mix..."
        " This can take a while for long/large clips."
    )

    final = CompositeVideoClip(
        positioned_clips,
        size=target_res,
    ).with_duration(total_duration)

    try:
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=target_fps if target_fps else 30,
            threads=4,
            preset="medium",
        )

        if normalize_loudness and ffmpeg_available():
            normalized_path = (
                str(Path(output_path).with_suffix(""))
                + "_normalized.mp4"
            )

            print(
                "\nApplying final loudness normalization..."
            )

            success = normalize_audio_file(
                output_path,
                normalized_path,
            )

            if success:
                try:
                    os.replace(
                        normalized_path,
                        output_path,
                    )
                    print(
                        f"  Loudness normalized to "
                        f"approximately {TARGET_LUFS} LUFS."
                    )
                except OSError:
                    print(
                        "  Normalized file created but "
                        "could not replace the original."
                    )
            else:
                print(
                    "  Loudness normalization skipped."
                )
        elif normalize_loudness:
            print(
                "\nFFmpeg was not found on PATH. "
                "Skipping loudness normalization."
            )

    finally:
        for clip in base_clips:
            try:
                clip.close()
            except Exception:
                pass

        try:
            final.close()
        except Exception:
            pass

    print(
        f"\nDone. Mix saved to: {output_path}"
    )