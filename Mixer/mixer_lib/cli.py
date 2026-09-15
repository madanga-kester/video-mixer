import argparse

from .config import DEFAULT_SEGMENT_SECONDS, DEFAULT_VISUALIZER_BARS
from .mixer_core import build_mix


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a humanized, phrase-aware "
            "automatic music video mix."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing input video song files",
    )

    parser.add_argument(
        "--output",
        default="mix.mp4",
        help="Output mixed video file path",
    )

    parser.add_argument(
        "--segment",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help=(
            "Base segment length in seconds. "
            "Actual lengths vary automatically."
        ),
    )

    parser.add_argument(
        "--crossfade",
        type=float,
        default=3.0,
        help=(
            "Legacy/default transition length. "
            "Actual transition lengths are selected dynamically "
            "in bars, not seconds."
        ),
    )

    parser.add_argument(
        "--no-bpm-sort",
        action="store_true",
        help=(
            "Keep clips in filename order instead "
            "of using intelligent musical ordering."
        ),
    )

    parser.add_argument(
        "--no-skip-intro",
        action="store_true",
        help=(
            "Play from the start of each clip "
            "instead of estimating the intro."
        ),
    )

    parser.add_argument(
        "--no-beat-lock",
        action="store_true",
        help=(
            "Disable beat/phrase locking."
        ),
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help=(
            "Skip final FFmpeg loudness normalization."
        ),
    )

    parser.add_argument(
        "--no-smart-audio",
        action="store_true",
        help=(
            "Disable the bass-swap EQ / tempo-match / equal-power blend "
            "and fall back to simple linear audio fades on every "
            "transition."
        ),
    )

    parser.add_argument(
        "--no-dj-effects",
        action="store_true",
        help=(
            "Disable synthesized DJ effects (siren, air horn, scratch, "
            "riser, impact) that are occasionally layered under "
            "transitions, each fading in and out."
        ),
    )

    parser.add_argument(
        "--visualizer",
        action="store_true",
        help=(
            "Replace raw video footage with a generated spectrum "
            "visualizer plus a title card held at the top of the "
            "screen for the whole segment. Reads titles.json from "
            "the input folder if present."
        ),
    )

    parser.add_argument(
        "--visualizer-bars",
        type=int,
        default=DEFAULT_VISUALIZER_BARS,
        help="Number of frequency bars in the visualizer.",
    )

    parser.add_argument(
        "--title-font",
        default=None,
        help=(
            "Path to a .ttf font file for title cards. Omit to use a "
            "plain built-in font."
        ),
    )

    parser.add_argument(
        "--no-tracklist",
        action="store_true",
        help=(
            "Skip the tracklist card normally shown before the mix "
            "begins when --visualizer is used."
        ),
    )

    args = parser.parse_args()

    if args.segment <= 0:
        parser.error("--segment must be greater than 0")

    if args.visualizer_bars < 4:
        parser.error("--visualizer-bars must be at least 4")

    build_mix(
        input_dir=args.input,
        output_path=args.output,
        segment_length=args.segment,
        crossfade_duration=args.crossfade,
        order_by_bpm=not args.no_bpm_sort,
        skip_intro=not args.no_skip_intro,
        beat_lock=not args.no_beat_lock,
        normalize_loudness=not args.no_normalize,
        smart_audio=not args.no_smart_audio,
        use_visualizer=args.visualizer,
        visualizer_bars=args.visualizer_bars,
        title_font=args.title_font,
        dj_effects=not args.no_dj_effects,
        show_tracklist=not args.no_tracklist,
    )