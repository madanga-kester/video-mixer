import json
import os
import subprocess
import sys

from .config import (
    SUPPORTED_EXTENSIONS,
    TARGET_LUFS,
    TARGET_TRUE_PEAK,
    TARGET_LRA,
    TITLES_JSON_FILENAME,
)
from .utils import ffmpeg_available


def collect_clips(input_dir: str) -> list:
    if not os.path.isdir(input_dir):
        print(f"Input folder does not exist: {input_dir}")
        sys.exit(1)

    files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not files:
        print(f"No video files found in {input_dir}")
        sys.exit(1)

    return files


def normalize_audio_file(
    input_path: str,
    output_path: str,
) -> bool:
    """
    Normalize audio with FFmpeg loudnorm.

    Returns True when normalization succeeded.
    Returns False if FFmpeg is unavailable or the command fails.
    """
    if not ffmpeg_available():
        return False

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        (
            f"loudnorm="
            f"I={TARGET_LUFS}:"
            f"TP={TARGET_TRUE_PEAK}:"
            f"LRA={TARGET_LRA}"
        ),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        return result.returncode == 0 and os.path.exists(output_path)

    except (OSError, subprocess.SubprocessError):
        return False


def load_titles_map(input_dir: str) -> dict:
    """
    Read titles.json from the input folder if present.

    Expected format:
      {
        "filename.mp4": {"title": "Song Name", "artist": "Artist Name"}
      }
    """
    path = os.path.join(input_dir, TITLES_JSON_FILENAME)

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as titles_file:
            data = json.load(titles_file)

        if not isinstance(data, dict):
            print(
                f"Warning: {TITLES_JSON_FILENAME} did not contain a "
                "JSON object. Ignoring it."
            )
            return {}

        return data

    except (OSError, json.JSONDecodeError) as error:
        print(
            f"Warning: could not read {TITLES_JSON_FILENAME} ({error}). "
            "Falling back to filenames for any missing entries."
        )
        return {}


def resolve_title(entry_path: str, titles_map: dict) -> str:
    """
    Look up the display title/artist for a clip from titles_map.

    Falls back to a cleaned-up filename, with a printed warning, when
    the file has no entry in titles.json.
    """
    filename = os.path.basename(entry_path)
    info = titles_map.get(filename)

    if isinstance(info, dict):
        title = str(info.get("title", "")).strip()
        artist = str(info.get("artist", "")).strip()

        if title and artist:
            return f"{title} - {artist}"

        if title:
            return title

    print(
        f"Warning: no titles.json entry for {filename}. "
        "Using a cleaned-up filename instead."
    )

    cleaned = os.path.splitext(filename)[0]
    cleaned = cleaned.replace("_", " ").replace("-", " ")

    return cleaned.title()