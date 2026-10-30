
import os
import subprocess
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv


load_dotenv()

FFMPEG_PATH = Path(os.getenv("FFMPEG_PATH", ""))


def validate_ffmpeg():
    if not FFMPEG_PATH.is_file():
        raise FileNotFoundError(
            f"FFmpeg executable not found: {FFMPEG_PATH}"
        )


def download_video(url: str, output_dir: str | Path) -> Path:
    """
    Download a video from a URL supported by yt-dlp.
    Saves it as sample.<extension>.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not url or not url.strip():
        raise ValueError("Video URL cannot be empty.")

    validate_ffmpeg()

    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(output_dir / "sample.%(ext)s"),
        "merge_output_format": "mp4",
        "ffmpeg_location": str(FFMPEG_PATH),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        raise RuntimeError(
            f"Video download failed: {e}"
        ) from e

    supported_extensions = {
        ".mp4", ".webm", ".mkv",
        ".mov", ".avi", ".m4v"
    }

    files = [
        path
        for path in output_dir.glob("sample.*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    ]

    if not files:
        raise FileNotFoundError(
            "Downloaded video file was not found."
        )

    # Prefer MP4 if available
    mp4_files = [
        path for path in files
        if path.suffix.lower() == ".mp4"
    ]

    return mp4_files[0] if mp4_files else files[0]


def extract_audio(
    video_path: str | Path,
    output_dir: str | Path
) -> Path:
    """
    Extract audio from video and save as sample.mp3.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    validate_ffmpeg()

    audio_path = output_dir / "sample.mp3"

    try:
        subprocess.run(
            [
                str(FFMPEG_PATH),
                "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "libmp3lame",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path)
            ],
            check=True
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Audio extraction failed: {e}"
        ) from e

    if not audio_path.is_file():
        raise FileNotFoundError(
            "Audio extraction did not create sample.mp3."
        )

    return audio_path


if __name__ == "__main__":

    import tempfile

    test_url = "https://youtu.be/l7tQ1v4TCeY"

    with tempfile.TemporaryDirectory() as temp_dir:

        video = download_video(test_url, temp_dir)

        print("Downloaded video:", video)

        audio = extract_audio(video, temp_dir)

        print("Extracted audio:", audio)
