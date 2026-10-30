
import logging
import time
import os

from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

logger = logging.getLogger("video_summary.transcription")

MODEL_ID = os.getenv(
    "MODEL_ID",
    "openai/whisper-small"
)

HF_TOKEN = os.getenv("HF_TOKEN")


# --------------------------------------------------
# 2. Load Whisper model once
# --------------------------------------------------

logger.info("Loading Whisper model: %s", MODEL_ID)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_ID,
    token=HF_TOKEN
)

logger.info("Whisper model loaded successfully.")


# --------------------------------------------------
# 3. Audio → Transcript
# --------------------------------------------------

def transcribe_audio(audio_path: str | Path) -> str:

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    logger.info(
        "[WHISPER] Starting transcription: %s",
        audio_path.name
    )

    start_time = time.perf_counter()

    result = pipe(
        str(audio_path),
        return_timestamps=True
    )

    transcript = result["text"].strip()

    elapsed = time.perf_counter() - start_time

    logger.info(
        "[WHISPER] Transcription completed in %.2f seconds",
        elapsed
    )

    logger.info(
        "[WHISPER] Transcript length: %d characters",
        len(transcript)
    )

    return transcript


# --------------------------------------------------
# 4. Local testing
# --------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    audio_path = "audio/Original_recording5.mp3"

    transcript = transcribe_audio(audio_path)

    print("\n--- TRANSCRIPT ---")
    print(transcript)

    print("\nTranscript length:", len(transcript))
