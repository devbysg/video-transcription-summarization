
import os
import shutil
import tempfile
import logging
import time

from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Environment and logging configuration
# --------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger("video_summary")


# --------------------------------------------------
# 2. Imports
# --------------------------------------------------

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from langchain_groq import ChatGroq

from video_processor import download_video, extract_audio
from transcription import transcribe_audio
from summarizer import summarize_transcript


# --------------------------------------------------
# 3. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Video Transcription and Summarization API",
    description="Upload a video or provide a supported video URL.",
    version="1.0.0"
)


# --------------------------------------------------
# 4. Initialize Groq LLM once
# --------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=groq_api_key
)


# --------------------------------------------------
# 5. Health check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Video Summarization API is running."
    }


@app.get("/health")
def health_check():

    return {
        "status": "ok"
    }


# --------------------------------------------------
# 6. Main video processing endpoint
# --------------------------------------------------

@app.post("/process")
def process_video(
    url: str | None = Form(default=None),
    file: UploadFile | None = File(default=None)
):

    request_start = time.perf_counter()

    # Normalize URL
    url = url.strip() if url and url.strip() else None

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------

    if url is None and file is None:

        raise HTTPException(
            status_code=400,
            detail="Provide either a video URL or upload a video."
        )

    if url is not None and file is not None:

        raise HTTPException(
            status_code=400,
            detail="Provide either a URL or a file, not both."
        )

    source = "upload" if file is not None else "url"

    logger.info("=" * 60)
    logger.info("NEW VIDEO PROCESSING REQUEST")
    logger.info("Source: %s", source)

    try:

        # --------------------------------------------------
        # Temporary processing directory
        # --------------------------------------------------

        with tempfile.TemporaryDirectory(
            prefix="video_summary_"
        ) as temp_dir:

            temp_dir = Path(temp_dir)

            # --------------------------------------------------
            # STEP 1: Obtain video
            # --------------------------------------------------

            stage_start = time.perf_counter()

            logger.info("[1/5] Obtaining video...")

            if file is not None:

                extension = Path(
                    file.filename or "video.mp4"
                ).suffix.lower() or ".mp4"

                allowed_extensions = {
                    ".mp4",
                    ".mov",
                    ".mkv",
                    ".webm",
                    ".avi",
                    ".m4v"
                }

                if extension not in allowed_extensions:

                    raise HTTPException(
                        status_code=400,
                        detail="Unsupported video format."
                    )

                video_path = temp_dir / f"sample{extension}"

                with open(video_path, "wb") as buffer:

                    shutil.copyfileobj(
                        file.file,
                        buffer
                    )

                logger.info(
                    "[1/5] Uploaded video saved: %s",
                    video_path.name
                )

            else:

                video_path = download_video(
                    url,
                    temp_dir
                )

                logger.info(
                    "[1/5] Video downloaded: %s",
                    video_path.name
                )

            logger.info(
                "[1/5] Video ready | Duration: %.2f seconds",
                time.perf_counter() - stage_start
            )

            # --------------------------------------------------
            # STEP 2: Video → Audio
            # --------------------------------------------------

            stage_start = time.perf_counter()

            logger.info("[2/5] Starting FFmpeg audio extraction...")

            audio_path = extract_audio(
                video_path,
                temp_dir
            )

            logger.info(
                "[2/5] Audio extracted: %s | Time: %.2f seconds",
                audio_path.name,
                time.perf_counter() - stage_start
            )

            # --------------------------------------------------
            # STEP 3: Audio → Transcript
            # --------------------------------------------------

            stage_start = time.perf_counter()

            logger.info("[3/5] Starting Whisper transcription...")

            transcript = transcribe_audio(
                audio_path
            )

            logger.info(
                "[3/5] Transcription completed | "
                "Characters: %d | Time: %.2f seconds",
                len(transcript),
                time.perf_counter() - stage_start
            )

            # --------------------------------------------------
            # STEP 4: Transcript → Summary
            # --------------------------------------------------

            stage_start = time.perf_counter()

            if len(transcript) < 200:

                logger.info(
                    "[4/5] Transcript shorter than 200 characters. "
                    "Skipping LLM summarization."
                )

            else:

                logger.info(
                    "[4/5] Starting Groq summarization..."
                )

            result = summarize_transcript(
                transcript,
                llm
            )

            logger.info(
                "[4/5] Summarization stage completed | Time: %.2f seconds",
                time.perf_counter() - stage_start
            )

            # --------------------------------------------------
            # STEP 5: Prepare response
            # --------------------------------------------------

            logger.info("[5/5] Preparing JSON response...")

            response = {
                "status": "success",
                "source": source,
                "video_name": video_path.name,
                "audio_name": audio_path.name,
                "transcript_length": len(transcript),
                **result
            }

        # TemporaryDirectory has now been deleted.

        logger.info(
            "[5/5] Temporary video and audio files deleted."
        )

        logger.info(
            "REQUEST COMPLETED SUCCESSFULLY | Total time: %.2f seconds",
            time.perf_counter() - request_start
        )

        logger.info("=" * 60)

        return response

    except HTTPException as e:

        logger.warning(
            "Request rejected: %s",
            e.detail
        )

        raise

    except Exception as e:

        logger.exception("Video processing failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        ) from e

    finally:

        if file is not None:
            file.file.close()
