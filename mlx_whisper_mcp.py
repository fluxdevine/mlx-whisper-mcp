# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]",
#     "mlx-whisper",
#     "rich",
#     "yt-dlp",
# ]
# ///

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP
from rich.logging import RichHandler
from rich.console import Console

console = Console(stderr=True)

logging.basicConfig(
    level="NOTSET",
    format="%(filename)s:%(lineno)d - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)

log = logging.getLogger("whisper-mcp")
log.setLevel(logging.INFO)

try:
    import mlx_whisper

    log.info("[green]Successfully imported mlx_whisper[/green]", extra={"markup": True})
except ImportError:
    log.warning(
        "Error: MLX Whisper not installed. Install with 'uv pip install mlx-whisper'."
    )

try:
    from yt_dlp import YoutubeDL

    log.info("[green]Successfully imported yt-dlp[/green]", extra={"markup": True})
except ImportError:
    log.warning("Error: yt-dlp not installed. Install with 'uv pip install yt-dlp'.")


server = FastMCP(
    name = "mlx-whisper",
    dependencies = ["mlx-whisper", "yt-dlp"]
)

MODEL_PATH = "mlx-community/whisper-large-v3-turbo"

HOME_DIR = Path.home()
DATA_DIR = HOME_DIR / ".mlx-whisper-mcp" / "downloads"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@server.tool()
async def transcribe_file(
    file_path: str | Path,
    language: Optional[str] = "en",
    task: Literal["transcribe", "translate"] = "transcribe",
) -> str:
    """Transcribe an audio file from disk using MLX Whisper.

    Args:
        file_path: Path to the audio file
        language: Optional language code to force a specific language
        task: Task to perform (transcribe or translate)

    Returns:
        Transcription text
    """
    assert Path(file_path).exists(), f"File not found: {file_path}"
    try:
        transcript = mlx_whisper.transcribe(
            file_path, path_or_hf_repo=MODEL_PATH, language=language, task=task
        )["text"]

        output_file = str(Path(file_path).with_suffix(".txt"))
        output_file = Path(DATA_DIR) / Path(output_file).name

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        log.info(f"Saving transcript to: {output_file}")

        return f"Transcription:\n\n{transcript}"

    except Exception as e:
        log.error(f"Error transcribing audio file: {str(e)}")
        return f"Error transcribing audio file: {str(e)}"


@server.tool()
async def transcribe_audio(
    audio_data: str,
    language: Optional[str] = "en",
    file_format: str = "wav",
    task: Literal["transcribe", "translate"] = "transcribe",
) -> str:
    """Transcribe audio using MLX Whisper.

    Args:
        audio_data: Base64 encoded audio data
        language: Optional language code to force a specific language. Optional language code (e.g., "en", "fr")
        file_format: Audio file format (wav, mp3, etc.)
        task: Task to perform (transcribe or translate)

    Returns:
        Transcription text
    """
    try:
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_format}", delete=False
        ) as temp_file:
            audio_path = temp_file.name
            audio_bytes = base64.b64decode(audio_data)
            temp_file.write(audio_bytes)

        transcript = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=MODEL_PATH, language=language, task=task
        )["text"]
        output_file = str(Path(audio_path).with_suffix(".txt"))
        output_file = Path(DATA_DIR) / Path(output_file).name
        log.info(f"Saving transcript to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)

        os.unlink(audio_path)

        return f"Transcription:\n\n{transcript}"

    except Exception as e:
        log.error(f"Error transcribing audio: {str(e)}")
        return f"Error transcribing audio: {str(e)}"


@server.tool()
async def download_youtube(url: str, keep_file: bool = True) -> str:
    """Download a YouTube video or extract its audio.

    Args:
        url: YouTube video URL
        keep_file: If True, keeps the file after transcription (stored in ~/.mlx-whisper-mcp/downloads)

    Returns:
        Path to the downloaded file or error message
    """
    try:
        video_id = url.split("v=")[-1].split("&")[0]

        output_path = str(DATA_DIR / f"{video_id}.mp4")
        if Path(output_path).exists():
            log.info(f"File already exists: {output_path}")
            return output_path

        ydl_opts = {
            "format": "best[ext=mp4]",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "merge_output_format": "mp4",
        }

        log.info(f"Downloading audio from: {url}")

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if Path(output_path).exists():
            log.info(f"Download successful: {output_path}")
            return output_path
        else:
            log.error(f"Download failed, file not found: {output_path}")
            return None

    except Exception as e:
        log.error(f"Error downloading from YouTube: {str(e)}")
        return None


@server.tool()
async def transcribe_youtube(
    url: str,
    language: Optional[str] = "en",
    task: str = "transcribe",
    keep_file: bool = True,
) -> str:
    """Transcribe a YouTube video using MLX Whisper."""
    try:
        log.info(f"Transcribing YouTube video: {url}")

        audio_path = await download_youtube(url, keep_file=True)

        if audio_path is None or not Path(audio_path).exists():
            log.error(f"Error downloading video: {audio_path}")
            return f"Error: {audio_path}"

        log.info(f"Beginning transcription of {audio_path}")

        transcript = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=MODEL_PATH, language=language, task=task
        )["text"]

        output_file = str(Path(audio_path).with_suffix(".txt"))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)

        if not keep_file and os.path.exists(audio_path):
            os.unlink(audio_path)
            log.info(f"Deleted temporary file: {audio_path}")
        else:
            log.info(f"Keeping file: {audio_path}")

        return f"Transcription of YouTube video ({url}):\n\n{transcript}"

    except Exception as e:
        log.error(f"Error transcribing YouTube video: {str(e)}")
        return f"Error transcribing YouTube video: {str(e)}"


if __name__ == "__main__":
    log.info(
        "[bold green]Starting MLX Whisper MCP Server[/bold green]",
        extra={"markup": True},
    )
    log.info(f"[yellow]Using model:[/yellow] {MODEL_PATH}", extra={"markup": True})
    log.info("[bold]Running with stdio transport...[/bold]", extra={"markup": True})
    log.info(f"[blue]Download directory:[/blue] {DATA_DIR}", extra={"markup": True})
    server.run(transport="stdio")
