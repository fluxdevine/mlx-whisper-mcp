# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]",
#     "mlx-whisper",
#     "rich",
# ]
# ///

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("whisper-mcp")

console = Console()

try:
    import mlx_whisper

    console.log("[green]Successfully imported mlx_whisper[/green]")
except ImportError:
    console.log("[red]Failed to import mlx_whisper[/red]")
    raise ImportError(
        "Error: MLX Whisper not installed. Install with 'uv pip install mlx-whisper'."
    )


server = FastMCP("mlx-whisper")

MODEL_PATH = "mlx-community/whisper-large-v3-turbo"


@server.tool()
async def transcribe_file(
    file_path: str | Path,
    language: Optional[str] = "en",
    task: str = "transcribe",  # Task: transcribe or translate
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
        result = mlx_whisper.transcribe(
            file_path, path_or_hf_repo=MODEL_PATH, language=language, task=task
        )

        return f"Transcription:\n\n{result['text']}"

    except Exception as e:
        log.error(f"Error transcribing audio file: {str(e)}")


@server.tool()
async def transcribe_audio(
    audio_data: str,  # Base64 encoded audio data
    language: Optional[str] = None,  # Optional language code (e.g., "en", "fr")
    file_format: str = "wav",  # Audio file format
    task: str = "transcribe",  # Task: transcribe or translate
) -> str:
    """Transcribe audio using MLX Whisper.

    Args:
        audio_data: Base64 encoded audio data
        language: Optional language code to force a specific language
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

        result = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=MODEL_PATH, language=language, task=task
        )

        os.unlink(audio_path)

        return f"Transcription:\n\n{result['text']}"

    except Exception as e:
        log.error(f"Error transcribing audio: {str(e)}")


if __name__ == "__main__":
    console.print("[bold green]Starting MLX Whisper MCP Server[/bold green]")
    console.print(f"[yellow]Using model:[/yellow] {MODEL_PATH}")

    console.print("[bold]Running with stdio transport...[/bold]")
    server.run(transport="stdio")
