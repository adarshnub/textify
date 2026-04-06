"""CLI entry point for Textify."""

from __future__ import annotations

import sys

import click

from . import __version__
from .transcriber import transcribe
from .utils import validate_audio_file


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("audio_file", type=click.Path(exists=False))
@click.option(
    "-m",
    "--model",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2"]),
    default="tiny",
    show_default=True,
    help="Whisper model size.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file path. Defaults to stdout.",
)
@click.option(
    "-l",
    "--language",
    type=str,
    default=None,
    help='Language code (e.g. "en", "es"). Auto-detects if not specified.',
)
@click.option(
    "--min-speakers",
    type=int,
    default=None,
    help="Minimum expected number of speakers (diarization hint).",
)
@click.option(
    "--max-speakers",
    type=int,
    default=None,
    help="Maximum expected number of speakers (diarization hint).",
)
@click.option(
    "--no-diarize",
    is_flag=True,
    default=False,
    help="Skip speaker diarization (faster, no HuggingFace token needed).",
)
@click.option(
    "-d",
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
    help="Compute device.",
)
@click.option(
    "--hf-token",
    type=str,
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace token for speaker diarization. Can also set HF_TOKEN env var.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print progress messages to stderr.",
)
@click.version_option(version=__version__, prog_name="textify")
def main(
    audio_file: str,
    model: str,
    output: str | None,
    language: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    no_diarize: bool,
    device: str,
    hf_token: str | None,
    verbose: bool,
) -> None:
    """Transcribe an audio file with word-level timestamps and speaker diarization.

    \b
    Examples:
      textify recording.mp3
      textify meeting.wav -m small -o result.json
      textify podcast.mp3 --max-speakers 3 --verbose
      textify audio.wav --no-diarize -m large-v2
    """
    # Validate input
    audio_path = validate_audio_file(audio_file)

    # Progress callback
    def on_progress(msg: str) -> None:
        if verbose:
            click.echo(f"[textify] {msg}", err=True)

    try:
        result = transcribe(
            audio_path=str(audio_path),
            model_name=model,
            device=device,
            language=language,
            diarize=not no_diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
            on_progress=on_progress,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Serialize to JSON
    json_output = result.model_dump_json(indent=2)

    # Write output
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(json_output)
            f.write("\n")
        if verbose:
            click.echo(f"[textify] Output written to {output}", err=True)
    else:
        click.echo(json_output)


if __name__ == "__main__":
    main()
