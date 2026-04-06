"""Utility functions for device detection, audio validation, and token resolution."""

from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma"}


def detect_device(preferred: str = "auto") -> str:
    """Return the compute device to use.

    Args:
        preferred: One of "auto", "cpu", or "cuda".

    Returns:
        "cuda" if available and requested, otherwise "cpu".
    """
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda"
    # auto-detect
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_compute_type(device: str) -> str:
    """Return the optimal compute type for the given device.

    Args:
        device: "cuda" or "cpu".

    Returns:
        "float16" for CUDA, "int8" for CPU.
    """
    return "float16" if device == "cuda" else "int8"


def get_batch_size(device: str) -> int:
    """Return the optimal batch size for the given device.

    Args:
        device: "cuda" or "cpu".

    Returns:
        16 for CUDA, 1 for CPU.
    """
    return 16 if device == "cuda" else 1


def validate_audio_file(path: str) -> Path:
    """Validate that the audio file exists and has a supported extension.

    Args:
        path: Path to the audio file.

    Returns:
        Resolved Path object.

    Raises:
        click.BadParameter: If the file doesn't exist or has an unsupported extension.
    """
    import click

    audio_path = Path(path).resolve()
    if not audio_path.exists():
        raise click.BadParameter(f"File not found: {audio_path}")
    if not audio_path.is_file():
        raise click.BadParameter(f"Not a file: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise click.BadParameter(
            f"Unsupported format '{audio_path.suffix}'. Supported: {supported}"
        )
    return audio_path


def get_audio_duration(path: str) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    import librosa
    return librosa.get_duration(path=path)


def resolve_hf_token(cli_token: str | None = None) -> str | None:
    """Resolve the HuggingFace token from CLI arg, env var, or .env file.

    Priority: CLI argument > HF_TOKEN env var > .env file > None.

    Args:
        cli_token: Token passed via CLI flag.

    Returns:
        The resolved token, or None if not found.
    """
    if cli_token:
        return cli_token
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    # Try loading from .env file
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                value = value.strip().strip("'\"")
                if value and value != "your_huggingface_token_here":
                    return value
    return None
