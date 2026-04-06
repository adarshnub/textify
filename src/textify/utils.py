"""Utility functions for device detection, audio validation, and token resolution."""

from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma"}

# Unicode script ranges for detecting non-Latin languages
SCRIPT_LANGUAGE_MAP = {
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "hi": (0x0900, 0x097F),  # Hindi (Devanagari)
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "bn": (0x0980, 0x09FF),  # Bengali
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Punjabi (Gurmukhi)
    "or": (0x0B00, 0x0B7F),  # Odia
    "ar": (0x0600, 0x06FF),  # Arabic
    "zh": (0x4E00, 0x9FFF),  # Chinese (CJK)
    "ja": (0x3040, 0x309F),  # Japanese (Hiragana)
    "ko": (0xAC00, 0xD7AF),  # Korean (Hangul)
    "th": (0x0E00, 0x0E7F),  # Thai
    "ru": (0x0400, 0x04FF),  # Cyrillic (Russian)
}


def detect_text_language(text: str, fallback: str = "en") -> str:
    """Detect language from text using Unicode script analysis.

    Reliable for languages with distinct scripts (Indian languages, CJK,
    Arabic, etc.). Falls back to the given default for Latin-script text.

    Args:
        text: The text to analyze.
        fallback: Language code to return for Latin-script text.

    Returns:
        ISO 639-1 language code.
    """
    script_counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for lang, (start, end) in SCRIPT_LANGUAGE_MAP.items():
            if start <= code <= end:
                script_counts[lang] = script_counts.get(lang, 0) + 1
                break

    if script_counts:
        return max(script_counts, key=script_counts.get)
    return fallback


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
