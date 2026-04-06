"""Core transcription pipeline using WhisperX."""

from __future__ import annotations

import sys
import time
from typing import Callable

from .models import Metadata, Segment, TranscriptionResult, Word
from .utils import (
    detect_device,
    detect_text_language,
    get_audio_duration,
    get_batch_size,
    get_compute_type,
    resolve_hf_token,
)


def _log(msg: str, on_progress: Callable[[str], None] | None) -> None:
    """Send a progress message if verbose mode is enabled."""
    if on_progress:
        on_progress(msg)


def transcribe(
    audio_path: str,
    model_name: str = "tiny",
    device: str = "auto",
    language: str | None = None,
    diarize: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Run the full transcription pipeline.

    Steps:
        1. Detect device (CPU/CUDA)
        2. Load Whisper model
        3. Load and transcribe audio
        4. Align words for word-level timestamps
        5. (Optional) Run speaker diarization
        6. Build structured result

    Args:
        audio_path: Path to the audio file.
        model_name: Whisper model size (tiny/base/small/medium/large-v2).
        device: Compute device (auto/cpu/cuda).
        language: Language code or None for auto-detect.
        diarize: Whether to run speaker diarization.
        min_speakers: Minimum expected number of speakers.
        max_speakers: Maximum expected number of speakers.
        hf_token: HuggingFace token for diarization models.
        on_progress: Optional callback for progress messages.

    Returns:
        TranscriptionResult with segments, words, and metadata.
    """
    import whisperx

    start_time = time.time()

    # Step 1: Detect device
    device = detect_device(device)
    compute_type = get_compute_type(device)
    batch_size = get_batch_size(device)
    _log(f"Using device: {device} (compute_type={compute_type})", on_progress)

    # Step 2: Load Whisper model
    _log(f"Loading Whisper model '{model_name}'...", on_progress)
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # Step 3: Load and transcribe audio
    _log("Loading audio...", on_progress)
    audio = whisperx.load_audio(audio_path)
    duration = get_audio_duration(audio_path)

    _log("Transcribing...", on_progress)
    result = model.transcribe(audio, batch_size=batch_size)

    detected_language = language or result.get("language", "en")
    _log(f"Detected language: {detected_language}", on_progress)

    # Step 4: Detect per-segment language and align with language-specific models
    for seg in result["segments"]:
        seg["language"] = detect_text_language(
            seg.get("text", ""), fallback=detected_language
        )

    # Group segments by language for separate alignment
    lang_groups: dict[str, list[tuple[int, dict]]] = {}
    for i, seg in enumerate(result["segments"]):
        lang = seg["language"]
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append((i, seg))

    all_aligned_segments: list[dict] = []
    for lang, indexed_segs in lang_groups.items():
        _log(f"Aligning words for language '{lang}'...", on_progress)
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=lang, device=device
            )
            segs_to_align = [seg for _, seg in indexed_segs]
            aligned = whisperx.align(
                segs_to_align,
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            for aligned_seg in aligned.get("segments", []):
                aligned_seg["language"] = lang
                all_aligned_segments.append(aligned_seg)
        except Exception as e:
            _log(
                f"Alignment failed for '{lang}': {e}. Keeping unaligned.",
                on_progress,
            )
            for _, seg in indexed_segs:
                all_aligned_segments.append(seg)

    # Sort by start time to maintain chronological order
    all_aligned_segments.sort(key=lambda s: s.get("start", 0.0))
    result = {"segments": all_aligned_segments}

    # Step 5: Speaker diarization (optional)
    num_speakers = None
    if diarize:
        token = resolve_hf_token(hf_token)
        if token is None:
            _log(
                "WARNING: No HuggingFace token found. Skipping diarization. "
                "Set HF_TOKEN env var or use --hf-token flag.",
                on_progress,
            )
            # Also print to stderr so the user sees it even without --verbose
            print(
                "Warning: No HuggingFace token found. Skipping speaker diarization.\n"
                "Set HF_TOKEN env var or use --hf-token. See --help for details.",
                file=sys.stderr,
            )
            diarize = False
        else:
            _log("Running speaker diarization...", on_progress)
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                token=token, device=device
            )

            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Count unique speakers
            speakers = set()
            for seg in result.get("segments", []):
                if seg.get("speaker"):
                    speakers.add(seg["speaker"])
                for w in seg.get("words", []):
                    if w.get("speaker"):
                        speakers.add(w["speaker"])
            num_speakers = len(speakers) if speakers else None
            _log(f"Detected {num_speakers} speaker(s)", on_progress)

    # Step 6: Build structured result
    _log("Building output...", on_progress)
    segments = _build_segments(result.get("segments", []))

    processing_time = round(time.time() - start_time, 2)
    _log(f"Done in {processing_time}s", on_progress)

    metadata = Metadata(
        file=str(audio_path),
        duration=round(duration, 2),
        language=detected_language,
        model=model_name,
        device=device,
        num_speakers=num_speakers,
        diarization=diarize,
        processing_time=processing_time,
    )

    return TranscriptionResult(metadata=metadata, segments=segments)


def _build_segments(raw_segments: list[dict]) -> list[Segment]:
    """Convert WhisperX segment dicts to Pydantic Segment models."""
    segments = []
    for seg in raw_segments:
        words = []
        for w in seg.get("words", []):
            # WhisperX may omit start/end for some words (edge cases)
            if "start" not in w or "end" not in w:
                continue
            words.append(
                Word(
                    word=w.get("word", ""),
                    start=round(w["start"], 3),
                    end=round(w["end"], 3),
                    score=round(w.get("score", 0.0), 3),
                    speaker=w.get("speaker"),
                )
            )

        segments.append(
            Segment(
                start=round(seg.get("start", 0.0), 3),
                end=round(seg.get("end", 0.0), 3),
                text=seg.get("text", "").strip(),
                speaker=seg.get("speaker"),
                language=seg.get("language"),
                words=words,
            )
        )
    return segments
