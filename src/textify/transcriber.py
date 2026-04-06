"""Core transcription pipeline using WhisperX."""

from __future__ import annotations

import sys
import time
from typing import Callable

from .models import Metadata, Segment, TranscriptionResult, Word
from .utils import (
    NATIVE_SCRIPT_PROMPTS,
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


def _transcribe_with_prompt(pipeline, audio, language: str, prompt: str) -> dict:
    """Transcribe using faster-whisper directly with initial_prompt.

    WhisperX's pipeline doesn't support initial_prompt, but the underlying
    faster-whisper model does.  Passing a prompt in the target script
    (e.g. Malayalam Unicode) strongly biases output toward that script.
    """
    segments_iter, _info = pipeline.model.transcribe(
        audio,
        language=language,
        initial_prompt=prompt,
        word_timestamps=True,
    )
    segments: list[dict] = []
    for seg in segments_iter:
        seg_dict: dict = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [],
        }
        if seg.words:
            for w in seg.words:
                seg_dict["words"].append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "score": w.probability,
                })
        segments.append(seg_dict)
    return {"segments": segments, "language": language}


def _do_transcribe(pipeline, audio, batch_size: int, language: str,
                   on_progress: Callable[[str], None] | None = None) -> dict:
    """Transcribe with native-script prompt when available, else WhisperX."""
    prompt = NATIVE_SCRIPT_PROMPTS.get(language)
    if prompt:
        _log(f"Using native script prompt for '{language}'", on_progress)
        return _transcribe_with_prompt(pipeline, audio, language, prompt)
    return pipeline.transcribe(audio, batch_size=batch_size, language=language)


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

    # Step 3b: Transcribe — multi-language, single-language, or auto-detect
    explicit_languages = (
        [l.strip() for l in language.split(",") if l.strip()]
        if language and "," in language
        else None
    )

    if explicit_languages:
        # --- Explicit multi-pass: user specified e.g. --language en,ml ---
        primary_lang = explicit_languages[0]
        _log(f"Pass 1: transcribing with language '{primary_lang}'...", on_progress)
        result = _do_transcribe(model, audio, batch_size, primary_lang, on_progress)
        for seg in result.get("segments", []):
            seg["language"] = primary_lang
        _log(f"Pass 1: got {len(result.get('segments', []))} segment(s)", on_progress)

        all_segments = list(result.get("segments", []))

        for pass_num, lang in enumerate(explicit_languages[1:], start=2):
            gaps = _find_time_gaps(all_segments, duration)
            if not gaps:
                _log(f"No gaps remaining, skipping pass {pass_num}", on_progress)
                break
            _log(f"Pass {pass_num}: filling gaps with language '{lang}'...", on_progress)
            pass_result = _do_transcribe(model, audio, batch_size, lang, on_progress)
            filled = 0
            for seg in pass_result.get("segments", []):
                if _segment_falls_in_gap(seg, gaps):
                    seg["language"] = lang
                    all_segments.append(seg)
                    filled += 1
            _log(f"Pass {pass_num}: filled {filled} segment(s)", on_progress)

        all_segments.sort(key=lambda s: s.get("start", 0.0))
        result = {"segments": all_segments}
        detected_language = primary_lang

    else:
        # --- Auto-detect or single forced language ---
        if language:
            # User forced a single language (e.g. --language ml)
            _log(f"Transcribing with language '{language}'...", on_progress)
            result = _do_transcribe(model, audio, batch_size, language, on_progress)
            detected_language = language
        else:
            # Full auto-detect
            _log("Transcribing (auto-detecting language)...", on_progress)
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result.get("language", "en")
            _log(f"Detected language: {detected_language}", on_progress)

            # If non-English detected, check if output is romanized instead
            # of native script. If so, re-transcribe with a native script prompt
            # using faster-whisper directly.
            if detected_language != "en":
                prompt = NATIVE_SCRIPT_PROMPTS.get(detected_language)
                if prompt:
                    all_text = " ".join(
                        seg.get("text", "") for seg in result.get("segments", [])
                    )
                    text_lang = detect_text_language(all_text, fallback="en")
                    if text_lang != detected_language:
                        _log(
                            f"Output is romanized — re-transcribing with "
                            f"native '{detected_language}' script prompt...",
                            on_progress,
                        )
                        result = _transcribe_with_prompt(
                            model, audio, detected_language, prompt
                        )

        # Tag every segment with its detected language
        for seg in result.get("segments", []):
            seg["language"] = detect_text_language(
                seg.get("text", ""), fallback=detected_language
            )

        # Auto gap-fill: if there are large uncovered regions, try the
        # "other" major language (English ↔ detected non-English)
        if not language:
            gaps = _find_time_gaps(result.get("segments", []), duration, min_gap=1.0)
            total_gap = sum(end - start for start, end in gaps) if gaps else 0
            if total_gap > 2.0:
                secondary = "en" if detected_language != "en" else None
                if secondary:
                    _log(
                        f"Found {total_gap:.1f}s of uncovered audio — "
                        f"trying '{secondary}' for gaps...",
                        on_progress,
                    )
                    pass_result = _do_transcribe(
                        model, audio, batch_size, secondary, on_progress
                    )
                    filled = 0
                    for seg in pass_result.get("segments", []):
                        if _segment_falls_in_gap(seg, gaps):
                            seg["language"] = secondary
                            result["segments"].append(seg)
                            filled += 1
                    if filled:
                        _log(f"Filled {filled} gap segment(s) with '{secondary}'", on_progress)
                        result["segments"].sort(key=lambda s: s.get("start", 0.0))

    # Step 4: Align words per language group
    lang_groups: dict[str, list[dict]] = {}
    for seg in result["segments"]:
        lang = seg.get("language", detected_language)
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append(seg)

    all_aligned_segments: list[dict] = []
    for lang, segs in lang_groups.items():
        _log(f"Aligning words for language '{lang}'...", on_progress)
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=lang, device=device
            )
            aligned = whisperx.align(
                segs,
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
            all_aligned_segments.extend(segs)

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


def _find_time_gaps(
    segments: list[dict], duration: float, min_gap: float = 0.5
) -> list[tuple[float, float]]:
    """Find time regions not covered by any segment.

    Args:
        segments: List of segment dicts with 'start' and 'end' keys.
        duration: Total audio duration in seconds.
        min_gap: Minimum gap duration to consider (seconds).

    Returns:
        List of (start, end) tuples representing uncovered time regions.
    """
    if not segments:
        return [(0.0, duration)]

    sorted_segs = sorted(segments, key=lambda s: s.get("start", 0.0))
    gaps: list[tuple[float, float]] = []

    # Gap before first segment
    first_start = sorted_segs[0].get("start", 0.0)
    if first_start > min_gap:
        gaps.append((0.0, first_start))

    # Gaps between consecutive segments
    for i in range(len(sorted_segs) - 1):
        end_cur = sorted_segs[i].get("end", 0.0)
        start_next = sorted_segs[i + 1].get("start", 0.0)
        if start_next - end_cur > min_gap:
            gaps.append((end_cur, start_next))

    # Gap after last segment
    last_end = sorted_segs[-1].get("end", 0.0)
    if duration - last_end > min_gap:
        gaps.append((last_end, duration))

    return gaps


def _segment_falls_in_gap(
    seg: dict, gaps: list[tuple[float, float]]
) -> bool:
    """Check if a segment's midpoint falls within any gap."""
    mid = (seg.get("start", 0.0) + seg.get("end", 0.0)) / 2
    return any(gap_start <= mid <= gap_end for gap_start, gap_end in gaps)


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
