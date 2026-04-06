"""Pydantic models for the Textify JSON output schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single transcribed word with timing and speaker info."""

    word: str = Field(description="The transcribed word text")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    score: float = Field(default=0.0, description="Confidence score (0-1)")
    speaker: str | None = Field(default=None, description="Speaker label, e.g. SPEAKER_00")


class Segment(BaseModel):
    """A transcribed segment (sentence/phrase) with word-level detail."""

    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Full segment text")
    speaker: str | None = Field(default=None, description="Speaker label for the segment")
    words: list[Word] = Field(default_factory=list, description="Word-level timestamps")


class Metadata(BaseModel):
    """Metadata about the transcription job."""

    file: str = Field(description="Input audio filename")
    duration: float = Field(description="Audio duration in seconds")
    language: str = Field(description="Detected or specified language code")
    model: str = Field(description="Whisper model size used")
    device: str = Field(description="Compute device used (cpu/cuda)")
    num_speakers: int | None = Field(default=None, description="Number of speakers detected")
    diarization: bool = Field(description="Whether speaker diarization was performed")
    processing_time: float = Field(description="Total processing time in seconds")


class TranscriptionResult(BaseModel):
    """Top-level transcription output."""

    metadata: Metadata
    segments: list[Segment]
