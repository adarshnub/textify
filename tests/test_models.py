"""Tests for Pydantic output models."""

from textify.models import Metadata, Segment, TranscriptionResult, Word


def test_word_model():
    word = Word(word="hello", start=0.5, end=0.8, score=0.95, speaker="SPEAKER_00")
    assert word.word == "hello"
    assert word.start == 0.5
    assert word.end == 0.8
    assert word.score == 0.95
    assert word.speaker == "SPEAKER_00"


def test_word_defaults():
    word = Word(word="hi", start=1.0, end=1.2)
    assert word.score == 0.0
    assert word.speaker is None


def test_segment_model():
    words = [
        Word(word="hello", start=0.5, end=0.8, score=0.95),
        Word(word="world", start=0.9, end=1.2, score=0.90),
    ]
    segment = Segment(
        start=0.5, end=1.2, text="hello world", speaker="SPEAKER_00", words=words
    )
    assert segment.text == "hello world"
    assert len(segment.words) == 2
    assert segment.speaker == "SPEAKER_00"


def test_segment_no_speaker():
    segment = Segment(start=0.0, end=1.0, text="test")
    assert segment.speaker is None
    assert segment.words == []


def test_metadata_model():
    meta = Metadata(
        file="test.mp3",
        duration=60.5,
        language="en",
        model="tiny",
        device="cpu",
        num_speakers=2,
        diarization=True,
        processing_time=5.3,
    )
    assert meta.file == "test.mp3"
    assert meta.duration == 60.5
    assert meta.num_speakers == 2


def test_metadata_no_speakers():
    meta = Metadata(
        file="test.wav",
        duration=30.0,
        language="en",
        model="small",
        device="cuda",
        diarization=False,
        processing_time=2.1,
    )
    assert meta.num_speakers is None
    assert meta.diarization is False


def test_transcription_result_json():
    result = TranscriptionResult(
        metadata=Metadata(
            file="test.mp3",
            duration=10.0,
            language="en",
            model="tiny",
            device="cpu",
            diarization=False,
            processing_time=1.0,
        ),
        segments=[
            Segment(
                start=0.0,
                end=2.0,
                text="hello world",
                words=[
                    Word(word="hello", start=0.0, end=0.8, score=0.95),
                    Word(word="world", start=0.9, end=1.8, score=0.90),
                ],
            )
        ],
    )
    json_str = result.model_dump_json(indent=2)
    assert '"hello world"' in json_str
    assert '"metadata"' in json_str
    assert '"segments"' in json_str


def test_transcription_result_roundtrip():
    original = TranscriptionResult(
        metadata=Metadata(
            file="audio.wav",
            duration=5.0,
            language="es",
            model="medium",
            device="cuda",
            num_speakers=3,
            diarization=True,
            processing_time=3.5,
        ),
        segments=[
            Segment(
                start=0.0,
                end=2.5,
                text="hola mundo",
                speaker="SPEAKER_00",
                words=[
                    Word(
                        word="hola",
                        start=0.0,
                        end=0.5,
                        score=0.98,
                        speaker="SPEAKER_00",
                    ),
                    Word(
                        word="mundo",
                        start=0.6,
                        end=1.2,
                        score=0.96,
                        speaker="SPEAKER_00",
                    ),
                ],
            ),
        ],
    )
    json_str = original.model_dump_json()
    restored = TranscriptionResult.model_validate_json(json_str)
    assert restored.metadata.language == "es"
    assert restored.segments[0].words[0].word == "hola"
    assert restored.segments[0].speaker == "SPEAKER_00"
