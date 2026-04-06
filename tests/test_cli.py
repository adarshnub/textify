"""Tests for CLI argument parsing and behavior."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from textify.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_help_flag(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "AUDIO_FILE" in result.output
    assert "--model" in result.output
    assert "--no-diarize" in result.output


def test_version_flag(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "textify" in result.output


def test_missing_audio_file(runner):
    result = runner.invoke(main, [])
    assert result.exit_code != 0


def test_nonexistent_file(runner):
    result = runner.invoke(main, ["/nonexistent/audio.mp3"])
    assert result.exit_code != 0
    assert "File not found" in result.output


def test_unsupported_format(runner, tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("not audio")
    result = runner.invoke(main, [str(txt_file)])
    assert result.exit_code != 0
    assert "Unsupported format" in result.output


@patch("textify.cli.transcribe")
def test_successful_transcription(mock_transcribe, runner, tmp_path):
    from textify.models import Metadata, Segment, TranscriptionResult, Word

    # Create a fake audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    # Mock the transcribe function
    mock_transcribe.return_value = TranscriptionResult(
        metadata=Metadata(
            file=str(audio_file),
            duration=5.0,
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

    result = runner.invoke(main, [str(audio_file), "--no-diarize"])
    assert result.exit_code == 0
    assert '"hello world"' in result.output
    assert '"metadata"' in result.output


@patch("textify.cli.transcribe")
def test_output_to_file(mock_transcribe, runner, tmp_path):
    from textify.models import Metadata, Segment, TranscriptionResult

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio data")
    output_file = tmp_path / "result.json"

    mock_transcribe.return_value = TranscriptionResult(
        metadata=Metadata(
            file=str(audio_file),
            duration=3.0,
            language="en",
            model="tiny",
            device="cpu",
            diarization=False,
            processing_time=0.5,
        ),
        segments=[],
    )

    result = runner.invoke(
        main, [str(audio_file), "--no-diarize", "-o", str(output_file)]
    )
    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert '"metadata"' in content
