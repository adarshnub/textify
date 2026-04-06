"""Tests for utility functions."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from textify.utils import (
    SUPPORTED_EXTENSIONS,
    detect_device,
    get_batch_size,
    get_compute_type,
    resolve_hf_token,
    validate_audio_file,
)


class TestDetectDevice:
    def test_cpu_explicit(self):
        assert detect_device("cpu") == "cpu"

    def test_cuda_explicit(self):
        assert detect_device("cuda") == "cuda"

    def test_auto_falls_back_to_cpu(self):
        # When torch reports no CUDA, should fall back to cpu
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert detect_device("auto") == "cpu"


class TestGetComputeType:
    def test_cuda_returns_float16(self):
        assert get_compute_type("cuda") == "float16"

    def test_cpu_returns_int8(self):
        assert get_compute_type("cpu") == "int8"


class TestGetBatchSize:
    def test_cuda_batch(self):
        assert get_batch_size("cuda") == 16

    def test_cpu_batch(self):
        assert get_batch_size("cpu") == 1


class TestValidateAudioFile:
    def test_nonexistent_file(self, tmp_path):
        import click

        with pytest.raises(click.BadParameter, match="File not found"):
            validate_audio_file(str(tmp_path / "nonexistent.mp3"))

    def test_directory_not_file(self, tmp_path):
        import click

        with pytest.raises(click.BadParameter, match="Not a file"):
            validate_audio_file(str(tmp_path))

    def test_unsupported_extension(self, tmp_path):
        import click

        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not audio")
        with pytest.raises(click.BadParameter, match="Unsupported format"):
            validate_audio_file(str(bad_file))

    def test_valid_mp3(self, tmp_path):
        mp3 = tmp_path / "test.mp3"
        mp3.write_bytes(b"fake audio")
        result = validate_audio_file(str(mp3))
        assert isinstance(result, Path)
        assert result.suffix == ".mp3"

    def test_valid_wav(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"fake audio")
        result = validate_audio_file(str(wav))
        assert result.suffix == ".wav"

    def test_all_supported_extensions(self, tmp_path):
        for ext in SUPPORTED_EXTENSIONS:
            f = tmp_path / f"test{ext}"
            f.write_bytes(b"fake")
            result = validate_audio_file(str(f))
            assert result.suffix == ext


class TestResolveHfToken:
    def test_cli_token_takes_priority(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            assert resolve_hf_token("cli_token") == "cli_token"

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            assert resolve_hf_token(None) == "env_token"

    def test_returns_none_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_hf_token(None) is None

    def test_empty_cli_token_uses_env(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            assert resolve_hf_token("") == "env_token"
