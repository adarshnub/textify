# Textify — Implementation Plan

## Context

**Problem**: Transcribing audio files with word-level timestamps and speaker identification currently requires paid APIs (ElevenLabs, AssemblyAI, etc.) or manually stitching together multiple open-source tools. There is no simple, free CLI tool that does transcription + word-level timestamps + speaker diarization in one command.

**Solution**: Build Textify — a Python CLI tool where you run `textify audio.mp3` and get a JSON output with word-level timestamps and speaker labels, powered entirely by free open-source models (WhisperX + pyannote.audio).

**Why WhisperX over raw Whisper + pyannote**: WhisperX already integrates faster-whisper (transcription) + wav2vec2 (word alignment) + pyannote.audio (diarization) into a single pipeline. No need to wire them together manually.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Core Engine | WhisperX | Pre-integrated pipeline: transcription + alignment + diarization |
| Transcription | faster-whisper (inside WhisperX) | 70x realtime, CTranslate2 backend |
| Word Alignment | wav2vec2 (inside WhisperX) | Precise per-word timestamps via forced alignment |
| Speaker Diarization | pyannote.audio 3.1 (inside WhisperX) | Best free diarization model (DER ~11-19%) |
| CLI Framework | `click` | Clean CLI with help text, better than argparse |
| Audio Loading | WhisperX (uses librosa/soundfile internally) | Handles MP3/WAV/FLAC/etc. |
| Packaging | pyproject.toml + pip | Modern Python packaging |
| Containerization | Docker + docker-compose | GPU passthrough support |

---

## Project Structure

```
textify/
├── pyproject.toml              # Package config, dependencies, CLI entry point
├── Dockerfile                  # Container build (CPU + GPU)
├── docker-compose.yml          # Easy container orchestration
├── .gitignore                  # Python/audio ignores
├── .env.example                # HuggingFace token template
├── README.md                   # Usage docs, setup, examples
├── LICENSE                     # MIT (already exists)
├── src/
│   └── textify/
│       ├── __init__.py         # Package version
│       ├── cli.py              # Click CLI entry point
│       ├── transcriber.py      # Core transcription pipeline
│       ├── models.py           # Pydantic models for JSON output schema
│       └── utils.py            # Audio validation, device detection, helpers
└── tests/
    ├── __init__.py
    ├── test_cli.py             # CLI argument parsing tests
    ├── test_models.py          # JSON schema validation tests
    └── test_utils.py           # Utility function tests
```

---

## Dependencies

```toml
[project]
requires-python = ">=3.9,<3.12"

[project.dependencies]
whisperx = ">=3.1.0"          # Core engine (pulls in faster-whisper, pyannote, torch)
click = ">=8.0"                # CLI framework
pydantic = ">=2.0"             # JSON output schema validation

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]

[project.scripts]
textify = "textify.cli:main"
```

Note: WhisperX pulls in PyTorch, faster-whisper, pyannote.audio, transformers, etc. as transitive dependencies.

---

## Module Details

### 1. `src/textify/cli.py` — CLI Entry Point

```
main()  — Click group/command entry point

Command: textify <audio_file>
Arguments:
  audio_file          Path to MP3/WAV file (required, positional)

Options:
  --model, -m         Whisper model size [tiny/base/small/medium/large-v2] (default: tiny)
  --output, -o        Output file path (default: stdout)
  --language, -l      Language code e.g. "en" (default: auto-detect)
  --min-speakers      Minimum expected speakers for diarization (optional)
  --max-speakers      Maximum expected speakers for diarization (optional)
  --no-diarize        Skip speaker diarization (faster, no HF token needed)
  --device, -d        Compute device [auto/cpu/cuda] (default: auto)
  --hf-token          HuggingFace token (or set HF_TOKEN env var)
  --verbose, -v       Print progress to stderr
```

Flow:
1. Parse arguments
2. Validate audio file exists and is supported format
3. Call `transcriber.transcribe()` with options
4. Serialize result to JSON via pydantic model
5. Write to output file or stdout

### 2. `src/textify/transcriber.py` — Core Pipeline

```python
def transcribe(
    audio_path: str,
    model_name: str = "tiny",
    device: str = "auto",
    language: str | None = None,
    diarize: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    on_progress: Callable | None = None,   # callback for --verbose
) -> TranscriptionResult:
```

Pipeline steps (sequential):
1. **Detect device**: auto → check `torch.cuda.is_available()` → "cuda" or "cpu"
2. **Load model**: `whisperx.load_model(model_name, device, compute_type)` — compute_type = "float16" for GPU, "int8" for CPU
3. **Load audio**: `whisperx.load_audio(audio_path)`
4. **Transcribe**: `model.transcribe(audio, batch_size)` — batch_size=16 for GPU, 1 for CPU. Returns segments with approximate timestamps.
5. **Align**: `whisperx.load_align_model(language, device)` → `whisperx.align(segments, ...)` — produces word-level timestamps
6. **Diarize** (if enabled): `whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)` → `diarize_model(audio)` → `whisperx.assign_word_speakers(diarize_segments, result)` — assigns speaker labels to each word/segment
7. **Build result**: Map WhisperX output to our Pydantic `TranscriptionResult` model

### 3. `src/textify/models.py` — JSON Output Schema (Pydantic)

```python
class Word(BaseModel):
    word: str               # "hello"
    start: float            # 0.52 (seconds)
    end: float              # 0.84 (seconds)
    score: float            # 0.97 (confidence 0-1)
    speaker: str | None     # "SPEAKER_00" or null if no diarization

class Segment(BaseModel):
    start: float            # 0.52
    end: float              # 3.21
    text: str               # "Hello, how are you doing today?"
    speaker: str | None     # "SPEAKER_00"
    words: list[Word]       # word-level detail

class TranscriptionResult(BaseModel):
    metadata: Metadata
    segments: list[Segment]

class Metadata(BaseModel):
    file: str               # "meeting.mp3"
    duration: float         # 342.5 (seconds)
    language: str           # "en"
    model: str              # "tiny"
    device: str             # "cuda" or "cpu"
    num_speakers: int | None  # 3
    diarization: bool       # true
    processing_time: float  # 12.3 (seconds)
```

**Example JSON output:**
```json
{
  "metadata": {
    "file": "meeting.mp3",
    "duration": 65.2,
    "language": "en",
    "model": "tiny",
    "device": "cpu",
    "num_speakers": 2,
    "diarization": true,
    "processing_time": 8.4
  },
  "segments": [
    {
      "start": 0.52,
      "end": 3.21,
      "text": "Hello, how are you doing today?",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello,", "start": 0.52, "end": 0.84, "score": 0.97, "speaker": "SPEAKER_00"},
        {"word": "how", "start": 0.86, "end": 1.02, "score": 0.95, "speaker": "SPEAKER_00"},
        {"word": "are", "start": 1.04, "end": 1.18, "score": 0.98, "speaker": "SPEAKER_00"},
        {"word": "you", "start": 1.20, "end": 1.36, "score": 0.96, "speaker": "SPEAKER_00"},
        {"word": "doing", "start": 1.38, "end": 1.72, "score": 0.94, "speaker": "SPEAKER_00"},
        {"word": "today?", "start": 1.74, "end": 2.10, "score": 0.99, "speaker": "SPEAKER_00"}
      ]
    },
    {
      "start": 3.80,
      "end": 5.44,
      "text": "I'm doing great, thanks!",
      "speaker": "SPEAKER_01",
      "words": [
        {"word": "I'm", "start": 3.80, "end": 4.02, "score": 0.91, "speaker": "SPEAKER_01"},
        {"word": "doing", "start": 4.04, "end": 4.38, "score": 0.93, "speaker": "SPEAKER_01"},
        {"word": "great,", "start": 4.40, "end": 4.78, "score": 0.97, "speaker": "SPEAKER_01"},
        {"word": "thanks!", "start": 4.80, "end": 5.44, "score": 0.95, "speaker": "SPEAKER_01"}
      ]
    }
  ]
}
```

### 4. `src/textify/utils.py` — Helpers

```python
def detect_device(preferred: str = "auto") -> str:
    """Returns 'cuda' or 'cpu' based on preference and availability."""

def get_compute_type(device: str) -> str:
    """Returns 'float16' for cuda, 'int8' for cpu."""

def validate_audio_file(path: str) -> Path:
    """Check file exists, extension is supported (.mp3, .wav, .flac, .m4a, .ogg, .wma)."""

def get_audio_duration(path: str) -> float:
    """Get duration in seconds using soundfile/librosa."""

def resolve_hf_token(cli_token: str | None) -> str | None:
    """Check CLI arg → HF_TOKEN env var → .env file → None."""
```

---

## Docker Configuration

### Dockerfile
- Base image: `python:3.11-slim` for CPU; `nvidia/cuda:12.1-runtime-ubuntu22.04` for GPU
- Use multi-stage or single image with conditional GPU support
- Install system deps: `ffmpeg`, `libsndfile1`
- Copy project, `pip install .`
- Entry point: `textify`

### docker-compose.yml
Two services:
- `textify` (CPU): basic run
- `textify-gpu` (GPU): with `nvidia` runtime and GPU reservation

Both mount `./input:/input` and `./output:/output` volumes.

---

## Implementation Order

### Step 1: Project scaffolding
- Create `pyproject.toml`, `.gitignore`, `.env.example`, `src/textify/__init__.py`
- Set up package structure

### Step 2: Models (`models.py`)
- Define Pydantic schemas: `Word`, `Segment`, `Metadata`, `TranscriptionResult`

### Step 3: Utils (`utils.py`)
- Device detection, audio validation, HF token resolution

### Step 4: Core pipeline (`transcriber.py`)
- Implement the 6-step WhisperX pipeline
- Handle diarization as optional step
- Map WhisperX output to Pydantic models

### Step 5: CLI (`cli.py`)
- Click command with all options
- Progress reporting via `--verbose`
- JSON output to stdout or file

### Step 6: Docker
- Dockerfile with ffmpeg + Python deps
- docker-compose.yml with CPU and GPU profiles

### Step 7: README
- Installation instructions (pip + Docker)
- HuggingFace token setup for diarization
- Usage examples
- JSON output format documentation

### Step 8: Tests
- Unit tests for models, utils
- Integration test with a short sample audio (if feasible)

---

## Verification Plan

1. **Install locally**: `pip install -e .` in a venv
2. **Test with sample audio** (no diarization): `textify sample.wav --no-diarize` → verify JSON has word timestamps
3. **Test with diarization**: `textify sample.wav --hf-token <token>` → verify speaker labels appear
4. **Test stdout vs file**: `textify sample.wav -o result.json` → verify file created
5. **Test model selection**: `textify sample.wav -m small` → verify different model loads
6. **Test device auto-detect**: Run on machine with/without GPU, check device in metadata
7. **Docker test**: `docker-compose run textify /input/sample.wav` → verify output
8. **Edge cases**: Empty audio, unsupported format, missing HF token with diarization enabled

---

## Key Caveats

1. **HuggingFace Token**: Speaker diarization requires a free HuggingFace account + accepting pyannote model terms. Without it, `--no-diarize` still works perfectly for transcription + timestamps.
2. **First Run**: Model weights are downloaded on first use (~39MB for tiny). Subsequent runs use cache.
3. **Python Version**: WhisperX requires Python 3.9-3.11 (not 3.12+ due to torch compatibility).
4. **Memory**: tiny model uses ~1-2GB RAM. With diarization, add ~500MB. Safe for 8GB+ machines.
