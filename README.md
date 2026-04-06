# Textify

CLI tool for audio transcription with word-level timestamps and speaker diarization. Powered by [WhisperX](https://github.com/m-bain/whisperX) (faster-whisper + wav2vec2 + pyannote.audio).

## Features

- Transcribe MP3, WAV, FLAC, M4A, OGG, and WMA audio files
- Word-level timestamps with confidence scores
- Speaker diarization (identify who said what)
- JSON output ready for downstream processing
- Auto-detects GPU (CUDA) or falls back to CPU
- Docker support for easy deployment

## Installation

### Prerequisites

- Python 3.9 - 3.11
- ffmpeg (`sudo apt install ffmpeg` on Ubuntu, `brew install ffmpeg` on macOS, or [download](https://ffmpeg.org/download.html) for Windows)

### Install with pip

```bash
pip install .
```

### For development

```bash
pip install -e ".[dev]"
```

## Setup for Speaker Diarization

Speaker diarization requires a free HuggingFace account:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the terms for these models (click "Agree" on each page):
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Get your token from [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
4. Set it as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

Or pass it directly: `textify audio.mp3 --hf-token your_token_here`

> You can skip this entirely by using `--no-diarize` — transcription and word timestamps still work without it.

## Usage

### Basic transcription (no diarization)

```bash
textify recording.mp3 --no-diarize
```

### Full transcription with speaker diarization

```bash
textify meeting.wav
```

### Save to file

```bash
textify interview.mp3 -o transcript.json
```

### Use a larger model for better accuracy

```bash
textify podcast.mp3 -m small -o result.json
```

### Specify number of speakers

```bash
textify conference.wav --min-speakers 2 --max-speakers 5
```

### Verbose output (see progress)

```bash
textify long_audio.mp3 -v -o output.json
```

### Force CPU or GPU

```bash
textify audio.wav -d cpu
textify audio.wav -d cuda
```

## CLI Options

```
Usage: textify [OPTIONS] AUDIO_FILE

Options:
  -m, --model [tiny|base|small|medium|large-v2]
                                  Whisper model size.  [default: tiny]
  -o, --output PATH               Output JSON file path. Defaults to stdout.
  -l, --language TEXT              Language code (e.g. "en"). Auto-detects if
                                  not specified.
  --min-speakers INTEGER           Min expected speakers (diarization hint).
  --max-speakers INTEGER           Max expected speakers (diarization hint).
  --no-diarize                     Skip speaker diarization.
  -d, --device [auto|cpu|cuda]     Compute device.  [default: auto]
  --hf-token TEXT                  HuggingFace token for diarization.
  -v, --verbose                    Print progress messages to stderr.
  --version                        Show the version and exit.
  -h, --help                       Show this message and exit.
```

## Output Format

Textify outputs JSON with this structure:

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
        {
          "word": "Hello,",
          "start": 0.52,
          "end": 0.84,
          "score": 0.97,
          "speaker": "SPEAKER_00"
        },
        {
          "word": "how",
          "start": 0.86,
          "end": 1.02,
          "score": 0.95,
          "speaker": "SPEAKER_00"
        }
      ]
    },
    {
      "start": 3.8,
      "end": 5.44,
      "text": "I'm doing great, thanks!",
      "speaker": "SPEAKER_01",
      "words": [
        {
          "word": "I'm",
          "start": 3.8,
          "end": 4.02,
          "score": 0.91,
          "speaker": "SPEAKER_01"
        }
      ]
    }
  ]
}
```

## Docker

### Build

```bash
docker compose build
```

### Run (CPU)

```bash
# Place audio files in ./input/
docker compose run textify /input/audio.mp3 -o /output/result.json
```

### Run (GPU)

```bash
docker compose run textify-gpu /input/audio.mp3 -d cuda -o /output/result.json
```

## Model Sizes

| Model | Size | RAM (approx) | Speed | Accuracy |
|-------|------|-------------|-------|----------|
| tiny | 39 MB | ~1 GB | Fastest | Basic |
| base | 74 MB | ~1.5 GB | Fast | Fair |
| small | 244 MB | ~2 GB | Moderate | Good |
| medium | 769 MB | ~3.5 GB | Slow | Great |
| large-v2 | 1.5 GB | ~6 GB | Slowest | Best |

> The default is `tiny` for quick processing. Use `small` or `medium` for better accuracy.

## License

MIT
