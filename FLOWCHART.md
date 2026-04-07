# Textify Flow Chart

```mermaid
flowchart TD
    A[User runs: textify audio.mp3] --> B[CLI validates file + options]
    B --> C[Detect device: CUDA or CPU]
    C --> D[Load Whisper model]
    D --> E[Transcribe audio]
    E --> F[Word alignment for precise timestamps]
    F --> G{Diarization enabled?}
    G -- No --> I[Build JSON output]
    G -- Yes --> H[Load pyannote model with HF token]
    H --> I[Build JSON output]
    I --> J[Write output.json or stdout]
```

## Simple Explanation

1. You run one command with an audio file.
2. Textify validates input and loads models on CPU/GPU.
3. WhisperX transcribes speech to text.
4. Alignment adds word-level timestamps.
5. Optional diarization adds speaker labels.
6. Final structured JSON is written to file/stdout.
