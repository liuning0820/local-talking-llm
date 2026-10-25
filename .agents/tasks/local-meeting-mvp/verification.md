# Verification record

This verification covers the local meeting MVP without pretending that physical
hardware, Whisper inference, Ollama, or a remote service is available.

## Commands and results

- `python -m compileall meeting_mvp tests stt.py` — passed.
- `python -m unittest discover -s tests -p "test_*.py"` — passed, 11 tests.
- PowerShell temporary-data synthetic check with
  `python -m meeting_mvp self-check --synthetic` — passed. It created two WAV
  chunks, persisted two transcripts, generated an evidence-linked summary, and
  wrote JSON and Markdown exports.
- The same temporary-data run followed by `python -m meeting_mvp list` —
  passed; the synthetic meeting appeared as `completed`.
- `python -m meeting_mvp --help` — passed and listed all meeting commands,
  including `--yes` on `start` and `retry` for remote upload confirmation.
- `python -m meeting_mvp devices --source system` — returned the expected
  actionable `WINDOWS_LOOPBACK_BACKEND_MISSING` error because SoundCard is not
  installed in this environment. Microphones were not presented as loopback
  devices.
- `git diff --check` — passed. No git commit was created.

## Implemented reliability checks

Recovery keeps `.tmp` WAV files belonging to fresh `recording` or `stopping`
meetings and only removes them after stale/interrupted recovery or a finished
writer. Removal failures are retained for a later recovery attempt. Ctrl+C,
Enter, and cross-process `stop ID` use the same durable stop request; the final
bounded chunk is flushed before transcription and summary processing.

Empty meetings remain `partial` and receive an explicit no-audio result. The
selected source is probed for approximately 500 ms before a recording row is
created. Windows WASAPI loopback and macOS virtual-input boundaries return
explicit device errors.

Whisper and compatible ASR segment timestamps are stored relative to each WAV
chunk and translated to the meeting timeline. Backends without timestamps use
one full-chunk segment. Remote audio or transcript upload requires a displayed
target and `YES` or `--yes`; local loopback remains the default path. SQLite
constraints and Store validation reject invalid status transitions, chunk
ranges, sizes, sequences, hashes, duplicate paths, and duplicate sequences.

## Not verified on this host

A real microphone capture, Windows WASAPI loopback playback capture, Windows
permission behavior, macOS CoreAudio capture, macOS BlackHole/Loopback/
Soundflower routing, Whisper inference, Ollama summary generation, and real
remote network requests require corresponding hardware, permissions, model
files, or services. These are recorded as limitations rather than claimed
passes.
