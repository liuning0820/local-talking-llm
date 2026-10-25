# local-talking-llm

This repository contains a local voice assistant and a separate local meeting
MVP. The meeting workflow records device audio into bounded WAV chunks, performs
per-chunk transcription, creates evidence-linked minutes, and keeps history in
SQLite. It does not add a bot to a meeting or upload audio by default.

The meeting worker persists each transcription and summary job before it starts
processing. If the recording process exits after audio is written, run `retry`
after restarting the CLI; closed chunks are recovered and already successful
chunks are not processed twice.

## Meeting MVP

Use the new CLI when you need meeting recording and minutes. The CLI does not
import the legacy TTS or chat paths unless you run those legacy entry points.

```bash
python -m meeting_mvp --help
python -m meeting_mvp devices --source all
python -m meeting_mvp start --source microphone --device-id 2 --title "Weekly planning"
# In another terminal, request a clean stop with the printed meeting ID.
python -m meeting_mvp stop MEETING_ID
python -m meeting_mvp list
python -m meeting_mvp show MEETING_ID --transcript
python -m meeting_mvp retry MEETING_ID
python -m meeting_mvp export MEETING_ID --format markdown
```

Run a hardware-, model-, and network-free check with the following command:

```bash
python -m meeting_mvp self-check --synthetic
```

Read [the meeting MVP guide](docs/meeting-mvp.md) before recording. It explains
installation, permissions, device selection, loopback limits, model settings,
data deletion, and known platform boundaries.

## Installation

The project requires Python 3.11 or newer. For the current locked legacy
PyTorch/torchaudio set, Python 3.11 through 3.13 is the tested installation
range; CPython 3.14 currently cannot install `torchaudio==2.6.0`. If `.venv`
does not exist, create it with Python's built-in `venv` module:

```bash
python -m venv .venv
```

Activate the environment before running project commands:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

If `uv` is installed, synchronize dependencies from `pyproject.toml` and the
committed `uv.lock` file:

```bash
uv sync
```

`uv sync` creates or reuses the project `.venv` and installs the locked
runtime dependencies. It also works with a `.venv` created by Python's
built-in `venv` command.

If `uv` is not installed, install the compatibility requirements instead:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is a pinned compatibility requirements file for
installations that do not use `uv`; it is not a second dependency source when
using `uv sync`. Do not regenerate it with an unrestricted environment freeze,
because that can add unrelated packages. Update it deliberately when a direct
dependency changes.
`SoundCard==0.4.5` is used only for Windows WASAPI loopback discovery and
capture; microphone capture uses `sounddevice`. `openai-whisper` is the declared
local Whisper dependency. Install that package; its Python import name is
`whisper`. Do not install the unrelated PyPI package named `whisper`. Whisper
and Ollama remain optional at runtime for history and synthetic checks, but a
real recording needs an installed and configured ASR and LLM backend. If your
interpreter is CPython 3.14, use a Python 3.11–3.13 virtual environment for the
pinned `torch`/`torchaudio` set. If you cannot use that environment, configure
the already supported OpenAI-compatible `llama-asr` backend instead of
claiming that Whisper is installed.

## Legacy voice assistant

The original short-turn assistant remains available for compatibility. It is
not part of the meeting pipeline and still loads Whisper, LangChain, and TTS at
startup:

```bash
python app_mac.py
python app_win.py
```

The old scripts use their existing `.env` settings and can record microphone
input for conversational replies. They do not provide meeting history or
bounded long-recording storage.

## Project layout

The `meeting_mvp/` package owns meeting configuration, SQLite persistence,
chunked audio, device adapters, transcription, evidence validation, summaries,
exports, and the CLI. The existing `app_mac.py`, `app_win.py`, `stt.py`,
`tts.py`, and `query_local_llm.py` remain legacy paths.

## Privacy defaults

The default data directory is `%LOCALAPPDATA%\\local-talking-llm` on Windows,
`~/Library/Application Support/local-talking-llm` on macOS, and
`~/.local/share/local-talking-llm` on other systems. Set `MEETING_DATA_DIR` to
choose another directory. Audio, transcripts, and summaries stay there, and
remote model URLs are rejected while `MEETING_ALLOW_NETWORK=false`. The
meeting MVP defaults to local Ollama at `http://127.0.0.1:11434` and uses its
native `/api/chat` endpoint; use `MEETING_LLM_BACKEND=openai` only for an
OpenAI-compatible `/v1/chat/completions` server. A remote upload requires
both `MEETING_ALLOW_NETWORK=true` and an explicit `YES` confirmation.

The recorder captures other people only with appropriate notice and consent.
Use `python -m meeting_mvp delete ID --audio-only` to remove WAV files while
keeping the transcript and minutes. Use `python -m meeting_mvp delete ID --all
--confirm` to remove the complete meeting.
