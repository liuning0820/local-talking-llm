# Meeting MVP guide

This guide describes the local meeting workflow, its setup, and its platform
limits. It is intentionally explicit about the difference between microphone
capture and system-audio loopback.

## Install and configure

Install Python 3.11 or newer. For the current locked legacy
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

If `uv` is installed, synchronize the project environment from
`pyproject.toml` and the committed `uv.lock` file:

```bash
uv sync
```

`uv sync` creates or reuses the project `.venv` and installs the dependencies
resolved from `pyproject.toml` and `uv.lock`. It also works with a `.venv`
created by Python's built-in `venv` command.

If `uv` is not installed, install the compatibility requirements instead:

```bash
python -m pip install -r requirements.txt
```

After installing or removing a dependency in the activated `.venv`, update the
snapshot with:

```bash
python -m pip freeze > requirements.txt
```

This command replaces the file with every package currently installed in the
virtual environment, including transitive dependencies. It does not update
`pyproject.toml` or `uv.lock`; if you later use `uv`, add intentional direct
dependencies to `pyproject.toml` and regenerate the lockfile with `uv lock`.

The repository includes `requirements.txt` for environments that do not have
`uv`. It is not a second dependency source when using `uv sync`. To refresh it
from the lockfile when needed, run:

```bash
uv export --format requirements.txt --output-file requirements.txt
```

Install and configure the ASR before recording. For local Whisper, use a
Python 3.11–3.13 environment and install the declared package:

```powershell
# In an activated Python 3.11–3.13 environment
python -m pip install -r requirements.txt
# Verify the package name and import name
python -c "import whisper; print(whisper.__file__)"
```

The dependency is named `openai-whisper` in the package manager but is imported
as `whisper`; installing the separate package named `whisper` does not satisfy
this project. Select a model with `MEETING_WHISPER_MODEL=base` or another model
supported by `openai-whisper`.

If the available interpreter is CPython 3.14 and the pinned PyTorch/torchaudio
wheels cannot be installed, use the supported compatible ASR endpoint instead:

```powershell
$env:MEETING_STT_BACKEND = "llama-asr"
$env:MEETING_ASR_URL = "http://127.0.0.1:8080/v1"
$env:MEETING_ASR_MODEL = "Qwen3-ASR-0.6B-Q8_0.gguf"
```

Start the compatible local server first. The repository's legacy `.env`
variables such as `STT_BACKEND` and `LLAMA_ASR_BASE_URL` are not read by the
meeting MVP; use the `MEETING_*` names above or copy them into `.env`.

The meeting CLI reads environment variables and never stores tokens in SQLite.
The main settings are:


- `MEETING_DATA_DIR` selects the local data root.
- `MEETING_STT_BACKEND=whisper` uses lazy-loaded local Whisper. Set it to
  `llama-asr` for an OpenAI-compatible local transcription endpoint.
- `MEETING_WHISPER_MODEL`, `MEETING_ASR_URL`, and `MEETING_ASR_MODEL` select
  the transcription model.
- `MEETING_LLM_BACKEND=ollama` uses Ollama's native `/api/chat` endpoint.
  Set it to `openai` for a compatible `/v1/chat/completions` endpoint. For
  Ollama, set `MEETING_LLM_URL` to `http://127.0.0.1:11434`; the legacy
  `/v1` base form is also normalized to the native endpoint. For OpenAI
  compatibility, set the URL to the server host or its `/v1` base; don't mix
  the two backend modes.
- `MEETING_LLM_URL` and `MEETING_LLM_MODEL` select the minutes model. The
  default is local Ollama with `MEETING_ALLOW_NETWORK=false`; use a non-loopback
  URL only when you intentionally enable networking and confirm the upload.
- `MEETING_ALLOW_NETWORK=false` rejects non-loopback model URLs. Set it to
  `true` only when you intentionally permit network traffic.
- `MEETING_CHUNK_SECONDS` accepts 30 through 900 seconds and defaults to 300.

The summary prompt requests JSON and validates evidence segment IDs. Missing
facts use `未提及`; owner and due date are not inferred. Long transcripts are
summarized in map/reduce windows rather than sent as one oversized request.

## Record and review a meeting

Run the following commands from the repository root:

```bash
python -m meeting_mvp devices --source all
python -m meeting_mvp start --source microphone --device-id 2 --title "Project review"
python -m meeting_mvp stop MEETING_ID
python -m meeting_mvp status MEETING_ID
python -m meeting_mvp list
python -m meeting_mvp show MEETING_ID --transcript
python -m meeting_mvp export MEETING_ID --format markdown
python -m meeting_mvp export MEETING_ID --format json
```

`start` prints a meeting ID and records until `Ctrl+C` or a separate `stop`
command requests termination. Audio is written to
`meetings/<id>/audio/` as atomic WAV chunks, not accumulated for the whole
meeting in memory. The stop boundary creates durable per-chunk jobs. If ASR
setup fails, the audio remains on disk, the meeting becomes `partial`, the
transcription jobs become retryable failed jobs, and the CLI prints the exact
`retry` command to run after you fix the setup. Processing creates the summary
job only after the transcription jobs reach a terminal state. A failed chunk
remains visible and produces a partial meeting when other chunks succeeded.

History and `show` do not require Whisper, Ollama, or TTS to be available.
`start` itself requires a usable input device; final processing requires the
configured ASR and LLM backends.

## Web UI

The Streamlit page runs the same workflow as three separate steps:

```bash
python -m meeting_mvp ui            # http://127.0.0.1:8501
python -m meeting_mvp ui --port 8600
```

1. **Record**: pick microphone or system audio (loopback), choose a device
   (WASAPI devices only by default), then start and stop recording.
2. **ASR**: choose local Whisper (lists all models and marks the downloaded
   ones), or an OpenAI-compatible ASR address. Models are detected from
   `<address>/models`, and you can type one in manually if detection fails.
3. **LLM**: choose Ollama or an OpenAI-compatible server and enter its
   address. Models are detected from `/api/tags` or `/v1/models`. Generate or
   regenerate the minutes, then download Markdown or JSON.

Each step can be re-run with different settings. For example, you can
re-transcribe every chunk with another ASR model, or regenerate the minutes
with another LLM; summaries are kept as numbered versions. Settings chosen on
the page apply to that run only and are not written back to `.env`. The page
uses the same data directory, network policy (`MEETING_ALLOW_NETWORK`), and
remote-upload confirmation as the CLI. It has no login, so `ui` binds to
`127.0.0.1` only.

## Select audio devices

The `devices` command prints a stable ID, name, host API, channel count,
sample rate, loopback marker, and virtual-device marker. Always select the
printed ID instead of relying on an implicit default device.

### Microphone input

Microphone capture uses PortAudio through `sounddevice` with 16 kHz, mono,
16-bit PCM. A missing device, permission denial, occupied device, unsupported
sample rate, or channel mismatch produces an actionable error. On macOS,
allow the terminal or Python application under **System Settings > Privacy &
Security > Microphone**. Windows users must allow microphone access in system
privacy settings and check the selected input in Sound settings.

### Windows system audio

Use `--source system` only with a device reported as a WASAPI loopback device:

```bash
python -m meeting_mvp devices --source system
python -m meeting_mvp start --source system --device-id "loopback:Speakers"
```

`SoundCard==0.4.5` reads the selected playback device through WASAPI. This is
not microphone capture. If loopback is unavailable, select Windows Stereo Mix
or configure a virtual mixer, then select that input explicitly. Exclusive
mode, driver restrictions, playback-device changes, and privacy permissions
can still prevent capture. The microphone list is never presented as proof of
loopback support.

### macOS system audio

macOS does not provide a general system-output loopback through this CLI. To
record system audio, install and explicitly route audio through BlackHole,
Loopback, Soundflower, or an Aggregate/Multi-Output device. Then grant
Microphone permission and select the resulting input shown by:

```bash
python -m meeting_mvp devices --source system
```

Without a recognized virtual or aggregate input, the command returns
`MAC_SYSTEM_AUDIO_REQUIRES_VIRTUAL_DEVICE`. A regular Mac microphone cannot be
represented as system audio.

## Data, deletion, and recovery

The data root contains `meetings.db`, `meetings/<id>/audio/`,
`meetings/<id>/exports/`, and `logs/`. SQLite enables WAL mode, foreign keys,
ordered chunk metadata, SHA-256 hashes, transcript attempts, and summary
versions. Temporary WAV files are removed on recovery; closed chunks remain
available after an interrupted process. Expired jobs can be retried.

Use these deletion commands:

```bash
python -m meeting_mvp delete MEETING_ID --audio-only
python -m meeting_mvp delete MEETING_ID --all --confirm
```

The first preserves the transcript and minutes. The second removes database
records and the meeting directory. Do not delete an active meeting. Back up or
remove the configured data directory manually only after stopping the CLI.

## Models and limitations

Whisper model loading is lazy and can use substantial disk, RAM, and CPU/GPU.
Ollama must already be running with the selected model for final minutes. The
MVP does not diarize speakers, guarantee perfect transcription, or invent
missing owners and dates. It retains chunk-level timestamps when an ASR
backend does not return word-level timestamps. A machine without audio
hardware can only run `self-check --synthetic`; that check does not prove
microphone permissions, Windows loopback, macOS virtual routing, Whisper
inference, or Ollama availability.

On Windows PowerShell, check the local Ollama service and model without
sending meeting text:

```powershell
ollama list
Invoke-WebRequest http://127.0.0.1:11434/api/tags -UseBasicParsing
ollama pull MODEL_NAME  # only when ollama list does not show the model
```

For an OpenAI-compatible server, check its model endpoint instead:

```powershell
Invoke-WebRequest http://127.0.0.1:8080/v1/models -UseBasicParsing
```

A 404 summary error reports the resolved endpoint and model without response
bodies or credentials. Correct `MEETING_LLM_BACKEND`, `MEETING_LLM_URL`, or
`MEETING_LLM_MODEL`, then run `python -m meeting_mvp retry MEETING_ID`.
The existing transcript is retained when summary generation fails, so retry
can generate a new summary after the model setup is fixed.

Recording may capture private conversations. Obtain consent and follow local
recording laws before using microphone or system-audio capture.
