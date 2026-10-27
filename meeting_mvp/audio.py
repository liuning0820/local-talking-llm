"""Bounded audio capture, explicit device discovery, and atomic WAV chunks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import threading
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .config import Settings
from .errors import DeviceError
from .paths import safe_relative
from .store import Store, utc_now


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    name: str
    host_api: str
    input_channels: int
    output_channels: int
    default_sample_rate: float
    is_loopback: bool = False
    requires_virtual_device: bool = False
    source: str = "microphone"

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@contextmanager
def _com_initialized() -> Iterator[None]:
    """Initialize COM on the current thread for WASAPI.

    PortAudio's WASAPI backend and SoundCard fail to start streams on threads that
    never initialized COM (e.g. Streamlit's script thread or a recording worker);
    the main thread of the CLI is already initialized, which hid this.
    """
    if platform.system() != "Windows":
        yield
        return
    import ctypes
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.restype = ctypes.c_long
    # S_OK / S_FALSE need a matching CoUninitialize; RPC_E_CHANGED_MODE means the
    # thread is already initialized in another mode, which is also usable.
    hr = ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED
    try:
        yield
    finally:
        if hr in (0, 1):
            ole32.CoUninitialize()


def list_devices(source: str = "all") -> list[DeviceInfo]:
    if source not in {"microphone", "system", "all"}:
        raise DeviceError("source must be microphone, system, or all", "INVALID_AUDIO_SOURCE")
    result: list[DeviceInfo] = []
    if source in {"microphone", "all"}:
        try:
            import sounddevice as sd
            host_apis = sd.query_hostapis()
            for index, item in enumerate(sd.query_devices()):
                if int(item.get("max_input_channels", 0)) <= 0:
                    continue
                host_index = int(item.get("hostapi", 0))
                host_name = host_apis[host_index].get("name", "PortAudio") if 0 <= host_index < len(host_apis) else "PortAudio"
                result.append(DeviceInfo(
                    str(index), str(item["name"]), str(host_name),
                    int(item["max_input_channels"]), int(item.get("max_output_channels", 0)),
                    float(item.get("default_samplerate", 16000)), source="microphone",
                ))
        except ImportError as exc:
            raise DeviceError("sounddevice is not installed; install the project dependencies", "AUDIO_BACKEND_MISSING") from exc
        except Exception as exc:
            raise DeviceError(f"Unable to enumerate input devices: {exc}. Check OS audio permissions.", "AUDIO_ENUMERATION_FAILED") from exc
    if source in {"system", "all"} and platform.system() == "Windows":
        try:
            import soundcard as sc
            for speaker in sc.all_speakers():
                result.append(DeviceInfo(
                    f"loopback:{speaker.name}", str(speaker.name), "WASAPI", 2, 2,
                    48000, True, False, "system",
                ))
        except ImportError as exc:
            if source == "system":
                raise DeviceError("Install pinned SoundCard 0.4.5 for Windows WASAPI loopback; a microphone is not system audio.", "WINDOWS_LOOPBACK_BACKEND_MISSING") from exc
        except Exception as exc:
            raise DeviceError(f"WASAPI loopback enumeration failed: {exc}. Try Stereo Mix or a virtual mixer.", "WINDOWS_LOOPBACK_ENUMERATION_FAILED") from exc
    elif source == "system" and platform.system() == "Darwin":
        virtual: list[DeviceInfo] = []
        try:
            import sounddevice as sd
            names = ("blackhole", "loopback", "soundflower", "multi-output", "aggregate")
            for index, item in enumerate(sd.query_devices()):
                name = str(item["name"])
                if int(item.get("max_input_channels", 0)) > 0 and any(token in name.lower() for token in names):
                    virtual.append(DeviceInfo(
                        str(index), name, "CoreAudio", int(item["max_input_channels"]),
                        int(item.get("max_output_channels", 0)), float(item.get("default_samplerate", 48000)),
                        False, True, "system",
                    ))
        except ImportError:
            pass
        if not virtual:
            raise DeviceError("macOS system audio requires a selected BlackHole, Loopback, Soundflower, or aggregate input device; grant Microphone permission in System Settings.", "MAC_SYSTEM_AUDIO_REQUIRES_VIRTUAL_DEVICE")
        result.extend(virtual)
    elif source == "system":
        raise DeviceError("System audio loopback is implemented only for Windows WASAPI and macOS virtual input devices.", "SYSTEM_AUDIO_UNSUPPORTED_PLATFORM")
    return result


def choose_device(devices: list[DeviceInfo], device_id: str | None) -> DeviceInfo:
    if not devices:
        raise DeviceError("No usable input devices found. Check microphone permissions and connect a device.", "NO_INPUT_DEVICE")
    if device_id is None:
        return devices[0]
    for device in devices:
        if device.id == device_id:
            return device
    raise DeviceError(f"Device '{device_id}' was not found. Run 'python -m meeting_mvp devices'.", "AUDIO_DEVICE_NOT_FOUND")


class SoundDeviceSource:
    def __init__(self, device: DeviceInfo, settings: Settings):
        self.device, self.settings = device, settings
        self.dropped_blocks = 0
        self.frames_received = 0
        self.received_bytes = 0
        self.last_status = ""
        self.selected_format: tuple[int, int] | None = None

    def _formats(self) -> list[tuple[int, int]]:
        requested = (self.settings.sample_rate, self.settings.channels)
        native = (max(1, round(self.device.default_sample_rate)), max(1, self.device.input_channels))
        options = [requested, native, (native[0], 1)]
        return list(dict.fromkeys(options))

    def _convert(self, samples: np.ndarray, capture_rate: int) -> bytes:
        values = np.asarray(samples, dtype=np.int16)
        if values.ndim == 1:
            values = values[:, None]
        values = values.mean(axis=1)
        if capture_rate != self.settings.sample_rate and len(values) > 1:
            target_length = max(1, round(len(values) * self.settings.sample_rate / capture_rate))
            values = np.interp(
                np.linspace(0, len(values) - 1, target_length),
                np.arange(len(values)), values,
            )
        return np.clip(values, -32768, 32767).astype(np.int16).tobytes()

    def frames(self, stop: threading.Event) -> Iterator[tuple[bytes, float]]:
        with _com_initialized():
            yield from self._frames(stop)

    def _frames(self, stop: threading.Event) -> Iterator[tuple[bytes, float]]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise DeviceError("sounddevice is not installed", "AUDIO_BACKEND_MISSING") from exc
        data: queue.Queue[tuple[bytes, float]] = queue.Queue(maxsize=self.settings.queue_chunks)
        last_open_error: Exception | None = None
        for capture_rate, capture_channels in self._formats():
            if stop.is_set():
                return
            data = queue.Queue(maxsize=self.settings.queue_chunks)

            def callback(indata, _frames, _time, status):
                timestamp = time.monotonic()
                self.last_status = str(status) if status else ""
                if status:
                    self.dropped_blocks += 1
                try:
                    pcm = self._convert(indata, capture_rate)
                    if pcm:
                        data.put_nowait((pcm, timestamp))
                except queue.Full:
                    self.dropped_blocks += 1

            try:
                with sd.InputStream(device=int(self.device.id), samplerate=capture_rate,
                                    channels=capture_channels, dtype="int16", callback=callback):
                    deadline = time.monotonic() + 1.0
                    while not stop.is_set():
                        try:
                            frame = data.get(timeout=0.2)
                            self.frames_received += 1
                            self.received_bytes += len(frame[0])
                            self.selected_format = (capture_rate, capture_channels)
                            yield frame
                        except queue.Empty:
                            if self.frames_received == 0 and time.monotonic() >= deadline:
                                break
                    if self.frames_received:
                        return
            except Exception as exc:
                last_open_error = exc
                continue
        detail = f" Last callback status: {self.last_status}." if self.last_status else ""
        if last_open_error is not None:
            detail += f" Last open error: {last_open_error}."
        if self.frames_received == 0:
            raise DeviceError(
                f"'{self.device.name}' produced no audio frames or bytes after trying "
                f"{self._formats()} using {self.device.host_api}.{detail} Check Windows microphone "
                "permission, mute/routing, or run `python -m meeting_mvp devices` and test the "
                "same microphone under another listed host API.",
                "AUDIO_NO_FRAMES",
            ) from last_open_error


class WindowsLoopbackSource:
    def __init__(self, device: DeviceInfo, settings: Settings):
        self.device, self.settings = device, settings
        self.dropped_blocks = 0

    def frames(self, stop: threading.Event) -> Iterator[tuple[bytes, float]]:
        with _com_initialized():
            yield from self._frames(stop)

    def _frames(self, stop: threading.Event) -> Iterator[tuple[bytes, float]]:
        try:
            import numpy as np
            import soundcard as sc
        except ImportError as exc:
            raise DeviceError("Windows system audio needs pinned SoundCard 0.4.5; it is not microphone capture.", "WINDOWS_LOOPBACK_BACKEND_MISSING") from exc
        speaker = next((item for item in sc.all_speakers() if item.name == self.device.name), None)
        if speaker is None:
            raise DeviceError("Selected playback device disappeared; choose it again.", "AUDIO_DEVICE_GONE")
        try:
            # SoundCard exposes loopback capture as a "microphone" bound to the speaker.
            loopback = sc.get_microphone(id=str(speaker.id), include_loopback=True)
        except Exception as exc:
            raise DeviceError(f"No WASAPI loopback endpoint for '{speaker.name}': {exc}", "WINDOWS_LOOPBACK_UNAVAILABLE") from exc
        try:
            with loopback.recorder(samplerate=self.settings.sample_rate, channels=self.settings.channels) as recorder:
                while not stop.is_set():
                    samples = recorder.record(numframes=max(1600, self.settings.sample_rate // 5))
                    if samples is not None and len(samples):
                        values = np.asarray(samples)
                        if values.ndim > 1:
                            values = values.mean(axis=1)
                        yield (np.clip(values, -1, 1) * 32767).astype(np.int16).tobytes(), time.monotonic()
        except Exception as exc:
            raise DeviceError(f"WASAPI loopback failed: {exc}. Try Stereo Mix or a virtual mixer.", "WINDOWS_LOOPBACK_STREAM_FAILED") from exc


def source_for(device: DeviceInfo, settings: Settings):
    if device.is_loopback:
        return WindowsLoopbackSource(device, settings)
    return SoundDeviceSource(device, settings)


def _frame_parts(frame: bytes | tuple[bytes, float], logical_clock: float, bytes_per_second: float) -> tuple[bytes, float, float, bool]:
    if isinstance(frame, tuple):
        pcm, started = frame
        started = float(started)
        return pcm, started, started + len(pcm) / bytes_per_second, True
    pcm = bytes(frame)
    return pcm, logical_clock, logical_clock + len(pcm) / bytes_per_second, False


def write_frames(store: Store, meeting_id: str, frames: Iterable[bytes | tuple[bytes, float]], settings: Settings, dropped_blocks: int = 0) -> int:
    """Write one bounded WAV at a time and preserve capture-clock gaps."""
    directory = store.root / "meetings" / meeting_id / "audio"
    bytes_per_second = settings.sample_rate * settings.channels * settings.sample_width
    target = int(bytes_per_second * settings.chunk_seconds)
    pending = bytearray()
    pending_start: float | None = None
    pending_end: float | None = None
    logical_clock = 0.0
    previous_end: float | None = None
    sequence = len(store.chunks(meeting_id, include_deleted=True)) + 1
    missing = False

    interrupted = False
    try:
        for frame in frames:
            pcm, started, ended, timed = _frame_parts(frame, logical_clock, bytes_per_second)
            if not pcm:
                continue
            if timed and previous_end is not None and started > previous_end + 0.05:
                store.note_gap(meeting_id, int(previous_end * 1000), int(started * 1000))
                missing = True
            previous_end = ended
            logical_clock = max(logical_clock, ended)
            if pending_start is None:
                pending_start = started
            pending.extend(pcm)
            pending_end = ended
            while len(pending) >= target:
                block = bytes(pending[:target])
                del pending[:target]
                start = pending_start or 0.0
                end = start + len(block) / bytes_per_second
                _write_chunk(store, meeting_id, directory, sequence, block, int(start * 1000), int(end * 1000), settings)
                sequence += 1
                pending_start = end if pending else None
                pending_end = pending_end if pending else None
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if pending:
            start = pending_start or 0.0
            end = pending_end if pending_end is not None else start + len(pending) / bytes_per_second
            _write_chunk(store, meeting_id, directory, sequence, bytes(pending), int(start * 1000), int(end * 1000), settings)
            sequence += 1
    if interrupted:
        raise KeyboardInterrupt
    dropped = max(dropped_blocks, 0)
    store.heartbeat(meeting_id, written_bytes=sum(item["size_bytes"] for item in store.chunks(meeting_id, include_deleted=True)), dropped_blocks=dropped)
    if dropped:
        store.note_gap(meeting_id, 0, 0, "audio queue overflow")
        missing = True
    if missing:
        store.mark_partial(meeting_id, "Audio capture contained dropped blocks or clock gaps")
    return sequence - 1


def _write_chunk(store: Store, meeting_id: str, directory: Path, sequence: int, pcm: bytes,
                 start_ms: int, end_ms: int, settings: Settings) -> None:
    if not pcm:
        return
    name = f"chunk-{sequence:06d}.wav"
    destination = directory / name
    temporary = directory / (name + ".tmp")
    directory.mkdir(parents=True, exist_ok=True)
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(settings.channels)
        handle.setsampwidth(settings.sample_width)
        handle.setframerate(settings.sample_rate)
        handle.writeframes(pcm)
    with temporary.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    os.replace(temporary, destination)
    relative = safe_relative(store.root, destination)
    store.add_audio_chunk(meeting_id, sequence, relative, start_ms, end_ms, destination.stat().st_size, digest)


def probe_source(source, settings: Settings, duration: float = 0.5) -> None:
    """Open and close a source briefly without creating meeting data."""
    stop = threading.Event()
    frames = source.frames(stop)
    deadline = time.monotonic() + duration
    received = False
    try:
        while time.monotonic() < deadline:
            try:
                frame = next(frames)
            except StopIteration:
                break
            pcm = frame[0] if isinstance(frame, tuple) else frame
            if pcm:
                received = True
                break
    except Exception as exc:
        if isinstance(exc, DeviceError):
            raise
        raise DeviceError(f"Audio device probe failed: {exc}", "AUDIO_PROBE_FAILED") from exc
    finally:
        stop.set()
        close = getattr(frames, "close", None)
        if close:
            close()
    if not received:
        raise DeviceError(
            "The selected device opened but produced no audio in 500 ms; "
            "check permissions, routing, and device selection.",
            "AUDIO_PROBE_NO_FRAMES",
        )


def record_meeting(store: Store, meeting_id: str, source, settings: Settings) -> int:
    stop = threading.Event()
    watcher_done = threading.Event()

    def watch_stop() -> None:
        while not watcher_done.wait(0.25):
            meeting = store.get_meeting(meeting_id)
            if not meeting or meeting.get("stop_requested"):
                stop.set()
                return
            store.heartbeat(meeting_id, dropped_blocks=getattr(source, "dropped_blocks", 0))

    watcher = threading.Thread(target=watch_stop, daemon=True)
    watcher.start()
    try:
        chunks = write_frames(store, meeting_id, source.frames(stop), settings)
        meeting = store.get_meeting(meeting_id) or {}
        dropped = int(getattr(source, "dropped_blocks", 0))
        missing = json.loads(meeting.get("missing_ranges") or "[]")
        final_status = "failed" if chunks == 0 else ("partial" if dropped or missing else "stopping")
        error = (
            "No audio frames were received and no audio bytes were written; "
            "recording was not sent to ASR. Run `python -m meeting_mvp devices` "
            "and test another host API/device."
            if chunks == 0 else
            "Audio capture was incomplete" if final_status == "partial" else None
        )
        store.update_meeting(meeting_id, status=final_status, ended_at=utc_now(), dropped_blocks=dropped, error=error)
        return chunks
    except KeyboardInterrupt:
        # Ctrl+C is a stop request, not an unrecoverable recording failure.
        # write_frames has already flushed its current bounded chunk in its
        # cleanup path before this handler persists the common stop state.
        stop.set()
        store.update_meeting(
            meeting_id, stop_requested=1, status="stopping", ended_at=utc_now(),
        )
        return len(store.chunks(meeting_id))
    except Exception as exc:
        store.update_meeting(meeting_id, status="interrupted", ended_at=utc_now(), error=str(exc))
        raise
    finally:
        watcher_done.set()
        stop.set()
        watcher.join(timeout=1)
