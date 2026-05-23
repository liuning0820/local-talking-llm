import io
import wave

import requests
import numpy as np


def numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = 16000) -> bytes:
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def transcribe_llama_asr(
    audio_np: np.ndarray,
    *,
    base_url: str,
    model: str,
    language: str | None = None,
    api_key: str = "sk-no-key-required",
    timeout: float | None = 600.0,
) -> str:
    if audio_np.size == 0:
        return ""

    wav_bytes = numpy_to_wav_bytes(audio_np)
    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model": model,
        "response_format": "json",
    }
    if language:
        data["language"] = language

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(url, files=files, data=data, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Provide more actionable diagnostics when server returns non-2xx
        resp = e.response if getattr(e, "response", None) is not None else response
        msg = (
            f"LLAMA-ASR HTTP error {resp.status_code} for url {resp.url}\n"
            f"Response headers: {dict(resp.headers)}\n"
            f"Response body: {resp.text}"
        )
        raise RuntimeError(msg) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request to ASR server failed: {e}") from e

    payload = response.json()
    if isinstance(payload, dict):
        return str(payload.get("text", "")).strip()
    return str(payload).strip()


def transcribe_whisper(stt_model, audio_np: np.ndarray) -> str:
    result = stt_model.transcribe(audio_np, fp16=False)
    return result["text"].strip()
