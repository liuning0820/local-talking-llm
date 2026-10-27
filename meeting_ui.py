"""Streamlit UI for the local meeting workflow: ① record -> ② ASR -> ③ LLM minutes.

Start with ``python -m meeting_mvp ui`` (binds to 127.0.0.1 only), or
``python -m streamlit run meeting_ui.py`` from the repository root.
"""

from __future__ import annotations

import json
import platform
from dataclasses import replace
from datetime import datetime, timezone

import streamlit as st

from meeting_mvp.audio import DeviceInfo, list_devices
from meeting_mvp.config import Settings
from meeting_mvp.discovery import (
    asr_models_endpoint, fetch_models, is_local_url, llm_models_endpoint,
    normalize_openai_base, whisper_models,
)
from meeting_mvp.errors import MeetingError
from meeting_mvp.recovery import recover
from meeting_mvp.service import create_store
from meeting_mvp.store import Store
from meeting_mvp.workflow import (
    BackgroundTask, error_text, recording_finished, start_recording, summarize_step, transcribe_step,
)

st.set_page_config(page_title="本地会议纪要", page_icon="🎙️", layout="wide")

STATUS_LABELS = {
    "recording": "🔴 录音中", "stopping": "⏹️ 已停止录音", "interrupted": "⚠️ 录音中断",
    "transcribing": "📝 转写中", "summarizing": "🤖 生成纪要中", "completed": "✅ 已完成",
    "partial": "🟡 部分完成", "failed": "❌ 失败", "deleting": "🗑️ 删除中",
}
SOURCE_LABELS = {"microphone": "麦克风", "system": "系统声音（扬声器回环）"}
ASR_BACKENDS = {"whisper": "本地 Whisper", "llama-asr": "OpenAI 兼容 ASR 接口（如 llama-server）"}
LLM_BACKENDS = {"ollama": "Ollama（/api/chat）", "openai": "OpenAI 兼容接口（/v1/chat/completions）"}


class AppState:
    """Process-wide state shared by all browser tabs; background work outlives page reruns."""

    def __init__(self) -> None:
        self.settings = Settings.from_env()
        self.store: Store = create_store(self.settings)
        recover(self.store)
        self.recorder: BackgroundTask | None = None
        self.processor: BackgroundTask | None = None

    @property
    def recording(self) -> bool:
        return self.recorder is not None and self.recorder.running

    @property
    def processing(self) -> bool:
        return self.processor is not None and self.processor.running


@st.cache_resource
def app_state() -> AppState:
    return AppState()


@st.cache_data(ttl=60, show_spinner="正在枚举音频设备…")
def cached_devices(source: str) -> list[DeviceInfo]:
    return list_devices(source)


@st.cache_data(ttl=30, show_spinner=False)
def cached_models(endpoint: str, allow_network: bool) -> tuple[list[str], str | None]:
    """Errors are cached too, so an offline server does not slow every rerun."""
    probe = Settings(data_dir=state.settings.data_dir, allow_network=allow_network)
    try:
        return fetch_models(endpoint, probe, timeout=3.0), None
    except Exception as exc:
        return [], error_text(exc)


@st.cache_data(show_spinner=False)
def cached_whisper_models() -> tuple[list[tuple[str, bool]], str | None]:
    try:
        return whisper_models(), None
    except Exception as exc:
        return [], error_text(exc)


state = app_state()
store = state.store
base = state.settings
ss = st.session_state


# --------------------------------------------------------------------------- helpers

def fmt_ms(value: int) -> str:
    seconds = int(value) // 1000
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def local_time(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return iso


def meeting_label(meeting: dict) -> str:
    title = meeting.get("title") or "未命名"
    return f"{title} · {local_time(meeting['started_at'])} · {STATUS_LABELS.get(meeting['status'], meeting['status'])}"


def remote_guard(url: str, data_type: str, key: str) -> tuple[bool, bool]:
    """Return (usable, network_confirmed) for a model URL, rendering any warning."""
    if not url.strip():
        st.warning("请填写模型地址")
        return False, False
    if is_local_url(url):
        return True, False
    if not base.allow_network:
        st.error(
            f"{url} 不是本机地址。当前 MEETING_ALLOW_NETWORK=false，远程地址被拒绝；"
            "如确需使用，请在 .env 中设置 MEETING_ALLOW_NETWORK=true 后重启页面。"
        )
        return False, False
    confirmed = st.checkbox(f"我确认将{data_type}发送到远程地址 {url}", key=key)
    if not confirmed:
        st.warning(f"使用远程地址前需要确认上传{data_type}")
    return confirmed, confirmed


def model_picker(label: str, endpoint: str, default: str, key: str, enabled: bool) -> str:
    """Auto-detected model list with a manual-entry fallback."""
    if not enabled:
        return st.text_input(f"{label}（手动输入）", value=default, key=f"{key}_manual", disabled=True).strip()
    cols = st.columns([5, 1], vertical_alignment="bottom")
    with cols[1]:
        if st.button("🔄 检测", key=f"{key}_refresh", help=f"重新请求 {endpoint}"):
            cached_models.clear()
    models, error = cached_models(endpoint, base.allow_network)
    with cols[0]:
        if models:
            index = models.index(default) if default in models else 0
            # Key includes the endpoint so a new address gets a fresh selection.
            choice = st.selectbox(label, models, index=index, key=f"{key}_select_{endpoint}")
        else:
            choice = st.text_input(f"{label}（手动输入）", value=default, key=f"{key}_manual").strip()
    if models:
        st.caption(f"✅ 从 `{endpoint}` 检测到 {len(models)} 个模型")
    else:
        st.caption(f"⚠️ 无法自动检测模型，可手动输入：{error}")
    return choice


# --------------------------------------------------------------------------- callbacks

def on_start_recording(device: DeviceInfo, title: str, chunk_seconds: int) -> None:
    ss.pop("record_error", None)
    if state.recording:
        ss["record_error"] = "已有一场录音正在进行"
        return
    try:
        settings = replace(base, chunk_seconds=int(chunk_seconds))
        meeting, task = start_recording(store, settings, device, title)
    except Exception as exc:
        ss["record_error"] = error_text(exc)
        return
    state.recorder = task
    ss["meeting_id"] = meeting["id"]


def on_stop_recording() -> None:
    task = state.recorder
    if task is not None and task.running:
        try:
            store.request_stop(task.meeting_id)
        except MeetingError as exc:
            ss["record_error"] = error_text(exc)


def launch(kind: str, meeting_id: str, run) -> None:
    if state.processing:
        ss[f"{kind}_error"] = "已有转写或总结任务在运行，请等待它结束"
        return
    state.processor = BackgroundTask(kind, meeting_id, run).start()


# --------------------------------------------------------------------------- live status fragments

@st.fragment(run_every=1.0)
def recording_panel() -> None:
    task = state.recorder
    if task is None or not task.running:
        st.rerun()
    meeting = store.get_meeting(task.meeting_id) or {}
    started = datetime.fromisoformat(meeting["started_at"]) if meeting.get("started_at") else None
    elapsed = int((datetime.now(timezone.utc) - started).total_seconds()) if started else 0
    stopping = meeting.get("status") == "stopping"
    st.markdown(
        f"### {'⏳ 正在停止…' if stopping else '🔴 正在录音'}  `{fmt_ms(elapsed * 1000)}`\n"
        f"设备：**{meeting.get('device_name', '?')}** · 已写入片段：{len(store.chunks(task.meeting_id))} 个"
        + (f" · ⚠️ 丢失音频块 {meeting['dropped_blocks']} 个" if meeting.get("dropped_blocks") else "")
    )
    st.button("⏹️ 停止录音", type="primary", on_click=on_stop_recording, disabled=stopping, key="stop_recording")


@st.fragment(run_every=1.0)
def processing_panel() -> None:
    task = state.processor
    if task is None or not task.running:
        st.rerun()
    st.info(f"⏳ {task.progress or '准备中…'}（已用时 {int(task.elapsed)} 秒）")


def show_task_result(kind: str, meeting_id: str) -> None:
    task = state.processor
    if ss.get(f"{kind}_error"):
        st.error(ss.pop(f"{kind}_error"))
    if task is None or task.kind != kind or task.meeting_id != meeting_id:
        return
    if task.running:
        processing_panel()
    elif task.error:
        st.error(f"上次运行失败（{int(task.elapsed)} 秒）：{task.error}")
    elif kind == "transcribe" and isinstance(task.result, dict):
        result = task.result
        if result["transcribed"] or result["failed"]:
            message = f"转写完成：{result['transcribed']} 个片段成功，{result['failed']} 个失败，共 {result['segments']} 条文本（{int(task.elapsed)} 秒）"
            (st.warning if result["failed"] else st.success)(message)
        else:
            st.info(task.progress)
    elif kind == "summarize":
        st.success(f"纪要已生成（{int(task.elapsed)} 秒）")


# --------------------------------------------------------------------------- sidebar: meeting history

with st.sidebar:
    st.title("🎙️ 本地会议纪要")
    meetings = store.list_meetings()
    ids = [item["id"] for item in meetings]
    by_id = {item["id"]: item for item in meetings}
    if ss.get("meeting_id") not in by_id:
        ss["meeting_id"] = ids[0] if ids else None
    if ids:
        st.selectbox("当前会议", ids, key="meeting_id", format_func=lambda mid: meeting_label(by_id[mid]))
    else:
        st.info("还没有会议记录，先在第 ① 步开始录音。")
    st.divider()
    st.caption(f"数据目录：`{base.data_dir}`")
    st.caption("所有音频、转写和纪要都只保存在本机。")

meeting_id: str | None = ss.get("meeting_id")
meeting = store.get_meeting(meeting_id) if meeting_id else None
chunks = store.chunks(meeting_id) if meeting else []
transcripts = store.transcripts(meeting_id) if meeting else []
summary = store.summary(meeting_id) if meeting else None
finished = bool(meeting) and recording_finished(meeting) and not (state.recording and state.recorder.meeting_id == meeting_id)

st.title("会议录音 → 转写 → 纪要")
# Chunk/segment times come from the capture clock (time since boot), so show them
# relative to the first chunk instead of as absolute values.
origin = min((item["start_ms"] for item in chunks), default=0)


def rel(ms: int) -> str:
    return fmt_ms(max(0, int(ms) - origin))


if meeting:
    duration = sum(max(0, item["end_ms"] - item["start_ms"]) for item in chunks)
    st.markdown(
        f"**{meeting.get('title') or '未命名'}** · {STATUS_LABELS.get(meeting['status'], meeting['status'])} · "
        f"开始于 {local_time(meeting['started_at'])} · 录音时长 {fmt_ms(duration)} · 设备 {meeting.get('device_name') or '—'}"
    )
    if meeting.get("error") and meeting["status"] != "completed":
        st.caption(f"状态说明：{meeting['error']}")


# --------------------------------------------------------------------------- ① recording

with st.container(border=True):
    st.subheader("① 录音")
    record_error = ss.pop("record_error", None)
    if record_error:
        st.error(record_error)
    recorder = state.recorder
    if recorder is not None and not recorder.running and recorder.error and recorder.meeting_id == meeting_id:
        st.error(f"录音异常结束：{recorder.error}（已写入的片段仍可转写）")
    if state.recording:
        recording_panel()
    else:
        cols = st.columns(2)
        with cols[0]:
            source = st.radio("音频来源", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get, horizontal=True, key="rec_source")
        with cols[1]:
            show_all = st.checkbox("显示全部音频接口（MME / DirectSound / WDM-KS）", key="rec_show_all",
                                   help="默认只显示 WASAPI 设备：同一个硬件在每种接口下都会出现一次，WASAPI 最稳定")
        try:
            devices = cached_devices(source)
        except Exception as exc:
            devices = []
            st.error(error_text(exc))
        if not show_all and platform.system() == "Windows":
            wasapi = [item for item in devices if "WASAPI" in item.host_api]
            devices = wasapi or devices
        if devices:
            cols = st.columns([5, 1], vertical_alignment="bottom")
            with cols[0]:
                device = st.selectbox(
                    "录音设备", devices, key=f"rec_device_{source}_{show_all}",
                    format_func=lambda item: f"{item.name}  ·  {item.host_api}  ·  ID {item.id}",
                )
            with cols[1]:
                if st.button("🔄 刷新设备"):
                    cached_devices.clear()
                    st.rerun()
            if source == "system":
                st.caption("系统声音录的是该扬声器/耳机正在播放的内容（例如会议中对方的声音），不包含你的麦克风。")
            cols = st.columns([3, 2])
            with cols[0]:
                title = st.text_input("会议标题（可选）", max_chars=base.title_max_length, key="rec_title")
            with cols[1]:
                chunk_seconds = st.slider("分段时长（秒）", 30, 900, base.chunk_seconds, step=30, key="rec_chunk",
                                          help="录音按此时长切分成多个 WAV 文件，转写也按片段进行")
            st.button("🎙️ 开始录音", type="primary", on_click=on_start_recording, args=(device, title, chunk_seconds))
            st.caption("开始前会先试录 0.5 秒检查设备。录音前请确认已取得参会者同意。")
        else:
            st.warning("没有找到可用的录音设备。")
    if meeting and chunks and finished:
        with st.expander(f"本场会议的录音片段（{len(chunks)} 个）"):
            for chunk in chunks:
                st.caption(f"片段 #{chunk['sequence']} · {rel(chunk['start_ms'])} – {rel(chunk['end_ms'])}")
                path = store.root / chunk["relative_path"]
                if path.is_file():
                    st.audio(str(path), format="audio/wav")
                else:
                    st.caption("（音频文件已删除）")


# --------------------------------------------------------------------------- ② ASR

with st.container(border=True):
    st.subheader("② 语音转文字（ASR）")
    asr_backend = st.radio("ASR 方式", list(ASR_BACKENDS), format_func=ASR_BACKENDS.get, horizontal=True,
                           index=list(ASR_BACKENDS).index(base.stt_backend) if base.stt_backend in ASR_BACKENDS else 0,
                           key="asr_backend")
    asr_usable, asr_confirmed = True, False
    whisper_name, asr_url, asr_model = base.whisper_model, base.asr_url, base.asr_model
    if asr_backend == "whisper":
        options, whisper_error = cached_whisper_models()
        if whisper_error:
            asr_usable = False
            st.error(whisper_error)
        else:
            names = [name for name, _ in options]
            downloaded = dict(options)
            whisper_name = st.selectbox(
                "Whisper 模型", names, index=names.index(base.whisper_model) if base.whisper_model in names else 0,
                format_func=lambda name: f"{name}{'  ✅ 已下载' if downloaded[name] else '  ⬇️ 首次使用会自动下载'}",
                key="asr_whisper_model", help="模型越大越准确也越慢；当前 torch 为 CPU 版本",
            )
    else:
        asr_url = normalize_openai_base(st.text_input(
            "ASR 服务地址", value=normalize_openai_base(base.asr_url), key="asr_url",
            help="OpenAI 兼容的 /v1 地址；转写请求发往 <地址>/audio/transcriptions，模型列表来自 <地址>/models",
        ))
        asr_usable, asr_confirmed = remote_guard(asr_url, "录音音频", "asr_remote_ok")
        asr_model = model_picker("ASR 模型", asr_models_endpoint(asr_url), base.asr_model, "asr_model", asr_usable)
        if not asr_model:
            asr_usable = False

    rerun_all = False
    if transcripts:
        rerun_all = st.checkbox("重新转写全部片段（会覆盖现有转写）", key="asr_rerun_all")
    pending = [job for job in store.jobs(meeting_id) if job["kind"] == "transcribe" and job["status"] != "completed"] if meeting else []
    asr_ready = bool(meeting) and finished and bool(chunks) and asr_usable and not state.processing
    if meeting and not chunks:
        st.caption("当前会议没有录音片段。")
    elif meeting and not finished:
        st.caption("录音结束后才能转写。")

    asr_settings = replace(
        base, stt_backend=asr_backend, whisper_model=whisper_name, asr_url=asr_url, asr_model=asr_model,
        network_confirmed=asr_confirmed,
    )
    label = "▶️ 开始转写" if not transcripts else ("🔁 重新转写全部" if rerun_all else f"▶️ 转写剩余片段（{len(pending)} 个）")
    if st.button(label, type="primary", disabled=not asr_ready or (bool(transcripts) and not rerun_all and not pending), key="asr_run"):
        launch("transcribe", meeting_id, lambda task, s=asr_settings, mid=meeting_id, again=rerun_all:
               transcribe_step(store, mid, s, rerun_all=again, progress=task.report))
        st.rerun()
    if meeting:
        show_task_result("transcribe", meeting_id)

    if transcripts:
        ok = [item for item in transcripts if item["status"] == "succeeded" and item.get("text")]
        bad = [item for item in transcripts if item["status"] == "failed"]
        st.markdown(f"**转写结果**：{len(ok)} 条文本" + (f"，{len(bad)} 个片段失败" if bad else ""))
        rows = [
            {"开始": rel(item["start_ms"]), "结束": rel(item["end_ms"]),
             "文本": item.get("text") or f"❌ {item.get('error') or '转写失败'}",
             "模型": f"{item.get('backend')}/{item.get('model') or '-'}"}
            for item in transcripts
        ]
        st.dataframe(rows, hide_index=True, width="stretch", height=min(400, 38 + 35 * len(rows)))
        plain = "\n".join(f"[{rel(item['start_ms'])}] {item['text']}" for item in ok)
        st.download_button("⬇️ 下载转写文本", plain, file_name=f"transcript-{meeting_id[:8]}.txt", mime="text/plain")


# --------------------------------------------------------------------------- ③ LLM

with st.container(border=True):
    st.subheader("③ LLM 生成会议纪要")
    llm_backend = st.radio("接口类型", list(LLM_BACKENDS), format_func=LLM_BACKENDS.get, horizontal=True,
                           index=list(LLM_BACKENDS).index(base.llm_backend) if base.llm_backend in LLM_BACKENDS else 0,
                           key="llm_backend")
    cols = st.columns([4, 1])
    with cols[0]:
        llm_url = st.text_input("LLM 服务地址", value=base.llm_url, key="llm_url",
                                help="Ollama 填 http://127.0.0.1:11434；OpenAI 兼容服务填主机或 /v1 地址").strip()
    with cols[1]:
        timeout = st.number_input("请求超时（秒）", 10, 3600, int(base.request_timeout), step=10, key="llm_timeout")
    llm_usable, llm_confirmed = remote_guard(llm_url, "转写文本和会议内容", "llm_remote_ok")
    llm_model = base.llm_model
    if llm_usable:
        try:
            endpoint = llm_models_endpoint(replace(base, llm_backend=llm_backend, llm_url=llm_url))
        except Exception as exc:
            endpoint, llm_usable = "", False
            st.error(error_text(exc))
        if endpoint:
            llm_model = model_picker("LLM 模型", endpoint, base.llm_model, "llm_model", True)
    if not llm_model:
        llm_usable = False

    usable_text = any(item["status"] == "succeeded" and item.get("text") for item in transcripts)
    transcription_pending = any(job["kind"] == "transcribe" and job["status"] in {"queued", "running"} for job in store.jobs(meeting_id)) if meeting else False
    llm_ready = bool(meeting) and finished and usable_text and not transcription_pending and llm_usable and not state.processing
    if meeting and not usable_text:
        st.caption("需要先在第 ② 步得到转写文本。")
    llm_settings = replace(
        base, llm_backend=llm_backend, llm_url=llm_url, llm_model=llm_model,
        request_timeout=float(timeout), network_confirmed=llm_confirmed,
    )
    has_summary = bool(summary and summary.get("status") == "ready")
    if st.button("🔁 重新生成纪要" if has_summary else "▶️ 生成纪要", type="primary", disabled=not llm_ready, key="llm_run"):
        launch("summarize", meeting_id, lambda task, s=llm_settings, mid=meeting_id:
               summarize_step(store, mid, s, progress=task.report))
        st.rerun()
    if meeting:
        show_task_result("summarize", meeting_id)

    if summary:
        if summary.get("status") == "ready" and summary.get("markdown"):
            st.caption(f"版本 {summary['version']} · 模型 {summary.get('model') or '—'} · 生成于 {local_time(summary.get('created_at'))}")
            with st.container(border=True):
                st.markdown(summary["markdown"])
            export = {
                "meeting": meeting,
                "summary": summary["payload"],
                "transcript": [{k: item[k] for k in ("id", "start_ms", "end_ms", "text", "status")} for item in transcripts],
            }
            cols = st.columns(2)
            cols[0].download_button("⬇️ 下载纪要 Markdown", summary["markdown"],
                                    file_name=f"minutes-{meeting_id[:8]}.md", mime="text/markdown")
            cols[1].download_button("⬇️ 下载完整 JSON", json.dumps(export, ensure_ascii=False, indent=2),
                                    file_name=f"meeting-{meeting_id[:8]}.json", mime="application/json")
        elif not (state.processor and state.processor.kind == "summarize" and state.processor.meeting_id == meeting_id):
            st.error(f"最近一次纪要生成失败：{summary.get('error') or '未知错误'}")
