"""Command line entry point: ``python -m meeting_mvp``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from urllib.parse import urlparse

from .audio import choose_device, list_devices, probe_source, record_meeting, source_for
from .config import Settings
from .errors import MeetingError
from .export import export_meeting
from .recovery import recover
from .self_check import run as synthetic_run
from .service import create_store, process_meeting


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local meeting recording, transcription, minutes, and history")
    sub = root.add_subparsers(dest="command", required=True)
    devices = sub.add_parser("devices", help="list selectable microphone/system devices")
    devices.add_argument("--source", choices=["microphone", "system", "all"], default="all")
    start = sub.add_parser("start", help="start recording until Ctrl+C or stop ID")
    start.add_argument("--source", choices=["microphone", "system"], default="microphone")
    start.add_argument("--device-id")
    start.add_argument("--title")
    start.add_argument("--yes", action="store_true", help="confirm configured remote ASR/LLM uploads")
    stop = sub.add_parser("stop", help="request a running recording to stop")
    stop.add_argument("id")
    status = sub.add_parser("status", help="show one or all meeting statuses")
    status.add_argument("id", nargs="?")
    listing = sub.add_parser("list", help="list meeting history")
    listing.add_argument("--status")
    show = sub.add_parser("show", help="show saved minutes and optional transcript")
    show.add_argument("id")
    show.add_argument("--transcript", action="store_true")
    retry = sub.add_parser("retry", help="retry failed transcription/summary work")
    retry.add_argument("id")
    retry.add_argument("--yes", action="store_true", help="confirm configured remote ASR/LLM uploads")
    export = sub.add_parser("export", help="export a meeting")
    export.add_argument("id")
    export.add_argument("--format", choices=["markdown", "json"], required=True)
    export.add_argument("--output")
    delete = sub.add_parser("delete", help="delete audio or a meeting")
    delete.add_argument("id")
    group = delete.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio-only", action="store_true")
    group.add_argument("--all", action="store_true")
    delete.add_argument("--confirm", action="store_true")
    check = sub.add_parser("self-check", help="run a no-hardware/no-model synthetic pipeline")
    check.add_argument("--synthetic", action="store_true")
    return root


def _confirm_remote(
    settings: Settings,
    assume_yes: bool,
    *,
    need_asr: bool = True,
    need_llm: bool = True,
) -> Settings:
    targets: list[tuple[str, str]] = []
    configured_targets = []
    if need_asr and settings.stt_backend == "llama-asr":
        configured_targets.append((settings.asr_url, "音频 WAV"))
    if need_llm:
        configured_targets.append((settings.llm_url, "转写文本和会议内容"))
    for url, data_type in configured_targets:
        host = (urlparse(url).hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            targets.append((url, data_type))
    if not targets:
        return settings
    if not settings.allow_network:
        settings.validate_url(targets[0][0])
    print("网络上传确认：" + "; ".join(f"{data_type} -> {urlparse(url).netloc}" for url, data_type in targets), flush=True)
    confirmed = assume_yes
    if not confirmed:
        answer = input("输入 YES 确认，其他输入取消: ").strip()
        confirmed = answer == "YES"
    if not confirmed:
        raise MeetingError("Remote upload was not confirmed; no audio or transcript was sent", "REMOTE_NETWORK_CONFIRMATION_REQUIRED")
    return replace(settings, network_confirmed=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = Settings.from_env()
    if args.command == "devices":
        for device in list_devices(args.source):
            print(json.dumps(device.as_dict(), ensure_ascii=False))
        return 0
    store = create_store(settings)
    recover(store)
    try:
        if args.command == "self-check":
            if not args.synthetic:
                raise MeetingError("Use --synthetic; real devices/models are not part of self-check", "SYNTHETIC_FLAG_REQUIRED")
            print(json.dumps(synthetic_run(settings), ensure_ascii=False, indent=2))
            return 0
        if args.command == "start":
            settings = _confirm_remote(settings, args.yes)
            devices = list_devices(args.source)
            device = choose_device(devices, args.device_id)
            meeting = store.create_meeting(
                title=args.title, input_mode=args.source, device_id=device.id,
                device_name=device.name, sample_rate=settings.sample_rate, channels=settings.channels,
            )
            try:
                probe_source(source_for(device, settings), settings)
            except Exception as exc:
                store.update_meeting(
                    meeting["id"], status="failed", ended_at=__import__("meeting_mvp.store", fromlist=["utc_now"]).utc_now(),
                    error=f"Audio short test failed: {exc}",
                )
                raise
            print(f"RECORDING {meeting['id']} title={args.title or '未命名'} device={device.name} started={meeting['started_at']}", flush=True)
            stop_input = __import__("threading").Event()

            def request_stop_on_enter() -> None:
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    return
                if not stop_input.is_set():
                    stop_input.set()
                    try:
                        store.request_stop(meeting["id"])
                    except MeetingError:
                        pass

            input_thread = __import__("threading").Thread(
                target=request_stop_on_enter, daemon=True
            )
            input_thread.start()
            try:
                record_meeting(store, meeting["id"], source_for(device, settings), settings)
            except KeyboardInterrupt:
                # record_meeting normally handles this; retain the same durable
                # stop protocol if an input backend interrupts before its loop.
                store.update_meeting(
                    meeting["id"], stop_requested=1, status="stopping", ended_at=__import__("meeting_mvp.store", fromlist=["utc_now"]).utc_now()
                )
            result = process_meeting(store, meeting["id"], settings)
            final = store.get_meeting(meeting["id"]) or {}
            status = final.get("status", result.get("status", "failed"))
            if status == "completed":
                print(f"completed {meeting['id']}")
            else:
                print(f"{status} {meeting['id']} error={final.get('error') or result.get('reason') or 'processing ended without a recorded error'}")
                if status in {"partial", "failed", "interrupted"}:
                    print(f"Next step: fix the audio/ASR/LLM issue, then run `python -m meeting_mvp retry {meeting['id']}`")
            return 0
        if args.command == "stop":
            store.request_stop(args.id)
            print(f"stop requested {args.id}")
            return 0
        if args.command == "status":
            rows = [store.get_meeting(args.id)] if args.id else store.list_meetings()
            print(json.dumps([row for row in rows if row], ensure_ascii=False, indent=2))
            return 0
        if args.command == "list":
            for item in store.list_meetings(args.status):
                print(f"{item['id']}\t{item['status']}\t{item['started_at']}\t{item['title'] or '未命名'}\t{item['device_name'] or '未选择'}")
            return 0
        if args.command == "show":
            meeting = store.get_meeting(args.id)
            if not meeting:
                raise MeetingError("Meeting not found", "MEETING_NOT_FOUND")
            result = {"meeting": meeting, "summary": store.summary(args.id)}
            if args.transcript:
                result["transcript"] = store.transcripts(args.id)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "retry":
            meeting = store.get_meeting(args.id)
            if not meeting:
                raise MeetingError(
                    f"Meeting not found in data directory {store.root}. "
                    "Run `python -m meeting_mvp list` with this configuration, "
                    "or set MEETING_DATA_DIR to the directory containing meetings.db.",
                    "MEETING_NOT_FOUND",
                )
            jobs = store.jobs(args.id)
            chunks = store.chunks(args.id)
            transcription_needed = bool(chunks) and (
                not jobs
                or any(
                    job["kind"] == "transcribe" and job["status"] != "completed"
                    for job in jobs
                )
            )
            summary_job = next((job for job in jobs if job["kind"] == "summarize"), None)
            summary_needed = bool(chunks) and (
                summary_job is None or summary_job["status"] != "completed"
            )
            settings = _confirm_remote(
                settings,
                args.yes,
                need_asr=transcription_needed,
                need_llm=summary_needed,
            )
            store.retry_jobs(args.id)
            result = process_meeting(store, args.id, settings)
            final = store.get_meeting(args.id) or {}
            status = final.get("status", result.get("status", "failed"))
            if status == "completed":
                print(f"retry completed {args.id}")
            else:
                print(f"retry {status} {args.id} error={final.get('error') or result.get('reason') or 'processing ended without a recorded error'}")
                print(f"Next step: fix the reported issue, then run `python -m meeting_mvp retry {args.id}`")
            return 0
        if args.command == "export":
            print(export_meeting(store, args.id, args.format, args.output))
            return 0
        if args.command == "delete":
            if args.all and not args.confirm:
                raise MeetingError("Deleting all meeting data requires --confirm", "CONFIRM_REQUIRED")
            store.delete_meeting(args.id, audio_only=args.audio_only)
            print(f"deleted {'audio' if args.audio_only else 'meeting'} {args.id}")
            return 0
        raise MeetingError("Unknown command", "UNKNOWN_COMMAND")
    finally:
        store.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeetingError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2)
