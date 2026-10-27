from __future__ import annotations

import hashlib
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from meeting_mvp.audio import write_frames
from meeting_mvp.config import Settings
from meeting_mvp.errors import MeetingError
from meeting_mvp.jobs import run_with_retry
from meeting_mvp.llm import LocalLLM, llm_endpoint
from meeting_mvp.__main__ import main
from meeting_mvp.recovery import recover
from meeting_mvp.service import process_meeting
from meeting_mvp.store import Store
from meeting_mvp.summary import SummaryEngine, summarize_meeting, validate_summary
from meeting_mvp.transcription import LlamaASRTranscriber, parse_llama_response


class FakeTranscriber:
    backend = "fake"
    model = "test"

    def transcribe(self, path: Path) -> str:
        return path.name


class EvidenceEngine(SummaryEngine):
    def __init__(self):
        super().__init__(lambda _prompt, _purpose: {}, "fake")

    def generate(self, segments):
        item = segments[0]
        payload = {
            "metadata": {"source": "transcript", "version": 1},
            "summary": "测试纪要",
            "topics": ["测试"],
            "decisions": [{"decision": "保留本地数据", "evidence_segment_ids": [item["id"]], "start_ms": item["start_ms"], "end_ms": item["end_ms"]}],
            "action_items": [{"action": "检查数据目录", "owner": "未提及", "due_date": "未提及", "status": "未提及", "evidence_segment_ids": [item["id"]], "start_ms": item["start_ms"], "end_ms": item["end_ms"]}],
            "unresolved_questions": ["未提及"],
            "inferences": ["未提及"],
            "missing_information": ["未提及"],
        }
        from meeting_mvp.markdown import render_summary
        return payload, render_summary(payload), hashlib.sha256(b"test").hexdigest()


class MeetingMvpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(data_dir=Path(self.temp.name), chunk_seconds=30)
        self.store = Store(self.settings.data_dir)
        self.meeting = self.store.create_meeting(title="Test", input_mode="synthetic")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_chunk_order_hash_and_transcription_persist(self):
        pcm = (100).to_bytes(2, "little", signed=True) * (16_000 * 31)
        write_frames(self.store, self.meeting["id"], [pcm], self.settings)
        chunks = self.store.chunks(self.meeting["id"])
        self.assertEqual([item["sequence"] for item in chunks], [1, 2])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in chunks))
        from meeting_mvp.transcription import transcribe_meeting
        succeeded, failed = transcribe_meeting(self.store, self.meeting["id"], FakeTranscriber())
        self.assertEqual((succeeded, failed), (2, 0))
        self.assertEqual([item["chunk_sequence"] for item in self.store.transcripts(self.meeting["id"])], [1, 2])

    def test_missing_whisper_marks_audio_meeting_partial_and_retryable(self):
        pcm = (100).to_bytes(2, "little", signed=True) * 16000
        write_frames(self.store, self.meeting["id"], [pcm], self.settings)
        self.store.update_meeting(self.meeting["id"], status="stopping")
        with patch.dict("sys.modules", {"whisper": None}):
            with self.assertRaises(MeetingError) as context:
                process_meeting(self.store, self.meeting["id"], self.settings)
        self.assertEqual(context.exception.code, "WHISPER_NOT_INSTALLED")
        self.assertIn("openai-whisper", str(context.exception))
        self.assertIn(f"retry {self.meeting['id']}", str(context.exception))
        self.assertEqual(self.store.get_meeting(self.meeting["id"])["status"], "partial")
        self.assertEqual(self.store.jobs(self.meeting["id"])[0]["status"], "failed")
        self.assertEqual(len(self.store.chunks(self.meeting["id"])), 1)

        payload = validate_summary({}, set())
        self.assertEqual(payload["summary"], ["未提及"])
        self.assertEqual(payload["decisions"], ["未提及"])
        self.assertEqual(payload["action_items"], ["未提及"])
        self.assertEqual(payload["unresolved_questions"], ["未提及"])

    def test_summary_list_fields_normalize_scalar_and_preserve_arrays(self):
        scalar = validate_summary(
            {
                "metadata": {"source": "transcript", "version": 1},
                "summary": "一句摘要",
                "topics": "发布计划",
                "decisions": ["未提及"],
                "action_items": None,
                "unresolved_questions": ["未提及"],
                "inferences": ["未提及"],
                "missing_information": "录音有缺口",
            },
            set(),
        )
        self.assertEqual(scalar["summary"], ["一句摘要"])
        self.assertEqual(scalar["missing_information"], ["录音有缺口"])
        self.assertEqual(scalar["action_items"], ["未提及"])

        normal = dict(scalar)
        normal["summary"] = ["第一点", "第二点"]
        normal["missing_information"] = ["未提及"]
        checked = validate_summary(normal, set())
        self.assertEqual(checked["summary"], ["第一点", "第二点"])
        self.assertEqual(checked["missing_information"], ["未提及"])

    def test_summary_tolerates_model_metadata_and_inference_drift(self):
        base = {
            "summary": ["测试"], "topics": [], "decisions": [], "action_items": [],
            "unresolved_questions": [], "missing_information": ["未提及"],
        }
        # Observed from DeepSeek-R1-Distill-Qwen-7B: string version, label-only inference object.
        payload = validate_summary(
            {**base, "metadata": {"source": "Transcript", "version": "1"},
             "inferences": [{"labels": ["模型推断", "需确认"], "evidence_segment_ids": []}]},
            set(),
        )
        self.assertEqual(payload["metadata"], {"source": "transcript", "version": 1})
        self.assertEqual(payload["inferences"], ["未提及"])
        payload = validate_summary(
            {**base, "metadata": {"source": "transcript", "version": 1},
             "inferences": ["可能是设备测试", {"text": "模型推断（需确认）：已标注"}]},
            set(),
        )
        self.assertEqual(payload["inferences"], ["模型推断（需确认）：可能是设备测试", "模型推断（需确认）：已标注"])
        with self.assertRaises(MeetingError):
            validate_summary({**base, "metadata": {"source": "audio", "version": "x"}}, set())

    def test_unbacked_model_facts_become_labelled_inferences(self):
        segments = [{"id": "s1", "start_ms": 0, "end_ms": 1000, "text": "Hello，这是啥测试？"}]
        raw = {
            "metadata": {"source": "transcript", "version": 1}, "summary": ["测试"],
            "decisions": [{"decision": "有依据", "evidence_segment_ids": ["s1"], "start_ms": 0, "end_ms": 500},
                          {"decision": "证据越界", "evidence_segment_ids": ["other"], "start_ms": 0, "end_ms": 500}],
            "action_items": ["安排相关人员进行测试分析"],
            "inferences": ["参与者在询问测试的名称"],
        }
        payload, markdown, _hash = SummaryEngine(lambda _p, _u: raw).generate(segments)
        self.assertEqual([item["decision"] for item in payload["decisions"]], ["有依据"])
        self.assertEqual(payload["action_items"], ["未提及"])
        self.assertEqual(payload["inferences"], [
            "模型推断（需确认）：参与者在询问测试的名称",
            "模型推断（需确认）：（转写中找不到依据的决策）证据越界",
            "模型推断（需确认）：（转写中找不到依据的行动项）安排相关人员进行测试分析",
        ])
        self.assertIn("安排相关人员进行测试分析", markdown.split("## 模型推断（需确认）")[1])
        # The validator itself stays strict for direct callers.
        with self.assertRaises(MeetingError):
            validate_summary(raw, segments)

    def test_summary_object_item_is_retained_for_explicit_validation(self):
        with self.assertRaisesRegex(MeetingError, "evidence"):
            validate_summary(
                {
                    "metadata": {"source": "transcript", "version": 1},
                    "summary": {"text": "摘要对象"},
                    "decisions": [{"decision": "未经证实", "evidence_segment_ids": []}],
                },
                set(),
            )

        segments = parse_llama_response({"text": "备用文本", "segments": [{"start": 0.25, "end": 1.5, "text": "  片段  "}]})
        self.assertEqual(segments, [{"start_ms": 250, "end_ms": 1500, "text": "片段"}])
        self.assertEqual(parse_llama_response({"text": "完整转写"}), "完整转写")
        with self.assertRaisesRegex(MeetingError, "without usable text"):
            parse_llama_response({"text": "", "segments": []})

    def test_retry_reclaims_running_job_and_preserves_audio(self):
        pcm = (100).to_bytes(2, "little", signed=True) * 16000
        write_frames(self.store, self.meeting["id"], [pcm], self.settings)
        self.store.enqueue_transcription_jobs(self.meeting["id"])
        claimed = self.store.claim_job(self.meeting["id"], "transcribe")
        self.assertIsNotNone(claimed)
        self.assertEqual(self.store.jobs(self.meeting["id"])[0]["status"], "running")
        with self.assertRaises(MeetingError) as context:
            self.store.retry_jobs(self.meeting["id"])
        self.assertEqual(context.exception.code, "JOB_ACTIVE")
        with self.store._lock, self.store.db:
            self.store.db.execute(
                "UPDATE processing_jobs SET lease_until=? WHERE meeting_id=?",
                ("2000-01-01T00:00:00+00:00", self.meeting["id"]),
            )
        self.store.retry_jobs(self.meeting["id"])
        self.assertEqual(self.store.jobs(self.meeting["id"])[0]["status"], "queued")
        self.assertTrue((self.store.root / "meetings" / self.meeting["id"] / "audio").exists())

    def test_llama_http_200_json_is_parsed(self):
        settings = Settings(data_dir=Path(self.temp.name), stt_backend="llama-asr", asr_url="http://127.0.0.1:8080/v1")
        wav = Path(self.temp.name) / "chunk.wav"
        wav.write_bytes(b"RIFF")
        response = MagicMock(status_code=200, headers={"content-type": "application/json"})
        response.json.return_value = {"text": "完整转写", "segments": []}
        with patch("requests.post", return_value=response) as post:
            result = LlamaASRTranscriber(settings).transcribe(wav)
        self.assertEqual(result, "完整转写")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8080/v1/audio/transcriptions")

    def test_llama_timeout_is_bounded_and_retryable(self):
        settings = Settings(
            data_dir=Path(self.temp.name), stt_backend="llama-asr",
            asr_url="http://127.0.0.1:8080/v1", request_timeout=2,
        )
        wav = Path(self.temp.name) / "chunk.wav"
        wav.write_bytes(b"RIFF")
        import requests
        with patch("requests.post", side_effect=requests.Timeout("read timed out")) as post:
            with self.assertRaisesRegex(MeetingError, "timed out") as context:
                LlamaASRTranscriber(settings).transcribe(wav)
        self.assertEqual(context.exception.code, "ASR_TIMEOUT")
        self.assertEqual(post.call_args.kwargs["timeout"], (2, 2.0))

    def test_empty_asr_response_is_partial_and_retryable(self):
        pcm = (100).to_bytes(2, "little", signed=True) * 16000
        write_frames(self.store, self.meeting["id"], [pcm], self.settings)
        self.store.update_meeting(self.meeting["id"], status="stopping")

        class EmptyTranscriber:
            backend = "llama-asr"
            model = "test"
            def transcribe(self, _path):
                from meeting_mvp.errors import RecoverableJobError
                raise RecoverableJobError("ASR returned HTTP 200 without usable text or segments", "ASR_EMPTY_RESPONSE")

        with patch("meeting_mvp.jobs.time.sleep"):
            result = process_meeting(self.store, self.meeting["id"], self.settings, EmptyTranscriber())
        meeting = self.store.get_meeting(self.meeting["id"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(meeting["status"], "partial")
        self.assertIn("ASR returned HTTP 200 without usable text", meeting["error"])
        self.assertEqual(self.store.jobs(self.meeting["id"])[0]["status"], "failed")
        self.assertTrue((self.store.root / "meetings" / self.meeting["id"] / "audio").exists())

        with self.assertRaises(MeetingError):
            validate_summary({"decisions": [{"decision": "x", "evidence_segment_ids": ["outside"]}]}, {"known"})
        with self.assertRaises(MeetingError):
            validate_summary({"metadata": {"source": "transcript", "version": 1}, "decisions": ["未经证实的决定"]}, set())

    def test_llm_endpoint_resolves_native_and_openai_paths(self):
        native = Settings(data_dir=Path(self.temp.name), llm_url="http://127.0.0.1:11434/v1")
        self.assertEqual(llm_endpoint(native), "http://127.0.0.1:11434/api/chat")
        compatible = Settings(data_dir=Path(self.temp.name), llm_backend="openai", llm_url="http://127.0.0.1:8080/v1")
        self.assertEqual(llm_endpoint(compatible), "http://127.0.0.1:8080/v1/chat/completions")

    def test_retry_checks_meeting_before_remote_confirmation(self):
        missing_id = "f19315a3-5872-4cfd-bb18-baab732079a1"
        with patch("meeting_mvp.__main__.Settings.from_env", return_value=self.settings), \
             patch("meeting_mvp.__main__._confirm_remote") as confirm:
            with self.assertRaises(MeetingError) as context:
                main(["retry", missing_id])
        self.assertEqual(context.exception.code, "MEETING_NOT_FOUND")
        self.assertIn(str(self.settings.data_dir.resolve()), str(context.exception))
        self.assertIn("MEETING_DATA_DIR", str(context.exception))
        confirm.assert_not_called()

    def test_audio_only_delete_preserves_meeting(self):
        path = self.store.root / "meetings" / self.meeting["id"] / "audio" / "chunk.wav"
        path.write_bytes(b"audio")
        self.store.add_audio_chunk(self.meeting["id"], 1, path.relative_to(self.store.root).as_posix(), 0, 1, 5, "hash")
        self.store.delete_audio(self.meeting["id"])
        self.assertIsNotNone(self.store.get_meeting(self.meeting["id"]))
        self.assertFalse(path.exists())

    def test_jobs_are_durable_and_summary_waits_for_transcription(self):
        pcm = (100).to_bytes(2, "little", signed=True) * 16000
        write_frames(self.store, self.meeting["id"], [pcm, pcm], self.settings)
        self.store.update_meeting(self.meeting["id"], status="stopping")
        process_meeting(self.store, self.meeting["id"], self.settings, FakeTranscriber(), EvidenceEngine())
        jobs = self.store.jobs(self.meeting["id"])
        self.assertEqual([job["kind"] for job in jobs], ["transcribe", "summarize"])
        self.assertTrue(all(job["status"] == "completed" for job in jobs))
        self.assertEqual(self.store.get_meeting(self.meeting["id"])["status"], "completed")

    def test_recovery_registers_closed_wav_left_before_database_insert(self):
        path = self.store.root / "meetings" / self.meeting["id"] / "audio" / "chunk-000001.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes((100).to_bytes(2, "little", signed=True) * 160)
        recover(self.store, stale_after_seconds=999)
        chunk = self.store.chunks(self.meeting["id"])[0]
        self.assertEqual(chunk["sha256"], self.store.file_hash(path))
        self.assertEqual(chunk["size_bytes"], path.stat().st_size)

    def test_timed_gap_is_persisted_as_partial_recording_evidence(self):
        pcm = (100).to_bytes(2, "little", signed=True) * 160
        write_frames(self.store, self.meeting["id"], [(pcm, 0.0), (pcm, 2.0)], self.settings)
        meeting = self.store.get_meeting(self.meeting["id"])
        self.assertNotEqual(meeting["missing_ranges"], "[]")
        self.assertEqual(meeting["dropped_blocks"], 0)

    def test_active_meeting_cannot_be_deleted(self):
        with self.assertRaises(MeetingError) as context:
            self.store.delete_meeting(self.meeting["id"])
        self.assertEqual(context.exception.code, "MEETING_ACTIVE")

    def test_summary_failure_is_saved_for_recovery(self):
        pcm = (100).to_bytes(2, "little", signed=True) * 16000
        write_frames(self.store, self.meeting["id"], [pcm], self.settings)
        from meeting_mvp.transcription import transcribe_meeting
        transcribe_meeting(self.store, self.meeting["id"], FakeTranscriber())
        bad = SummaryEngine(lambda _prompt, _purpose: (_ for _ in ()).throw(ValueError("bad JSON")), "fake")
        with self.assertRaises(ValueError):
            summarize_meeting(self.store, self.meeting["id"], bad)
        self.assertEqual(self.store.summary(self.meeting["id"])["status"], "failed")

    def test_long_summary_uses_bounded_map_reduce(self):
        segments = [{"id": "seg-1", "start_ms": 0, "end_ms": 1000, "text": "长" * 35000}]
        purposes = []
        def respond(_prompt, purpose):
            purposes.append(purpose)
            if purpose != "reduce":
                return {}
            return {"metadata": {"source": "transcript", "version": 1}, "summary": "完成", "topics": ["未提及"], "decisions": ["未提及"], "action_items": ["未提及"], "unresolved_questions": ["未提及"], "inferences": ["未提及"], "missing_information": ["未提及"]}
        payload, _markdown, _source_hash = SummaryEngine(respond).generate(segments)
        self.assertEqual(payload["summary"], ["完成"])
        self.assertIn("map", purposes)
        self.assertIn("reduce", purposes)


if __name__ == "__main__":
    unittest.main()
