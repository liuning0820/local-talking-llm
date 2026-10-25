# meeting_mvp ASR 文字稿位置调查

## 摘要答案

会议 `e6818839-fa72-4b33-a505-5c2aa90282f1` 的当前实际数据根目录是：

```text
C:\Users\toby0\AppData\Local\local-talking-llm
```

数据库路径是：

```text
C:\Users\toby0\AppData\Local\local-talking-llm\meetings.db
```

ASR 转写**不保存为独立的 transcript 文件**；它保存于上述 SQLite 数据库的
`transcript_segments` 表。该会议当前数据库中有 1 条转写记录，状态为
`succeeded`。会议目录下当前只发现原始音频：

```text
C:\Users\toby0\AppData\Local\local-talking-llm\meetings\e6818839-fa72-4b33-a505-5c2aa90282f1\audio\chunk-000001.wav
```

查看 ASR 文字稿：

```powershell
python -m meeting_mvp show e6818839-fa72-4b33-a505-5c2aa90282f1 --transcript
```

该命令输出 JSON，其中 `transcript` 数组就是 ASR 结果。导出包含 ASR 文字稿的 JSON：

```powershell
python -m meeting_mvp export e6818839-fa72-4b33-a505-5c2aa90282f1 --format json
```

默认文件为：

```text
C:\Users\toby0\AppData\Local\local-talking-llm\meetings\e6818839-fa72-4b33-a505-5c2aa90282f1\exports\meeting.json
```

如需直接指定导出位置：

```powershell
python -m meeting_mvp export e6818839-fa72-4b33-a505-5c2aa90282f1 --format json --output 'D:\Exports\e6818839.json'
```

注意：`--format markdown` 只导出摘要/会议纪要 Markdown，不包含原始 ASR
片段；要导出文字稿本身应使用 `--format json`，或使用
`show --transcript` 查看。

## 证据

### 数据库路径与数据目录

- `meeting_mvp/config.py` 的 `default_data_dir()`：优先使用
  `MEETING_DATA_DIR`；没有有效覆盖时，Windows 使用
  `%LOCALAPPDATA%\\local-talking-llm`。本次只读检查未读取仓库 `.env` 内容，
  仅检查当前安全的路径配置结果；当前 `MEETING_DATA_DIR` 没有有效覆盖，实际
  解析为 `C:\Users\toby0\AppData\Local\local-talking-llm`。
- `meeting_mvp/store.py` 的 `Store.__init__()`：将 `data_dir` 解析为
  `self.root`，并打开 `self.root / "meetings.db"`；同时创建 `meetings` 和
  `logs` 目录。
- `meeting_mvp/service.py` 的 `create_store()`：CLI 创建 `Store` 前调用
  `Settings.ensure_dirs()`，所以所有 `show`、`export` 等命令都针对同一数据根目录。
- 只读 filesystem 检查确认上述 `meetings.db` 存在，目标会议目录存在。

### 转写表与字段

- `meeting_mvp/store.py` 的 `_migrate()` 创建表
  `transcript_segments`，字段为：
  - `id`：转写片段 UUID 主键；
  - `meeting_id`：所属会议 ID；
  - `chunk_sequence`：所属音频分片序号；
  - `start_ms`、`end_ms`：片段时间范围；
  - `text`：ASR 文字内容；
  - `backend`、`model`：ASR 后端与模型；
  - `attempts`：尝试次数；
  - `status`：`succeeded` 或 `failed`；
  - `error`：失败信息。
  该表按 `(meeting_id, start_ms, chunk_sequence)` 建有排序索引。
- `meeting_mvp/store.py` 的 `add_transcript()`、`replace_transcripts()`：把 ASR
  结果持久化到 `transcript_segments`；一个 WAV 分片可以产生多个带时间戳的片段。
- `meeting_mvp/store.py` 的 `transcripts()`：按时间顺序查询指定会议的全部片段。
- `meeting_mvp/transcription.py` 的 `transcribe_meeting()` 与
  `meeting_mvp/service.py` 的 `_process_transcription_jobs()`：ASR 对每个关闭的
  WAV 分片执行，并通过 `replace_transcripts()` 写入数据库。Whisper 返回的
  `segments` 或 Llama ASR 返回的 `segments` 会转换为毫秒时间戳；只有纯文本时则
  使用整个音频分片的时间范围。

### 目标会议的实际只读核对

对当前数据根目录的 `meetings.db` 使用 SQLite 只读连接核对得到：

- 会议存在：ID 为 `e6818839-fa72-4b33-a505-5c2aa90282f1`，标题为
  `Microphone test 3`，当前状态为 `failed`。
- `transcript_segments` 中该会议共有 1 条记录，`succeeded` 为 1，`failed` 为 0。
- 数据库包含 `transcript_segments` 和 `summary_versions` 表。
- 目标会议物理目录当前只有 `audio\chunk-000001.wav`，未发现
  `transcript.txt`、`transcript.json` 或其他独立 transcript 文件。

会议当前为 `failed` 不代表 ASR 文字稿不存在：原始请求中的失败是摘要校验错误
(`Summary field 'missing_information' must be a list`)，而转写记录已成功保存；应先
用 `show --transcript` 或 JSON 导出验证文字稿。

### 查看与导出 CLI

- `meeting_mvp/__main__.py` 的 `parser()`：`show` 支持 `--transcript`；`export`
  要求 `--format markdown` 或 `--format json`，并支持 `--output`。
- `meeting_mvp/__main__.py` 的 `main()`：`show` 先返回 `meeting` 和最新
  `summary`，传入 `--transcript` 时额外加入
  `store.transcripts(args.id)` 的结果；这就是读取数据库文字稿的 CLI 方法。
- `meeting_mvp/export.py` 的 `export_meeting()`：构造
  `{"meeting": ..., "transcript": ..., "summary": ...}`；JSON 导出包含完整
  transcript 数组。Markdown 分支只写 `summary["markdown"]`，没有 transcript。
- `README.md` 的 “Meeting MVP” 示例明确列出
  `python -m meeting_mvp show MEETING_ID --transcript` 和
  `python -m meeting_mvp export MEETING_ID --format markdown`。
- `docs/meeting-mvp.md` 的 “Record and review a meeting” 同时列出 show、带
  transcript 的 show，以及 markdown/json 两种导出；“Data, deletion, and
  recovery” 说明数据根目录含 `meetings.db`、`meetings/<id>/audio/`、
  `meetings/<id>/exports/` 和 `logs/`。

## 结论与建议

1. 给 LLM 生成 summary 的输入文字稿来自 SQLite 的
   `transcript_segments.text` 字段，不是某个 transcript 文本文件；摘要结果则
   保存于 `summary_versions`（`json_text` 和 `markdown` 字段）。
2. 对目标会议，建议先运行 `show ... --transcript`，确认 `transcript` 数组的
   `text`、时间戳、backend/model 和 status；需要留存时运行 JSON export。
3. 若只运行 markdown export，得到的是纪要，不是完整 ASR 文字稿；这是当前实现的
   预期行为，不应把 `meeting.md` 当作原始转写文件。
4. 当前 CLI 没有直接打印数据根目录的命令；如用户经常定位文件，建议未来增加
   `status` 或 `show` 中的 data-root/meeting-path 字段，或增加独立的路径诊断命令。
   本次为只读调查，未实施任何修改。
