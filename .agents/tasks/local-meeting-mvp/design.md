# 本地会议记录与纪要 MVP 技术设计

## 概述

本设计把 `local-talking-llm` 从“短语音问答助手”扩展为本地单用户会议记录工具。用户在命令行中创建会议、选择音频输入、开始和停止录音；录音按块可靠落盘，停止后由可恢复的任务队列逐块转写，再通过本地 Ollama 或 OpenAI 兼容的本地服务生成带证据标记的结构化纪要。历史会议、原始转写和纪要都从本地 SQLite 和会议目录读取，不需要机器人加入会议，也不引入复杂前端。

本文件是基于当前工作区实际代码的实现设计，不实现代码。当前没有发现 `.agents/tasks/local-meeting-mvp/design-verdict.json`，因此没有可回填的审查 finding；设计完成后由后续审查步骤生成或更新该文件，不能由本设计自行伪造 `APPROVED`。

## 现状事实

仓库目前不是会议应用，而是两个入口脚本组成的语音助手原型。`app_mac.py` 使用
`openai-whisper`、`sounddevice`、`langchain_ollama.OllamaLLM` 和 Chatterbox；
`app_win.py` 使用 Whisper 或 `stt.py` 中的 llama-ASR HTTP 适配器，并通过
`langchain_openai.ChatOpenAI` 连接 OpenAI 兼容服务。两者都把一次录音先积存在内存
`Queue`，停止后整体转写，再生成对话回复；它们没有会议、录音文件、转写、纪要、SQLite
或历史数据模型。

`stt.py` 已有 16 kHz、单声道、16-bit WAV 编码和 Whisper/llama-ASR 调用函数。
`query_local_llm.py` 和 `actor-critic-sample.py` 证明仓库可以访问 OpenAI 兼容服务，
但它们不是会议流水线。`tts.py` 和两个入口的 TTS、唤醒词、声音克隆、情绪参数均不属于
本 MVP，新的会议入口不得初始化这些重量级组件。

`pyproject.toml` 要求 Python `>=3.11`，但依赖均为开放下界；`requirements.txt` 是另一份
不完全一致的锁定风格清单，包含代码直接使用而未在 `pyproject.toml` 中声明的
`requests`、`soundfile` 和 `langchain-openai`。`uv.lock` 当前解析到例如
`chatterbox-tts 0.1.1`、`langchain-ollama 0.3.3`、`openai-whisper 20240930`、
`sounddevice 0.5.2`、`soundfile 0.13.1` 和 `requests 2.32.3`，但它没有把所有代码的
直接导入都表达为项目直接依赖。

当前录音实现只验证 `max_input_channels > 0` 的普通输入。实际安装的
`sounddevice` 的 `WasapiSettings` 只有 `exclusive`、`auto_convert` 和
`explicit_sample_format`，没有 loopback 参数；所以不能把现有 `RawInputStream` 静默描述为
Windows 系统音频采集。macOS 也没有现有系统音频捕获实现。README 仍有不存在的
`app.py` 示例和未实现参数，不能作为能力证明。

仓库没有项目自己的自动化测试配置。`stt-test.py` 需要 `sample-zh.wav` 和外部
llama-server，只能作为人工服务冒烟脚本。当前 `.env` 被忽略，但其中的服务地址可能是
远程地址；新会议入口不能因为已有 `.env` 就默认向远程服务发送会议内容。

## 目标、非目标和假设

MVP 必须实现不限产品数量的本地会议创建和管理、开始/停止录音、录音状态和时间、可选
标题、输入设备选择、Windows 麦克风与 WASAPI loopback 输入、macOS 麦克风以及用户明确
选择的系统音频输入、长会议分段落盘、停止后的后台逐块转写、失败重试和重启恢复、结构化
可靠纪要、历史查看以及 Markdown/JSON 导出。录音默认保留在本机，且在 UI 中始终显示是否
仍在录音。

“会议数量不限”表示应用不设置人为数量上限；实际限制是磁盘空间、SQLite 文件大小和
模型处理能力。MVP 不承诺说话人识别、实时字幕、自动 VAD 停止、降噪、回声消除、跨设备
自动混音、多人账户、云端同步、会议平台 API、机器人、日历集成、TTS、聊天回复或全文
语义搜索。一个会议选择一个实际音频输入设备；若用户需要麦克风和系统声道混合，MVP 要求
选择操作系统已经提供的混合输入、Stereo Mix、聚合设备或虚拟音频设备，不自行假设两个
设备时钟可以无损同步。

设计假设是单机单用户、一次只有一个 `recording` 会议、Python 3.11+、会议数据目录由
当前用户读写。默认 ASR 是本地 Whisper，默认 LLM 是本机 Ollama；兼容的本地
OpenAI API 可选。任何非本机 HTTP 地址都必须显式开启网络开关并在启动时提示。

## Summary

本 MVP 为单机单用户提供“不加入会议、直接采集本机音频”的会议记录闭环：选择输入设备，开始和停止录音，按块保留音频，自动逐块转写，生成可核对事实来源的结构化纪要，并从历史中查看或导出。实现优先复用现有 Whisper、`sounddevice`、本地 Ollama/兼容 LLM 和 Rich CLI；TTS 与旧语音聊天仍是独立 legacy 功能。

## Functional Requirements

功能要求如下，详细的接口、数据字段、状态和错误处理见后文。所有功能都必须能在 CLI 中观察到状态或错误，而不是只写日志。

- 必须创建和管理任意数量的本地会议，支持可选标题、开始时间、结束时间、输入模式和设备信息。
- 必须提供设备枚举、显式设备选择、麦克风录音、Windows WASAPI loopback，以及 macOS 对用户已安装虚拟/聚合系统音频输入的选择和错误提示。
- 必须按合理时长分块落盘，保存中间状态，在进程异常、设备异常、磁盘异常或模型失败时尽可能保留已完成块和已完成转写。
- 必须在停止后自动排队全部已关闭音频块，保存顺序、时间范围、后端、模型和失败状态，并支持重试和重启恢复。
- 必须生成包含摘要、讨论主题、决策、行动项、负责人、截止日期和未解决问题的结构化纪要，并分开表达原文事实、模型推断和待确认内容。
- 必须提供历史列表、会议详情、完整转写查看、状态查看、失败重试、音频/会议删除和 Markdown/JSON 导出。
- 默认不得向非本机地址发送音频或转写；启用联网后必须显式配置、确认并记录所使用的后端元数据。

## Non-Functional Requirements

这些质量要求直接影响 MVP 是否可日常使用，因此由架构和验收共同约束：

- **可靠性：** 录音不能等待 ASR 或 LLM；已关闭音频块必须在崩溃后可恢复，状态转换和任务领取必须具有事务与租约语义。
- **可核对性：** 纪要事实、决策、行动项和日期必须引用本会议转写段；缺少证据、负责人或日期时不得臆造。
- **隐私：** 默认本地处理，日志不写会议正文、音频或 token，音频、转写、纪要必须可分别保留或删除。
- **可移植性：** Python 3.11+ 下提供 Windows 和 macOS 的明确支持边界；平台不具备系统音频能力时必须在枚举和文档中显式说明。
- **可测试性：** 核心 store、写盘、状态机、时间换算、证据校验和配置校验必须可用标准库 fake source/model 在无硬件、无模型、无网络环境验证。

## Acceptance Criteria

验收必须满足以下可测试条件；真实设备和本地模型相关条目同时受后文 smoke 矩阵约束：

1. 在临时 `MEETING_DATA_DIR` 中运行 synthetic self-check，能创建会议、写入至少一个有效 WAV 块、完成 fake 转写、生成 fake 纪要、列出历史并导出 Markdown/JSON，且不连接网络。
2. `devices` 能列出普通输入设备及设备 ID、名称、通道和 Host API；Windows 显示 loopback 能力，macOS 没有虚拟系统输入时返回 `MAC_SYSTEM_AUDIO_REQUIRES_VIRTUAL_DEVICE`，不切换为麦克风。
3. 使用 fake source 录制超过一个块长后，数据库中的块序号连续、每个关闭块可由 `wave` 打开、文件 hash 与数据库一致，内存不会保存整场会议。
4. 在录音继续期间人为延迟 fake 转写，音频仍持续落盘；`status` 能显示录音状态、已关闭块数、写入字节、丢块数和转写进度。
5. 通过 `stop`、Enter 或 Ctrl+C 停止后，当前块安全关闭并进入转写队列；重复 stop 不会生成第二组块或任务。
6. 模拟进程崩溃和过期 job 后，下一次启动只恢复完整 WAV，标记会议 `interrupted`，把过期任务重新排队，并保留已经成功的转写。
7. ASR 临时超时或 5xx 最多按 5 秒、30 秒、300 秒重试三次；4xx、模型不存在或损坏 WAV 被标为不可恢复，`retry` 能单独重排可重试项。
8. 短会议和超过输入预算的长会议分别走直接总结和 map/reduce；长会议不会把超出预算的完整转写一次性发送给 LLM。
9. 纪要 schema 至少产生摘要、主题、决策、行动项、负责人、截止日期和未解决问题；每个事实性条目都有属于本会议的证据段 ID，缺失负责人/日期为 `null` 或“未明确”。
10. 模型输出的推断只出现在 `inferences` 或 `needs_review`，并带“模型推断/需确认”标识；坏 JSON、无效证据或越权证据不会成为 `ready` 纪要。
11. `list`、`show`、`status`、`retry`、`export` 和删除命令在模型服务不可用时仍能访问历史和已保存原文；删除音频不会删除转写/纪要，删除全部会议会清理数据库记录和会议目录。
12. `MEETING_ALLOW_NETWORK=false` 时，配置远程 ASR/LLM 会在发送任何音频或转写前失败；开启后 CLI 显示目标主机并要求确认，token 不进入 SQLite、日志或导出。
13. Python 标准库 `unittest` 覆盖迁移幂等、路径安全、块恢复、任务租约、时间换算、重试分类和纪要证据校验；平台 smoke 另外验证真实麦克风、Windows loopback 和 macOS 虚拟输入。

## Out of Scope

以下内容不阻塞本 MVP，也不应通过增加无关依赖混入实现：说话人分离和自动姓名识别、实时字幕、可靠自动 VAD 停止、降噪/回声消除、跨设备自动混音、会议平台 API 或机器人、云端同步、账号权限、日历、全文语义搜索、纪要协作编辑、TTS/聊天回复、移动端、自动安装虚拟音频驱动，以及保证任意 Windows 声卡或 macOS 无虚拟设备都能获取系统音频。

## 功能要求（详细约束）

功能按以下边界实现。每项都必须能在 CLI 中观察到状态或错误，而不是只写日志。

1. 用户运行 `python -m meeting_mvp start [--title ...] --source ...` 创建会议。
   标题可省略，最长 200 个 Unicode 字符；缺省标题为本地时区下的开始时间。应用生成
   UUID，记录 UTC 开始时间、平台、输入模式、设备标识和设备名称。
2. `devices` 命令枚举当前平台可用的普通输入、Windows loopback 和其他可读输入，显示
   稳定 ID、名称、Host API、输入通道、默认采样率以及是否为 loopback。开始录音必须使用
   明确的设备配置或可验证的系统默认设备，不允许在所选设备不可用时静默换设备。
3. `start` 只负责录音控制和落盘，不加载 TTS、不调用 LLM，也不因转写变慢而阻塞音频
   回调。`status` 显示会议状态、录音时长、已关闭块数、写入字节、丢块计数和后处理进度。
   `stop <meeting-id>` 设置持久化停止请求；录音进程在最多 1 秒内发现它并安全关闭当前块。
   Ctrl+C 走同一停止路径，而不是直接退出。
4. 音频以 16 kHz、单声道、signed 16-bit PCM WAV 块写入会议目录，默认每块 300 秒，
   以时间和序号命名。写盘使用有限内存队列和独立 writer；块关闭后原子更新数据库，
   不把整场会议留在内存。磁盘不足、设备断开、回调溢出或写入失败时保留已经关闭的块，
   标记会议为 `interrupted` 或 `partial`，并给出可操作错误。
5. 停止后，所有已关闭音频块按序进入转写队列。每个块和每个转写任务有状态、尝试次数、
   错误和时间范围；成功结果保存原文、模型、后端、语言和尽可能细的 ASR 时间戳。失败块
   可以单独重试，重启应用会接管过期任务。
6. 纪要只在转写任务达到终态后生成。短会议直接输入完整转写；长会议先按字符预算做
   分块摘要，再对有序分块摘要做总结合并。纪要至少包含摘要、讨论主题、决策、行动项、
   负责人、截止日期和未解决问题，并保留转写证据段 ID 和相对时间。
7. 所有模型声称的事实、决策、行动项和截止日期都必须有 `evidence_segment_ids`。
   没有明确负责人或截止日期时保存 `null` 或“未明确”，不能让模型补齐。模型推断必须
   单独放入 `inferences`，UI 和导出明确标为“模型推断/需确认”；不能把推断渲染到事实列表。
8. `list` 按开始时间倒序显示历史会议和状态；`show <id>` 显示纪要、状态、失败提示和
   按时间排序的转写；`retry <id>` 只重试可恢复失败的任务；`export <id> --format
   markdown|json` 导出到用户指定目录。删除操作必须有明确确认，并分别说明删除音频、转写和
   纪要的影响。
9. 默认 `MEETING_KEEP_AUDIO=true`，因为音频是转写重试和事实核对的证据。用户可以在
   删除操作中只删音频而保留转写/纪要，也可以删除整个会议；删除必须清理数据库记录和
   数据目录中的文件，不接受目录外路径。

## 技术栈和文件边界

技术栈锁定为 Python 3.11+、标准库 `sqlite3`/`wave`/`threading`/`queue`/`argparse`、
现有 `numpy`、`sounddevice`、`openai-whisper`、`requests`、`rich`，以及 Ollama 或
OpenAI 兼容 HTTP API。命令行和状态输出使用现有 `rich`，不引入 Flask、FastAPI、React、
Streamlit 或桌面 GUI。SQLite 采用 WAL、外键和事务；会议目录与数据库分离但由同一数据根
目录管理。

为提供 Windows WASAPI loopback，新增唯一一个运行时外部依赖
`SoundCard==0.4.5`，理由是当前 `sounddevice` 版本没有 loopback flag，而 SoundCard 提供
`all_microphones(include_loopback=True)` 和 WASAPI loopback 设备读取。该依赖必须在
`pyproject.toml`、`requirements.txt` 和 `uv.lock` 中以同一明确版本记录；如果目标平台验证
证明该版本不可用，实施者必须先解决适配，不得退回为“默认输入就是系统声音”。测试不新增
pytest：可靠性测试使用 Python 标准库 `unittest`，避免为测试引入未必要依赖。

新增 `meeting_mvp/` 包，文件职责固定如下：

- `meeting_mvp/__main__.py` 解析子命令、打印 Rich UI，不承载录音或模型逻辑。
- `meeting_mvp/config.py` 解析环境变量、路径和数值边界，默认使用操作系统用户数据目录。
- `meeting_mvp/store.py` 管理 SQLite schema、迁移、事务和查询；所有状态变化由此层提交。
- `meeting_mvp/audio.py` 定义 `AudioSource`、`DeviceInfo`、`AudioBlock` 和 `Recorder`，实现
  有限队列、块轮换、原子 WAV 写入、心跳和停止协议。
- `meeting_mvp/audio_windows.py` 用 `sounddevice` 实现麦克风，用 `SoundCard` 实现
  WASAPI loopback，并把设备统一为 `AudioSource`。
- `meeting_mvp/audio_macos.py` 用 `sounddevice` 读取 CoreAudio 输入设备。系统声音只接受
  用户已经安装并选中的虚拟/聚合输入，不实现未经授权的系统级捕获。
- `meeting_mvp/jobs.py` 实现任务租约、重试、过期任务回收和 `recording` 到处理终态的状态机。
- `meeting_mvp/transcription.py` 复用并扩展 `stt.py` 的音频编码和 Whisper/llama-ASR 思路，
  提供统一的 `Transcriber` 接口。
- `meeting_mvp/summary.py` 实现分层摘要、JSON schema 校验、证据引用校验和 Markdown 渲染。
- `meeting_mvp/llm.py` 使用标准库 HTTP 客户端调用 Ollama `/api/chat` 或 OpenAI 兼容
  `/v1/chat/completions`，不复用现有的对话历史链；请求必须是一次会议一个无历史的总结任务。
- `meeting_mvp/export.py` 负责脱离 UI 的 Markdown/JSON 输出和路径安全。

`stt.py` 只增加可复用的带时间戳结果类型或适配器辅助函数；不把会议状态写入现有短问答
入口。`app_mac.py`、`app_win.py`、`tts.py`、`query_local_llm.py` 和
`actor-critic-sample.py` 保持旧功能，README 中明确它们是 legacy voice-assistant 或
诊断脚本。更新 `pyproject.toml`、`requirements.txt`、`uv.lock`、`.gitignore`、
`README.md`，新增 `.env.example` 和 `docs/meeting-mvp.md`。

## 数据目录、数据模型和迁移

默认数据根目录为 Windows 的 `%LOCALAPPDATA%/local-talking-llm`、macOS 的
`~/Library/Application Support/local-talking-llm`；`MEETING_DATA_DIR` 可以覆盖。目录下
固定有 `meetings.db`、`meetings/<uuid>/audio/`、`meetings/<uuid>/exports/` 和 `logs/`。
数据库只保存相对路径，启动时验证 realpath 仍在数据根目录内。音频、转写和纪要默认不进
Git；`.gitignore` 增加该数据根目录被放在仓库内时的忽略规则。

第一版 schema 通过 `PRAGMA user_version` 迁移。迁移在独占事务中按版本顺序运行，失败就
回滚并阻止应用继续写入；不通过 ORM 或隐式自动改表。数据库启用 `foreign_keys=ON`、
`journal_mode=WAL`、忙等待 5 秒。未来字段只能追加迁移，不重写已有会议。数据库损坏或
迁移失败是 fatal，UI 显示备份/恢复路径，日志只记录错误类型和会议 ID，不记录正文。

核心表如下，字段名是实现约束而不是建议：

- `meetings`：`id TEXT PRIMARY KEY`、`title TEXT NULL`、`status TEXT`、`started_at`
  和 `ended_at TEXT NULL`（UTC ISO 8601）、`source_mode`、`device_id`、`device_name`、
  `platform`、`sample_rate`、`channels`、`keep_audio INTEGER`、`stop_requested INTEGER`、
  `dropped_blocks INTEGER DEFAULT 0`、`bytes_written INTEGER DEFAULT 0`、`last_heartbeat TEXT`、
  `error_code`、`error_message`、`created_at`。状态和 `started_at <= ended_at` 由 store
  层的事务校验，标题长度由 config/store 双重校验；`stop` 只在事务中将
  `stop_requested` 设为 1，不直接终止录音线程。
- `audio_chunks`：`id INTEGER PRIMARY KEY`、`meeting_id`、`seq`、`relative_path`、
  `start_ms`、`end_ms`、`sample_rate`、`channels`、`byte_size`、`sha256`、`status`、
  `closed_at`、`error_message`。`(meeting_id, seq)` 唯一，只有 `closed` 块才可进入转写。
- `transcript_segments`：`id TEXT PRIMARY KEY`、`meeting_id`、`chunk_id`、`seq`、
  `start_ms`、`end_ms`、`text`、`status`、`backend`、`model`、`language`、
  `attempts`、`error_message`、`updated_at`。同一块重试用同一逻辑块标识，成功写入新
  `segment` 行时先删除该块旧的失败结果，不能重复拼接。
- `processing_jobs`：`id TEXT PRIMARY KEY`、`meeting_id`、`chunk_id NULL`、`kind`、
  `status`、`attempts`、`available_at`、`lease_until`、`last_error`、`created_at`、
  `updated_at`。`kind` 至少为 `transcribe`、`summarize_map`、`summarize_reduce`，并对
  `(kind, chunk_id)` 做唯一约束。
- `summary_versions`：`id TEXT PRIMARY KEY`、`meeting_id`、`status`、`model`、
  `prompt_version`、`source_digest`、`json_payload`、`markdown_payload`、`error_message`、
  `created_at`。只把通过 schema 和证据校验的结果标为 `ready`；失败的原始模型响应
  不进入正常纪要展示。
- `settings`：`key TEXT PRIMARY KEY`、`value_json TEXT`、`updated_at`，只存非秘密偏好，
  不存 API key。索引覆盖 `meetings(started_at)`、`meetings(status)`、
  `audio_chunks(meeting_id, seq)` 和 `transcript_segments(meeting_id, seq, start_ms)`。

## 音频设备抽象和平台边界

`AudioSource.enumerate()` 返回统一的 `DeviceInfo`，包含 `id`、`name`、`backend`、
`host_api`、`input_channels`、`default_samplerate`、`is_loopback` 和
`requires_virtual_device`。设备 ID 原样存储，设备名称只用于显示；开始时如果保存的 ID
不再出现，应用返回设备不可用错误并要求重新运行 `devices`，不能按名称模糊匹配或切到默认设备。

Windows 麦克风使用 `sounddevice.InputStream`/`RawInputStream`，显式传入用户选中的
设备 ID。Windows 系统音频使用 `SoundCard.all_microphones(include_loopback=True)` 枚举
WASAPI loopback，使用对应 loopback microphone 的 recorder 读取默认播放设备音频。若
SoundCard 未安装、没有 loopback、输出设备被独占或采样率不支持，错误码分别为
`LOOPBACK_BACKEND_MISSING`、`LOOPBACK_DEVICE_UNAVAILABLE`、`AUDIO_DEVICE_BUSY` 或
`AUDIO_FORMAT_UNSUPPORTED`，UI 显示选择麦克风、Windows Stereo Mix 或虚拟混音输入的
替代方法。loopback 模式不会伪装成麦克风模式。

macOS 麦克风使用 `sounddevice` 的 CoreAudio 输入设备，启动失败时明确提示在“系统设置 →
隐私与安全性 → 麦克风”允许当前 Python/终端。macOS 系统声音不能由当前普通
`sounddevice` 输入流直接获得；`--source system` 必须指定用户已安装并授权的
BlackHole、Loopback、Soundflower 或聚合音频设备。`devices` 会列出所有输入设备并标出
疑似虚拟设备，但最终由用户明确选择。找不到合适设备时返回
`MAC_SYSTEM_AUDIO_REQUIRES_VIRTUAL_DEVICE`，文档说明安装、路由和权限，不静默退化为
麦克风。MVP 不声称无额外设备即可捕获 macOS 系统音频。

设备自检只打开流并读取至少 500 ms，然后关闭，不创建会议。它检查通道、采样率、首块
是否有数据和权限。真实设备可用性取决于机器和 OS，不能由静态测试替代；`devices` 和
合成音频 self-check 则不需要硬件。

## 录音、分段和恢复状态机

创建会议时，store 先写入 `created`，设备自检成功后同一事务改为 `recording` 并写入
心跳。录音循环的 callback 只复制数据并放入有界队列，不做 SQLite、文件 IO 或日志输出。
writer 每次从队列取块并写临时 `chunk-000001.wav.tmp`，完成后 flush、关闭、计算 SHA-256，
用 `os.replace` 改成 `.wav`，再以事务插入 `audio_chunks(status='closed')`。临时文件在启动
恢复时删除或重新校验，不把未完成文件当作可转写音频。

默认块长为 300 秒，可由 `MEETING_CHUNK_SECONDS` 设置在 30 到 900 秒之间；超过该范围
配置校验失败。writer 在块边界先落盘再开启下一块，时间以录音单调时钟累计，不依赖系统时钟
跳变。队列容量和磁盘最低剩余空间也由配置限制；队列满不阻塞 callback，记录丢块计数并将
会议标为 `partial`，但继续保存后续数据。若持续无法写盘，停止 recorder 并保留已有块。

生命周期为：`created → recording → stopping → transcribing → summarizing → completed`。
任何录音异常进入 `interrupted`，只要已有可读块就允许 `retry`；没有可读块则为
`failed`。停止时先写 `stopping`，关闭流和 writer，确认最后一个块，再创建每个块的
转写 job，随后进入 `transcribing`。所有块成功或明确不可恢复失败后，若存在至少一段成功
转写则进入 `summarizing`；有失败块也可生成 `partial` 纪要，但纪要顶部必须显示缺失块
和时间范围。没有任何成功转写则进入 `failed`，保留音频供修复。

应用启动时扫描 `recording`、`stopping`、`transcribing`、`summarizing` 会议。超过
`MEETING_HEARTBEAT_TIMEOUT=30` 秒没有心跳的录音标记 `interrupted`，扫描目录中的 WAV
并校验后补登记可读块；`running` 且 `lease_until` 已过期的 job 改回 `queued`。这使进程
崩溃后能够保留已经关闭的音频和已完成的转写，不尝试恢复一个未完整写入的 WAV。

## 转写队列和错误处理

默认只运行一个转写 worker，避免 Whisper 模型重复占用内存；它可以在录音仍继续时处理
已经关闭的块。worker 每次以事务领取一个 job，租约默认 10 分钟并定期续租；完成后事务写入
带序号的结果。Whisper 结果优先使用返回的 `segments[].start/end/text`；相对块时间加上
`audio_chunks.start_ms` 后保存为会议时间。llama-ASR 请求优先使用 `verbose_json` 获取
时间段；服务只返回普通 JSON 时写一条覆盖整个块的 segment，并在 UI 标注“块级时间戳”。

可恢复错误包括临时网络断开、服务忙、模型加载超时、短暂设备文件锁和数据库忙；每块最多
重试 3 次，间隔 5 秒、30 秒、300 秒，用户也可通过 `retry` 重置失败 job。不可恢复错误
包括音频校验失败、模型名称不存在、输入格式损坏和配置非法；它们不自动重试，但保留音频
和错误信息。ASR 服务返回 4xx 时显示后端 URL 的主机、状态码和模型名，不打印音频或完整
响应；5xx/超时按可恢复处理。所有尝试、错误码和最后时间写入 job，日志级别为 warning；
fatal 的数据库迁移/损坏为 error 并阻止新的会议。

转写结果不会用空字符串覆盖之前成功的文本。空文本是成功的“未识别语音”，仍记录块和
时间范围；用户查看时可区分“无语音”和“转写失败”。停止、重启和重试都不删除原始音频。

## 纪要生成、事实约束和长度控制

`summary.py` 的输入是按 `chunk.seq`、`segment.start_ms` 排序的转写，以及失败块的明确
占位标记。单次模型输入最多 `MEETING_SUMMARY_CHUNK_CHARS=12000` 个 Unicode 字符；
长会议按连续时间窗口分组，每组生成 map JSON，再把 map 结果合并为 reduce 输入。Reduce
仍有 `MEETING_SUMMARY_MAX_INPUT_CHARS=30000` 上限；超过时只保留各 map 结果和必要证据摘录，
绝不截断到没有说明的半句话。配置范围由 `config.py` 校验，实际模型上下文不足时按错误
处理或由用户降低预算，不无限重试。

模型输出固定为 JSON，核心结构为：`summary`、`topics`、`decisions`、`action_items`、
`unresolved_questions`、`inferences`、`missing_audio_or_transcript` 和 `metadata`。
每个事实性条目包含 `text`、`evidence_segment_ids`、`evidence_time_range` 和
`classification='fact'`；行动项还包含 `owner`、`due_date`、`raw_owner`、`raw_due_date`
和 `status`。未明确的 owner/due date 必须为 `null`，只保留原文中出现的相对日期或称呼，
不根据上下文猜测人员、日期、优先级或完成状态。

`inferences` 可以记录模型认为有用但不能直接由原文证明的解释，每项必须有
`classification='inference'`、证据段和 `needs_confirmation=true`。没有证据 ID 的事实、
决策、行动项会被校验器拒绝；证据 ID 不属于本会议或引用失败块的条目会被移到
`needs_review`，不能渲染为事实。模型被明确要求使用“原文未说明”，而不是填写常见会议
模板内容。纪要渲染器再加上“由模型生成，原文证据见转写”的提示，原始转写始终可查看。

JSON 解析失败先做一次不改变内容的格式重试；仍失败则保存 `summary_versions(status='failed')`
的错误元数据，会议保留在 `summarizing`/`partial`，用户可修复模型后 `retry`，不展示一份
看似完整的自由文本。模型连接、超时、5xx 是可恢复 warning；4xx、模型不存在和证据校验
失败在该版本上是 fatal，但不损坏转写。成功版本记录模型名、prompt 版本、输入摘要 hash
和创建时间，重新生成产生新版本而不覆盖历史版本。

## 本地 LLM、联网开关和隐私

默认配置为 `MEETING_STT_BACKEND=whisper`、`MEETING_WHISPER_MODEL=base`、
`MEETING_LLM_BACKEND=ollama`、`MEETING_LLM_BASE_URL=http://127.0.0.1:11434`，以及
`MEETING_LLM_MODEL=gemma3:1b`。Ollama 适配器调用 `/api/chat`，兼容适配器调用配置的
`/v1/chat/completions`；两者都关闭流式输出、设置明确连接/读取超时，并只发送当前会议
转写。现有 `OLLAMA_*`、`LLAMA_*` 变量只作为兼容迁移输入，不覆盖会议入口的安全默认值。

`MEETING_ALLOW_NETWORK=false` 是默认且强制的本地隐私开关。关闭时只接受 loopback 主机
`127.0.0.1`、`localhost` 和 `::1`；配置其他主机在创建会议前失败，并显示“当前会发送
音频/转写，需显式设置 `MEETING_ALLOW_NETWORK=true`”。开启时 CLI 启动显示目标主机、
ASR/LLM 类型并要求用户确认；仍不上传音频到第三方，除非用户选择 llama-ASR 远程服务。
API token 只从环境变量读取，不写 SQLite、日志或导出文件；日志不打印原文、模型 prompt
或完整 HTTP body。项目不能保证本地服务本身不留日志，文档必须说明用户需自行检查服务配置。

录音时 CLI 显示红色 `RECORDING`、会议 ID、输入设备和开始时间，并提供停止提示。这是
“不让机器人加入会议”的本地桌面采集方式，但不等于录音合规；文档提醒用户遵守当地对
录音通知和同意的要求。导出文件由用户主动指定，删除操作不可恢复，文档说明应先备份。

## CLI 和历史查看

CLI 是 MVP 的唯一用户界面，避免前端构建和额外服务。命令及行为固定为：

- `python -m meeting_mvp devices [--source microphone|system|all]` 枚举和短测设备。
- `python -m meeting_mvp start --source microphone|system [--device-id ID] [--title TITLE]`
  前台开始录音；它等待 `stop` 命令、Enter 或 Ctrl+C，并持续写心跳。
- `python -m meeting_mvp stop ID` 设置持久化停止请求；若会议不在 `recording`，返回当前
  状态而不重复停止。
- `python -m meeting_mvp status [ID]` 显示正在录音会议、块数、字节数和转写/纪要进度。
- `python -m meeting_mvp list [--status STATUS]` 列出历史，显示 ID 前缀、标题、时间、时长、
  输入设备和状态。
- `python -m meeting_mvp show ID [--transcript]` 显示纪要各栏、证据时间和失败块；加
  `--transcript` 才显示完整转写，避免默认刷屏。
- `python -m meeting_mvp retry ID` 重新排入失败块或失败纪要任务，不重新录音。
- `python -m meeting_mvp export ID --format markdown|json --output PATH` 导出；输出路径
  必须是用户指定的绝对或数据根目录内路径。
- `python -m meeting_mvp delete ID --audio-only|--all --confirm` 执行明确删除。
- `python -m meeting_mvp self-check --synthetic` 运行无硬件、无模型、无网络的最小闭环。

`start` 前台进程和 `stop` 的跨进程协调依靠 SQLite 的 `stop_requested` 设置和短轮询，
而不是 PID 文件；store 对一个活动会议加唯一活动约束。若调用者关闭终端，心跳恢复机制
会留下 `interrupted` 会议，下一次启动可继续转写已关闭块。`show` 不把模型推断混在事实
列表里，历史列表只查询 SQLite，不要求 Whisper、Ollama 或 TTS 可用。

## 具体错误处理和不变量

配置层拒绝非法路径、空模型名、标题超过 200、块长不在 30–900 秒、未知后端和非本地
URL（当联网关闭）。用户得到带错误码的 stderr 和修复提示，配置错误是 fatal，不创建会议。

设备层把没有输入通道、权限拒绝、采样率/通道不支持、设备忙和 loopback 缺失分开报告；
这些是当前会议的 fatal，但不影响历史查看。打开成功后检测不到数据只警告并保留会议，停止
时根据块是否可读标记 partial，避免把安静会议误判成失败。

写盘层在磁盘不足、权限拒绝、队列溢出、文件校验失败时先停止新写入，再安全关闭能关闭的
WAV。已提交块保留，临时文件清理；错误写入会议 `error_code/error_message` 和日志。SQLite
事务失败则不报告块已完成；应用重启用文件 hash 和大小决定是否补登记。

任务层严格禁止未 `closed` 音频进入 ASR，禁止未成功转写进入 reduce，禁止未经 schema/证据
验证的 summary 标为 ready。可恢复异常按退避重试，不可恢复异常保留材料供人工修复。所有
job 更新使用租约防止两个进程同时处理同一块；唯一索引防止重复转写和重复纪要。

删除层先验证会议 ID 和路径，再在数据库事务中标记 deleting，文件删除成功后删除行；文件
删除失败保留 deleting 状态和错误，下一次可以重试，不删除数据库记录冒充成功。导出层只读
已保存结果，文件写临时文件后原子替换；权限或空间失败只影响导出，不改变会议状态。

## 测试和 smoke 验证矩阵

测试分为不依赖真实设备/模型的自动化测试和明确标注的本机集成冒烟。由于当前项目没有
测试框架，新增的必要测试使用标准库 `unittest`，只覆盖可靠性边界，不为 TTS、旧聊天入口
复制测试。

| 验证 | 方法 | 通过条件 |
| --- | --- | --- |
| 语法和导入 | `python -m compileall meeting_mvp stt.py` | 无语法错误；导入会议模块不加载 TTS/模型 |
| Store 与迁移 | 临时 SQLite，运行迁移两次和状态转换测试 | 幂等；外键、唯一约束和非法状态生效 |
| 录音落盘 | fake `AudioSource` 产生正弦波，模拟块边界、停止、队列满和重启 | WAV 可读、序号连续、hash 正确、丢块被标记 |
| 恢复 | 构造过期 heartbeat、临时 WAV 和 running job | 只补登记完整块，任务回到 queued，会议为 interrupted |
| 转写 | fake Whisper/ASR 返回 segments、空文本、超时和 4xx/5xx | 时间换算正确；重试/致命分类正确；无文本不覆盖旧文本 |
| 纪要校验 | 提交合法 JSON、无证据 ID、外部证据 ID、超长输入和坏 JSON | 只有合法版本 ready；推断与事实分离；失败可重试 |
| 安全配置 | 非本地 URL、路径穿越、未提供 token、非法标题/块长 | 在执行录音或发送请求前拒绝，并不写会议内容 |
| 无硬件最小流程 | `self-check --synthetic` | 临时目录中完成创建、写一个 WAV、fake 转写、fake 纪要、历史查看和导出 |
| 设备枚举 | Windows/macOS 实机运行 `devices` | 至少能列出普通输入；Windows 单独显示 loopback；macOS 无虚拟设备时给清晰错误 |
| 麦克风集成 | 真实设备录 10 秒并停止 | 音频可回放，状态转入转写，不因模型慢而丢整段 |
| Windows loopback 集成 | 播放固定测试音并选择 loopback，录 10 秒 | WAV 含播放音；无 loopback 时不静默切换 |
| macOS 系统音频集成 | 已授权虚拟设备和未安装虚拟设备各测一次 | 已安装时能选中并录音；未安装时出现 `MAC_SYSTEM_AUDIO_REQUIRES_VIRTUAL_DEVICE` |
| 本地后端集成 | 本机 Whisper 与 Ollama/兼容服务，使用短 fixture | 转写和纪要完整；服务不可用时历史仍可查看且任务可 retry |

真实模型不可用时，`self-check --synthetic` 使用项目内确定性的 fake transcriber 和 fake
summarizer；它验证持久化、状态机、队列、恢复和导出，但不声称验证 Whisper/LLM 质量。
真实设备不可用时，`devices` 仍验证枚举路径；fake source 验证写盘。只有平台集成项能证明
权限、驱动和 loopback，不能用合成测试替代。

## 依赖、配置和迁移交付物

实施时先统一安装来源。`pyproject.toml` 继续作为 uv 的权威声明，所有代码直接 import
的 `requests`、`langchain-openai`（若旧入口仍需它）等现有依赖必须显式列出；
`requirements.txt` 与 `uv.lock` 由同一组版本更新。会议 MVP 自身用标准库 HTTP，不因新
功能再引入 LangChain。唯一新增运行时包为固定的 `SoundCard==0.4.5`；TTS 依赖保留用于
legacy 入口，但会议入口不加载它。实施后必须运行 `uv lock` 并审阅 lock diff，不能只手改
requirements。

新增 `.env.example` 只包含以下非秘密键及安全默认值：
`MEETING_DATA_DIR`、`MEETING_STT_BACKEND`、`MEETING_WHISPER_MODEL`、
`MEETING_ASR_BASE_URL`、`MEETING_ASR_MODEL`、`MEETING_LLM_BACKEND`、
`MEETING_LLM_BASE_URL`、`MEETING_LLM_MODEL`、`MEETING_ALLOW_NETWORK`、
`MEETING_LLM_API_KEY`、`MEETING_CHUNK_SECONDS` 和 `MEETING_KEEP_AUDIO`。API key 示例
使用空值或明显假值，不能复制当前 `.env` 的实际地址。

README 更新为先介绍会议 MVP，再单列 legacy voice assistant。`docs/meeting-mvp.md` 必须
完整说明 Windows/macOS Python 安装、`uv sync`/requirements、设备枚举和选择、麦克风权限、
Windows loopback、macOS 虚拟音频设备与权限、Whisper/Ollama/兼容模型配置、联网开关、
数据目录和删除、已知限制、录音合规提醒、错误码和排障。文档必须明确 macOS 无虚拟设备
时不支持系统音频，Windows loopback 受驱动和播放设备影响。

## 风险和实施顺序

最大风险是平台音频差异：Windows loopback 可能因独占模式、驱动或 SoundCard 后端失败；
macOS 系统音频需要额外虚拟设备和权限。第一阶段先完成 `devices`、SoundCard 版本验证、
macOS 虚拟设备错误路径和 synthetic self-check，再实现完整会议；不要在模型功能完成后才
验证采集。

第二个风险是本地模型资源和长会议耗时。Whisper 模型启动、RAM/VRAM 和模型上下文都可能
超出个人机器能力，因此转写 worker 与录音解耦、map/reduce 预算、任务恢复和原始音频保留
都是强制项。第三个风险是事实可靠性不能由提示词单独保证；证据 ID、空值规则、事实/推断
分离和原文可核对性必须在 `summary.py` 的代码校验层执行。

建议实施顺序为：先加配置/store/schema 和 synthetic self-check；再做普通输入录音、块落盘
和恢复；接入转写队列；接入带证据 JSON 的层级纪要；实现历史/导出；最后接入 Windows
loopback、macOS 虚拟输入文档和平台冒烟。每阶段都保留可读的部分结果，不因后续 LLM 失败
丢失音频或转写。

## 审查回填说明

本次读取时没有发现 `design-verdict.json`，因此没有 reviewDoc 或 finding 可以逐项回应。
后续设计审查若将 verdict 设为 `CHANGES_REQUESTED`，下一轮必须读取其 `findings` 和
`reviewDoc`，在本节逐条写明“已修改、加入 backlog 的理由或依据原需求忽略”，并同步修改
正文；只有外部审查文件的顶层 verdict 为 `APPROVED` 时，设计循环才满足停止契约。
