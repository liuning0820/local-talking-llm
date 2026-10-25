# Implementation Plan

本计划基于实际读取的 `app_mac.py`、`app_win.py`、`stt.py`、
`query_local_llm.py`、`tts.py`、`README.md`、`pyproject.toml`、
`requirements.txt`、`uv.lock` 和 `.gitignore`。当前项目只有面向短问答的
一次性内存录音，没有会议、SQLite、历史、转写任务或纪要实现；`stt.py`
提供 Whisper/llama-server 转写适配思路，两个旧入口提供普通输入设备枚举，
但都不支持长录音落盘或系统音频 loopback。

实施选择：使用 `python -m meeting_mvp` 作为唯一新的 MVP 入口，而不引入
Web/桌面 UI。当前仓库已经以 Python CLI 为主，CLI 可以直接显示录音和处理
状态，也避免 Flask、前端构建和额外服务；`app_mac.py`、`app_win.py` 和
TTS 保持 legacy 功能，避免在录音进程中加载无关的 TTS/聊天模型。

数据根目录默认使用 Windows `%LOCALAPPDATA%/local-talking-llm`、macOS
`~/Library/Application Support/local-talking-llm`，由 `MEETING_DATA_DIR`
覆盖。根目录包含 `meetings.db`、`meetings/<uuid>/audio/`、
`meetings/<uuid>/exports/` 和 `logs/`；数据库只保存根目录内的相对路径，
音频、转写、纪要默认不上传，用户可分别删除音频或整场会议。

任务拆分建议：实现顺序拆为“基础存储与配置 → 音频采集与恢复 → 转写任务
→ 可靠纪要 → CLI 历史/删除/导出与文档”。这些是同一条流水线的依赖阶段，
不是并行独立产品；后续单一实现/审查循环必须按以下顺序落地，且每项完成后
运行该项验证。

- [ ] 1. 建立会议 MVP 包、配置边界和依赖基线。
      创建 `meeting_mvp/` 包和 `__main__.py` 的子命令骨架，先支持
      `devices`、`start`、`stop`、`status`、`list`、`show`、`retry`、
      `export`、`delete`、`self-check` 的参数解析和统一错误退出码；CLI
      不在导入时加载 Whisper、TTS 或 LLM。创建 `config.py`，解析数据目录、
      16 kHz/单声道/16-bit 音频、30–900 秒块长（默认 300 秒）、有限队列、
      job 租约/重试、Whisper/ASR/LLM 后端、输入预算和
      `MEETING_ALLOW_NETWORK=false`。远程 URL 在联网开关关闭时必须在发送
      音频或文本前拒绝，API token 只从环境读取。仅为 Windows WASAPI loopback
      新增固定版本 `soundcard==0.4.5`；不新增 Web、TTS、聊天或测试框架依赖。
      Files: `meeting_mvp/__init__.py`, `meeting_mvp/__main__.py`,
      `meeting_mvp/config.py`, `pyproject.toml`, `requirements.txt`,
      `uv.lock`, `.env.example`
      Verify: `uv sync` 成功；`python -m compileall meeting_mvp` 无语法错误；
      `python -m meeting_mvp --help` 展示所有子命令；配置单元测试覆盖默认
      数据目录、标题上限、块长范围、loopback/远程 URL 安全边界。

- [ ] 2. 实现 SQLite 数据模型、迁移和本地数据目录安全。
      在 `store.py` 使用标准库 `sqlite3`，开启 WAL、外键和忙等待，使用
      `PRAGMA user_version` 做幂等、事务化迁移。创建 `meetings`、
      `audio_chunks`、`transcript_segments`、`processing_jobs`、
      `summary_versions`、`settings` 表及状态/时间/唯一约束和索引。会议至少
      保存 UUID、可选标题、UTC 开始/结束时间、状态、输入模式、设备 ID/名称、
      平台、采样率/声道、停止请求、心跳、写入字节/丢块计数和错误信息；音频块
      保存序号、相对路径、起止毫秒、大小、SHA-256 和关闭状态；转写保存块序、
      时间段、文本、后端/模型、尝试次数和失败状态；job 保存类型、租约、重试
      和错误；纪要保存版本、模型、提示版本、源摘要哈希、JSON/Markdown 和
      ready/failed 状态。所有相对路径必须 realpath 校验在数据根目录内，删除和
      导出不能路径穿越；活动会议由事务约束为至多一个。提供会议/块/job/转写/
      纪要的查询、状态迁移、任务领取/续租/过期回收和删除接口。
      Files: `meeting_mvp/store.py`, `meeting_mvp/paths.py`,
      `tests/test_store.py`, `tests/test_paths.py`
      Verify: `python -m unittest discover -s tests -p "test_*.py"` 通过；测试
      将迁移执行两次仍成功，非法状态/重复序号/目录外路径被拒绝，删除整场
      会议清理记录和会议目录而音频单独删除保留转写与纪要。

- [ ] 3. 实现设备抽象和明确的 Windows/macOS 音频输入边界。
      创建统一 `DeviceInfo`/`AudioSource` 接口和设备短测。普通麦克风使用
      `sounddevice` 的显式设备 ID、CoreAudio/PortAudio host API、通道数和默认
      采样率；设备消失、无输入通道、权限拒绝、占用或采样率不支持必须给出
      带错误码的修复提示，不能静默切换默认设备。Windows 的 `system` 源使用
      `soundcard` 的 WASAPI `all_microphones(include_loopback=True)` 和选定
      loopback 播放设备，明确标记 `is_loopback`；没有 SoundCard、没有 loopback、
      输出设备被独占或格式不支持时显示替代的麦克风、Stereo Mix 或虚拟混音
      方案，不能把麦克风冒充系统音频。macOS 普通源使用 CoreAudio 输入设备；
      macOS 系统声音只接受用户已安装并明确选择的 BlackHole、Loopback、
      Soundflower 或聚合音频输入，未安装虚拟设备时返回
      `MAC_SYSTEM_AUDIO_REQUIRES_VIRTUAL_DEVICE`，并提示“系统设置 → 隐私与
      安全性 → 麦克风”权限。设备枚举输出稳定 ID、名称、host API、通道、采样率、
      loopback 和是否需要虚拟设备；短测打开约 500 ms 后关闭，不创建会议。
      Files: `meeting_mvp/audio.py`, `meeting_mvp/audio_windows.py`,
      `meeting_mvp/audio_macos.py`, `meeting_mvp/errors.py`,
      `meeting_mvp/__main__.py`, `tests/test_devices.py`
      Verify: `python -m unittest tests.test_devices` 通过，使用 fake 枚举器
      验证设备选择和错误码；`python -m meeting_mvp devices` 在当前机器能列出
      可用输入或清晰的无设备提示。真实 Windows/macOS loopback、权限和虚拟
      设备只能在对应机器做后文平台 smoke，不以静态测试声称已支持。

- [ ] 4. 实现有界队列、按时间片 WAV 落盘和录音崩溃恢复。
      在 `audio.py` 的录音器中让 PortAudio/SoundCard 回调只复制当前帧并写入
      有界队列，不执行 SQLite、文件 IO、模型调用或 Rich 日志；独立 writer
      线程按 16 kHz、单声道、signed 16-bit PCM 写
      `chunk-000001.wav.tmp`，flush/关闭后计算 SHA-256，用 `os.replace` 原子
      改为 `.wav`，再事务登记 `audio_chunks(status='closed')`。按单调时钟记录
      起止毫秒，块轮换默认为 300 秒；队列满记录 dropped_blocks 并标记
      `partial`，磁盘/设备异常安全关闭可完成的当前块并保留已关闭块。停止、
      Enter 和 Ctrl+C 都走同一停止协议：持久化 `stop_requested`，最多约 1 秒
      关闭流、writer 和最后块；重复 stop 不重复生成块或 job。启动恢复删除残留
      `.tmp`，只校验并补登记完整 WAV，将无心跳录音标为 `interrupted`，保留已
      关闭块，并让过期 job 回到 queued；录音线程不得因 ASR/LLM 变慢而等待。
      Files: `meeting_mvp/audio.py`, `meeting_mvp/store.py`,
      `meeting_mvp/recovery.py`, `tests/test_recording.py`,
      `tests/fixtures/`（仅放小型合成音频或 fake source）
      Verify: `python -m unittest tests.test_recording` 通过；fake source 产生
      超过一个块长后每个 WAV 可由 `wave` 打开、序号连续、数据库大小和 hash
      一致，模拟队列满/磁盘异常/重启后只恢复完整文件；无硬件 smoke 不保留
      整场 NumPy 音频数组。

- [ ] 5. 实现逐块转写、时间信息、重试和恢复任务。
      创建统一 `Transcriber` 接口，复用 `stt.py` 的 WAV 格式和 Whisper/
      llama-ASR 思路但按文件逐块处理；Whisper 在 worker 中懒加载，优先保存
      `segments[].start/end/text`，以块起始时间换算为会议绝对相对时间；普通
      ASR JSON 没有时间段时登记覆盖整块的 segment 并在 UI 标注块级时间戳。
      默认一个转写 worker，按事务领取带租约 job，写入成功结果后才提交完成；
      空文本是“未识别语音”的成功结果，不能覆盖已有成功文本。超时、临时网络
      断开、服务忙和短暂文件锁按 5/30/300 秒最多三次重试；损坏 WAV、非法配置、
      模型不存在和 4xx 为不可恢复失败；`retry` 只重排可恢复块或纪要 job，
      重启接管过期租约，已成功块不重复拼接。若部分块失败仍保留音频和已成功
      转写，并以缺失时间范围进入 partial 处理状态。
      Files: `meeting_mvp/transcription.py`, `meeting_mvp/jobs.py`,
      `meeting_mvp/stt_adapters.py`, `stt.py`（仅在确有复用需要时增加带时间戳
      辅助，不改变旧入口行为）, `tests/test_transcription.py`,
      `tests/test_jobs.py`
      Verify: `python -m unittest tests.test_transcription tests.test_jobs` 通过；
      fake Whisper/HTTP 客户端验证绝对时间换算、空文本、成功结果幂等、3 次退避、
      4xx 致命分类、过期租约恢复；`python -m compileall meeting_mvp stt.py`
      无语法错误。外部 ASR 不可用时测试必须使用 fake，不连接网络。

- [ ] 6. 实现本地 LLM 结构化纪要、长文本分层和证据校验。
      创建不带对话历史的 LLM 适配器：默认调用本机 Ollama `/api/chat`，可选
      OpenAI 兼容 `/v1/chat/completions`，使用标准库 HTTP/JSON，明确超时和
      `MEETING_ALLOW_NETWORK` 检查；非 loopback 目标在开关关闭时拒绝，开启时
      在 CLI 显示目标主机并要求确认，token 不写数据库、日志或导出。输入按块
      序和 segment 时间排序，短文本直接总结；超过约 12,000 Unicode 字符按
      连续时间窗口 map 摘要，再对 map 结果按不超过约 30,000 字符 reduce，绝不
      把完整超预算转写一次发送给模型。固定 JSON 至少包含 `summary`、`topics`、
      `decisions`、`action_items`（含 `owner`、`due_date`、状态）、
      `unresolved_questions`、`inferences`、缺失音频/转写信息和 metadata；事实、
      决策、行动项必须有本会议 `evidence_segment_ids` 与时间范围，owner/date
      未提及使用 `null` 或“未明确”，没有事实时明确写“未提及”，禁止模板臆造。
      `inferences` 必须标记“模型推断/需确认”，无证据、外部 evidence ID 或坏
      JSON 不得成为 ready；格式重试一次仍失败则保存 failed 版本而不覆盖原文。
      生成 Markdown 时把事实和推断分栏，并始终保留可核对的转写引用。
      Files: `meeting_mvp/llm.py`, `meeting_mvp/summary.py`,
      `meeting_mvp/prompts.py`, `meeting_mvp/markdown.py`,
      `tests/test_summary.py`, `tests/test_llm_privacy.py`
      Verify: `python -m unittest tests.test_summary tests.test_llm_privacy` 通过；
      fake summarizer 覆盖短/长输入、map/reduce 顺序、合法 JSON、坏 JSON、无效
      证据、未提及负责人/日期和推断分离；测试确认联网关闭时不会发出任何远程
      请求，合格结果才写入 `summary_versions.status='ready'`。

- [ ] 7. 串联会议生命周期和后台处理，不加载 legacy TTS/聊天功能。
      创建 `meeting_mvp/service.py`，串联创建会议、设备短测、recording →
      stopping → transcribing → summarizing → completed/partial/interrupted/
      failed 状态，停止后为所有 closed 块建立唯一转写 job，全部可用结果后才
      建立纪要 job；部分转写可生成明确标有缺失块的 partial 纪要。`start` 前台
      显示红色 RECORDING、会议 ID、标题、设备和开始时间，并持续心跳；`stop ID`
      跨进程只设置数据库停止请求，历史/status 不应要求 Whisper、Ollama 或 TTS
      可用。应用启动扫描未完成会议、回收过期任务并允许从最后成功块继续。
      Files: `meeting_mvp/service.py`, `meeting_mvp/jobs.py`,
      `meeting_mvp/store.py`, `meeting_mvp/__main__.py`,
      `tests/test_service.py`
      Verify: `python -m unittest tests.test_service` 通过；fake source/transcriber/
      summarizer 完成创建、录音停止、逐块任务、恢复和部分失败状态；
      `python -m meeting_mvp self-check --synthetic` 的核心流水线不导入
      `tts.py`、`chatterbox` 或旧对话历史。

- [ ] 8. 完成 CLI 历史查看、转写查看、导出和删除。
      在 `__main__.py` 接入真实命令：`devices [--source microphone|system|all]`、
      `start --source ... [--device-id ID] [--title TITLE]`、`stop ID`、
      `status [ID]`、`list [--status STATUS]`、`show ID [--transcript]`、
      `retry ID`、`export ID --format markdown|json --output PATH` 和
      `delete ID --audio-only|--all --confirm`。历史按开始时间倒序显示标题、时间、
      输入设备和状态；show 显示纪要、失败块、证据和可选完整转写，不依赖模型服务。
      导出只能写数据根目录内或用户明确提供的安全绝对路径，使用临时文件原子
      替换；删除音频保留转写/纪要，删除全部会议清理 DB 和会议目录，删除失败
      保留 deleting 状态供下次恢复。所有命令用 Rich 输出可行动错误，不显示 token、
      完整 prompt 或完整会议正文到日志。
      Files: `meeting_mvp/__main__.py`, `meeting_mvp/export.py`,
      `meeting_mvp/service.py`, `tests/test_cli.py`, `tests/test_export_delete.py`
      Verify: `python -m unittest tests.test_cli tests.test_export_delete` 通过；在
      临时 `MEETING_DATA_DIR` 中先用 synthetic fixture 创建会议，再验证 list/show/
      status/retry/export/delete；模型服务不可用时仍能查看已保存会议和转写。

- [ ] 9. 添加无硬件、无模型、无网络的最小 self-check 和相关可靠性测试。
      实现 `self-check --synthetic` 使用确定性的 fake source、fake transcriber 和
      fake summarizer，在临时或 `MEETING_DATA_DIR` 下完成会议创建、至少两个合成
      WAV 块、持久化、逐块转写、结构化纪要、历史列表和 Markdown/JSON 导出，并
      明确输出未验证真实设备/模型。测试只覆盖本 MVP 的状态、落盘、恢复、时间换算、
      重试、证据校验、隐私和路径安全，不为旧 TTS/聊天入口添加无关测试。
      Files: `meeting_mvp/self_check.py`, `meeting_mvp/__main__.py`,
      `tests/test_self_check.py`, `tests/test_recording.py`,
      `tests/test_transcription.py`, `tests/test_summary.py`
      Verify: Windows PowerShell 使用
      `$env:MEETING_DATA_DIR=(Join-Path $env:TEMP "local-meeting-smoke"); python -m meeting_mvp self-check --synthetic`
      成功并输出 completed/导出结果；macOS 使用
      `MEETING_DATA_DIR=$(mktemp -d) python -m meeting_mvp self-check --synthetic`；
      随后运行 `python -m unittest discover -s tests -p "test_*.py"` 全部通过。

- [ ] 10. 更新安装、运行、设备、权限、模型、隐私和限制文档。
      重写 `README.md` 的首要入口为会议 MVP，保留旧的 `app_mac.py`/`app_win.py`
      语音助手为 legacy，并移除/改正当前不存在的 `app.py` 用法。新增
      `docs/meeting-mvp.md`，明确 Python 3.11+、`uv sync`/requirements 安装、
      `python -m meeting_mvp` 启动方式、`devices`/设备 ID 选择、标题和状态、
      Windows 麦克风与 WASAPI loopback 的 SoundCard/驱动边界、Windows Stereo Mix
      替代方式、macOS 麦克风权限、BlackHole/Loopback/Soundflower/聚合设备路由、
      没有虚拟设备时不支持系统音频、Whisper/Ollama/兼容模型配置、联网开关、
      数据目录、音频/整场删除、录音同意提醒、磁盘/模型/权限限制和错误码。新增
      `.gitignore` 规则，防止仓库内覆盖数据目录时提交 SQLite、音频、导出和日志；
      `.env.example` 只含安全默认值和空 token，不复制当前 `.env`。
      Files: `README.md`, `docs/meeting-mvp.md`, `.env.example`, `.gitignore`
      Verify: `python -m compileall meeting_mvp` 通过；按文档命令执行 `python -m meeting_mvp --help`、
      `devices` 和 synthetic self-check；人工检查文档没有把普通麦克风声称为
      Windows loopback，也没有把 macOS 无虚拟设备声称为系统音频支持。

- [ ] 11. 完成整体验证并记录真实环境限制；不创建 git commit。
      先运行 `uv sync`，再运行 `python -m compileall meeting_mvp tests stt.py`、
      `python -m unittest discover -s tests -p "test_*.py"` 和临时目录下的
      `python -m meeting_mvp self-check --synthetic`。若本机有设备，运行
      `python -m meeting_mvp devices --source all` 和短时麦克风录音；Windows
      额外验证已选 WASAPI loopback 播放固定声音后确实写入 WAV，macOS 分别验证
      麦克风权限和已安装虚拟输入。若 Whisper/Ollama、硬件、权限或 SoundCard
      不可用，只记录具体命令、错误和未验证边界，不伪造通过；不要运行需要远程
      服务的现有 `stt-test.py` 作为无网络回归测试，也不要提交 commit。
      Files: `pyproject.toml`, `requirements.txt`, `uv.lock`, `meeting_mvp/`,
      `tests/`, `README.md`, `docs/meeting-mvp.md`, `.env.example`, `.gitignore`
      Verify: 静态编译和标准库 unittest 通过；synthetic smoke 在无硬件/无模型/
      无网络下通过；真实设备/模型结果按平台分别记录，`git status --short`
      显示修改但没有新增提交。
