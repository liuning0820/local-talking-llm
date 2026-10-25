# 项目当前功能调查报告

## 总结

`local-talking-llm` 当前是一个以 Python 编写的本地/局域网语音助手原型，主流程为“麦克风录音 → 语音识别 → 带会话历史的 LLM 对话 → 文本转语音 → 播放或保存音频”。项目有两个主要入口：`app_mac.py` 面向 macOS/Linux，使用 Ollama 的 LangChain 接口；`app_win.py` 面向 Windows，使用 OpenAI 兼容接口连接 `llama-server` 或其他兼容服务，并额外支持连续唤醒、中文系统 TTS 和两种 STT 后端。

项目还提供三个独立脚本：`query_local_llm.py` 用于测试 llama-server 的流式聊天和 reasoning/final answer 输出；`stt-test.py` 用于直接测试 Qwen3-ASR 的 OpenAI 兼容转写接口；`actor-critic-sample.py` 是基于 LangGraph 的 Actor-Critic 文本工作流示例。核心 TTS 是 Chatterbox，可自动选择 CUDA、Apple MPS 或 CPU，支持短音频合成、长文本分句合成、参考音频声音克隆和保存 WAV。

实际代码与 README 存在明显偏差：README 仍描述部分不存在的 `app.py` 参数和入口；`app_mac.py` 的若干参数只打印或参与合成但没有完整实现对应的保存/播放开关；Windows 的中文 TTS 依赖本机指定的微软 Huihui 语音注册项；项目没有自动化单元测试，`stt-test.py` 需要外部服务和本地 `sample-zh.wav`，所有主程序也都需要麦克风、模型和相应服务才能运行。

## 证据与实际调用链

### 主程序调用链

`app_mac.py` 的模块初始化顺序是：读取 `.env`，从 `OLLAMA_LLM_MODEL` 和 `OLLAMA_LLM_BASE_URL` 取得 Ollama 配置，加载 Whisper 模型，初始化 `TextToSpeechService`，创建包含 system prompt、会话历史和用户输入的 `ChatPromptTemplate`，再通过 `OllamaLLM` 和 `RunnableWithMessageHistory` 构造对话链。

运行 `app_mac.py` 后，主循环在 `__main__` 中等待用户按 Enter，通过 `record_audio()` 使用 `sounddevice.RawInputStream` 以 16 kHz、单声道、16-bit 录音；录音数据由队列拼接并转换为 `float32`。`transcribe()` 调用 Whisper 的 `stt.transcribe(..., fp16=False)`，`get_llm_response()` 使用固定会话 ID `voice_assistant_session` 调用 LangChain 链并保留历史。响应经过 `analyze_emotion()` 计算 0.3 到 0.9 的情绪强度，再由 `tts.long_form_synthesize()` 生成音频，最后无条件调用 `play_audio()` 播放。

`app_win.py` 采用相同的录音、LangChain 会话历史和情绪调节基本结构，但 LLM 使用 `langchain_openai.ChatOpenAI`，通过 `LLAMA_SERVER_BASE_URL` 访问 OpenAI 兼容端点，API key 固定为不需要真实鉴权的 `sk-no-key-required`。`get_llm_response()` 对 ChatOpenAI 返回对象读取 `.content`，适配了聊天模型响应格式。

Windows 入口的手动模式仍是“按 Enter 开始，再按 Enter 停止”。启用 `--continuous` 后，调用链变为：`continuous_listen_for_wake_word()` 按短时段录音并转写 → 检查唤醒词 → `listen_in_wake_mode()` 在唤醒窗口内监测音量 → 检测到语音后调用 `record_until_silence()` → 转写问题、生成回答、合成/播放语音 → 窗口超时后清空聊天历史并返回唤醒词监听。`is_junk_transcription()` 会过滤空文本、纯数字、去除点和空格后长度小于 2 的识别结果。

### 语音识别能力

`app_mac.py` 直接使用 `openai-whisper`，默认模型为 `base`，可通过 `--whisper-model` 选择 `tiny`、`base`、`small`、`medium`、`large`、`base.en` 或 `small.en` 等模型。代码固定使用 `fp16=False`，因此 CPU 可运行，但没有在主程序中提供 GPU 精度开关。

`app_win.py` 支持命令行或环境变量选择两个 STT 后端：

- `whisper`：启动时加载 Whisper；默认模型来自 `--whisper-model`，默认值为 `small`。
- `llama-asr`：调用 `stt.transcribe_llama_asr()`，把 NumPy 音频编码为 16 kHz、单声道、16-bit WAV，通过 `POST {base_url}/audio/transcriptions` 上传，默认模型为 `Qwen3-ASR-0.6B-Q8_0.gguf`，可传递 `language`。

`stt.py` 中的 `numpy_to_wav_bytes()`、`transcribe_llama_asr()` 和 `transcribe_whisper()` 分别实现音频封装、OpenAI 兼容 ASR 请求及 Whisper 结果提取。ASR 请求带有默认 600 秒超时，并把 HTTP 和网络错误转换为带响应信息的 `RuntimeError`。

`stt-test.py` 是最小化 ASR 接口测试：直接打开项目根目录的 `sample-zh.wav`，向 `http://localhost:8080/v1/audio/transcriptions` 上传，模型固定为 `Qwen3-ASR-0.6B-Q8_0.gguf`、语言固定为 `zh`。它不是自动化测试，运行前必须存在音频文件和兼容的 llama-server。

### LLM 与对话能力

macOS/Linux 主程序使用 `langchain_ollama.OllamaLLM`，默认 Ollama URL 是 `http://localhost:11434`，默认模型是 `gemma3:1b`；提示模板要求助手保持友善并尽量少于 20 个单词。`RunnableWithMessageHistory` 使用进程内字典保存单一固定会话，因此同一进程中会保留上下文，但没有持久化、用户隔离或会话 ID 命令行选项。

Windows 主程序使用 `ChatOpenAI`，默认服务地址是 `http://localhost:8080/v1`，默认模型名是 `DeepSeek-R1-Distill-Qwen-1.5B`；配置可由 `.env` 覆盖。提示词为中文，要求回答通常为 30–80 字，并强调准确性、结构清晰、实用性和承认知识局限。代码实际只需要兼容 Chat Completions 的服务，不验证服务是否真的是 llama.cpp。

`query_local_llm.py` 使用官方 `openai.OpenAI` 客户端连接 `http://127.0.0.1:8080/v1`，发起流式聊天请求，分别读取 delta 中的 `reasoning_content` 和 `content`，把推理过程及最终答案分段打印。模型名写成 `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`，注释说明 llama.cpp 服务端可能忽略该名称。

`actor-critic-sample.py` 是独立的文本质量迭代示例。`AgentState` 保存任务、草稿、批评、修订次数和最终输出；`actor_node()` 生成或根据反馈修订草稿，`critic_node()` 检查逻辑、事实、格式和遗漏，`route()` 在反馈包含 `PASS` 或修订次数超过 3 次时结束，否则回到 Actor。LangGraph 图按 Actor → Critic → Actor/END 执行，`run_actor_critic()` 以流式方式打印每次草稿和批评。它同样通过 OpenAI 兼容接口访问 llama-server，不与语音助手主流程连接。

### 文本转语音、声音和音频输出

`tts.py` 的 `TextToSpeechService` 使用 `ChatterboxTTS.from_pretrained()`。未指定设备时按 CUDA、Apple MPS、CPU 顺序自动选择；显式请求不可用 CUDA 时回退到 CPU。`_patch_torch_load()` 为模型加载增加设备映射，以处理模型权重原本保存于 CUDA 的情况。

`TextToSpeechService.synthesize()` 调用 Chatterbox 的 `generate()`，支持 `audio_prompt_path` 参考音频、`exaggeration` 情绪强度和 `cfg_weight` 节奏/引导权重，返回模型采样率与 NumPy 音频数组。`long_form_synthesize()` 使用 NLTK `sent_tokenize()` 切分句子，逐句合成并插入 0.25 秒静音；`save_voice_sample()` 使用 `torchaudio.save()` 保存生成音频。因此 README 所说的参考音频声音克隆和情绪控制在 `tts.py` 与两个主入口中确实有调用路径，但实际质量、语言覆盖和速度取决于 Chatterbox 模型及本机硬件。

macOS/Linux 主程序始终使用 Chatterbox 生成并用 `sounddevice` 播放。`--voice`、`--exaggeration` 和 `--cfg-weight` 会影响生成；`--save-voice` 会创建 `voices/` 并把回答保存为 `voices/response_NNN.wav`。不过 `--voice` 只作为 Chatterbox 参考音频传入，未在启动时验证文件；保存路径和音频播放失败也没有专门的异常处理。

Windows 主程序通过 `--language` 分支输出语音：非 `zh` 使用 Chatterbox；`zh` 使用 `pyttsx3`，并硬编码 Windows 注册表中的 `TTS_MS_ZH-CN_HUIHUI_11.0` 语音 token，然后把结果写入临时 WAV，再用 `soundfile` 读入。只有设置 `--play-voice` 才实际调用 `sounddevice` 播放；不设置时仍会生成音频但不播放。中文分支要求 Windows 上存在该系统语音 token，不能据此推断所有 Windows 或 Linux 环境都可用中文 TTS。

### 平台、安装和依赖

`pyproject.toml` 要求 Python `>=3.11`，声明的主要依赖包括 `chatterbox-tts`、`openai-whisper`、`langchain`、`langchain-ollama`、`pyttsx3`、`sounddevice`、`torchaudio`、`nltk`、`rich`、`peft` 和 `setuptools`。同时配置了清华 PyPI 镜像作为 uv 默认索引。

`requirements.txt` 是一份锁定版本风格的依赖列表，包含 PyTorch、TorchAudio、Whisper、LangChain/OpenAI 兼容栈、声音设备和 Windows 相关 `pywin32`/`comtypes`。它与 `pyproject.toml` 并不完全一致，例如锁定文件中出现 `langchain-openai`、`requests`、`soundfile` 等传递或代码实际使用的依赖，而 `pyproject.toml` 没有逐项声明所有这些包。

README 的安装步骤是创建虚拟环境、执行 `pip install -r requirements.txt` 或 `uv sync`、下载 NLTK `punkt` 数据、安装并启动 Ollama，然后拉取例如 `gemma3:1b`。代码实际还依赖：Ollama 或 OpenAI 兼容服务可访问、Whisper/Chatterbox 模型可下载或已缓存、麦克风和音频输出设备可用；使用 llama-ASR 时还必须启动支持 `/v1/audio/transcriptions` 的服务。

仓库没有发现 `AGENTS.md`、`CONTRIBUTING.md` 或 `.kiro/steering` 文件，也没有发现项目自己的自动化单元测试配置。对请求中的七个 Python 文件执行了只读 AST 解析，结果全部通过；这只能证明语法可解析，不能证明外部模型服务、音频设备或运行时依赖可用。

## 运行方式

推荐按实际代码使用以下入口：

1. 安装 Python 3.11 或更高版本及依赖，下载 NLTK `punkt` 数据。
2. 启动与入口匹配的模型服务：macOS/Linux 使用 Ollama；Windows 主入口和独立 LLM/ASR 测试使用 OpenAI 兼容的 llama-server 或兼容 API。
3. macOS/Linux 执行 `python app_mac.py`，可加 `--model`、`--whisper-model`、`--voice`、`--exaggeration`、`--cfg-weight` 和 `--save-voice`。
4. Windows 手动模式执行 `python app_win.py --play-voice`；连续唤醒模式执行 `python app_win.py --play-voice --continuous --wake-word "你好"`。可用 `--stt-backend whisper` 或 `--stt-backend llama-asr` 选择识别后端，并用 `--asr-base-url`、`--asr-model` 和 `--language` 调整 ASR。
5. 仅测试 LLM 流式接口时执行 `python query_local_llm.py`；仅测试 ASR 接口时准备 `sample-zh.wav` 后执行 `python stt-test.py`；测试 Actor-Critic 文本流程时执行 `python actor-critic-sample.py`。

## 已知限制与文档偏差

- README 的“With Voice Cloning”和“Custom Settings”示例使用不存在的 `app.py`；仓库当前入口实际是 `app_mac.py` 和 `app_win.py`。
- README 描述的 `--record-voice` 在两个主程序中都没有实现；`--continuous`、`--wake-word`、`--play-voice`、`--record-voice` 和若干默认值并非两个入口共有，README 没有清楚区分平台。
- `app_mac.py` 定义了 `--exaggeration`，但主循环实际把 `analyze_emotion(response)` 的结果传给 TTS，而不是使用命令行传入的 `args.exaggeration`；该参数目前只在启动信息中显示。它也没有 `--play-voice` 或 `--record-voice`，默认总是播放。
- `app_mac.py` 的 `--save-voice` 调用 `save_voice_sample()` 时没有传入命令行的情绪和 CFG 参数，因此保存版本与实际播放版本的生成参数可能不同。
- `app_win.py` 的静音检测主要检查队列是否持续收到数据，不是基于音量阈值的真正语音活动检测；`listen_in_wake_mode()` 才额外检查音频峰值。环境噪声可能使录音持续到最大时长。
- 两个主程序都在导入和模型初始化阶段加载重量级模型，没有惰性加载或服务健康检查；第一次启动可能耗时很长并需要网络下载模型。
- `app_win.py` 中文 TTS 的硬编码语音 token、临时文件和 `soundfile` 读取使其强依赖 Windows 语音组件；README 对这一平台限制说明不足。
- `.env` 被 `.gitignore` 忽略，但当前工作树存在本地 `.env`，其中包含实际的远程 LLM 地址和模型配置。报告不复述完整地址；部署时应继续避免提交或公开该文件，并检查远程服务的访问控制。
- `stt-test.py` 没有使用 `with open()`，也没有参数化 URL、模型或音频路径；它适合人工冒烟测试，不适合作为可重复测试。
- `actor-critic-sample.py` 的 `final_output` 状态字段没有被节点填充，结束时主要依赖打印结果；它是示例工作流，不是主产品功能。
- 当前 Git 状态显示一个未跟踪文件 `bash.exe.stackdump`。本次调查没有创建、修改或删除它。

## 结论与建议

项目已经实现了一个可运行的本地语音助手原型：Whisper 或 llama-ASR 负责识别，Ollama/llama-server 负责带内存的对话，Chatterbox 或 Windows `pyttsx3` 负责语音输出，并提供 Windows 连续唤醒与独立 LLM/ASR/Actor-Critic 示例。核心能力在代码中存在，但是否能端到端运行取决于本机音频设备、模型下载、服务地址和平台语音组件。

建议按以下顺序修复或澄清：

1. 先更新 README，使入口、参数、默认模型和平台差异与 `app_mac.py`/`app_win.py` 一致，删除或改写不存在的 `app.py`、`--record-voice` 和 Chatterbox 参数示例。
2. 统一依赖来源，明确 `requirements.txt` 和 `pyproject.toml` 哪一个是权威安装入口，并显式声明代码直接导入的 `requests`、`soundfile` 和 `langchain-openai`。
3. 修正 `app_mac.py` 中 `--exaggeration` 未生效的问题，并明确保存音频使用的生成参数；同时为模型服务、麦克风、TTS 和文件写入增加启动检查及清晰错误信息。
4. 把 LLM/ASR 基础地址、模型名、中文 TTS voice token 和音频路径配置化，避免硬编码；对远程服务增加鉴权、连通性和超时说明。
5. 增加不依赖真实麦克风或模型服务的单元测试，至少覆盖 `numpy_to_wav_bytes()`、`is_junk_transcription()`、情绪分数边界、LLM 响应内容提取和配置分支；再增加可选的外部服务冒烟测试。
6. 处理并确认未跟踪的 `bash.exe.stackdump` 是否应保留，避免将崩溃转储误纳入后续提交。
