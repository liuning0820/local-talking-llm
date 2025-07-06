import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import argparse
import os
from queue import Queue
from rich.console import Console
# Updated imports for modern LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM
from tts import TextToSpeechService
from dotenv import load_dotenv
import re

# Load environment variables from .env
load_dotenv()
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:1b")
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://localhost:11434")
console = Console()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Local Voice Assistant with TTS")
parser.add_argument("--voice", type=str, help="Path to voice sample for cloning")
parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0)")
parser.add_argument("--model", type=str, default=OLLAMA_LLM_MODEL, help="Ollama model to use")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples")
parser.add_argument("--play-voice", action="store_true", help="Play generated voice samples")
parser.add_argument("--continuous", action="store_true", help="Enable continuous listening mode with wake word")
parser.add_argument("--wake-word", type=str, default="你好", help="Wake word to activate listening")
parser.add_argument("--listen-duration", type=float, default=3.0, help="Duration to listen for wake word (seconds)")
parser.add_argument("--wake-timeout", type=float, default=60.0, help="Time to stay awake after wake word detection (seconds)")
parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model to use (tiny, base, small, medium, large, base.en, small.en)")
parser.add_argument("--language", type=str, default="zh", help="Voice Language(e.g., en, zh)")
args = parser.parse_args()


# Initialize Whisper with selected model
stt = whisper.load_model(args.whisper_model)

# Initialize TTS with ChatterBox only if not using zh (Chinese)
tts = None
if args.language != "zh":
    tts = TextToSpeechService()

# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM
llm = OllamaLLM(model=args.model, base_url=OLLAMA_LLM_BASE_URL)

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Now includes device checks, error handling, and device selection if needed.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    import sounddevice as sd
    from sounddevice import PortAudioError

    # List available input devices
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    if not input_devices:
        console.print("[red]No input devices (microphones) found. Please check your system settings.")
        return
    default_device = sd.default.device[0]
    if default_device is None or default_device < 0:
        console.print("[red]No default input device set. Please set a default microphone in your OS.")
        # Prompt user to select a device
        console.print("[yellow]Available audio input devices:")
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                console.print(f"  [{idx}] {dev['name']} (inputs: {dev['max_input_channels']})")
        try:
            device_idx = int(console.input("Enter the device index to use for recording: "))
            if device_idx < 0 or device_idx >= len(devices) or devices[device_idx]['max_input_channels'] == 0:
                console.print("[red]Invalid device index selected.")
                return
            sd.default.device = (device_idx, sd.default.device[1])
        except Exception as e:
            console.print(f"[red]Error selecting device: {e}")
            return
    try:
        def callback(indata, frames, time, status):
            if status:
                console.print(status)
            data_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=16000, dtype="int16", channels=1, callback=callback
        ):
            while not stop_event.is_set():
                time.sleep(0.1)
    except PortAudioError as e:
        console.print(f"[red]PortAudioError: {e}\nTry checking your microphone connection and permissions.")
        # Optionally, print available devices for debugging
        console.print("[yellow]Available audio devices:")
        for idx, dev in enumerate(devices):
            console.print(f"  [{idx}] {dev['name']} (inputs: {dev['max_input_channels']}, outputs: {dev['max_output_channels']})")
        return
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}")
        return


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    # Use a default session ID for this simple voice assistant
    session_id = "voice_assistant_session"

    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    # The response is now a string from the LLM, no need to remove "Assistant:" prefix
    # since we're using a proper chat model setup
    return response.strip()


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


def analyze_emotion(text: str) -> float:
    """
    Simple emotion analysis to dynamically adjust exaggeration.
    Returns a value between 0.3 and 0.9 based on text content.
    """
    # Keywords that suggest more emotion
    emotional_keywords = ['amazing', 'terrible', 'love', 'hate', 'excited', 'sad', 'happy', 'angry', 'wonderful', 'awful', '!', '?!', '...']

    emotion_score = 0.5  # Default neutral

    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    # Cap between 0.3 and 0.9
    return min(0.9, max(0.3, emotion_score))



def continuous_listen_for_wake_word(wake_word: str, listen_duration: float = 3.0):
    """
    Continuously listens for wake word in short audio chunks.
    
    Args:
        wake_word (str): The wake word to listen for
        listen_duration (float): Duration of each listening chunk in seconds
        
    Returns:
        bool: True if wake word detected, False otherwise
    """
    data_queue = Queue()
    stop_event = threading.Event()
    
    # Record for a short duration
    recording_thread = threading.Thread(
        target=record_audio,
        args=(stop_event, data_queue),
    )
    recording_thread.start()
    
    # Let it record for the specified duration
    time.sleep(listen_duration)
    stop_event.set()
    recording_thread.join()
    
    # Process the audio
    audio_data = b"".join(list(data_queue.queue))
    if not audio_data:
        return False
        
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    if audio_np.size > 0:
        try:
            text = transcribe(audio_np)
            if not is_junk_transcription(text):
                console.print(f"[dim]Heard: {text}[/dim]")

            
            # Check if wake word is in the transcribed text
            wake_word_variants = [
                wake_word.lower(),
                wake_word.lower().replace(" ", "")
            ]
            
            text_lower = text.lower().replace(" ", "")
            for variant in wake_word_variants:
                if variant.replace(" ", "") in text_lower:
                    return True
                    
        except Exception as e:
            console.print(f"[dim red]Transcription error: {e}[/dim red]")
    
    return False


def record_until_silence(max_duration: float = 10.0, silence_threshold: float = 2.0):
    """
    Records audio until silence is detected or max duration is reached.
    
    Args:
        max_duration (float): Maximum recording duration in seconds
        silence_threshold (float): Seconds of silence before stopping
        
    Returns:
        np.ndarray: The recorded audio data
    """
    data_queue = Queue()
    stop_event = threading.Event()
    
    recording_thread = threading.Thread(
        target=record_audio,
        args=(stop_event, data_queue),
    )
    recording_thread.start()
    
    console.print("[green]🎤 Listening... (speak your question)[/green]")
    
    start_time = time.time()
    last_audio_time = start_time
    silence_detected = False
    
    # Simple silence detection by checking if we're getting audio data
    while not silence_detected and (time.time() - start_time) < max_duration:
        time.sleep(0.1)
        
        # Check if we have recent audio data
        if not data_queue.empty():
            last_audio_time = time.time()
        elif (time.time() - last_audio_time) > silence_threshold:
            silence_detected = True
            console.print("[dim]Silence detected, stopping recording...[/dim]")
    
    stop_event.set()
    recording_thread.join()
    
    # Process the collected audio
    audio_data = b"".join(list(data_queue.queue))
    if audio_data:
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np
    else:
        return np.array([])

def collect_audio_from_queue(data_queue):
    audio_data = b"".join(list(data_queue.queue))
    if audio_data:
        return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return np.array([])

def listen_in_wake_mode(wake_timeout: float = 60.0):
    """
    Listen for questions while in wake mode (no wake word needed).
    
    Args:
        wake_timeout (float): Time to stay in wake mode (seconds)
        
    Returns:
        tuple: (audio_np, is_timeout) - audio data and whether timeout occurred
    """
    console.print(f"[green]🎤 Wake mode active for {wake_timeout}s - just speak your question![/green]")
    start_time = time.time()
    
    data_queue = Queue()
    stop_event = threading.Event()
    recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
    recording_thread.start()

    last_printed_second = None  # 新增变量，记录上次打印的整秒数
    try:
        while (time.time() - start_time) < wake_timeout:
            # 检查是否接收到足够大的音频信号
            if not data_queue.empty():
                audio_np = collect_audio_from_queue(data_queue)
                # 增加安全检查
                if audio_np is None:
                    console.print("[red dim]⚠️ No audio data collected from queue.[/red dim]")
                    continue  # 继续下一轮监听循环
                if audio_np.size > 0 and np.max(np.abs(audio_np)) > 0.01:
                    console.print("[green]Speech detected! Recording full question...[/green]")
                    full_audio = record_until_silence(max_duration=10.0, silence_threshold=2.0)
                    if full_audio.size > 0:
                        combined_audio = np.concatenate([audio_np, full_audio])
                    else:
                        combined_audio = audio_np
                    stop_event.set()  # 停止录音线程
                    recording_thread.join()
                    return combined_audio, False
            
            remaining_time = wake_timeout - (time.time() - start_time)
            current_second = int(remaining_time)
            if remaining_time > 0 and current_second % 10 == 0 and current_second != last_printed_second:
                console.print(f"[dim]Wake mode: {current_second}s remaining...[/dim]")
                last_printed_second = current_second # 更新已打印的秒数

            time.sleep(0.5)  # 避免 CPU 占用过高

        # 超时处理
        console.print("[dim]Wake mode timeout - returning to wake word listening...[/dim]")
        stop_event.set()
        recording_thread.join()
        return np.array([]), True
    
    except Exception as e:
        console.print(f"[red]Error during wake mode: {e}[/red]")
        stop_event.set()
        recording_thread.join()
        return np.array([]), True


active_threads = []
active_stop_events = []

def start_recording_thread(target, args):
    stop_event = threading.Event()
    data_queue = Queue()
    thread = threading.Thread(target=target, args=(stop_event, data_queue))
    thread.start()
    active_threads.append(thread)
    active_stop_events.append(stop_event)
    return thread, stop_event, data_queue

def is_junk_transcription(text: str) -> bool:
    """
    Returns True if the transcription is likely junk (e.g., only numbers, dots, or empty).
    """
    cleaned = text.strip().replace('.', '').replace(' ', '')
    # If empty after removing dots and spaces, or only numbers, it's junk
    if not cleaned or re.fullmatch(r'\d+', cleaned):
        return True
    # Optionally, filter out very short transcriptions
    if len(cleaned) < 2:
        return True
    return False

if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with ChatterBox TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.voice:
        console.print(f"[green]Using voice cloning from: {args.voice}")
    else:
        console.print("[yellow]Using default voice (no cloning)")

    console.print(f"[blue]Emotion exaggeration: {args.exaggeration}")
    console.print(f"[blue]CFG weight: {args.cfg_weight}")
    console.print(f"[blue]LLM model: {args.model}")
    
    if args.continuous:
        console.print(f"[green]Mode: Continuous listening with wake word")
        console.print(f"[green]Wake word: '{args.wake_word}'")
        console.print(f"[green]Listen duration: {args.listen_duration}s")
        console.print(f"[green]Wake timeout: {args.wake_timeout}s")
    else:
        console.print("[yellow]Mode: Manual (press Enter to record)")
    
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    # Create voices directory if saving voices
    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0

    try:
        if args.continuous:
            # Continuous listening mode with wake word detection
            console.print(f"[green]🎧 Continuous listening mode enabled[/green]")
            console.print(f"[green]Say '{args.wake_word}' to activate the assistant[/green]")
            console.print("[dim]Listening for wake word...[/dim]")
            
            while True:
                # Listen for wake word
                if continuous_listen_for_wake_word(args.wake_word, args.listen_duration):
                    console.print(f"[green]✅ Wake word detected! Assistant is now active.[/green]")
                    
                    # Enter wake mode - listen for questions without wake word for specified timeout
                    while True:
                        audio_np, is_timeout = listen_in_wake_mode(args.wake_timeout)
                        
                        if is_timeout:
                            # Wake mode timeout, return to wake word listening
                            break
                        
                        if audio_np.size > 0:
                            with console.status("Transcribing your question...", spinner="dots"):
                                text = transcribe(audio_np)
                            # Add this check:
                            if is_junk_transcription(text):
                                console.print("[dim]No clear speech detected, continuing to listen...[/dim]")
                                continue
                            console.print(f"[yellow]You: {text}[/yellow]")

                            with console.status("Generating response...", spinner="dots"):
                                response = get_llm_response(text)
                                console.print(f"[cyan]Assistant: {response}[/cyan]")
                                # Analyze emotion and adjust exaggeration dynamically
                                dynamic_exaggeration = analyze_emotion(response)
                                dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                                if args.language == "zh":
                                    import pyttsx3
                                    engine = pyttsx3.init()
                                    engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0')
                                    temp_filename = "voices/temp_response.wav" if args.save_voice else "temp_response.wav"
                                    engine.save_to_file(response, temp_filename)
                                    engine.runAndWait()
                                    import soundfile as sf
                                    audio_array, sample_rate = sf.read(temp_filename, dtype='float32')
                                else:
                                    sample_rate, audio_array = tts.long_form_synthesize(
                                        response,
                                        audio_prompt_path=args.voice,
                                        exaggeration=dynamic_exaggeration,
                                        cfg_weight=dynamic_cfg
                                    )


                            console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                            # Save voice sample if requested
                            if args.save_voice:
                                response_count += 1
                                filename = f"voices/response_{response_count:03d}.wav"
                                if args.language == "zh":
                                    # Already saved by pyttsx3 above, just rename
                                    import shutil
                                    shutil.move(temp_filename, filename)
                                else:
                                    tts.save_voice_sample(response, filename, args.voice)
                                console.print(f"[dim]Voice saved to: {filename}[/dim]")

                            if args.play_voice:
                                play_audio(sample_rate, audio_array)

                            # Continue in wake mode for more questions
                            console.print(f"[green]🎤 Still listening... ({args.wake_timeout}s window refreshed)[/green]")
                        else:
                            console.print("[dim]No audio detected, continuing to listen...[/dim]")
                    
                    # Wake mode ended, return to listening for wake word
                    console.print("[dim]Returning to wake word listening...[/dim]")
                
                # Small delay before listening for wake word again
                time.sleep(0.1)
        
        else:
            # Manual mode (original functionality)
            while True:
                console.input(
                    "🎤 Press Enter to start recording, then press Enter again to stop."
                )

                data_queue = Queue()  # type: ignore[var-annotated]
                stop_event = threading.Event()
                recording_thread = threading.Thread(
                    target=record_audio,
                    args=(stop_event, data_queue),
                )
                recording_thread.start()

                input()
                stop_event.set()
                recording_thread.join()

                audio_data = b"".join(list(data_queue.queue))
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                )

                if audio_np.size > 0:
                    with console.status("Transcribing your question...", spinner="dots"):
                        text = transcribe(audio_np)
                    if is_junk_transcription(text):
                        console.print("[dim]No clear speech detected, continuing to listen...[/dim]")
                        continue
                    console.print(f"[yellow]You: {text}")

                    with console.status("Generating response...", spinner="dots"):
                        response = get_llm_response(text)
                        console.print(f"[cyan]Assistant: {response}")
                        # Analyze emotion and adjust exaggeration dynamically
                        dynamic_exaggeration = analyze_emotion(response)
                        dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                        if args.language == "zh":
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0')
                            temp_filename = "voices/temp_response.wav" if args.save_voice else "temp_response.wav"
                            engine.save_to_file(response, temp_filename)
                            engine.runAndWait()
                            import soundfile as sf
                            audio_array, sample_rate = sf.read(temp_filename, dtype='float32')
                        else:
                            sample_rate, audio_array = tts.long_form_synthesize(
                                response,
                                audio_prompt_path=args.voice,
                                exaggeration=dynamic_exaggeration,
                                cfg_weight=dynamic_cfg
                            )

                    console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                    # Save voice sample if requested
                    if args.save_voice:
                        response_count += 1
                        filename = f"voices/response_{response_count:03d}.wav"
                        if args.language == "zh":
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0')
                            engine.save_to_file(response, filename)
                            engine.runAndWait()
                        else:
                            tts.save_voice_sample(response, filename, args.voice)
                        console.print(f"[dim]Voice saved to: {filename}[/dim]")

                    if args.play_voice:
                        play_audio(sample_rate, audio_array)
                else:
                    console.print(
                        "[red]No audio recorded. Please ensure your microphone is working."
                    )



    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
        # Stop all threads
        for stop_event in active_stop_events:
            stop_event.set()
        for thread in active_threads:
            thread.join(timeout=1)
        console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
        os._exit(0)  # Force exit if threads are still running