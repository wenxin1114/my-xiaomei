from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import pyaudio
import numpy as np
import time
from openai import OpenAI

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.15  # 提高阈值
MIN_SPEECH_FRAMES = 15
SILENCE_FRAMES = 30  # 减少静音判断时间

def detect_speech(audio_chunk):
    volume = np.max(np.abs(audio_chunk))
    return volume > SILENCE_THRESHOLD

def ask_ai_model(text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    print(f"Model answer: {response.choices[0].message.content}")

def continuous_listen():
    p = pyaudio.PyAudio()

    # 使用默认输入设备
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("\n使用系统默认输入设备开始监听...(按Ctrl+C退出)")
    print(f"当前阈值: {SILENCE_THRESHOLD}")
    
    frames = []
    silent_frames = 0
    is_speaking = False
    
    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # 计算并显示音量
                volume = np.max(np.abs(audio_chunk))
                volume_bar = '=' * int(volume * 1000)
                print(f'\r音量: [{volume_bar:<50}] {volume:.6f}', end='', flush=True)
                
                if volume > SILENCE_THRESHOLD:
                    if not is_speaking:
                        print("\n检测到语音...")
                        is_speaking = True
                    frames.append(audio_chunk)
                    silent_frames = 0
                elif is_speaking:
                    frames.append(audio_chunk)
                    silent_frames += 1
                    
                    if silent_frames >= SILENCE_FRAMES:
                        if len(frames) >= MIN_SPEECH_FRAMES:
                            print("\n处理语音片段...")
                            audio = np.concatenate(frames)
                            yield audio
                        frames = []
                        silent_frames = 0
                        is_speaking = False
                        print("\n语音片段结束")
                
            except IOError as e:
                print(f"\n音频读取错误: {e}")
                continue
                
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
    )
    
    try:
        for audio in continuous_listen():
            res = model.generate(
                input=audio,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            text = rich_transcription_postprocess(res[0]["text"])
            if text:
                print("问:", text)
                ask_ai_model(text)
    except KeyboardInterrupt:
        print("\n程序已终止")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()