from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import pyaudio
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import os
import edge_tts
import asyncio

from playsound import playsound

# 加载 .env 文件
load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_KEY"), base_url="https://api.deepseek.com")

# 初始化文本转语音引擎
VOICE = "zh-CN-YatingNeural"

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.01  # 提高阈值
MIN_SPEECH_FRAMES = 15
SILENCE_FRAMES = 30  # 减少静音判断时间

def detect_speech(audio_chunk):
    volume = np.max(np.abs(audio_chunk))
    return volume > SILENCE_THRESHOLD

chat_history = [{"role": "system", "content": os.getenv("CALL_WORD")}]


async def play_audio(text):
    if not text.strip():  # Check if the text is empty
        print("没有接收到有效的文本。")
        return
    # zh-CN-XiaoxiaoNeural：中文（简体）女声
    # zh-CN-YunxiNeural：中文（简体）男声
    # zh-CN-XiaoyiNeural：中文（简体）女声
    # zh-TW-HsiaoChenNeural：中文（繁体）女声
    # zh-TW-YunJheNeural：中文（繁体）男声
    # zh-HK-HiuGaaiNeural：粤语女声
    # zh-HK-WanLungNeural：粤语男声
    # zh-CN-shaanxi-XiaoniNeural：陕西方言女声
    # zh-CN-liaoning-XiaobeiNeural：东北方言女声
    # zh-TW-HsiaoYuNeural：台湾口音女声
    # 选择适合女朋友的音色
    VOICE = "zh-CN-XiaoyiNeural"  # 选择音色
    communicate = edge_tts.Communicate(
        text,
        VOICE,
        rate='+0%',
        volume='+0%',
        pitch='+0Hz'
    )
    
    # 创建目录并保存音频到文件
    output_dir = "output_directory"  # 你想要的目录名
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    output_file = os.path.abspath(os.path.join(output_dir, "output.mp3"))
    
    # 使用 save 方法直接保存音频到文件
    await communicate.save(output_file)

    # 打印文件路径
    print(f"音频文件保存路径: {output_file}")

    # 检查音频文件是否存在
    if os.path.exists(output_file):
        print("音频文件已成功保存。")
        playsound(output_file)
    else:
        print("音频文件未能保存。")
        return  # 退出函数



def ask_ai_model(text):
    chat_history.append({"role": "user", "content": text})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=chat_history,
        stream=False
    )
    answer = response.choices[0].message.content
    print(f"Model answer: {answer}")
    
    if not answer.strip():  # Check if the answer is empty
        print("模型没有返回有效的答案。")
        return
    
    asyncio.run(play_audio(answer))


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






