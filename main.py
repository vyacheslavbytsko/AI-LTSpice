from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
from pydub.utils import which
import wave
import json
import os

# Указываем путь к ffmpeg
AudioSegment.converter = which("ffmpeg")

# Отключаем лишние логи
SetLogLevel(0)

# Проверяем наличие модели
if not os.path.exists("model"):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

# Функция для конвертации MP3 в WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Одноканальный 16 kHz
    audio.export(wav_path, format="wav")

# Функция для распознавания речи
def transcribe_audio(wav_path, model_path="/Users/polinapopova/AI-LTSpice/model"):
    model_path = "/Users/polinapopova/AI-LTSpice/model"
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "")

# Основная функция
def main():
    mp3_file = os.path.join(os.getcwd(), "bobr.mp3") # Укажи путь к своему MP3 файлу
    wav_file = "converted.wav"
    text_file = "transcription.txt"

    print("Конвертируем MP3 в WAV...")
    convert_mp3_to_wav(mp3_file, wav_file)

    print("Распознаем речь...")
    text = transcribe_audio(wav_file)

    print("Сохраняем результат...")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)

    print("Готово! Распознанный текст сохранен в", text_file)

if __name__ == "__main__":
    main()
