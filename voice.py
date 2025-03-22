import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import ffmpeg

def convert_oga_to_wav(oga_path: str, wav_path: str):
    """Конвертирует .oga файл в .wav с помощью ffmpeg."""
    (
        ffmpeg
        .input(oga_path)
        .output(wav_path, format='wav')
        .run(overwrite_output=True)
    )

def speech_to_text(filename: str) -> str:
    """Преобразует аудиофайл в текст."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text
    except sr.UnknownValueError:
        return "Речь не распознана"
    except sr.RequestError:
        return "Ошибка сервиса распознавания речи"

def text_to_speech(text: str) -> str:
    """Преобразует текст в аудиофайл и возвращает путь к нему."""
    tts = gTTS(text=text, lang='ru')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        return temp_audio_file.name

def process_voice_message(voice_file_path: str) -> str:
    """Обрабатывает голосовое сообщение: конвертирует в .wav и распознает текст."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        wav_path = temp_wav_file.name

    # Конвертируем .oga в .wav
    convert_oga_to_wav(voice_file_path, wav_path)

    # Распознаем текст
    text = speech_to_text(wav_path)

    # Удаляем временный .wav файл
    os.remove(wav_path)

    return text