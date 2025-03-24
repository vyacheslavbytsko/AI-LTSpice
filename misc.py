import os.path
import tempfile

import ffmpeg
from gtts import gTTS
from langchain_core.rate_limiters import InMemoryRateLimiter
from speech_recognition import Recognizer, AudioFile, UnknownValueError, RequestError
from telebot import TeleBot
from telebot.types import Message


def multiline_input() -> str:
    inputs = []
    while True:
        line = input()
        if line == "0":
            break
        inputs.append(line)
    return "\n".join(inputs)


def get_rate_limiter():
    return InMemoryRateLimiter(
        requests_per_second=0.05,  # Можно делать запрос только раз в 20 секунд
        check_every_n_seconds=10,  # Проверять, доступны ли токены каждые 10 с
        max_bucket_size=1,  # Контролировать максимальный размер всплеска запросов
    )


def get_groq_key() -> str:
    try:
        return open("groq_key.txt", "r").read().strip()
    except:
        raise Exception("Нужно создать файл groq_key.txt, в который вставить ключ Groq.")


def get_tg_token() -> str:
    try:
        return open("tg_token.txt", "r").read().strip()
    except:
        raise Exception("Нужно создать файл tg_token.txt, в который вставить токен телеграм бота.")


def convert_ogg_to_wav(ogg_path: str, wav_path: str):
    """Конвертирует .ogg файл в .wav с помощью ffmpeg."""
    (
        ffmpeg
        .input(ogg_path)
        .output(wav_path, format='wav')
        .run(overwrite_output=True)
    )

def convert_mp3_to_ogg(mp3_path: str, ogg_path: str):
    """Конвертирует .mp3 файл в .ogg с помощью ffmpeg."""
    (
        ffmpeg
        .input(mp3_path)
        .output(ogg_path, format='ogg', **{"c:a": "libopus", "b:a": "32k", "ar": "48000"})
        .run(overwrite_output=True)
    )

def voice_message_to_text(message: Message, bot: TeleBot) -> str:
    voice_info = bot.get_file(message.voice.file_id)
    voice_path = voice_info.file_path

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
        ogg_file_name = ogg_file.name
        downloaded_file = bot.download_file(voice_path)

        with open(ogg_file_name, 'wb') as f:
            f.write(downloaded_file)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_file_name = wav_file.name
            convert_ogg_to_wav(ogg_file_name, wav_file_name)

            text = speech_to_text(wav_file_name)

            os.remove(wav_file_name)
            os.remove(ogg_file_name)

            return text


def speech_to_text(filename: str) -> str:
    """Преобразует аудиофайл в текст."""
    recognizer = Recognizer()
    with AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text
    except UnknownValueError:
        return "Речь не распознана"
    except RequestError:
        return "Ошибка сервиса распознавания речи"


def text_to_speech(text: str) -> str:
    """Преобразует текст в аудиофайл и возвращает путь к нему."""
    tts = gTTS(text=text, lang='ru')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        return temp_audio_file.name


def text_to_voice_message(chat_id: int, bot: TeleBot, text: str):
    mp3_path = text_to_speech(text)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as voice_file:
        voice_path = voice_file.name
        convert_mp3_to_ogg(mp3_path, voice_path)
        with open(voice_path, 'rb') as audio_file:
            bot.send_voice(chat_id, audio_file)
            os.remove(voice_path)
            os.remove(mp3_path)

