import glob
import math
import os.path
import pickle
import tempfile

import ffmpeg
from gtts import gTTS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from speech_recognition import Recognizer, AudioFile, UnknownValueError, RequestError
from spicelib.simulators.ltspice_simulator import LTspice
from telebot import TeleBot
from telebot.types import Message


def round16(val):
    """Округляет значение до ближайшего числа, кратного 16."""
    return int(round(val / 16.0)) * 16


def get_pin_positions(center, pin_count, offset=20):
    """
    Возвращает список координат для пинов компонента относительно центра.
    Для 2 контактов – слева и справа; для 3,4 – предопределённые позиции;
    для остальных – равномерное распределение по окружности.
    Все координаты округляются до кратных 16.
    """
    cx, cy = center
    if pin_count == 2:
        pos = [(cx - offset, cy), (cx + offset, cy)]
    elif pin_count == 3:
        pos = [(cx - offset, cy), (cx, cy + offset), (cx + offset, cy)]
    elif pin_count == 4:
        pos = [(cx - offset, cy), (cx, cy + offset), (cx + offset, cy), (cx, cy - offset)]
    else:
        pos = []
        for i in range(pin_count):
            angle = 2 * math.pi * i / pin_count
            pos.append((cx + offset * math.cos(angle), cy + offset * math.sin(angle)))
    return [(round16(x), round16(y)) for x, y in pos]


def get_default_mapping(inst_name, full_line):
    """
    Возвращает словарь с настройками для компонента по его имени.
    Для универсальных схем подбираются:
      V*: voltage (2 контакта, оконные настройки)
      R*: res (2 контакта, оконные настройки)
      C*: cap (2 контакта)
      J*: njf (3 контакта)
      Q*: npn (3 контакта)
      X*: если содержит "opamp" – Opamps\opamp (3 контакта), иначе unknown (2 контакта)
      D*: для диодов – по умолчанию symbol "diode" (2 контакта)
    """
    mapping = {}
    if inst_name.startswith("V"):
        mapping["symbol"] = "voltage"
        mapping["windows"] = ["WINDOW 123 24 124 Left 2", "WINDOW 39 0 0 Left 2"]
        mapping["pin_count"] = 2
        if "AC" in full_line:
            mapping["extra"] = "SYMATTR Value2 AC 1"
    elif inst_name.startswith("R"):
        mapping["symbol"] = "res"
        mapping["windows"] = ["WINDOW 0 0 56 VBottom 2", "WINDOW 3 32 56 VTop 2"]
        mapping["pin_count"] = 2
    elif inst_name.startswith("C"):
        mapping["symbol"] = "cap"
        mapping["windows"] = []
        mapping["pin_count"] = 2
    elif inst_name.startswith("J"):
        mapping["symbol"] = "njf"
        mapping["windows"] = []
        mapping["pin_count"] = 3
    elif inst_name.startswith("Q"):
        mapping["symbol"] = "npn"
        mapping["windows"] = []
        mapping["pin_count"] = 3
    elif inst_name.startswith("X"):
        if "opamp" in full_line.lower():
            mapping["symbol"] = "Opamps\\opamp"
            mapping["windows"] = []
            mapping["pin_count"] = 3
        else:
            mapping["symbol"] = "unknown"
            mapping["windows"] = []
            mapping["pin_count"] = 2
    elif inst_name.startswith("D"):
        # Универсальный диод – для обычного диода можно задать symbol "diode"
        mapping["symbol"] = "diode"
        mapping["windows"] = []
        mapping["pin_count"] = 2
    else:
        mapping["symbol"] = "unknown"
        mapping["windows"] = []
        mapping["pin_count"] = 2
    return mapping


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
        requests_per_second=0.2,  # Можно делать запрос только раз в 5 секунд
        check_every_n_seconds=2,  # Проверять, доступны ли токены каждые 2 с
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


def get_vector_store_as_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        k=1,
        score_threshold=None,
    )


def make_netlists() -> None:
    for filepath in glob.iglob(os.path.join("circuits", "**", "*.asc"), recursive=True):
        if not os.path.isfile(filepath.removesuffix(".asc") + ".net"):
            print(f"Генерируем netlist для файла {filepath}")
            LTspice.create_netlist(filepath)


def get_netlists_descriptions(llm: ChatGroq):
    result = []
    for filepath in glob.iglob(os.path.join("circuits", "**", "*.net"), recursive=True):
        description_filepath = filepath.removesuffix(".net") + ".desc.txt"
        if not os.path.isfile(description_filepath):
            print(f"Генерируем русское описание для файла {filepath}")

            result = llm.invoke(
                [HumanMessage(
                    "На основе названия файла и его содержимого поясни, "
                    "что именно за схема spice представлена, и из чего она состоит. "
                    "Свой ответ напиши на русском языке.\n\n"
                    f"Название файла: {filepath}\n\n"
                    f"Содержимое файла:\n{open(filepath).read()}"
                )]
            )

            with open(description_filepath, "w") as f:
                f.write(result.content)
        with open(description_filepath, 'r') as f:
            result.append(Document(page_content=f.read(), metadata={"netlist_filename": filepath, "description_filename": description_filepath}))
    return result


def get_split_netlists_descriptions(netlists_descriptions):
    if not os.path.isfile("split_netlists_descriptions.pkl"):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "."]
        )

        split_netlists_descriptions_texts = [splitter.split_text(netlists_description.page_content) for netlists_description in netlists_descriptions]

        split_netlists_descriptions = [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc, chunks in zip(netlists_descriptions, split_netlists_descriptions_texts)
            for chunk in chunks
        ]

        with open("split_netlists_descriptions.pkl", "wb") as f:
            pickle.dump(split_netlists_descriptions, f)

    with open("split_netlists_descriptions.pkl", "rb") as f:
        split_netlists_descriptions = pickle.load(f)
        print(f"len(split_netlists_descriptions) = {len(split_netlists_descriptions)}")
        return split_netlists_descriptions


def get_netlists_descriptions_vector_store(split_netlists_descriptions) -> FAISS:
    emb_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    if not os.path.isdir("netlists_descriptions_vector_store"):
        vector_store = FAISS.from_documents(
            split_netlists_descriptions, emb_model
        )

        vector_store.save_local("netlists_descriptions_vector_store")
    return FAISS.load_local("netlists_descriptions_vector_store", emb_model, allow_dangerous_deserialization=True)


def get_known_circuits_names_str():
    filepaths = []
    for filepath in glob.iglob(os.path.join("circuits", "**", "*.asc"), recursive=True):
        filepaths.append(filepath.split("/")[-1].removesuffix(".asc"))
    return "\n".join(filepaths)


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

