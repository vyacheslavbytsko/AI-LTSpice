import glob
import math
import os.path
import pickle

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from spicelib.simulators.ltspice_simulator import LTspice


def round16(val):
    """Округляет значение до ближайшего числа, кратного 16."""
    return int(round(val / 16.0)) * 16


def get_pin_positions(center, pin_count, offset=20):
    """
    Возвращает список координат пинов компонента относительно центра.
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
    Подбирает символ, предопределённые окна и количество контактов.
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
        # Если в строке содержится слово "opamp", выбираем символ операционного усилителя
        if "opamp" in full_line.lower():
            mapping["symbol"] = "Opamps\\opamp"
            mapping["windows"] = []
            mapping["pin_count"] = 3
        else:
            mapping["symbol"] = "unknown"
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

def combine_netlists_tool(llm):
    def combine_netlists(netlists, description):
        response = llm.invoke([
            HumanMessage(
                "Объедини данные netlist'ы в один netlist, следуя предоставленному описанию схемы. "
                "Убедись, что соединения между элементами корректны и логичны. "
                "Если необходимо, добавь соединения для согласованности схемы.\n\n"
                f"Netlist'ы:\n{netlists}\n\nОписание:\n{description}"
            )
        ])
        return response.content

