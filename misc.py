import glob
import os.path
import pickle

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool, Tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from spicelib.simulators.ltspice_simulator import LTspice


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
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
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


def description_to_filenames_tool(vector_store: FAISS):

    def get_relevant_filenames_and_descriptions(query):
        retriever = vector_store.as_retriever(
            search_type="similarity",
            k=3,
            score_threshold=None,
        )

        relevant_descriptions = retriever.invoke(query)

        relevant_descriptions_text = ""
        for relevant_description in relevant_descriptions:
            relevant_descriptions_text += f"Filename \"{relevant_description.metadata["description_filename"]}\": \"{relevant_description.page_content}\"\n\n"

        return f"Here are three filenames and descriptions which fit the best to the description you provided:\n\n{relevant_descriptions_text}Use them to choose filename to get full description of the circuit."

    return Tool(
        name="description_to_filenames",
        description="Searches and returns filenames and incomplete descriptions of circuits which fits best to the description provided by human to then get full description using another tool.",
        func=lambda query: get_relevant_filenames_and_descriptions(query)
    )


def filename_to_full_circuit_description_tool():
    def get_content(query):
        return f"Here is the full description of circuit:\n\n\"{open(query).read()}\""
    return Tool(
        name="filename_to_full_circuit_description",
        description="Returns the full description of circuit based on the filename of description.",
        func=lambda query: get_content(query)
    )
