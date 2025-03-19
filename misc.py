import os.path
import pickle

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# пока не используем
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.03,  # Можно делать запрос только раз в 30 секунд
    check_every_n_seconds=10,  # Проверять, доступны ли токены каждые 10 с
    max_bucket_size=1,  # Контролировать максимальный размер всплеска запросов
)


def get_groq_key() -> str:
    return open("groq_key.txt", "r").read().strip()


def get_tg_token() -> str:
    return open("tg_token.txt", "r").read().strip()


def get_split_circuits():
    if not os.path.isfile("split_circuits.pkl"):
        loader = DirectoryLoader("circuits", glob="**/*.asc")
        circuits_files = loader.load()

        circuits_contents = [file.page_content for file in circuits_files]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

        split_circuits = splitter.create_documents(circuits_contents)
        with open("split_circuits.pkl", "wb") as f:
            pickle.dump(split_circuits, f)

    with open("split_circuits.pkl", "rb") as f:
        split_circuits = pickle.load(f)
        print(f"len(split_circuits) = {len(split_circuits)}")
        return split_circuits


def get_vector_store(split_circuits):
    emb_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    if not os.path.isdir("vector_store"):
        vector_store = FAISS.from_documents(
            split_circuits, emb_model
        )

        vector_store.save_local("vector_store")
    return FAISS.load_local("vector_store", emb_model, allow_dangerous_deserialization=True)


def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        k=3,
        score_threshold=None,
    )
