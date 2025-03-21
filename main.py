# используемые библиотеки
# langgraph langchain langchain-groq langchain-community langchain-huggingface faiss-cpu pyTelegramBotAPI sentence-transformers spicelib unstructured

import os

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from tg import start_tg_bot
from misc import get_groq_key, get_tg_token, get_rate_limiter, make_netlists, get_netlists_descriptions, \
    get_split_netlists_descriptions, get_netlists_descriptions_vector_store

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

rate_limiter = get_rate_limiter()
model = "llama-3.3-70b-versatile"
llm = ChatGroq(model=model, temperature=1)
llm_limited = ChatGroq(model=model, temperature=1, rate_limiter=rate_limiter)

make_netlists()
netlists_descriptions = get_netlists_descriptions(llm_limited)
split_netlists_descriptions = get_split_netlists_descriptions(netlists_descriptions)
netlists_descriptions_vector_store = get_netlists_descriptions_vector_store(split_netlists_descriptions)

system_message = SystemMessage("Ты инженер LTSpice. Спроси у пользователя, "
                               "какую схему он хочет получить, раздели её на "
                               "простые схемы, получи netlist'ы каждой, объедини "
                               "netlist'ы, проверь netlist на ошибки и отправь "
                               "этот файл пользователю. Используй доступные инструменты.")

start_tg_bot(tg_token, llm, netlists_descriptions_vector_store, system_message)