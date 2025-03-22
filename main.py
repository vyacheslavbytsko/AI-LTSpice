# используемые библиотеки
# langgraph langchain langchain-groq langchain-community langchain-huggingface faiss-cpu pyTelegramBotAPI sentence-transformers spicelib unstructured pydub gTTS ffmpeg-python SpeechRecognition

import os

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from tg import start_tg_bot
from misc import get_groq_key, get_tg_token, get_rate_limiter, make_netlists, get_netlists_descriptions, \
    get_split_netlists_descriptions, get_netlists_descriptions_vector_store, get_known_circuits_names_str

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

rate_limiter = get_rate_limiter()
model = "llama3-70b-8192"
llm = ChatGroq(model=model, temperature=0.5)
llm_limited = ChatGroq(model=model, temperature=0.5, rate_limiter=rate_limiter)

make_netlists()
known_circuits_names_str = get_known_circuits_names_str()
netlists_descriptions = get_netlists_descriptions(llm_limited)
split_netlists_descriptions = get_split_netlists_descriptions(netlists_descriptions)
netlists_descriptions_vector_store = get_netlists_descriptions_vector_store(split_netlists_descriptions)

system_message = SystemMessage("Ты инженер LTSpice. Спроси у пользователя, "
                               "какую схему он хочет получить, разбей эту схему на составляющие, "
                               "получи netlist'ы каждой схемы (у тебя для этого есть инструмент "
                               "simple_circuit_description_to_descriptions_and_filenames), "
                               "и если окажется, что netlist'ов из описания оказалось несколько, "
                               "объедини эти netlist'ы в один netlist (передавай в инструмент combine_netlists "
                               "base64 репрезентации netlist'ов полностью!!!), "
                               "потом примени к netlist'у изначальные параметры и получи final netlist, "
                               "преобразуй final netlist в .asc файл, отправь пользователю netlist при "
                               "помощи инструмента send_netlist_to_user, "
                               "отправь пользователю .asc файл при помощи инструмента send_asc_to_user, "
                               "затем напиши пользователю сообщение "
                               "в духе \"Ура, всё получилось!\". Точно удостоверься, что отправил netlist пользователю. Не забудь отправить netlist пользователю. "
                               "Используй доступные инструменты. Когда пользователь сообщает, что ты что-то "
                               "сделал неверно, не стесняйся использовать инструменты ещё раз. "
                               "Разговаривай с пользователем на русском языке.")

start_tg_bot(tg_token, llm_limited, netlists_descriptions_vector_store, system_message, known_circuits_names_str)