# используемые библиотеки
# langgraph langchain langchain-groq langchain-community langchain-huggingface faiss-cpu pyTelegramBotAPI sentence-transformers spicelib unstructured pydub gTTS ffmpeg-python SpeechRecognition

import os

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from tg import start_tg_bot
from misc import get_groq_key, get_tg_token, get_rate_limiter

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

rate_limiter = get_rate_limiter()
model = "llama3-70b-8192"
llm = ChatGroq(model=model, temperature=0.5)
#llm_limited = ChatGroq(model=model, temperature=0.5, rate_limiter=rate_limiter)

system_message = SystemMessage("Ты инженер LTSpice. Ты умеешь составлять netlistы схем на основе запроса пользователя. Общайся с пользователем на русском языке.")

start_tg_bot(tg_token, llm, system_message)