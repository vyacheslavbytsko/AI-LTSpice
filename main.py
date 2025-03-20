import os
import threading

import telebot
from langchain_community.tools import HumanInputRun
from langchain_core.tools import create_retriever_tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from telebot import StateMemoryStorage
from telebot.custom_filters import StateFilter
from telebot.states import StatesGroup, State
from telebot.states.sync import StateContext, StateMiddleware
from telebot.types import Message, BotCommand

from misc import get_groq_key, get_tg_token, get_rate_limiter, make_netlists, get_netlists_descriptions, \
    get_split_netlists_descriptions, get_netlists_descriptions_vector_store, get_vector_store_as_retriever, \
    multiline_input, description_to_filenames_tool, filename_to_full_circuit_description_tool, filename_to_netlist_tool


groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

rate_limiter = get_rate_limiter()
llm = ChatGroq(model="llama3-70b-8192", temperature=1)
llm_limited = ChatGroq(model="llama3-70b-8192", temperature=1, rate_limiter=rate_limiter)

make_netlists()
netlists_descriptions = get_netlists_descriptions(llm_limited)
split_netlists_descriptions = get_split_netlists_descriptions(netlists_descriptions)
netlists_descriptions_vector_store = get_netlists_descriptions_vector_store(split_netlists_descriptions)

agent = create_react_agent(
    llm,
    [
        HumanInputRun(
            input_func=multiline_input
        ),
        description_to_filenames_tool(netlists_descriptions_vector_store),
        filename_to_netlist_tool()
    ], debug=True
)

messages = [{"role": "system", "content": "Ты инженер LTSpice. Спроси у пользователя, какую схему он хочет получить, и отправь netlist этой схемы. Используй доступные инструменты."}]


result = agent.invoke({
    "messages": messages
})

print(result["messages"][-1].content)

