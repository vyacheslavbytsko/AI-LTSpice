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

from misc import get_rate_limiter, get_groq_key, get_tg_token
from ltspice_code_gen import process_circuit_description  

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

rate_limiter = get_rate_limiter()
llm = ChatGroq(model="llama3-70b-8192", temperature=1)

bot = telebot.TeleBot(tg_token, parse_mode="Markdown", use_class_middlewares=True)


class States(StatesGroup):
    in_conversation = State()


def handle_circuit_request(message: Message, state: StateContext):
    description = message.text
    bot.send_message(message.chat.id, "Генерирую схему...")
    asc_file, simulation_result = process_circuit_description(description, llm)
    bot.send_document(message.chat.id, open(asc_file, "rb"))
    bot.send_message(message.chat.id, f"Результат симуляции: {simulation_result}")



@bot.message_handler(commands=['new'])
def start_new_conversation(message: Message, state: StateContext):
    state.set(States.in_conversation)
    bot.send_message(message.chat.id, "Введите описание схемы для генерации.")


@bot.message_handler(func=lambda msg: True, state=States.in_conversation)
def handle_conversation_message(message: Message, state: StateContext):
    handle_circuit_request(message, state)
    state.delete()


@bot.message_handler(commands=['start'])
def send_start_message(message):
    bot.send_message(message.chat.id, "*Привет!* Опиши схему, и я сгенерирую её для LTSpice.")


bot.add_custom_filter(StateFilter(bot))
bot.setup_middleware(StateMiddleware(bot))

bot.set_my_commands([
    BotCommand("start", "Стартовое сообщение"),
    BotCommand("new", "Начать новый диалог")
])

print("Бот запущен!")
bot.infinity_polling()
