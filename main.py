import os
import threading

import telebot
from langchain_community.tools import HumanInputRun
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from telebot import StateMemoryStorage
from telebot.states import StatesGroup, State

from misc import get_groq_key, get_tg_token
from tools.spice_tools import spice_tool

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

state_storage = StateMemoryStorage()
bot = telebot.TeleBot(tg_token, parse_mode="Markdown", state_storage=state_storage)
llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)


class InputStates(StatesGroup):
    in_conversation = State()


@bot.message_handler(commands=['start'])
def send_start(message):
    bot.send_message(message.chat.id,
                     "*Привет!* Я бот генерации схем LTSpice. Я помогу тебе сгенерировать схему по текстовому или голосовому описанию.\n\nИнтерфейс бота очень похож на интерфейс ChatGPT:\n/new - создать новый чат\n/continue - продолжить старый чат")


@bot.message_handler(commands=['new'])
def start_new_conversation(message):
    agent = create_react_agent(
        llm,
        [
            HumanInputRun(
                prompt_func=get_prompt_func(bot, message.chat.id),
                input_func=get_input_func(message.chat.id)
            ),
            spice_tool
        ]
    )

    result = agent.invoke({
        "messages": [{"role": "system", "content": "Make analysis of circuit. "
                                                   "First of all, ask for circuit description like this: \"Привет! Предоставь мне, пожалуйста, описание схемы.\". "
                                                   "Then ask what to find like this: \"Отлично! Подскажи, что ты хочешь найти? Это может быть, например, напряжение на участке цепи.\", "
                                                   "then run circuit simulation. Interact with human on Russian language."}]
    })

    print(result)

    bot.send_message(message.chat.id, result["messages"][-1].content)


#@bot.message_handler(func=lambda message: True)
#def echo_all(message):
#    bot.send_message(message.chat.id, message.text)


def ask_prompt(bot: telebot.TeleBot, chat_id: int, text: str) -> None:
    bot.send_message(chat_id, text)


def get_prompt_func(bot: telebot.TeleBot, chat_id: int):
    return lambda text: ask_prompt(bot, chat_id, text)


user_responses = {}


def get_user_input(chat_id: int):
    event = threading.Event()
    user_responses[chat_id] = {"event": event, "response": None}

    while not event.wait(timeout=0.1):
        if chat_id not in user_responses:
            return None  # Пользователь отменил ввод

    response = user_responses[chat_id]["response"]
    del user_responses[chat_id]
    return response


def get_input_func(chat_id: int):
    return lambda: get_user_input(chat_id)


@bot.message_handler(func=lambda message: message.chat.id in user_responses)
def handle_response(message):
    chat_id = message.chat.id
    user_responses[chat_id]["response"] = message.text
    user_responses[chat_id]["event"].set()


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    file_name_ogg = f"{message.voice.file_id}.ogg"

    # Скачать голосовое сообщение
    downloaded_file = bot.download_file(file_path)
    with open(file_name_ogg, 'wb') as new_file:
        new_file.write(downloaded_file)

    with open(file_name_ogg, 'rb') as audio_file:
        bot.send_audio(message.chat.id, audio_file)

    os.remove(file_name_ogg)


print("Бот запущен!")
bot.polling(non_stop=True)
