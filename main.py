import os
import threading
import uuid

import telebot
from langchain_community.tools import HumanInputRun
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from telebot import StateMemoryStorage
from telebot.custom_filters import StateFilter
from telebot.states import StatesGroup, State
from telebot.states.sync import StateContext, StateMiddleware
from telebot.types import Message

from misc import get_groq_key, get_tg_token, rate_limiter
from tools.spice_tools import spice_tool

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

state_storage = StateMemoryStorage()
bot = telebot.TeleBot(tg_token, parse_mode="Markdown", state_storage=state_storage, use_class_middlewares=True)
llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)

user_inputs = {}
conversations = {}


class States(StatesGroup):
    in_conversation = State()


# Ниже работаем с Human Tool

def ask_prompt(bot: telebot.TeleBot, chat_id: int, text: str) -> None:
    bot.send_message(chat_id, text)


def get_prompt_func_for_chat_id(bot: telebot.TeleBot, chat_id: int):
    return lambda text: ask_prompt(bot, chat_id, text)


def get_user_input(chat_id: int):
    event = threading.Event()
    user_inputs[chat_id] = {"event": event, "input": None}

    while not event.wait(timeout=0.1):
        if chat_id not in user_inputs:
            return None  # Пользователь отменил ввод

    response = user_inputs[chat_id]["input"]
    del user_inputs[chat_id]
    return response


def get_input_func_for_chat_id(chat_id: int):
    return lambda: get_user_input(chat_id)


@bot.message_handler(func=lambda message: message.chat.id in user_inputs)
def handle_input(message):
    chat_id = message.chat.id
    user_inputs[chat_id]["input"] = message.text
    user_inputs[chat_id]["event"].set()


# Отвечаем на сообщения в переписке

def answer_in_conversation(message: Message, conversation_id: str):
    agent = create_react_agent(
        llm,
        [
            HumanInputRun(
                prompt_func=get_prompt_func_for_chat_id(bot, message.chat.id),
                input_func=get_input_func_for_chat_id(message.chat.id)
            ),
            spice_tool
        ]
    )

    result = agent.invoke({
        "messages": conversations[message.chat.id][conversation_id]
    })

    bot.send_message(message.chat.id, result["messages"][-1].content)

    num_of_messages_now = len(conversations[message.chat.id][conversation_id])

    conversations[message.chat.id][conversation_id].extend(result["messages"][num_of_messages_now:])


@bot.message_handler(commands=['new'])
def start_new_conversation(message: Message, state: StateContext):
    state.set(States.in_conversation)

    if message.chat.id not in conversations:
        conversations[message.chat.id] = {}

    conversation_id = str(uuid.uuid4())
    conversations[message.chat.id][conversation_id] = [{"role": "system", "content": "Make analysis of circuit. "
                                                                                     "First of all, ask for circuit description like this: \"Привет! Предоставь мне, пожалуйста, описание схемы.\". "
                                                                                     "Then ask what to find like this: \"Отлично! Подскажи, что ты хочешь найти? Это может быть, например, напряжение на участке цепи.\", "
                                                                                     "then run circuit simulation. Interact with human on Russian language."}]

    state.add_data(conversation_id=conversation_id)
    answer_in_conversation(message, conversation_id)


@bot.message_handler(func=lambda message: True, state=States.in_conversation)
def handle_conversation_message(message: Message, state: StateContext):
    with state.data() as data:
        conversations[message.chat.id][data.get("conversation_id")].append(
            {"role": "user", "content": message.text})
        answer_in_conversation(message, data.get("conversation_id"))


# Скоро будем работать с голосом

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


# Команда /start

@bot.message_handler(commands=['start'])
def send_start_message(message):
    bot.send_message(message.chat.id,
                     "*Привет!* Я бот генерации схем LTSpice. Я помогу тебе сгенерировать схему по текстовому или голосовому описанию.\n\nИнтерфейс бота очень похож на интерфейс ChatGPT:\n/new - создать новый чат\n/continue - продолжить старый чат")


# Запускаем бота

print("Бот запущен!")
bot.add_custom_filter(StateFilter(bot))
bot.setup_middleware(StateMiddleware(bot))
bot.polling(non_stop=True)
