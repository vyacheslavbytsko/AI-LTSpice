import threading
import uuid
from functools import partial

from langchain_community.tools import HumanInputRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from telebot import TeleBot, StateMemoryStorage
from telebot.custom_filters import StateFilter
from telebot.states import StatesGroup, State
from telebot.states.sync import StateContext, StateMiddleware
from telebot.types import Message, BotCommand

from tools import netlist_to_asc_tool, simple_circuits_description_to_filenames_tool, filename_to_netlist_tool, \
    description_to_simple_circuits_descriptions_tool, combine_netlists_tool


class States(StatesGroup):
    in_conversation = State()


user_inputs = {}


# Работа с /start

def send_start_message(message: Message, bot: TeleBot):
    bot.send_message(message.chat.id,
                     "*Привет!* Я бот генерации схем LTSpice. "
                     "Я помогу тебе сгенерировать схему по текстовому "
                     "или голосовому описанию.\n\nЧат-бот работает по "
                     "подобию ChatGPT, то есть, общение происходит "
                     "в диалогах. Чтобы начать новый диалог с ботом, "
                     "пропиши /new. Чтобы закончить диалог, напиши /end.", parse_mode="markdown")


# Работа с диалогом

def start_conversation(message: Message, bot: TeleBot, state: StateContext,
                       llm: ChatGroq, netlists_descriptions_vector_store: FAISS, system_message: SystemMessage,
                       known_circuits_names_str: str):
    print("Получили команду /new")

    if state.get() is not None:
        bot.send_message(message.chat.id, "_Сначала закончи прошлый диалог - /end._", parse_mode="markdown")
        return

    state.set(States.in_conversation)

    with state.data() as data:
        data["messages"] = [system_message]
        data["id"] = str(uuid.uuid4())

    answer_in_conversation(message, bot, llm, netlists_descriptions_vector_store, known_circuits_names_str, state)


def answer_in_conversation(message: Message, bot: TeleBot,
                           llm: ChatGroq, netlists_descriptions_vector_store: FAISS,
                           known_circuits_names_str: str, state: StateContext):
    with state.data() as data:
        print(f"chat {message.chat.id}: перед ответом LLM было {len(data["messages"])} сообщений")
        agent = create_react_agent(
            llm,
            [
                HumanInputRun(
                    prompt_func=get_prompt_func_for_chat_id(bot, message.chat.id),
                    input_func=get_input_func_for_chat_id(message.chat.id)
                ),
                description_to_simple_circuits_descriptions_tool(llm, known_circuits_names_str),
                simple_circuits_description_to_filenames_tool(netlists_descriptions_vector_store),
                filename_to_netlist_tool(),
                combine_netlists_tool(llm),
                # TODO: check_for_errors_tool()
                netlist_to_asc_tool()
            ], debug=True
        )

        result = agent.invoke({
            "messages": data["messages"]
        })

        if state.get() is not None:
            bot.send_message(message.chat.id, result["messages"][-1].content)
            num_of_messages_now = len(data["messages"])
            data["messages"].extend(result["messages"][num_of_messages_now:])
            print(f"chat {message.chat.id}: после ответа LLM стало {len(data["messages"])} сообщений")
        else:
            print(
                "Поскольку диалог завершён, не отправляем последнее сообщение (скорее всего оно о том, что ИИ больше не будет использовать инструментарий)")


def handle_conversation_message(message: Message, bot: TeleBot,
                                llm: ChatGroq, netlists_descriptions_vector_store: FAISS,
                                known_circuits_names_str: str, state: StateContext):
    print(f"Получили сообщение, будучи в диалоге. {state.get()}")
    with state.data() as data:
        data["messages"].append({"role": "user", "content": message.text})
    answer_in_conversation(message, bot, llm, netlists_descriptions_vector_store, known_circuits_names_str, state)


def end_conversation(message: Message, bot: TeleBot, state: StateContext):
    print(f"Получили команду /end, будучи в диалоге.")
    if message.chat.id in user_inputs.keys():
        user_inputs[message.chat.id]["event"].set()
    state.delete()
    bot.send_message(message.chat.id, "_Диалог завершён. Чтобы начать новый, напиши /new._", parse_mode="markdown")


# Работа с HumanInputRun

def ask_prompt(bot: TeleBot, chat_id: int, text: str) -> None:
    bot.send_message(chat_id, text)


def get_prompt_func_for_chat_id(bot: TeleBot, chat_id: int):
    return lambda text: ask_prompt(bot, chat_id, text)


def get_user_input(chat_id: int):
    print("Ждём ответа пользователя")
    event = threading.Event()
    user_inputs[chat_id] = {"input": None, "event": event}

    while not event.wait(timeout=1):
        print(f"Ждём... {event}")

    response = user_inputs[chat_id]["input"]
    del user_inputs[chat_id]
    if response is None:
        print("Ответа не получили. Говорим ИИ не использовать больше никакой инструментарий")
        return "Stop the operation, don't ask for human input again, don't use any of the tools."
    return response


def get_input_func_for_chat_id(chat_id: int):
    return lambda: get_user_input(chat_id)


def handle_input(message: Message, bot: TeleBot, state: StateContext):
    print(f"Получили сообщение-ответ, будучи в диалоге.")
    chat_id = message.chat.id
    if state.get() is None:
        user_inputs[chat_id]["event"].set()
        return
    user_inputs[chat_id]["input"] = message.text
    user_inputs[chat_id]["event"].set()


# Регистрация хэндлеров

def register_handlers(bot, llm, netlists_descriptions_vector_store, system_message, known_circuits_names_str):
    bot.register_message_handler(partial(send_start_message, bot=bot), commands=['start'])
    bot.register_message_handler(partial(start_conversation, bot=bot, llm=llm,
                                         netlists_descriptions_vector_store=netlists_descriptions_vector_store,
                                         system_message=system_message,
                                         known_circuits_names_str=known_circuits_names_str), commands=['new'])
    bot.register_message_handler(partial(handle_conversation_message, bot=bot, llm=llm,
                                         netlists_descriptions_vector_store=netlists_descriptions_vector_store,
                                         known_circuits_names_str=known_circuits_names_str),
                                 func=lambda message: message.chat.id not in user_inputs and message.text not in [
                                     "/new", "/end"], state=States.in_conversation)
    bot.register_message_handler(partial(end_conversation, bot=bot), commands=['end'], state=States.in_conversation)
    bot.register_message_handler(partial(handle_input, bot=bot),
                                 func=lambda message: message.chat.id in user_inputs and message.text not in ["/new",
                                                                                                              "/end"])


# Запуск бота

def start_tg_bot(tg_token, llm, netlists_descriptions_vector_store, system_message, known_circuits_names_str):
    state_storage = StateMemoryStorage()
    bot = TeleBot(tg_token, state_storage=state_storage, use_class_middlewares=True)

    register_handlers(bot, llm, netlists_descriptions_vector_store, system_message, known_circuits_names_str)

    bot.add_custom_filter(StateFilter(bot))
    bot.setup_middleware(StateMiddleware(bot))
    bot.set_my_commands([
        BotCommand("start", "Стартовое сообщение"),
        BotCommand("new", "Начать новый диалог"),
        BotCommand("end", "Завершить диалог")
    ])
    print("Бот запущен!")
    bot.infinity_polling()
