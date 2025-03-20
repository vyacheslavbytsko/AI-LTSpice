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
        #filename_to_full_circuit_description_tool()
    ], debug=True
)

messages = [{"role": "system", "content": "Ты инженер LTSpice. Спроси у пользователя, какую схему он хочет получить, и отправь netlist этой схемы. Используй доступные инструменты."}]


result = agent.invoke({
    "messages": messages
})

print(result["messages"][-1].content)

"""circuits = get_circuits()
split_circuits = get_split_circuits(circuits)
vector_store = get_vector_store(split_circuits)
retriever = get_retriever(vector_store)

#state_storage = StateMemoryStorage()
#bot = telebot.TeleBot(tg_token, parse_mode="Markdown", state_storage=state_storage, use_class_middlewares=True)


messages = [{"role": "system", "content": "Ты инженер, который проектирует LTspice схемы (.asc файлы). Спроси у пользователя, какую именно схему ему нужно создать, сгенерируй её и отправь её пользователю. Используй доступные инструменты для генерации схем."}]

agent = create_react_agent(
    llm,
    [
        HumanInputRun(
            input_func=multiline_input
        ),
        create_retriever_tool(
            retriever,
            "circuits description",
            "Searches and returns circuits",
        ),
        spice_tool
    ]#, debug=True
)

result = agent.invoke({
    "messages": messages
})

print(result["messages"][-1].content)

user_inputs = {}


class States(StatesGroup):
    in_conversation = State()


# Ниже работаем с Human Tool

def ask_prompt(bot: telebot.TeleBot, chat_id: int, text: str) -> None:
    bot.send_message(chat_id, text)


def get_prompt_func_for_chat_id(bot: telebot.TeleBot, chat_id: int):
    return lambda text: ask_prompt(bot, chat_id, text)


def get_user_input(chat_id: int):
    print("Ждём ответа пользователя")
    event = threading.Event()
    user_inputs[chat_id] = {"input": None, "event": event}

    while not event.wait(timeout=0.1):
        pass

    response = user_inputs[chat_id]["input"]
    if response is None:
        print("Ответа не получили. Говорим ИИ не использовать больше никакой инструментарий")
        return "Stop the operation, don't ask for human input again, don't use any of the tools."
    del user_inputs[chat_id]
    return response


def get_input_func_for_chat_id(chat_id: int):
    return lambda: get_user_input(chat_id)


@bot.message_handler(func=lambda message: message.chat.id in user_inputs and message.text not in ["/new", "/end"])
def handle_input(message: Message, state: StateContext):
    print(f"Получили сообщение-ответ, будучи в диалоге.")
    chat_id = message.chat.id
    if state.get() is None:
        user_inputs[chat_id]["event"].set()
        return
    user_inputs[chat_id]["input"] = message.text
    user_inputs[chat_id]["event"].set()


# Обрабатываем события в диалогах

def answer_in_conversation(message: Message, state: StateContext):
    agent = create_react_agent(
        llm,
        [
            HumanInputRun(
                prompt_func=get_prompt_func_for_chat_id(bot, message.chat.id),
                input_func=get_input_func_for_chat_id(message.chat.id)
            ),
            create_retriever_tool(
                retriever,
                "circuits description",
                "Searches and returns circuits",
            ),
            spice_tool
        ], debug=True
    )

    with state.data() as data:
        result = agent.invoke({
            "messages": data["messages"]
        })

        if state.get() is not None:
            bot.send_message(message.chat.id, result["messages"][-1].content)
            num_of_messages_now = len(data["messages"])
            data["messages"].extend(result["messages"][num_of_messages_now:])
        else:
            print("Поскольку диалог завершён, не отправляем последнее сообщение (скорее всего оно о том, что ИИ больше не будет использовать инструментарий)")

@bot.message_handler(commands=['new'])
def start_new_conversation(message: Message, state: StateContext):
    print("Получили команду /new")

    if state.get() is not None:
        bot.send_message(message.chat.id, "Сначала закончи прошлый диалог - /end.")
        return

    state.set(States.in_conversation)

    with state.data() as data:
        data["messages"] = [{"role": "system", "content": "Ты инженер, который проектирует LTspice схемы (.asc файлы). Спроси у пользователя, какую именно схему ему нужно создать, сгенерируй её и отправь её пользователю. Используй доступные инструменты для генерации схем."}]
        #data["messages"] = [{"role": "system", "content": "Ask for circuit description like this: \"Привет! Расскажи, пожалуйста, какую схему ты хочешь создать.\". "
        #                                                  "Then search similar circuits, analyze their contents, understand, из чего состоят эти записи, и создай свою circuit based on the provided description. Then send circuit contents to the user, because the user doesn't see Tool Messages. Interact in Russian."}]

    answer_in_conversation(message, state)


@bot.message_handler(func=lambda message: message.text not in ["/new", "/end"], state=States.in_conversation)
def handle_conversation_message(message: Message, state: StateContext):
    print(f"Получили сообщение, будучи в диалоге. {state.get()}")
    with state.data() as data:
        data["messages"].append({"role": "user", "content": message.text})
    answer_in_conversation(message, state)


@bot.message_handler(commands=['end'], state=States.in_conversation)
def end_conversation(message: Message, state: StateContext):
    print(f"Получили команду /end, будучи в диалоге.")
    if message.chat.id in user_inputs.keys():
        user_inputs[message.chat.id]["event"].set()
    state.delete()
    bot.send_message(message.chat.id, "Диалог завершён. Чтобы начать новый, напиши /new.")


# Скоро будем работать с голосом

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    print("Получили голосовое сообщение")
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
                     "*Привет!* Я бот генерации схем LTSpice. "
                     "Я помогу тебе сгенерировать схему по текстовому "
                     "или голосовому описанию.\n\nЧат-бот работает по "
                     "подобию ChatGPT, то есть, общение происходит "
                     "в диалогах. Чтобы начать новый диалог с ботом, "
                     "пропиши /new. Чтобы закончить диалог, напиши /end.")


# Запускаем бота

bot.add_custom_filter(StateFilter(bot))
bot.setup_middleware(StateMiddleware(bot))
print("Бот запущен!")
bot.set_my_commands([
    BotCommand("start", "Стартовое сообщение"),
    BotCommand("new", "Начать новый диалог"),
    BotCommand("end", "Завершить диалог")
])
bot.infinity_polling()"""
