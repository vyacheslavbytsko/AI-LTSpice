import os

from langchain_community.tools import HumanInputRun
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from misc import get_groq_key, get_tg_token, multiline_input
from tools.spice_tools import spice_tool

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)

agent = create_react_agent(
    llm,
    [HumanInputRun(input_func=multiline_input), spice_tool]
)

result = agent.invoke({
    "messages": [{"role": "system", "content": "Make analysis of circuit. "
                                               "First of all, ask for circuit description like this: \"Привет! Предоставь мне, пожалуйста, описание схемы.\". "
                                               "Then ask what to find like this: \"Отлично! Подскажи, что ты хочешь найти? Это может быть, например, напряжение на участке цепи.\", "
                                               "then run circuit simulation. Interact with human on Russian language."}]
})

print(result["messages"][-1].content)
