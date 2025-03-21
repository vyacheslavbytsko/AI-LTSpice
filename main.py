import os

from langchain_community.tools import HumanInputRun
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from misc import get_groq_key, get_tg_token, get_rate_limiter, make_netlists, get_netlists_descriptions, \
    get_split_netlists_descriptions, get_netlists_descriptions_vector_store, \
    multiline_input, simple_circuits_description_to_filenames_tool, filename_to_netlist_tool


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
        # TODO: description_to_simple_circuits_descriptions_tool(),
        simple_circuits_description_to_filenames_tool(netlists_descriptions_vector_store),
        filename_to_netlist_tool(),
        # TODO: combine_netlists_tool(),
        # TODO: check_for_errors_tool()
    ], debug=True
)

messages = [{"role": "system", "content": "Ты инженер LTSpice. Спроси у пользователя, "
                                          "какую схему он хочет получить, раздели её на "
                                          "простые схемы, получи netlist'ы каждой, объедини "
                                          "netlist'ы, проверь netlist на ошибки и отправь "
                                          "этот файл пользователю. Используй доступные инструменты."}]


result = agent.invoke({
    "messages": messages
})

print(result["messages"][-1].content)

