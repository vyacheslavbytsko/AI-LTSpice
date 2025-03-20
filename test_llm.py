import os

from langchain_community.tools import HumanInputRun
from langchain_core.tools import create_retriever_tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from misc import multiline_input, get_groq_key, get_circuits, get_split_circuits, get_vector_store, get_retriever

os.environ["GROQ_API_KEY"] = get_groq_key()
circuits = get_circuits()
split_circuits = get_split_circuits(circuits)
vector_store = get_vector_store(split_circuits)
retriever = get_retriever(vector_store)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

messages = [{"role": "system", "content": "Ты инженер, который проектирует LTspice схемы (.asc файлы). Спроси у пользователя, какую именно схему ему нужно создать, сгенерируй её и отправь её пользователю. Используй доступные инструменты для генерации схем. Спрашивай пользователя всё на русском."}]

agent = create_react_agent(
    llm,
    [
        HumanInputRun(
            input_func=multiline_input
        ),
        #create_retriever_tool(
        #    retriever,
        #    "circuits_description",
        #    "Searches and returns circuits",
        #),
    ], debug=True
)

result = agent.invoke({
    "messages": messages
})

print(result["messages"][-1].content)