import os

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from misc import get_groq_key, rate_limiter, get_tg_token

groq_key, tg_token = get_groq_key(), get_tg_token()
os.environ["GROQ_API_KEY"] = groq_key

llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
#llm = ChatGroq(model="llama3-70b-8192", temperature=0.3, rate_limiter=rate_limiter)

tools = load_tools(
    ["human"],
    llm=llm,
)

agent = create_react_agent(
    llm,
    tools
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Tell me something about Moscow."}]
})

print(result["messages"][-1].content)