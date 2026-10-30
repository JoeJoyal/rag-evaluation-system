import rag1.tools as t1
from prompts import system_agent_prompt
from langchain_openai import ChatOpenAI
from tools import collecting_info
from langchain_core.agents import create_agent
from langchain_core.messages import HumanMessage
from objects import Answer

llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    response_format=Answer
)

def ask(question) -> dict:
    """Answer the user's question about the pdf"""

    t1.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer": structured_response.answer,
        "context": structured_response.context
    }

print(ask(""))

