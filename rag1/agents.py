import tools as t1
from prompts import system_agent_prompt
from langchain_groq import ChatGroq
from tools import collecting_info
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from objects import Answer

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    response_format=Answer
)

def ask(question: str) -> dict:
    """Answer the user's question about the PDF."""

    # Reset previously retrieved context
    t1.last_retrieved_context = []

    # Invoke the agent
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        }
    )

    # Extract structured response
    structured_response = result["structured_response"]

    return {
        "answer": structured_response.answer,
        "context": structured_response.context
    }


# Test the agent
print(ask("What are Joe's technical skills?"))


