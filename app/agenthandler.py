from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Literal
import os
from retriever import tools,assistant,should_continue


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("openai_key")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found! Please set it in your .env file.")

print("✅ API key loaded successfully")

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    should_continue,
    {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "assistant")
memory=MemorySaver()
agent = builder.compile(checkpointer=memory)




def run_agent(user_input: str,thread_id: str = "test_session"):
    """
    Run the agent and display the conversation.
    """
    print(f"\n{'='*70}")
    print(f"👤 User: {user_input}")
    print(f"{'='*70}\n")


    result = agent.invoke(
        {"messages":[HumanMessage(content=user_input)]},
        config={"configurable":{"thread_id":thread_id}}
    )

    for message in reversed(result["messages"]):
        if isinstance(message,HumanMessage):
            continue
        elif isinstance(message,AIMessage):
            if message.tool_calls:
                print(f"🤖 Agent: [Calling tool: {message.tool_calls[0]['name']}]")
            else:
                return(f"🤖 Agent: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"🔧 Tool Result: {message.content[:100]}..." if len(message.content) > 100 else f"🔧 Tool Result: {message.content}")
    
    print(f"\n{'='*70}\n")

print("✅ Test function ready")


def get_conversation_history(session_id: str):
    state = agent.get_state(
        config={"configurable": {"thread_id": session_id}}
    )

    messages = state.values.get("messages", [])
    # print(messages)
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    return history