from duckduckgo_search import DDGS
from loader import vectorstore
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Literal
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("openai_key")

if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found! Please set it in your .env file.")

print("✅ API key loaded successfully")


@tool
def retrieve_tax_documents(query: str) -> str:
    """
    Retrieve official Nigerian tax policy documents.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 12}
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant official documents found."

    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'Official Document')}\n{doc.page_content}"
        for doc in docs
    )


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.Use this toool when you need to perform calculations.

    Args:
        expression: A mathematical expression like "2 + 2" or "15*37"01_Introduction_to_LangGraph.ipynb

    Returns:
        The calculated result as a string

    Examples:
    - "15% of 2 million naira" returns "300,000"
    - "2 + 2" returns "4"
    - "100 / 5" returns "20.0"
    - "2 ** 10" returns "1024"

    """
    try:
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception as e:
        return f"Error calculating: {str(e)}"


print("Calculator tool created")


TRUSTED_DOMAINS = [
    "oecd.org",
    "imf.org",
    "worldbank.org",
    "gov.za",
    "irs.gov",
    "firs.gov.ng",
    "finance.gov.ng"
]


@tool
def restricted_policy_search(query: str) -> str:
    """
    Search ONLY trusted government and policy websites
    for legal and policy comparisons.
    """
    domain_filter = " OR ".join([f"site:{d}" for d in TRUSTED_DOMAINS])
    search_query = f"{query} {domain_filter}"

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(search_query, max_results=5):
            results.append(
                f"Title: {r['title']}\n"
                f"Source: {r['href']}\n"
                f"Snippet: {r['body']}\n"
            )

    if not results:
        return "No trusted sources found."

    return "\n\n".join(results)


# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=openai_api_key
)

print(f"✅ LLM initialized: {llm.model_name}")


system_prompt = SystemMessage(content="""You are a helpful assistant with access to a document retrieval tool.
You are TaxBotNG, an AI assistant helping Nigerians understand the 2024 Nigerian Tax Reform Bills.

RULES:
DO NOT retrieve for:
- Greetings: "Hello", "Hi", "How are you"
- Questions about your capabilities: "What can you help with?", "What do you do?"
- If the user is asking for information (not just chatting), retrieve first.
- Use retrieval ONLY for questions about tax policy, VAT, PAYE, businesses, or state revenue.
- NEVER guess or hallucinate.
- If documents do not contain the answer, say so.
- Explain answers in simple Nigerian-friendly English.
- Cite sources when retrieval is used.
- Use the calculator tools when there is need to calculate tax for user and amount they will pay
DO NOT retrieve for greetings, math, dates, or general knowledge.
""")


tools = [retrieve_tax_documents, calculator, restricted_policy_search]
llm_with_tools = llm.bind_tools(tools)


def assistant(state: MessagesState):
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "__end__"
