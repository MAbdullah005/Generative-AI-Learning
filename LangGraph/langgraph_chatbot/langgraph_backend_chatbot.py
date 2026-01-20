
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm=ChatOllama(model='qwen2.5:7b',temperature=0.2)

# -------------------
# 2. Tools
# -------------------
# Tools
@tool(description="Search the web for real-time information using DuckDuckGo")
def web_search(query: str) -> str:
    return DuckDuckGoSearchRun(region="us-en").run(query)

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:

    """
    Perform a basic arithmetic operation a on two numbers.
    Supported operations: add, sub, mul, div
    """
    op_map = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div"
    }

    operation = op_map.get(operation, operation)

    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero"}
        result = first_num / second_num
    else:
        return {"error": "Unsupported operation"}

    return {"result": result}



@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=FJTGRV5AO4QXV69N"
    r = requests.get(url)
    return r.json()



tools = [web_search, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "Only call tools when explicitly required. "
                "Do NOT call tools for general questions."
            )
        )
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot_conv.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

def safe_tools_condition(state: ChatState):
    last_msg = state["messages"][-1]

    # Only go to tools if the model made a REAL tool call
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"

    return END


# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    safe_tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helpaer
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)