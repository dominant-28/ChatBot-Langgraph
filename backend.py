# ================= Backend =================
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3, os, requests
from dotenv import load_dotenv

# ---------------- Load environment ----------------
load_dotenv()
API_KEY = os.environ.get("API_KEY")
os.environ["GOOGLE_API_KEY"] = API_KEY

# ---------------- Initialize model ----------------
Model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# =================== Tools ===================
search_tool = DuckDuckGoSearchRun(region="us-en")
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Performs arithmetic: add, sub, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero not allowed."}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation."}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_weather_info(place: str) -> dict:
    """Fetches weather when we want to know the weather of any location."""
    api_key = os.environ.get("weather_api_key")
    if not api_key:
        raise ValueError("Missing WEATHER_API_KEY in environment variables.")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={place}"
    data = requests.get(url).json()
    return {
        "location": data.get("location", {}).get("name"),
        "country": data.get("location", {}).get("country"),
        "temperature": data.get("current", {}).get("temperature"),
        "humidity": data.get("current", {}).get("humidity"),
        "description": data.get("current", {}).get("weather_descriptions", []),
    }

tools = [calculator, get_weather_info, search_tool]
Model_with_tools = Model.bind_tools(tools)

# =================== Chat Graph ===================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    Name: str=""

def chat_node(state: ChatState):
    """LLM node that may respond or call tools."""
    messages = state["messages"]
    result: AIMessage = Model_with_tools.invoke(messages)
    output={"messages": [result]}
    if not state.get("Name"):
        output["Name"] = generate_conversation_name_with_llm(messages + [result])
    return output

tool_node = ToolNode(tools)

# ---------------- SQLite & Graph ----------------
connection = sqlite3.connect("ChatBot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")
chatbot = graph.compile(checkpointer=checkpointer)

# ---------------- Utility ----------------
def retrieve_all_threads() -> list[str]:
    return list({cp.config["configurable"]["thread_id"] for cp in checkpointer.list(None)})

def generate_conversation_name_with_llm(messages: list[BaseMessage]) -> str:
   
    snippet = "\n".join([m.content for m in messages[:3]])
    
    prompt = f"Generate a short, catchy title for this conversation (max 5 words):\n{snippet}"

    response: AIMessage = Model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()