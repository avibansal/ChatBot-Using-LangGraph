import os
from dotenv import load_dotenv
import streamlit as st

from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import Image, display

from langgraph.graph import StateGraph
from langgraph.constants import START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults

#Streamlit Title
st.title("ChatBot Using LangGraph")

# Load environment variables from .env file
load_dotenv()

# Access the variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#LLM model
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

#Tool
tool = TavilySearchResults(max_results=2)
llm_with_tools=llm.bind_tools([tool])

#LangFraph State
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

#Chatbot Function
def tool_calling_llm(state: MessagesState):
    ai_message = {"messages": [llm_with_tools.invoke(state["messages"])]}
    return ai_message

# graph builder
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([tool]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()


# For Displaying 
with st.expander("🔍 View LangGraph Structure"):
    st.image(graph.get_graph().draw_mermaid_png())

# Output
user_input = st.chat_input()
if user_input:
    messages = graph.invoke({"messages": user_input})
    for msg in messages["messages"]:
        st.chat_message(msg.type).markdown(msg.content)