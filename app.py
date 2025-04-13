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

#Streamlit Title
st.title("ChatBot Using LangGraph")

# Load environment variables from .env file
load_dotenv()

# Access the variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#LLM model
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

#LangFraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]

#Chatbot Function
def chatbot(state: State):
    ai_message = {"messages": [llm.invoke(state["messages"])]}
    return ai_message

# graph builder
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


# For Displaying 
with st.expander("🔍 View LangGraph Structure"):
    st.image(graph.get_graph().draw_mermaid_png())

# Output
user_input = st.chat_input()
if user_input:
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            st.write("Assistant:", value["messages"][-1].content)