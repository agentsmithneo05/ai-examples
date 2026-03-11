import yfinance as yf
import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


# --- STEP 1: Define the Scraping Tool ---
@tool
def get_stock_history(ticker: str):
    """
    Scrapes the last 3 months of daily closing prices for a given stock ticker.
    Example ticker: 'AAPL' for Apple, 'TSLA' for Tesla, 'NVDA' for Nvidia.
    """
    stock = yf.Ticker(ticker)
    # Fetch 3 months of daily data
    hist = stock.history(period="3mo")

    # Clean up the data to keep it small for the LLM
    hist = hist[['Close']].reset_index()
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')

    # Calculate day-over-day change
    hist['Daily Change'] = hist['Close'].diff()

    # Return as a simple string table
    return hist.to_string(index=False)


# --- STEP 2: Setup Agent ---
tools = [get_stock_history]
tool_node = ToolNode(tools)

# Initialize DeepSeek 1.5B
llm = ChatOllama(
    model="MFDoom/deepseek-r1-tool-calling:1.5b",
    base_url="http://127.0.0.1:11434", # Explicitly point to localhost
    temperature=0,
).bind_tools(tools)


def call_model(state: MessagesState):
    # We give the model a system prompt to guide its small 'brain'
    system_prompt = {
        "role": "system",
        "content": "You are a financial assistant. Use the stock tool to get data and then summarize the trends."
    }
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Logic to decide if we should stop or use a tool
def route(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# --- STEP 3: Build Graph ---
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- STEP 4: Run it! ---
query = {
    "messages": [{"role": "user", "content": "Show me the day-wise price changes for NVDA for the last 3 months."}]}
for event in app.stream(query, stream_mode="values"):
    event["messages"][-1].pretty_print()