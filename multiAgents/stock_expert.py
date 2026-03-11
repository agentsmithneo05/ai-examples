import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


# --- TOOLS ---

@tool
def get_stock_history(ticker: str):
    """Fetches price history. Use suffix .NS for Indian stocks."""
    print(f"\n--- [TOOL CALL]: Fetching history for {ticker} ---")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    if hist.empty: return "No data found."
    hist = hist[['Close']].reset_index()
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    return hist.tail(10).to_string(index=False)


@tool
def save_stock_chart(ticker: str):
    """Saves a 3-month chart. MUST be used if user asks for a chart."""
    print(f"\n--- [TOOL CALL]: Generating chart for {ticker} ---")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")

    if hist.empty: return "No data found."

    # Use Object-Oriented plotting for stability
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist.index, hist['Close'], color='tab:blue', linewidth=2)
    ax.set_title(f"3-Month Trend: {ticker}")
    ax.grid(True, alpha=0.3)

    file_path = os.path.join(os.getcwd(), "../stock_chart.png")
    fig.savefig(file_path)
    plt.close(fig)  # Important: Clear memory

    print(f"--- [SUCCESS]: Chart saved at {file_path} ---")
    return f"CHART_SAVED_AT: {file_path}"


# --- AGENT LOGIC ---

tools = [get_stock_history, save_stock_chart]
tool_node = ToolNode(tools)

llm = ChatOllama(
    model="MFDoom/deepseek-r1-tool-calling:1.5b",
    base_url="http://127.0.0.1:11434",
    temperature=0
).bind_tools(tools)


def call_model(state: MessagesState):
    # Aggressive system prompt to force tool usage
    system_msg = {
        "role": "system",
        "content": (
            "You are a strict financial bot. "
            "If the user asks for a chart, you MUST use the 'save_stock_chart' tool. "
            "If the user asks for Indian stocks, append '.NS' to the ticker."
        )
    }
    response = llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


def route(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END


# --- GRAPH ---
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route)
workflow.add_edge("tools", "agent")
app = workflow.compile()

if __name__ == "__main__":
    # Test query
    query = "Search for INFY.NS and save a chart for me."
    print(f"User Query: {query}")

    for event in app.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
        event["messages"][-1].pretty_print()