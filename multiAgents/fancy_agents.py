# fancy_agent.py
import asyncio
import os
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent

# 1. Setup Model
llm = ChatOllama(model="llama3.2", base_url="http://127.0.0.1:11434")


async def main():
    # 2. Initialize the Client with Server Config
    # Use an absolute path for the command to avoid 'File not found' errors
    server_script = os.path.abspath("math_server.py")

    client = MultiServerMCPClient({
        "math_server": {
            "transport": "stdio",
            "command": "python",
            "args": [server_script]
        }
    })

    # 3. Load Tools from the Client
    # This is the modern 2026 way to get tools for LangGraph
    mcp_tools = await client.get_tools()

    # 4. Define Agents
    math_agent = create_react_agent(llm, tools=mcp_tools)
    stock_agent = create_react_agent(llm, tools=[])  # Placeholder for your stock tools

    # 5. Routing Logic
    def router(state: MessagesState):
        user_msg = state["messages"][-1].content.lower()
        if "calculate" in user_msg or "interest" in user_msg:
            return "math_node"
        return "stock_node"

    # 6. Build Graph
    builder = StateGraph(MessagesState)
    builder.add_node("math_node", math_agent)
    builder.add_node("stock_node", stock_agent)
    builder.add_conditional_edges(START, router)
    builder.add_edge("math_node", END)
    builder.add_edge("stock_node", END)

    app = builder.compile()

    # 7. Run Test
    print("--- Running Multi-Agent Setup ---")
    query = {"messages": [{"role": "user", "content": "Calculate 5% interest on 1000 for 2 years"}]}
    async for chunk in app.astream(query):
        for node, data in chunk.items():
            print(f"\n[{node}] is thinking...")
            data["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())