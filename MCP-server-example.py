import asyncio
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from typing import Literal

# 1. Setup the Shared LLM (DeepSeek or Llama 3.2)
llm = ChatOllama(model="llama3.2", base_url="http://127.0.0.1:11434")


async def main():
    # 2. Connect to an MCP Server (Example: a local math server)
    # MCP allows this agent to 'discover' tools dynamically from the server
    mcp_tools = await load_mcp_tools(
        "stdio",
        command="python",
        args=["your_mcp_server.py"]
    )

    # 3. Define Specialist Agents
    # Research Agent (Uses standard LangChain tools)
    research_agent = create_react_agent(llm, tools=[], prompt="You are a Researcher.")

    # Tool Agent (Uses MCP-provided tools)
    tool_agent = create_react_agent(llm, tools=mcp_tools, prompt="You are a Tool Specialist.")

    # 4. Supervisor Logic
    def supervisor(state: MessagesState) -> Literal["researcher", "tool_expert", END]:
        # Simple routing logic: look at keywords or use a small LLM call to decide
        last_message = state["messages"][-1].content.lower()
        if "calculate" in last_message:
            return "tool_expert"
        elif "search" in last_message:
            return "researcher"
        return END

    # 5. Build the Graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("researcher", research_agent)
    workflow.add_node("tool_expert", tool_agent)

    workflow.add_conditional_edges(START, supervisor)
    workflow.add_edge("researcher", END)
    workflow.add_edge("tool_expert", END)

    app = workflow.compile()

    # Run a test
    async for chunk in app.astream({"messages": [{"role": "user", "content": "Calculate 500 * 22"}]}):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())