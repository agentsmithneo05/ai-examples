# insight_agent.py
import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# 1. SETUP MODEL - Optimized for speed
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0,
    num_predict=150  # Prevent long, 'stuck' responses
)


async def main():
    # 2. CONNECT VIA SSE (Standard Web Request)
    # This replaces the 'command' and 'args' logic which often hangs on Windows
    server_params = {
        "math_engine": {
            "transport": "sse",
            "url": "http://localhost:8888/sse"
        }
    }

    client = MultiServerMCPClient(server_params)

    try:
        mcp_tools = await client.get_tools()

        def call_model(state: MessagesState):
            system_prompt = (
                "You are a Math Assistant. Use 'calculate_growth' for all math. "
                "Once you get a result, state ONLY the final number. Be extremely brief."
            )
            model_with_tools = llm.bind_tools(mcp_tools)
            # Only send the last 3 messages to keep context window clean
            messages = [{"role": "system", "content": system_prompt}] + state["messages"][-3:]
            return {"messages": [model_with_tools.invoke(messages)]}

        # 3. GRAPH CONSTRUCTION
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(mcp_tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            lambda state: "tools" if state["messages"][-1].tool_calls else END
        )
        workflow.add_edge("tools", "agent")

        app = workflow.compile()

        # 4. EXECUTION
        print("--- SSE-Optimized Agent Active ---")
        query = {"messages": [{"role": "user", "content": "12% growth on 1900 for 5 years"}]}

        # Using a timeout to ensure the script never hangs forever
        try:
            async for event in app.astream(query, stream_mode="values"):
                event["messages"][-1].pretty_print()
        except asyncio.TimeoutError:
            print("Request timed out. The LLM might be struggling.")

    finally:
        # Proper cleanup
        pass


if __name__ == "__main__":
    asyncio.run(main())