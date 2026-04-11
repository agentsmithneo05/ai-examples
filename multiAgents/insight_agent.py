import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434", temperature=0)


async def main():
    client = MultiServerMCPClient({"engine": {"transport": "sse", "url": "http://localhost:"
                                                                         "/sse"}})
    mcp_tools = await client.get_tools()

    # Linear Nodes
    def finder(state: MessagesState):
        return {"messages": [llm.bind_tools(mcp_tools).invoke(
            [{"role": "system", "content": "Use get_stock_price only."}] + state["messages"])]}

    def calculator(state: MessagesState):
        return {"messages": [llm.bind_tools(mcp_tools).invoke(
            [{"role": "system", "content": "Use calculate_growth with the price found."}] + state["messages"])]}

    # Strict Linear Path
    workflow = StateGraph(MessagesState)
    workflow.add_node("finder", finder)
    workflow.add_node("calculator", calculator)
    workflow.add_node("tools", ToolNode(mcp_tools))

    workflow.add_edge(START, "finder")
    workflow.add_edge("finder", "tools")
    workflow.add_edge("tools", "calculator")
    workflow.add_edge("calculator", "tools")
    workflow.add_edge("tools", END)

    app = workflow.compile()

    query = {"messages": [{"role": "user", "content": "12% growth for INFY over 5 years."}]}
    async for chunk in app.astream(query, stream_mode="values"):
        chunk["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())