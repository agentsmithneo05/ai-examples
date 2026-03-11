from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END

# 1. Setup the local model
# Ensure the model name matches exactly what you see in 'ollama list'
llm = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://127.0.0.1:11434", # Explicitly point to localhost
    temperature=0,
)

# 2. Define a simple node
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 3. Build the Graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 4. Compile and Run
app = workflow.compile()
input_data = {"messages": [{"role": "user", "content": "Explain LangGraph in one sentence."}]}

for event in app.stream(input_data):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)