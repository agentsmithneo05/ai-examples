import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load the variables from .env into the system environment
load_dotenv()

# 2. Initialize the model
# LangChain automatically looks for an environment variable named GROQ_API_KEY
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# 3. Test it
response = llm.invoke("Explain the concept of 'latency' in one sentence.")
print(response.content)