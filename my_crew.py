from crewai import Agent, Task, Crew, Process, LLM

# Define your local models via Ollama
# We use the 'ollama/' prefix so CrewAI knows to look locally
reasoning_llm = LLM(model="ollama/deepseek-r1:1.5b", base_url="http://localhost:11434")
worker_llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

# Agent 1: The Logical Researcher (DeepSeek)
researcher = Agent(
    role='Lead Researcher',
    goal='Analyze the logical steps to solve: {topic}',
    backstory='You are a master of logic and chain-of-thought reasoning.',
    llm=reasoning_llm,
    verbose=True
)

# Agent 2: The Technical Writer (Llama 3.2)
writer = Agent(
    role='Technical Communicator',
    goal='Write a clear, beginner-friendly explanation based on the research.',
    backstory='You excel at taking complex logic and making it easy to read.',
    llm=worker_llm,
    verbose=True
)

# Define Tasks
task1 = Task(description='Break down the logic of {topic} step by step.', agent=researcher, expected_output="A logical breakdown.")
task2 = Task(description='Summarize the research into a 3-bullet point email.', agent=writer, expected_output="A 3-bullet point summary.")

# Assemble the Crew
my_crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential # Crucial for 4GB GPU!
)

# Start the process
result = my_crew.kickoff(inputs={'topic': 'How a refrigerator works using thermodynamics'})
print("\n\n########################")
print("FINAL OUTPUT:")
print(result)