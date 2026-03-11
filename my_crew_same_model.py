from crewai import Agent, Task, Crew, Process, LLM

# Define one single LLM for both
shared_llm = LLM(model="ollama/deepseek-r1:1.5b", base_url="http://localhost:11434")

# Agent 1: The Researcher
researcher = Agent(
    role='Expert Researcher',
    goal='Uncover the deep logic behind {topic}',
    backstory='You are a specialist in technical analysis.',
    llm=shared_llm,
    verbose=True
)

# Agent 2: The Writer
writer = Agent(
    role='Content Creator',
    goal='Write a simple summary based on the research provided.',
    backstory='You turn complex logic into easy-to-read text.',
    llm=shared_llm,
    verbose=True
)

# Tasks
t1 = Task(description='Research the core mechanics of {topic}.', agent=researcher, expected_output="A list of facts.")
t2 = Task(description='Summarize these facts into a tweet.', agent=writer, expected_output="A single tweet.")

# Crew
my_crew = Crew(
    agents=[researcher, writer],
    tasks=[t1, t2],
    process=Process.sequential
)

my_crew.kickoff(inputs={'topic': 'Quantum Computing for kids'})