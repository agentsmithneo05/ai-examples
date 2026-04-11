import ollama

# Replace 'llama3' with whatever model you have installed
response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response['message']['content'])