!pip install -q transformers accelerate

import torch

from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=0
)

print("🤖 Chatbot started!")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and friendly AI assistant."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = chatbot(
        messages,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7
    )

    answer = response[0]["generated_text"][-1]["content"]
    print("Bot:", answer)
