import os
import time
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import InferenceClient


# Connect to Hugging Face
client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

# Get topic from the user
topic = input("Enter a topic: ")

# Prompt for generating flashcards
prompt = f"""
Create 5 study flashcards about {topic}.

For each flashcard:
1. Write one clear question.
2. Write a short and accurate answer.
3. Include a useful concept, example, comparison, or application when suitable.

Keep the answers suitable for a 2nd-year computer science student.
Keep the flashcards concise and easy to revise.
"""

# Start latency timer
start_time = time.time()

# Generate flashcards using Model 1
completion = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Calculate response time
end_time = time.time()
latency = end_time - start_time

# Display results
print("\n" + "=" * 55)
print("              ✦ FLIP.AI ✦")
print("        AI-POWERED STUDY CARDS")
print("=" * 55)

print(f"\nTopic: {topic}")
print("Model: Qwen/Qwen3-8B")

print("\n" + "-" * 55)
print("                 FLASHCARDS")
print("-" * 55)

print(completion.choices[0].message.content)

print("\n" + "-" * 55)
print(f"Response time: {latency:.2f} seconds")
print("-" * 55)