import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
print("Groq Key:", groq_key)

client = Groq(api_key=groq_key)

try:
    # Test simple text completion
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Hi"}
        ],
        model="llama-3.3-70b-versatile",
    )
    print("Text completion works:", chat_completion.choices[0].message.content)
except Exception as e:
    print("Text completion failed:", e)

try:
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        print(f"- {m.id}")
except Exception as e:
    print("Listing models failed:", e)
