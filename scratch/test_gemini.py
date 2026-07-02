import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
print("Gemini Key:", gemini_key)

if gemini_key:
    genai.configure(api_key=gemini_key)

models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

for m in models:
    try:
        model = genai.GenerativeModel(m)
        response = model.generate_content("Hi, test")
        print(f"Model {m} works! Response: {response.text}")
    except Exception as e:
        print(f"Model {m} failed: {e}")
