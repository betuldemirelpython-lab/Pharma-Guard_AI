import os
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_key)

# Create simple test image with text
img = Image.new('RGB', (400, 100), color='white')
from PIL import ImageDraw
draw = ImageDraw.Draw(img)
draw.text((10, 30), "PAROL 500mg Parasetamol - Atabay", fill="black")
buffered = BytesIO()
img.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

prompt = 'Read the text in this image and output as JSON: {"ilac_adi": "...", "etken_madde": "...", "dozaj": "...", "uretici_firma": "...", "yazi_okunuyor_mu": true, "guven_puani": 9}'

groq_vision_models = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-90b-vision-preview",
]

for gmodel in groq_vision_models:
    try:
        print(f"\nTrying {gmodel}...")
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }],
            model=gmodel,
            temperature=0.1
        )
        print(f"SUCCESS with {gmodel}:")
        print(response.choices[0].message.content)
        break
    except Exception as e:
        print(f"FAILED {gmodel}: {e}")
