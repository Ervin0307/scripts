from google import genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the client
client = genai.Client(api_key=GEMINI_API_KEY)

# Upload the audio file using the File API
uploaded_file = client.files.upload(file="path/to/your/audio.mp3")

# Generate content using the uploaded file
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "Transcribe this audio content, identifying different speakers.",
        uploaded_file
    ]
)

print(response.text)
