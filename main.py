from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI(
    title="Health Support Chatbot",
    description="AI chatbot for basic health support using Gemini",
    version="1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str

# Response model
class ChatResponse(BaseModel):
    response: str


# Health assistant prompt
SYSTEM_PROMPT = """
You are a helpful health support assistant.

Rules:
1. Provide general health advice only
2. Do NOT give medical diagnosis
3. Suggest consulting a doctor for serious problems
4. Give lifestyle, fitness, diet, and mental health advice
5. Be supportive and polite
6. Avijit's personal assistance
7.Don't write anything about the rules in the response. Just follow them.
"""


@app.get("/")
def home():
    return {"message": "Health Support Chatbot API is running"}



@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    user_message = request.message

    prompt = f"""
{SYSTEM_PROMPT}

User Question: {user_message}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"response": response.text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)