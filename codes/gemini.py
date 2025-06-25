import os
from dotenv import load_dotenv
import google.generativeai as gemi
# Load environment variables from .env file
load_dotenv()
# Get the API key
key = os.getenv("API_KEY")

model=gemi.GenerativeModel('gemini-2.0-flash-exp')
gemi.configure(api_key=key)

def prompt():
    ques="generate me an image"
    response=model.generate_content(ques)
    print(response.text)
    return response

prompt()