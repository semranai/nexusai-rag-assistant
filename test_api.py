# test_api.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

print("🔑 Checking API key...")
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {'YES' if api_key else 'NO'}")

if api_key:
    print(f"First 10 chars: {api_key[:10]}...")
else:
    print("❌ No API key found in .env file")
    exit()

client = OpenAI(api_key=api_key)

try:
    print("\n📡 Testing OpenAI API...")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello world",
        encoding_format="float"
    )
    
    embedding = response.data[0].embedding
    print(f"✅ SUCCESS!")
    print(f"Dimension: {len(embedding)}")
    print(f"First 3 values: {embedding[:3]}")
    
except Exception as e:
    print(f"\n❌ OPENAI API ERROR:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")
    
    # Check specific error types
    if "authentication" in str(e).lower() or "api key" in str(e).lower():
        print("\n🔑 PROBLEM: Invalid API key")
        print("Check if Matthew's key is correct in .env file")
    elif "rate limit" in str(e).lower():
        print("\n⏰ PROBLEM: Rate limit exceeded")
        print("Matthew's key might have usage limits")
    elif "billing" in str(e).lower():
        print("\n💳 PROBLEM: Billing issue")
        print("Matthew's account might need payment")
    else:
        print("\n🌐 PROBLEM: Network or other issue")