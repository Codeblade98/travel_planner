#!/usr/bin/env python3
"""Test LLM connectivity and JSON response parsing"""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

# Load environment variables
load_dotenv()

# Check if API key is loaded
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API Key (first 10 chars): {api_key[:10]}...")
else:
    print("❌ GROQ_API_KEY not found in environment!")
    exit(1)

# Initialize LLM
print("\n" + "="*80)
print("Testing LLM Connection")
print("="*80)

try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        streaming=False
    )
    print("✓ LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    exit(1)

# Test 1: Simple query
print("\n" + "="*80)
print("Test 1: Simple Query")
print("="*80)

try:
    response = llm.invoke([SystemMessage(content="Say 'Hello, World!' and nothing else.")])
    print(f"✓ Response received: {response.content}")
except Exception as e:
    print(f"❌ Failed to get response: {e}")

# Test 2: JSON response
print("\n" + "="*80)
print("Test 2: JSON Response")
print("="*80)

json_prompt = """
You are a travel assistant AI. Analyze the user's query and extract information.

User query: "I want to visit Paris"

Return a JSON object with this structure:
{
    "user_name": "string or null",
    "city_found": true/false,
    "city_name": "extracted city name or null",
    "trip_type": "solo/family/business/other or null",
    "travel_preferences": "proper descriptive string or null",
    "summary": "brief summary of whatever has been updated"
}

IMPORTANT: Return ONLY the JSON object, no other text.
"""

try:
    response = llm.invoke([SystemMessage(content=json_prompt)])
    print(f"Raw response:\n{response.content}\n")
    
    # Try to parse the JSON
    content = response.content.strip()
    
    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    result = json.loads(content)
    print(f"✓ Successfully parsed JSON:")
    print(json.dumps(result, indent=2))
    
except json.JSONDecodeError as e:
    print(f"❌ Failed to parse JSON: {e}")
    print(f"Content that failed to parse:\n{content}")
except Exception as e:
    print(f"❌ Failed to get response: {e}")

# Test 3: Router decision
print("\n" + "="*80)
print("Test 3: Router Decision")
print("="*80)

router_prompt = """
You are a travel assistant chatting with a customer. You need to talk with the customer, ask clarifying questions
and verify if the user has responded properly to your previous question.

User Input: "I want to visit Paris"

Clarification questions: None

Task: Since this is the first query, ask the user for more details about their trip.

Output: Return your response in JSON format as:
{
    "thinking": "your internal thought process",
    "next_task": "query" or "user",
    "message": "Your message to the user or the query assistant"
}

IMPORTANT: Return ONLY the JSON object, no other text.
"""

try:
    response = llm.invoke([SystemMessage(content=router_prompt)])
    print(f"Raw response:\n{response.content}\n")
    
    # Try to parse the JSON
    content = response.content.strip()
    
    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    result = json.loads(content)
    print(f"✓ Successfully parsed JSON:")
    print(json.dumps(result, indent=2))
    
except json.JSONDecodeError as e:
    print(f"❌ Failed to parse JSON: {e}")
    print(f"Content that failed to parse:\n{content}")
except Exception as e:
    print(f"❌ Failed to get response: {e}")

print("\n" + "="*80)
print("All tests completed")
print("="*80)
