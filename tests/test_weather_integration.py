#!/usr/bin/env python3
"""
Test script to verify weather data integration with the travel assistant
"""
import asyncio
from datetime import datetime
from agent import run_agent
from langgraph.checkpoint.memory import MemorySaver

async def test_weather_integration():
    print("🌍 Testing Weather Data Integration")
    print("=" * 60)
    
    # Initialize memory for the session
    memory = MemorySaver()
    thread_id = "weather_test_session"
    
    # Test query asking for weather in different time periods
    queries = [
        "I want to plan a trip to Paris in late December 2025",
        "What's the weather like in Tokyo during March 2026?",
        "I'm considering New York for early June 2026",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}️⃣  Query: {query}")
        print("-" * 60)
        
        try:
            result = await run_agent(query, thread_id=f"{thread_id}_{i}")
            
            # Display result
            if result and "weather" in result.get("final_answer", "").lower():
                print("✅ Weather data included in response")
                # Extract and show date range if present
                if "2025" in result["final_answer"] or "2026" in result["final_answer"]:
                    print("✅ Future dates (2025-2026) confirmed in response")
            
            print(f"\n📝 Response preview:")
            response = result.get("final_answer", "")
            # Show first 300 characters
            print(response[:300] + "..." if len(response) > 300 else response)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Weather integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_weather_integration())
