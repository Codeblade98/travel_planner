#!/usr/bin/env python3
"""Test message summarization when exceeding 50k tokens"""

import asyncio
from agent import run_agent_interactive

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              MESSAGE SUMMARIZATION TEST (>50k tokens)                        ║
║                                                                              ║
║  This test demonstrates automatic message chain summarization               ║
║  when the conversation exceeds 50,000 tokens.                               ║
║                                                                              ║
║  Note: In this demo, we'll simulate a long conversation with                ║
║  detailed queries to show the token counting feature.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Test with multiple detailed queries to accumulate tokens
test_conversation = [
    "I want to plan an extensive family vacation to Paris, France. We are a family of four - myself, my wife, and our two children aged 7 and 10. We're looking for a comprehensive travel experience that combines culture, history, entertainment, and family-friendly activities.",
    
    "We're planning to visit during the spring season, specifically from April 15th to April 30th, 2025. We want to experience the famous Parisian springtime with blooming gardens and pleasant weather. Can you provide detailed information about what to expect during this time?",
    
    "Our family has diverse interests. My wife loves art and museums, especially impressionist paintings. I'm a history buff interested in Napoleon era and World War II sites. Our 10-year-old loves science and technology, while our 7-year-old is fascinated by fairy tales and Disney characters. Can you suggest activities for everyone?",
    
    "We also need recommendations for family-friendly accommodations in central Paris, preferably near metro stations for easy transportation. We'd like a place with at least two bedrooms and a kitchenette. Budget is flexible but we prefer good value for money.",
    
    "What are the best restaurants for families in Paris? We want to try authentic French cuisine but also need options that children will enjoy. Are there any restaurants with special kids' menus or entertainment?",
    
    "Tell me about the most important museums and attractions we should visit. How much time should we allocate for each? Should we buy skip-the-line tickets in advance? What about museum passes?",
    
    "quit"
]

input_index = 0

async def mock_input(prompt):
    global input_index
    if input_index < len(test_conversation):
        user_msg = test_conversation[input_index]
        
        # Show truncated message for long inputs
        display_msg = user_msg if len(user_msg) < 100 else user_msg[:97] + "..."
        
        print(f"\n{'╔' + '═'*78 + '╗'}")
        print(f"║ USER INPUT #{input_index + 1:02d}: {display_msg[:60]:<60} ║")
        print(f"{'╚' + '═'*78 + '╝'}\n")
        input_index += 1
        return user_msg
    return "quit"

async def main():
    await run_agent_interactive(
        thread_id="summarization_test_session",
        user_input_callback=mock_input
    )
    
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*22 + "✓ TEST COMPLETED" + " "*39 + "║")
    print("╚" + "═"*78 + "╝\n")
    
    print("Observations:")
    print("  • Token count displayed for each message exchange")
    print("  • Router automatically checks for >50k token threshold")
    print("  • If exceeded, older messages are summarized")
    print("  • Recent messages preserved for context")
    print("  • Conversation continues seamlessly after summarization\n")

if __name__ == "__main__":
    asyncio.run(main())
