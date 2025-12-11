#!/usr/bin/env python3
"""Simple test to verify the agent works end-to-end"""

import asyncio
from agent import run_agent_interactive

# Test inputs simulating a real conversation
test_conversation = [
    "Tell me about Tokyo",
    "I want to go there in spring with my family",
    "What about the weather?",
    "quit"
]

input_index = 0

async def mock_input(prompt):
    global input_index
    if input_index < len(test_conversation):
        user_msg = test_conversation[input_index]
        print(f"\n{'='*80}")
        print(f"USER: {user_msg}")
        print(f"{'='*80}\n")
        input_index += 1
        return user_msg
    return "quit"

async def main():
    print("\n" + "="*80)
    print("MULTI-TURN CONVERSATION TEST")
    print("="*80)
    await run_agent_interactive(
        thread_id="tokyo_test",
        user_input_callback=mock_input
    )
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
