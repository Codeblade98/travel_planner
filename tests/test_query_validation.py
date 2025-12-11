#!/usr/bin/env python3
"""Test query node with full validation"""

import asyncio
from agent import run_agent_interactive

# Test conversation that should trigger Query Node
test_conversation = [
    "I want to visit Tokyo for a family trip",
    "We're interested in cherry blossoms and traditional temples",
    "quit"
]

input_index = 0

async def mock_input(prompt):
    global input_index
    if input_index < len(test_conversation):
        user_msg = test_conversation[input_index]
        print(f"\n{'='*80}")
        print(f"USER INPUT #{input_index + 1}: {user_msg}")
        print(f"{'='*80}\n")
        input_index += 1
        return user_msg
    return "quit"

async def main():
    print("\n" + "="*80)
    print("QUERY NODE VALIDATION TEST - Tokyo Family Trip")
    print("="*80)
    await run_agent_interactive(
        thread_id="tokyo_family_test",
        user_input_callback=mock_input
    )
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
