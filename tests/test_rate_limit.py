#!/usr/bin/env python3
"""Test rate limit handling with model fallback"""

import asyncio
from agent import run_agent_interactive, AVAILABLE_MODELS, current_model_index

print("="*80)
print("RATE LIMIT HANDLING TEST")
print("="*80)
print(f"\nAvailable Models: {AVAILABLE_MODELS}")
print(f"Current Model Index: {current_model_index}")
print(f"Starting with: {AVAILABLE_MODELS[current_model_index]}")
print("="*80)

# Test conversation
test_conversation = [
    "I want to visit Paris",
    "Tell me about the best museums",
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
    print("\nStarting interactive session...")
    print("Rate limits will be automatically handled:")
    print("  • TPM (tokens per minute) → Wait 80s and retry")
    print("  • TPD (tokens per day) → Switch to next model")
    print("\n")
    
    try:
        await run_agent_interactive(
            thread_id="rate_limit_test",
            user_input_callback=mock_input
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✓ Test completed")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
