#!/usr/bin/env python3
"""Quick test to verify message chain stays reasonable"""

import asyncio
from agent import run_agent_interactive

test_inputs = [
    "I want to visit Paris",
    "I'm going in spring with my family",  
    "What about Tokyo?",
    "Tell me about New York",
    "quit"
]

index = 0

async def mock_input(prompt):
    global index
    if index < len(test_inputs):
        msg = test_inputs[index]
        print(f"\nUSER: {msg}\n")
        index += 1
        return msg
    return "quit"

async def main():
    await run_agent_interactive("test_session", mock_input)

if __name__ == "__main__":
    asyncio.run(main())
