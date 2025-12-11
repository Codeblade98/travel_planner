#!/usr/bin/env python3
"""Test script for multi-turn conversation"""

import asyncio
from agent import run_agent_interactive

test_inputs = [
    "I want to visit paris",
    "I am planning to visit on the 3rd of January, 2025 with my wife and two kids",
    "quit"
]

input_index = 0

async def mock_input(prompt):
    global input_index
    if input_index < len(test_inputs):
        user_input = test_inputs[input_index]
        print(f"\n🗨️  Your message: {user_input}")
        input_index += 1
        return user_input
    return "quit"

async def main():
    await run_agent_interactive(
        thread_id="test_session",
        user_input_callback=mock_input
    )

if __name__ == "__main__":
    asyncio.run(main())
