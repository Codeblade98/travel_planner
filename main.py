import asyncio
import json
from agent import run_agent_interactive, clear_session, SESSION_MEMORY
from vector_store import get_vector_store


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_header(text):
    """Print a formatted header."""
    print_separator()
    print(f"  {text}")
    print_separator()


async def main():
    """Main CLI interface."""
    print_header("🌍 Multi-Modal Travel Assistant - CLI Mode")
    
    # Initialize vector store
    print("\n📚 Initializing vector store...")
    vector_store = get_vector_store()
    print("✓ Vector store ready with pre-loaded cities: Paris, Tokyo, New York\n")
    
    print("💡 Features:")
    print("   - LLM-based city extraction")
    print("   - Session memory management")
    print("   - Interactive conversation flow")
    print("   - Type 'quit' to exit\n")
    
    # Interactive loop with callback
    thread_id = "cli_session"
    
    # Define the input callback
    async def get_user_input(prompt: str) -> str:
        """Get user input for the conversation."""
        if prompt:
            # Don't print prompt as it's already displayed by run_agent_interactive
            pass
        
        print_separator("-")
        user_input = input("\n🗨️  Your message: ").strip()
        
        # Handle special commands
        if user_input.lower() == 'session':
            session_data = SESSION_MEMORY.get_session(thread_id)
            print("\n💾 Current Session Memory:")
            print(json.dumps(session_data, indent=2))
            # Ask again
            return await get_user_input("")
        
        if user_input.lower() == 'clear':
            clear_session(thread_id)
            print("\n✓ Session cleared\n")
            # Ask again
            return await get_user_input("")
        
        return user_input
    
    try:
        # Run the interactive agent - it handles the entire conversation loop
        await run_agent_interactive(thread_id=thread_id, user_input_callback=get_user_input)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
