from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from logger import log_message


# ============================================================================
# In-Memory Session Storage
# ============================================================================

class SessionMemory:
    """In-memory session storage managed by LLM decisions."""
    
    def __init__(self):
        self._sessions = {}
    
    def get_session(self, thread_id: str) -> Dict:
        """Get session data for a thread."""
        if thread_id not in self._sessions:
            self._sessions[thread_id] = {}
        return self._sessions[thread_id]
    
    def update_session(self, thread_id: str, updates: Dict):
        """Update session data."""
        if thread_id not in self._sessions:
            self._sessions[thread_id] = {}
        self._sessions[thread_id].update(updates)
        print(f"💾 Session updated: {updates}")
    
    def clear_session(self, thread_id: str):
        """Clear session data."""
        if thread_id in self._sessions:
            del self._sessions[thread_id]
            log_message(f"Session cleared: {thread_id}", "INFO")


class TravelSessionMemory(SessionMemory):
    """Specialized session memory for travel assistant."""
    
    def __init__(self):
        super().__init__()
        
    def get_favorite_cities(self, thread_id: str) -> List[str]:
        """Get user's cities from session."""
        session = self.get_session(thread_id)
        return session.get("cities", [])
    
    def check_city_extracted(self, thread_id: str, city: str) -> bool:
        """Check if a city has been extracted before."""
        session = self.get_session(thread_id)
        extracted_cities = session.get("cities", [])
        return city in extracted_cities if isinstance(extracted_cities, list) else False


# Global session memory instance
SESSION_MEMORY = TravelSessionMemory()


# ============================================================================
# Token Counting and Message Summarization
# ============================================================================

def count_tokens_in_messages(messages: List[BaseMessage]) -> int:
    """
    Estimate token count for a list of messages.
    Uses a simple approximation: ~4 characters per token.
    For more accurate counting, could use tiktoken library.
    
    Args:
        messages: List of LangChain messages
        
    Returns:
        Estimated token count
    """
    total_chars = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            total_chars += len(str(msg.content))
    
    # Rough approximation: 4 characters per token
    estimated_tokens = total_chars // 4
    return estimated_tokens


def summarize_message_chain(messages: List[BaseMessage], llm) -> List[BaseMessage]:
    """
    Summarize a long message chain to reduce token count.
    Keeps the most recent messages and creates a summary of older ones.
    
    Args:
        messages: List of messages to summarize
        llm: LLM instance for generating summary
        
    Returns:
        Condensed list of messages with summary
    """
    if len(messages) <= 4:
        return messages
    
    # Keep the last 4 messages (most recent context)
    recent_messages = messages[-4:]
    
    # Summarize older messages
    older_messages = messages[:-4]
    
    # Limit the content we send to LLM to avoid "Request too large" errors
    conversation_summary = []
    for i, msg in enumerate(older_messages):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = str(msg.content)
        # Truncate very long messages
        if len(content) > 200:
            content = content[:200] + "..."
        conversation_summary.append(f"{role}: {content}")
    
    # Limit total summary text
    conversation_text = "\n\n".join(conversation_summary[-20:])  # Only last 20 exchanges
    
    # Generate summary
    summary_prompt = f"""
    You are summarizing a conversation between a user and a travel assistant.
    Create a concise summary (maximum 300 words) that captures:
    1. The cities discussed
    2. Travel preferences (dates, trip type, interests)
    3. Key questions asked and answered
    4. Any important context for continuing the conversation
    
    Conversation to summarize:
    {conversation_text}
    
    Provide a brief summary:
    """
    
    try:
        summary_response = llm.invoke([SystemMessage(content=summary_prompt)])
        summary_content = summary_response.content.strip()
        
        # Create a system message with the summary
        summary_message = SystemMessage(
            content=f"[CONVERSATION SUMMARY]\n{summary_content}\n[END SUMMARY]"
        )
        
        original_tokens = count_tokens_in_messages(older_messages)
        summary_tokens = count_tokens_in_messages([summary_message])
        
        log_message(
            f"Message chain summarized: {len(older_messages)} messages → 1 summary "
            f"(~{original_tokens} tokens → ~{summary_tokens} tokens)",
            "INFO"
        )
        
        # Return summary + recent messages
        return [summary_message] + recent_messages
        
    except Exception as e:
        log_message(f"Failed to summarize messages: {e}", "ERROR")
        # Fallback: just keep recent messages
        return recent_messages


def check_and_summarize_messages(
    messages: List[BaseMessage], 
    llm, 
    max_tokens: int = 8000
) -> List[BaseMessage]:
    """
    Check if message chain exceeds token limit and summarize if needed.
    
    Args:
        messages: List of messages
        llm: LLM instance for summarization
        max_tokens: Maximum token threshold (default 8k)
        
    Returns:
        Original or summarized message list
    """
    token_count = count_tokens_in_messages(messages)
    
    if token_count > max_tokens:
        log_message(
            f"Message chain exceeds {max_tokens} tokens ({token_count} tokens). "
            f"Triggering automatic summarization...",
            "WARNING"
        )
        return summarize_message_chain(messages, llm)
    
    return messages


# ============================================================================
# Utility Functions
# ============================================================================

def clear_session(thread_id: str):
    """
    Clear session memory for a thread.
    
    Args:
        thread_id: Thread identifier
    """
    SESSION_MEMORY.clear_session(thread_id)
    print(f"🗑️  Cleared session: {thread_id}")
