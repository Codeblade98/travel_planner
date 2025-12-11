"""
LangGraph State and Agent Architecture for Multi-Modal Travel Assistant.
Restructured with 4 nodes: Router, Query, Tool Calling, and User.
Implements LLM-based entity extraction and in-memory session management.
"""
import os
import json
import asyncio
import copy
import time
import re
from typing import TypedDict, List, Dict, Literal, Optional, Annotated
from operator import add
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from vector_store import get_vector_store
from mock_apis import fetch_weather_forecast, fetch_city_images, mock_web_search

# Load environment variables
load_dotenv()

# Log file path for LLM responses
LLM_LOG_FILE = "LLM_Response.txt"

# Global model list for fallback (ordered by preference)
AVAILABLE_MODELS = [
    "openai/gpt-oss-20b",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# Current model index (starts with first model)
current_model_index = 0


def log_llm_response(prompt: str, response: str, success: bool = True):
    """
    Log LLM prompt and response to a file.
    
    Args:
        prompt: The prompt sent to the LLM
        response: The response received from the LLM
        success: Whether the response was successfully parsed
    """
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}\n")
            f.write("-"*80 + "\n")
            f.write("PROMPT:\n")
            f.write(prompt[:500] + ("..." if len(prompt) > 500 else "") + "\n")
            f.write("-"*80 + "\n")
            f.write("RESPONSE:\n")
            f.write(response + "\n")
            f.write("="*80 + "\n")
    except Exception as e:
        print(f"⚠️  Failed to log LLM response: {e}")


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
        return city in extracted_cities.keys()

# Global session memory
SESSION_MEMORY = TravelSessionMemory()


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """
    The state that flows through the graph.
    Uses Annotated with add operator to append to messages list.
    """
    messages: List[BaseMessage]
    thread_id: Optional[str]
    session_data: Optional[Dict]  # LLM-managed session storage
    prev_node: Optional[str]
    next_node: Optional[str]
    clarification_question: Optional[str]
    last_extracted_data: Optional[str]
    image_urls: Optional[List[str]]  # List of image URLs, replaced on each query
    weather_data: Optional[List[Dict]]  # Weather data, replaced on each query


# ============================================================================
# Structured Output Schema
# ============================================================================

class WeatherData(BaseModel):
    """Schema for weather data point."""
    date: str
    temperature: float
    condition: str
    humidity: int
    wind_speed: int


class TravelAssistantOutput(BaseModel):
    """Final structured output from the agent."""
    city: str = Field(description="The city name")
    city_summary: str = Field(description="Detailed summary about the city")
    weather_forecast: List[WeatherData] = Field(description="Weather forecast data")
    image_urls: List[str] = Field(description="List of image URLs for the city")
    source: str = Field(description="Source of information: 'vector_store' or 'web_search'")


class CityExtractionResult(BaseModel):
    """Schema for LLM-based city extraction."""
    city_found: bool = Field(description="Whether a city was found in the query")
    city_name: Optional[str] = Field(description="The extracted city name")
    needs_clarification: bool = Field(description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(description="Question to ask user if clarification needed")
    session_updates: Dict = Field(description="Updates to session memory (LLM decides keys)")


# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_weather",
            "description": "Fetch weather forecast for a city for the next 7 days",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to fetch weather for"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days for forecast (default 7)",
                        "default": 7
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_images",
            "description": "Fetch high-quality images of a city or location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to fetch images for"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of images to fetch (default 5)",
                        "default": 5
                    }
                },
                "required": ["city"]
            }
        }
    }
]


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
    
    # Keep the last 4 messages (most recent context) instead of 2
    recent_messages = messages[-4:]
    
    # Summarize older messages
    older_messages = messages[:-4]
    
    # Limit the content we send to LLM to avoid "Request too large" errors
    # Only include message types and truncated content
    conversation_summary = []
    for i, msg in enumerate(older_messages):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = str(msg.content)
        # Truncate very long messages to prevent explosion
        if len(content) > 200:
            content = content[:200] + "..."
        conversation_summary.append(f"{role}: {content}")
    
    # Limit total summary text to prevent API errors
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
        
        print(f"Message chain summarized: {len(older_messages)} messages → 1 summary")
        print(f"Original tokens: ~{count_tokens_in_messages(older_messages)}")
        print(f"Summary tokens: ~{count_tokens_in_messages([summary_message])}")
        
        # Return summary + recent messages
        return [summary_message] + recent_messages
        
    except Exception as e:
        print(f"Failed to summarize messages: {e}")
        # Fallback: just keep recent messages (more than before)
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
        print(f"⚠️  Message chain exceeds {max_tokens} tokens ({token_count} tokens)")
        print(f"   Triggering automatic summarization...")
        return summarize_message_chain(messages, llm)
    
    return messages


# ============================================================================
# Rate Limit Handling
# ============================================================================

def parse_rate_limit_error(error_message: str) -> Dict[str, any]:
    """
    Parse rate limit error message to extract details.
    
    Args:
        error_message: Error message from API
        
    Returns:
        Dict with 'type' (TPM/TPD), 'retry_after' (seconds), and 'limit_type'
    """
    result = {
        'type': None,
        'retry_after': None,
        'limit_type': None
    }
    
    # Check for tokens per day (TPD)
    if 'tokens per day (TPD)' in error_message or 'TPD' in error_message:
        result['limit_type'] = 'TPD'
        result['type'] = 'tokens_per_day'
        
        # Extract retry time
        time_match = re.search(r'try again in ([\d]+h)?([\d]+m)?([\d.]+s)?', error_message)
        if time_match:
            hours = int(time_match.group(1).replace('h', '')) if time_match.group(1) else 0
            minutes = int(time_match.group(2).replace('m', '')) if time_match.group(2) else 0
            seconds = float(time_match.group(3).replace('s', '')) if time_match.group(3) else 0
            result['retry_after'] = hours * 3600 + minutes * 60 + seconds
    
    # Check for tokens per minute (TPM)
    elif 'tokens per minute (TPM)' in error_message or 'TPM' in error_message or 'Rate limit' in error_message:
        result['limit_type'] = 'TPM'
        result['type'] = 'tokens_per_minute'
        result['retry_after'] = 80  # Default to 80 seconds for TPM
        
        # Try to extract specific wait time
        time_match = re.search(r'try again in ([\d]+)s', error_message)
        if time_match:
            result['retry_after'] = int(time_match.group(1))
    
    # Check for request size error
    elif 'Request too large' in error_message or 'reduce your message size' in error_message:
        result['limit_type'] = 'REQUEST_SIZE'
        result['type'] = 'request_too_large'
    
    return result


def switch_to_next_model():
    """
    Switch to the next available model in the global list.
    
    Returns:
        New model name or None if no models left
    """
    global current_model_index
    
    current_model_index += 1
    
    if current_model_index < len(AVAILABLE_MODELS):
        new_model = AVAILABLE_MODELS[current_model_index]
        print(f"🔄 Switching to fallback model: {new_model}")
        return new_model
    else:
        print(f"❌ No more fallback models available!")
        return None


def handle_rate_limit(error, retry_count: int = 0, max_retries: int = 3):
    """
    Handle rate limit errors with appropriate retry logic.
    
    Args:
        error: Exception from API call
        retry_count: Current retry attempt
        max_retries: Maximum retries allowed
        
    Returns:
        Tuple of (should_retry: bool, wait_time: int, use_new_model: bool)
    """
    error_str = str(error)
    rate_info = parse_rate_limit_error(error_str)
    
    print(f"\n⚠️  Rate Limit Error Detected:")
    print(f"   Type: {rate_info['limit_type']}")
    print(f"   Retry count: {retry_count + 1}/{max_retries}")
    
    if rate_info['limit_type'] == 'TPM':
        # Tokens per minute - wait and retry
        wait_time = rate_info['retry_after'] or 80
        if retry_count < max_retries:
            print(f"   ⏳ Waiting {wait_time} seconds before retry...")
            return (True, wait_time, False)
        else:
            print(f"   ❌ Max retries reached for TPM limit")
            return (False, 0, False)
    
    elif rate_info['limit_type'] == 'TPD':
        # Tokens per day - switch model
        print(f"   🔄 Daily token limit reached, switching to fallback model...")
        return (True, 0, True)
    
    elif rate_info['limit_type'] == 'REQUEST_SIZE':
        # Request too large - cannot retry
        print(f"   ❌ Request size too large, cannot retry")
        return (False, 0, False)
    
    else:
        # Unknown error
        print(f"   ⚠️  Unknown rate limit type")
        return (False, 0, False)


# ============================================================================
# Initialize LLM
# ============================================================================

def get_llm(model_name: str = None):
    """
    Initialize the ChatGroq LLM with specified or default model.
    
    Args:
        model_name: Optional model name. If None, uses current global model.
    """
    if model_name is None:
        model_name = AVAILABLE_MODELS[current_model_index]
    
    return ChatGroq(
        model=model_name,
        temperature=0,
        streaming=False
    )


def parse_llm_json_response(content: str, max_retries: int = 3) -> dict:
    """
    Parse JSON from LLM response with retry logic.
    
    Args:
        content: Raw content from LLM response
        max_retries: Maximum number of parsing attempts
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If parsing fails after all retries
    """
    # Try to extract JSON from markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    # Try parsing with increasingly aggressive cleaning
    for attempt in range(max_retries):
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                # Try cleaning the content progressively
                if attempt == 0:
                    # First retry: Strip whitespace and newlines
                    content = content.strip()
                elif attempt == 1:
                    # Second retry: Remove trailing commas before closing braces
                    content = content.replace(",}", "}").replace(",]", "]")
                    # Remove comments
                    content = '\n'.join(line for line in content.split('\n') 
                                      if not line.strip().startswith('//'))
                print(f"⚠️  JSON parse attempt {attempt + 1} failed, retrying with cleaned content...")
            else:
                # Last attempt failed, raise the error
                print(f"❌ Failed to parse JSON after {max_retries} attempts")
                raise


def invoke_llm_with_retry(llm, prompt: str, max_retries: int = 3) -> dict:
    """
    Invoke LLM and parse JSON response with retry logic and rate limit handling.
    
    Args:
        llm: The LLM instance
        prompt: The prompt to send
        max_retries: Maximum number of retry attempts
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        Exception: If all retries fail
    """
    global current_model_index
    
    messages = [SystemMessage(content="You must only return valid JSON responses without any additional text, as per given instructions.")]
    messages.append(HumanMessage(content=prompt))
    
    current_llm = llm
    
    for attempt in range(max_retries):
        try:
            response = current_llm.invoke(messages)
            content = response.content.strip()
            
            # Log the response
            log_llm_response(prompt, content, success=True)
            
            # Parse the JSON response
            result = parse_llm_json_response(content, max_retries=2)
            return result
            
        except json.JSONDecodeError as e:
            # Log the failed response
            log_llm_response(prompt, content if 'content' in locals() else "No response", success=False)
            
            if attempt < max_retries - 1:
                print(f"⚠️  LLM returned invalid JSON (attempt {attempt + 1}/{max_retries}), retrying...")
                continue
            else:
                print(f"❌ LLM failed to return valid JSON after {max_retries} attempts")
                raise
                
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error
            if 'rate limit' in error_str.lower() or 'error code: 429' in error_str.lower() or 'error code: 413' in error_str.lower():
                should_retry, wait_time, use_new_model = handle_rate_limit(e, attempt, max_retries)
                
                if should_retry:
                    if use_new_model:
                        # Switch to next model
                        new_model = switch_to_next_model()
                        if new_model:
                            current_llm = get_llm(new_model)
                            print(f"   ✓ Retrying with model: {new_model}")
                            continue
                        else:
                            print(f"   ❌ No fallback models available")
                            raise
                    else:
                        # Wait and retry with same model
                        time.sleep(wait_time)
                        print(f"   ✓ Retrying after wait...")
                        continue
                else:
                    # Cannot retry
                    raise
            else:
                # Non-rate-limit error
                if attempt < max_retries - 1:
                    print(f"⚠️  LLM invocation error (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    raise


# ============================================================================
# Node 1: Router Node
# ============================================================================

async def router_node(state: AgentState) -> AgentState:
    """
    Router Node: Analyzes user input using LLM to extract city and decide routing.
    Updates session memory with LLM-determined keys.
    Automatically summarizes message chain if it exceeds 50k tokens.
    """
    messages = state["messages"]
    thread_id = state.get("thread_id", "default")
    last_message = messages[-1] if messages else None
    
    updated_state = copy.deepcopy(state)
    
    if not last_message or not isinstance(last_message, HumanMessage):
        return {"next_node": "user"}  # Don't return messages to avoid adding
    
    # Check and summarize messages if needed (>8k tokens)
    llm = get_llm()
    messages = check_and_summarize_messages(messages, llm, max_tokens=8000)
    # DON'T set messages in updated_state here - it will ADD to existing messages!
    # The summarization should happen BEFORE entering the graph
    
    prev_node = state.get("prev_node")   
    
    # When prev_node is "user" or None (initial input), process the user's query
    if prev_node == "user" or prev_node is None:
        user_query = last_message.content
        
        # Get current session data
        session_data = SESSION_MEMORY.get_session(thread_id)
        
        # Use LLM to extract city and determine session updates
        llm = get_llm()
        
        extraction_prompt = f"""
        You are a travel assistant AI. Analyze the user's query and extract information.

        Current session data: {json.dumps(session_data, indent=2)}
        User query: "{user_query}"

        **Task**:
        Update any session data as needed (such as new user preferences, change in preferences, etc).
        Your main objective is to identify and segment out any user preferences such as:
        - Name of user (if mentioned)
        - Current city of interest
        - Trip type (solo, family, business, etc)
        - Travel preferences (e.g., weather, activities)
        
        

        **Output**:
        Return only a VALID JSON object with this structure without any verbiage:
        {{
            "user_name": "string or null",
            "city_found": true/false,
            "city_name": "extracted city name or null",
            "trip_type": Literal["solo", "family", "business", "other"] or null,
            "travel_preferences": "proper descriptive string or null",
            "clarification_question": "string or null",
            "needs_clarification": true/false,
            "summary": "brief summary of whatever has been updated",
        }}"""
        extraction_prompt = ' '.join(extraction_prompt.split())
        
        try:
            extraction_result = invoke_llm_with_retry(llm, extraction_prompt, max_retries=3)
            
            print(f"🔀 Router: Extracted data: {extraction_result}")
            
            # Update session memory
            if extraction_result:
                updation_summary = extraction_result["summary"]
                extraction_result.pop("summary")
                SESSION_MEMORY.update_session(thread_id, extraction_result)
        
        except Exception as e:
            print(f"❌ Router error: {e}")
            updated_state["clarification_question"] = "I had trouble understanding your request. Could you please specify which city you're interested in?"
            updated_state["next_node"] = "user"
            return updated_state
    
        # Case 1: User has given some input 
        #    (a) User responded with clarification → query
        #    (b) User did not clarify → ask again
        # Case 2: User is doing randon chit-chat
        #     Ask user for his travel preferences 
        clarification_question = extraction_result.get("clarification_question")
        
        if state.get("clarification_question") or extraction_result.get("needs_clarification", False):
            updated_state["next_node"] = "user"
            updated_state["clarification_question"] = clarification_question
            if clarification_question is None:
                clarification_question = "Could you please clarify your travel preferences?"
            updated_state["messages"] = [AIMessage(content=clarification_question)]  # Only NEW message
            return updated_state          
        else:
            print("NO CLARIFICATION NEEDED, PROCEEDING TO QUERY NODE.........................")
            router_prompt = f"""
            You are a travel assistant chatting with a customer.
            The user had provided enough information about their travel preferences.
            Based on the information, generate a query to fetch relevant travel information.
            
            **Travel Preferences*:
            ```
            {session_data}
            ```
            
            **Task**:
            The query should be relevant to the travel location and preferences.
            It must cover all the important aspects of the user's travel needs.
            
            **Output**:
            Return ONLY a VALID JSON object with this structure:
            {{
                "thinking": "your internal thought process",
                "city": "city of interest of the user",
                "query": "query to send to the knowledge oracle to help user",
            }}
            
            **Example**:
            Let's say the user has provided a clear destination, Paris, I can proceed to gather more information about their travel preferences.
            
            """
        router_prompt = ' '.join(router_prompt.split())
        try:
            router_result = invoke_llm_with_retry(llm, router_prompt, max_retries=3)
            
            message = router_result.get("message", "Could you please clarify?")
            
            print(f"Router: Message: {message}")
            
            query_to_send = {
                "city": router_result.get("city", "Unknown"),
                "query": router_result.get("query", "")
            }
            updated_state["clarification_question"] = json.dumps(query_to_send)
            updated_state["next_node"] = "query"
            updated_state["messages"] = [AIMessage(content=message)]  # Only NEW message
            return updated_state
        
        except Exception as e:
            print(f"❌ Router (User prev) error: {e}")
            updated_state["clarification_question"] = "I had trouble understanding your request. Could you please specify which city you're interested in?"
            updated_state["next_node"] = "user"
            return updated_state
        
    elif prev_node == "query":
        updated_state["next_node"] = "user"
        return updated_state


# ============================================================================
# Node 2: Query Node
# ============================================================================

async def query_node(state: AgentState) -> AgentState:
    """
    Query Node: 
    1. Parse city and query from clarification_question (serialized JSON)
    2. Check session memory for city info first
    3. If not in session, query ChromaDB with similarity > 0.9
    4. If not in ChromaDB, fall back to web search
    5. For the query itself, check ChromaDB with similarity > 0.9
    6. Execute tools in parallel (weather + images)
    """
    thread_id = state.get("thread_id", "default")
    clarification_question = state.get("clarification_question", "{}")
    
    # Step 1: Parse city and query from clarification_question
    try:
        query_data = json.loads(clarification_question)
        city = query_data.get("city", "Unknown")
        user_query = query_data.get("query", "")
    except json.JSONDecodeError:
        # Fallback to old behavior if not JSON
        city = state.get("city", "Unknown")
        user_query = clarification_question
    
    print(f"🔍 Query Node: Processing city='{city}', query='{user_query}'")
    
    # Step 2: Check session memory for city info
    session_data = SESSION_MEMORY.get_session(thread_id)
    city_name_from_session = session_data.get("city_name")
    
    # Validate that the city matches extracted data from session
    if city_name_from_session and city.lower() != city_name_from_session.lower():
        print(f"⚠️  City mismatch detected!")
        print(f"   Router provided: '{city}'")
        print(f"   Session has: '{city_name_from_session}'")
    elif city_name_from_session:
        print(f"✓ City validated against session: '{city}' matches '{city_name_from_session}'")
    else:
        print(f"⚠️  No city in session memory, using router-provided city: '{city}'")
    
    city_summary = None
    source = None
    
    if city_name_from_session and city_name_from_session.lower() == city.lower():
        # Check if we have cached city info in session
        cached_summary = session_data.get("cached_city_summary")
        if cached_summary:
            city_summary = cached_summary
            source = "session_memory"
            print(f"📝 Session Memory: Using cached info for {city}")
    
    # Step 3: If not in session, try ChromaDB with high similarity threshold (0.9)
    if not city_summary:
        vector_store = get_vector_store()
        result = vector_store.search_city(city, n_results=1)
        
        if result["found"]:
            distance = result.get("distance", 1.0)
            # Lower distance = better match. Distance of 0.1 means 0.9 similarity
            similarity = 1.0 - distance if distance is not None else 0.0
            
            print(f"ChromaDB similarity for '{city}': {similarity:.2f}")
            
            if similarity > 0.9:
                city_summary = result["content"].strip()
                source = "vector_store_high_confidence"
                print(f"ChromaDB: High confidence match for {city} (similarity={similarity:.2f})")
                
                # Cache in session for future use
                SESSION_MEMORY.update_session(thread_id, {
                    "cached_city_summary": city_summary
                })
            else:
                print(f"ChromaDB: Low confidence match for {city} (similarity={similarity:.2f}), falling back to web search")
        else:
            print(f"ChromaDB: No results found for {city}")
    
    # Step 4: If not in ChromaDB with high confidence, fall back to web search
    if not city_summary:
        web_query = f"Tell me about {city}"
        city_summary = mock_web_search(web_query)
        source = "web_search"
        print(f"🌐 Web Search: Fetched info for {city}")
        
        # Cache in session
        SESSION_MEMORY.update_session(thread_id, {
            "cached_city_summary": city_summary
        })
    
    # Step 5: For the user query, check ChromaDB with similarity > 0.9
    query_info = None
    if user_query and user_query.strip():
        vector_store = get_vector_store()
        query_result = vector_store.search_city(user_query, n_results=1)
        
        if query_result["found"]:
            distance = query_result.get("distance", 1.0)
            similarity = 1.0 - distance if distance is not None else 0.0
            
            print(f"ChromaDB similarity for query '{user_query}': {similarity:.2f}")
            
            if similarity > 0.9:
                query_info = query_result["content"].strip()
                print(f"ChromaDB: High confidence match for query (similarity={similarity:.2f})")
            else:
                # Fall back to web search for the specific query
                query_info = mock_web_search(user_query)
                print(f"Web Search: Fetched info for query '{user_query}'")
        else:
            # Fall back to web search
            query_info = mock_web_search(user_query)
            print(f"Web Search: Fetched info for query '{user_query}'")
        
        # Append query-specific info to city summary
        if query_info:
            city_summary = f"{city_summary}\n\n---\nAdditional Information:\n{query_info}"
    
    # Step 6: Parallel Tool Execution (Weather + Images)
    print(f"⚡ Executing tools in parallel for {city}")
    
    # Create tasks for parallel execution
    weather_task = fetch_weather_forecast(city, days=7)
    images_task = fetch_city_images(city, count=5)
    
    # Execute concurrently
    try:
        weather_data, image_urls = await asyncio.gather(weather_task, images_task)
        print(f"Parallel execution completed: {len(weather_data)} days, {len(image_urls)} images")
    except Exception as e:
        print(f"Tool execution error: {e}")
        weather_data = []
        image_urls = []
    
    # Step 7: Format structured output
    output = TravelAssistantOutput(
        city=city,
        city_summary=city_summary,
        weather_forecast=[WeatherData(**w) for w in weather_data],
        image_urls=image_urls,
        source=source
    )
    
    # Update session with result
    SESSION_MEMORY.update_session(thread_id, {
        "last_queried_city": city,
        "last_query_time": str(asyncio.get_event_loop().time())
    })
    
    updated_state = copy.deepcopy(state)
    updated_state["last_extracted_data"] = output.model_dump_json()
    
    # Store images and weather separately in state (replaced on each query)
    updated_state["image_urls"] = image_urls
    updated_state["weather_data"] = weather_data
    
    import pprint
    pprint.pprint(output.model_dump())
    updated_state["next_node"] = "router"
    updated_state["messages"] = [AIMessage(content=f"Retrieved information about {city} from {source}.")]  # Only NEW message
    return updated_state


# ============================================================================
# Node 3: User Node
# ============================================================================

async def user_node(state: AgentState) -> AgentState:
    """
    User Node: Displays output and signals that user input is needed.
    This node ALWAYS ends the current graph execution.
    The next user input will restart from Router Node.
    """
    clarification_question = state.get("clarification_question")
    final_output = state.get("final_output")
    last_extracted_data = state.get("last_extracted_data")
    session_details = state.get("session_data", {}) 
    
    # Get images and weather separately from state (don't include in LLM prompt)
    image_urls = state.get("image_urls", [])
    weather_data = state.get("weather_data", [])
    
    presentation_prompt = f"""
    You are a travel assistant AI. Present the information to the user in a friendly manner.
    **Follow the previous conversation context properly while answering.**
    If you have a clarification question, ask it politely.
    Otherwise, summarize the information retrieved and steer for further conversation.
    
    First you are provided with the initial user preferences:
    {session_details}
    """
    
    if last_extracted_data:
        # Parse the extracted data and remove image_urls and weather_forecast
        # to avoid sending them to the LLM (they'll be displayed separately in UI)
        try:
            extracted_dict = json.loads(last_extracted_data)
            # Create a clean version without images and weather
            clean_extracted = {
                "city": extracted_dict.get("city"),
                "city_summary": extracted_dict.get("city_summary"),
                "source": extracted_dict.get("source")
            }
            extraction_message = f"**Extracted Data**\n{json.dumps(clean_extracted, indent=2)}"
        except:
            extraction_message = f"**Extracted Data**\n{last_extracted_data}"
        
        presentation_prompt += f"\n\n{extraction_message}"
        
    elif clarification_question:
        clarification_message = f"**Clarification Question**\n{clarification_question}"
        presentation_prompt += f"\n\n{clarification_message}"
        
    presentation_prompt = " ".join(presentation_prompt.split())
    
    messages = state["messages"]
    llm = get_llm()
    messages.append(SystemMessage(content=presentation_prompt))
    
    try:
        response = llm.invoke(messages)
        agent_response = response.content.strip()
        print(f"👤 User Node: Generated response to user.")
    except Exception as e:
        print(f"❌ User Node error: {e}")
        agent_response = "I encountered an error while preparing the response."
    
    updated_state = copy.deepcopy(state)
    updated_state["messages"] = messages + [AIMessage(content=agent_response)]
    updated_state["next_node"] = "__end__"  
    return updated_state
    

# ============================================================================
# Conditional Edge Functions
# ============================================================================

def route_from_router(state: AgentState) -> str:
    """Route from router to either query or user node."""
    next_node = state.get("next_node", "user")
    return next_node


def route_from_query(state: AgentState) -> str:
    """Route from query to user node."""
    return "user"


# ============================================================================
# Build Graph
# ============================================================================

def build_graph():
    """
    Construct the LangGraph StateGraph with 4 nodes:
    1. Router Node - Analyzes input and routes
    2. Query Node - Queries vector DB/web search + parallel tool execution
    3. User Node - User handoff and interaction
    """
    workflow = StateGraph(AgentState)
    
    # Add the 4 nodes
    workflow.add_node("router", router_node)
    workflow.add_node("query", query_node)
    workflow.add_node("user", user_node)
    
    # Start always goes to router
    workflow.add_edge(START, "router")
    
    # Router conditionally routes to query or user
    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {
            "query": "query",
            "user": "user"
        }
    )
        
    # User node ends the graph
    workflow.add_edge("user", END)
    workflow.add_edge("query", "router")

    # Add memory/checkpointer
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    print("✓ Graph compiled with 4 nodes: Router → Query → User")
    
    return app


# ============================================================================
# Helper Functions
# ============================================================================

async def run_agent_interactive(thread_id: str = "default", user_input_callback=None):
    """
    Run the agent in interactive mode with continuous conversation loop.
    
    Flow:
    1. User provides input
    2. Router Node processes and routes
    3. Query Node fetches data (if applicable)
    4. User Node displays output and awaits next input
    5. Loop back to step 1 with new user input
    
    Args:
        thread_id: Thread ID for session management
        user_input_callback: Async function to get user input.
                           Should return the user's input as a string.
                           Return None/empty to exit.
    
    Usage:
        async def get_input(prompt):
            return input(f"{prompt}\nYour input: ")
        
        await run_agent_interactive("my_thread", get_input)
    """
    app = build_graph()
    
    # Configuration for thread-based memory
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    print("✓ Agent ready. Type your queries or 'quit' to exit.\n")
    
    # Get initial user input
    if user_input_callback:
        initial_input = await user_input_callback("Welcome to the Travel Assistant!")
        if not initial_input or initial_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            return
    else:
        raise ValueError("user_input_callback is required for interactive mode")
    
    current_input = initial_input
    messages = []
    
    while True:
        # Check token count BEFORE adding new message
        token_count_before = count_tokens_in_messages(messages)
        
        # If approaching limit, summarize BEFORE adding new user message
        if token_count_before > 6000:  # Lower threshold to catch before it gets too big
            print(f"⚠️  Message chain approaching limit ({token_count_before:,} tokens)")
            print(f"   Applying preventive summarization...")
            llm = get_llm()
            messages = check_and_summarize_messages(messages, llm, max_tokens=6000)
        
        # Add new user message
        messages.append(HumanMessage(content=current_input))
        
        # Check token count and display
        token_count = count_tokens_in_messages(messages)
        print(f"📊 Message chain: {len(messages)} messages, ~{token_count:,} tokens")
        
        # Create fresh state for each iteration (starts from router)
        current_state = {
            "messages": messages.copy(),
            "thread_id": thread_id,
            "prev_node": "user"  # Signal that this is coming from user input
        }
        
        print(f"\nProcessing: '{current_input}'\n")
        
        # Run the graph once (will stop at User Node)
        final_state = None
        async for event in app.astream(current_state, config):
            for node_name, node_output in event.items():
                if node_name == "router":
                    print(f"  🔀 Router: Analyzing query...")
                elif node_name == "query":
                    print(f"  🔍 Query: Fetching data...")
                elif node_name == "user":
                    print(f"  👤 User Node: Preparing output...\n")
                    final_state = node_output
        
        if not final_state:
            print("❌ Error: No output from graph")
            break
        
        # Update messages list with all messages from final state
        messages = final_state.get("messages", messages)
        
        # Extract the agent's response message
        agent_messages = final_state.get("messages", [])
        if agent_messages and isinstance(agent_messages[-1], AIMessage):
            agent_response = agent_messages[-1].content
        else:
            agent_response = "I processed your request."
        
        # Display the output to user
        print("=" * 80)
        
        # If there's a final output, display it
        final_output = final_state.get("final_output")
        if final_output:
            city = final_output.get("city", "Unknown")
            summary = final_output.get("city_summary", "")
            weather = final_output.get("weather_forecast", [])
            images = final_output.get("image_urls", [])
            source = final_output.get("source", "")
            
            print(f"📍 {city}")
            print(f"📊 Source: {source.replace('_', ' ').title()}")
            print("-" * 80)
            print(f"\n{summary[:300]}..." if len(summary) > 300 else f"\n{summary}")
            print(f"\n🌤️  Weather: {len(weather)} days forecast available")
            print(f"� Images: {len(images)} images available")
        else:
            # Just display the agent's message (e.g., clarification question)
            print(f"� Agent: {agent_response}")
        
        print("=" * 80)
        
        # Get next user input
        next_input = await user_input_callback("")
        
        if not next_input or next_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        current_input = next_input


async def run_agent(query: str, thread_id: str = "default") -> Dict:
    """
    Single-turn run function for non-interactive use (e.g., Streamlit).
    Executes one cycle: User Input → Router → Query/User → Results
    """
    app = build_graph()
    
    # Configuration for thread-based memory
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Create state with user query
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "thread_id": thread_id
    }
    
    # Run graph once
    final_state = None
    async for event in app.astream(initial_state, config):
        for node_name, node_output in event.items():
            if node_name == "user":
                final_state = node_output
    
    if not final_state:
        return {}
    
    # Check if clarification is needed (router sets clarification_question when clarification needed)
    clarification_question = final_state.get("clarification_question")
    last_extracted_data = final_state.get("last_extracted_data")
    
    # If there's a clarification question but no extracted data, return clarification
    if clarification_question and not last_extracted_data:
        return {
            "status": "needs_clarification",
            "clarification_question": clarification_question,
            "city": "Unknown",
            "city_summary": clarification_question,
            "weather_forecast": [],
            "image_urls": [],
            "source": "clarification"
        }
    
    # Get the last extracted data and merge with images and weather from state
    image_urls = final_state.get("image_urls", [])
    weather_data = final_state.get("weather_data", [])
    
    if last_extracted_data:
        try:
            output = json.loads(last_extracted_data)
            # Replace the images and weather with the ones from state
            output["image_urls"] = image_urls
            output["weather_forecast"] = weather_data
            return output
        except:
            return {
                "city": "Unknown",
                "city_summary": "Unable to parse response",
                "weather_forecast": weather_data,
                "image_urls": image_urls,
                "source": "error"
            }
    
    # Return empty if no data
    return {}


def clear_session(thread_id: str):
    """Clear session memory for a thread."""
    SESSION_MEMORY.clear_session(thread_id)
    print(f"🗑️  Cleared session: {thread_id}")
