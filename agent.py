import os
import json
import asyncio
import copy
from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from logger import log_llm_response, log_node_execution, log_api_call, agent_logger, api_logger
from utils.llm_utils import get_llm, invoke_llm_with_retry, parse_llm_json_response, switch_to_next_model, handle_rate_limit
from utils.utils import SESSION_MEMORY, count_tokens_in_messages, check_and_summarize_messages, clear_session
from vector_store import get_vector_store
from utils.api_utils import fetch_weather_forecast, fetch_city_images, execute_web_search
from models import TravelAssistantOutput, CityExtractionResult, WeatherData

# Load environment variables
load_dotenv()


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


# Router Node

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
        
        **Instructions**:
        - If no city is found, you need not ask any clarification question.
        - Clarification questions build upon given session data. If enough data is present, do not ask for clarification.
        - Prefer not asking clarification questions if sufficient data is present.

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
            
            log_node_execution("router", f"Extracted data: {extraction_result}", "info")
            
            # Update session memory
            if extraction_result:
                updation_summary = extraction_result["summary"]
                extraction_result.pop("summary")
                SESSION_MEMORY.update_session(thread_id, extraction_result)
        
        except Exception as e:
            agent_logger.error(f"Router error: {e}", exc_info=True)
            updated_state["clarification_question"] = "I had trouble understanding your request. Could you please specify which city you're interested in?"
            updated_state["next_node"] = "user"
            return updated_state
    
        # Case 1: User has given some input 
        #    (a) User responded with clarification → query
        #    (b) User did not clarify → ask again
        # Case 2: User is doing randon chit-chat
        #     Ask user for his travel preferences 
        clarification_question = extraction_result.get("clarification_question")
        # print(extraction_result)
        # 2
        if not extraction_result.get("city_found", False):
            updated_state["next_node"] = "user"
            clarification_question = "Please specify the what are your travel preferences?"
            updated_state["clarification_question"] = clarification_question
            updated_state["messages"] = [AIMessage(content=clarification_question)] 
            return updated_state     
        # 1b
        elif state.get("clarification_question") and extraction_result.get("needs_clarification", False):
            updated_state["next_node"] = "user"
            updated_state["clarification_question"] = clarification_question
            if clarification_question is None:
                clarification_question = "Could you please clarify your travel preferences?"
            updated_state["messages"] = [AIMessage(content=clarification_question)] 
            return updated_state          
        # 1a
        else:
            agent_logger.info("No clarification needed, proceeding to query node")
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
            
            agent_logger.debug(f"Router message: {message}")
            
            query_to_send = {
                "city": router_result.get("city", "Unknown"),
                "query": router_result.get("query", "")
            }
            updated_state["clarification_question"] = json.dumps(query_to_send)
            updated_state["next_node"] = "query"
            updated_state["messages"] = [AIMessage(content=message)]  # Only NEW message
            return updated_state
        
        except Exception as e:
            agent_logger.error(f"Router (User prev) error: {e}", exc_info=True)
            updated_state["clarification_question"] = "I had trouble understanding your request. Could you please specify which city you're interested in?"
            updated_state["next_node"] = "user"
            return updated_state
        
    elif prev_node == "query":
        updated_state["next_node"] = "user"
        return updated_state


# Query Node
async def query_node(state: AgentState) -> AgentState:
    """
    Query Node: Fetches city information and executes tools in parallel.
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
    
    log_node_execution("query", f"Processing city='{city}', query='{user_query}'", "info")
    
    # Step 2: Check session memory for city info
    session_data = SESSION_MEMORY.get_session(thread_id)
    city_name_from_session = session_data.get("city_name")
    
    city_summary = None
    source = None
    
    if city_name_from_session and city_name_from_session.lower() == city.lower():
        # Check if we have cached city info in session
        cached_summary = session_data.get("cached_city_summary")
        if cached_summary:
            city_summary = cached_summary
            source = "session_memory"
            agent_logger.info(f"Session Memory: Using cached info for {city}")
    
    # Step 3: If not in session, try ChromaDB with high similarity threshold (0.9)
    if not city_summary:
        vector_store = get_vector_store()
        result = vector_store.search_city(city, n_results=1)
        
        if result["found"]:
            distance = result.get("distance", 1.0)
            # Lower distance = better match. Distance of 0.1 means 0.9 similarity
            similarity = 1.0 - distance if distance is not None else 0.0
            
            agent_logger.debug(f"ChromaDB similarity for '{city}': {similarity:.2f}")
            
            if similarity > 0.9:
                city_summary = result["content"].strip()
                source = "vector_store_high_confidence"
                agent_logger.info(f"ChromaDB: High confidence match for {city} (similarity={similarity:.2f})")
                
                # Cache in session for future use
                SESSION_MEMORY.update_session(thread_id, {
                    "cached_city_summary": city_summary
                })
            else:
                agent_logger.info(f"ChromaDB: Low confidence match for {city} (similarity={similarity:.2f}), falling back to web search")
        else:
            agent_logger.info(f"ChromaDB: No results found for {city}")
    
    # Step 4: If not in ChromaDB with high confidence, fall back to web search
    if not city_summary:
        web_query = f"Tell me about {city}"
        city_summary = execute_web_search(web_query)
        source = "web_search"
        log_api_call("WebSearch", f"Fetched info for {city}", success=True)
        
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
            
            agent_logger.debug(f"ChromaDB similarity for query '{user_query}': {similarity:.2f}")
            
            if similarity > 0.9:
                query_info = query_result["content"].strip()
                agent_logger.info(f"ChromaDB: High confidence match for query (similarity={similarity:.2f})")
            else:
                # Fall back to web search for the specific query
                query_info = execute_web_search(user_query)
                log_api_call("WebSearch", f"Fetched info for query '{user_query}'", success=True)
        else:
            # Fall back to web search
            query_info = execute_web_search(user_query)
            log_api_call("WebSearch", f"Fetched info for query '{user_query}'", success=True)
        
        # Append query-specific info to city summary
        if query_info:
            city_summary = f"{city_summary}\n\n---\nAdditional Information:\n{query_info}"
    
    # Step 6: Fetch weather and images
    log_node_execution("query", f"Executing tools in parallel for {city}", "info")
    
    # Create tasks for parallel execution
    weather_task = fetch_weather_forecast(city, days=7)
    images_task = fetch_city_images(city, count=5)
    
    # Execute concurrently
    try:
        weather_data, image_urls = await asyncio.gather(weather_task, images_task)
        agent_logger.info(f"Parallel execution completed: {len(weather_data)} days, {len(image_urls)} images")
    except Exception as e:
        agent_logger.error(f"Tool execution error: {e}", exc_info=True)
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
    
    agent_logger.debug(f"Output: {output.model_dump()}")
    updated_state["next_node"] = "router"
    updated_state["messages"] = [AIMessage(content=f"Retrieved information about {city} from {source}.")]  # Only NEW message
    return updated_state


# User Node
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
        log_node_execution("user", "Generated response to user", "info")
    except Exception as e:
        agent_logger.error(f"User Node error: {e}", exc_info=True)
        agent_response = "I encountered an error while preparing the response."
    
    updated_state = copy.deepcopy(state)
    updated_state["messages"] = messages + [AIMessage(content=agent_response)]
    updated_state["next_node"] = "__end__"  
    return updated_state
    

# Conditional Edge Functions
def route_from_router(state: AgentState) -> str:
    """Route from router to either query or user node."""
    next_node = state.get("next_node", "user")
    return next_node


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
    try:
        app = workflow.compile(checkpointer=memory)
    except Exception as e:
        agent_logger.error(f"Graph compilation error: {e}", exc_info=True)
        raise e
        
    return app

async def run_agent_interactive(thread_id: str = "default", user_input_callback=None):
    """
    Run the agent in interactive mode with continuous conversation loop in CLI
    """
    app = build_graph()
    
    # Configuration for thread-based memory
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    agent_logger.info("Agent ready. Type your queries or 'quit' to exit.")
    
    # Get initial user input
    if user_input_callback:
        initial_input = await user_input_callback("Welcome to the Travel Assistant!")
        if not initial_input or initial_input.lower() in ['quit', 'exit', 'q']:
            agent_logger.info("Goodbye!")
            return
    else:
        raise ValueError("user_input_callback is required for interactive mode")
    
    current_input = initial_input
    messages = []
    
    while True:
        token_count_before = count_tokens_in_messages(messages)
        
        if token_count_before > 6000:  # Lower threshold to catch before it gets too big
            agent_logger.warning(f"Message chain approaching limit ({token_count_before:,} tokens)")
            agent_logger.info("Applying preventive summarization...")
            llm = get_llm()
            messages = check_and_summarize_messages(messages, llm, max_tokens=6000)
        
        # Add new user message
        messages.append(HumanMessage(content=current_input))
        
        # Check token count and display
        token_count = count_tokens_in_messages(messages)
        agent_logger.debug(f"Message chain: {len(messages)} messages, ~{token_count:,} tokens")
        
        # Create fresh state for each iteration (starts from router)
        current_state = {
            "messages": messages.copy(),
            "thread_id": thread_id,
            "prev_node": "user"  # Signal that this is coming from user input
        }
        
        agent_logger.info(f"Processing: '{current_input}'")
        
        # Run the graph once 
        final_state = None
        async for event in app.astream(current_state, config):
            for node_name, node_output in event.items():
                if node_name == "router":
                    agent_logger.debug("Router: Analyzing query...")
                elif node_name == "query":
                    agent_logger.debug("Query: Fetching data...")
                elif node_name == "user":
                    agent_logger.debug("User Node: Preparing output...")
                    final_state = node_output
        
        if not final_state:
            agent_logger.error("Error: No output from graph")
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
        print(f"� Agent: {agent_response}")
        
        print("=" * 80)
        
        # Get next user input
        next_input = await user_input_callback("")
        
        if not next_input or next_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        current_input = next_input

