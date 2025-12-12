import asyncio
import streamlit as st
from agent import run_agent_interactive, clear_session, SESSION_MEMORY, build_graph
from langchain_core.messages import HumanMessage, AIMessage
import json

# Page configuration
st.set_page_config(
    page_title="🌍 Travel Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin: 15px 0;
    }
    .image-grid img {
        width: 100%;
        max-width: 250px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    /* Make images in chat more compact */
    .stImage {
        margin: 0.5rem 0;
    }
    }
    .weather-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session"

if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = build_graph()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "processing" not in st.session_state:
    st.session_state.processing = False


def display_images(image_urls, max_images=4, small=True):
    """Display images in a grid layout.
    
    Args:
        image_urls: List of image URLs
        max_images: Maximum number of images to display (default 4)
        small: If True, display smaller images (default True)
    """
    if not image_urls:
        return
    
    # Take top 4 images
    images_to_show = image_urls[:max_images]
    
    if len(images_to_show) > 0:
        st.markdown("### 📸 Images")
        
        # Use 3 columns for smaller images, or 2 for larger
        num_cols = 3 if small else 2
        cols = st.columns(num_cols)
        
        for idx, img_url in enumerate(images_to_show):
            with cols[idx % num_cols]:
                try:
                    # Set width to smaller value for compact display
                    if small:
                        st.image(img_url, width=250)
                    else:
                        st.image(img_url, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image {idx+1}: {str(e)}")



def display_weather(weather_data):
    """Display weather forecast data."""
    if not weather_data:
        return
    
    st.markdown("### 🌤️ Weather Forecast")
    
    # Display first 3 days in columns
    if len(weather_data) > 0:
        cols = st.columns(min(3, len(weather_data)))
        for idx, day in enumerate(weather_data[:3]):
            with cols[idx]:
                st.markdown(f"""
                <div class="weather-card">
                    <strong>{day.get('date', 'N/A')}</strong><br>
                    🌡️ {day.get('temperature', 'N/A')}°C<br>
                    💧 {day.get('humidity', 'N/A')}% humidity<br>
                    {day.get('condition', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
    
    # Show remaining days in expander
    if len(weather_data) > 3:
        with st.expander(f"Show {len(weather_data) - 3} more days"):
            for day in weather_data[3:]:
                st.markdown(f"""
                **{day.get('date', 'N/A')}** - {day.get('temperature', 'N/A')}°C, 
                {day.get('condition', 'N/A')}, {day.get('humidity', 'N/A')}% humidity
                """)


def is_weather_only_query(user_input: str) -> bool:
    """Check if the query is primarily about weather (should not show images)."""
    weather_keywords = [
        'weather', 'forecast', 'temperature', 'rain', 'sunny', 'cold', 'hot',
        'climate', 'humidity', 'wind', 'snow', 'storm', 'degrees', 'celsius',
        'fahrenheit', 'conditions', 'what\'s the weather', 'how\'s the weather'
    ]
    
    user_input_lower = user_input.lower()
    
    # Check if query contains weather keywords
    has_weather_keyword = any(keyword in user_input_lower for keyword in weather_keywords)
    
    # Check if query does NOT contain image/photo/picture keywords
    image_keywords = ['image', 'photo', 'picture', 'show me', 'see', 'look']
    has_image_keyword = any(keyword in user_input_lower for keyword in image_keywords)
    
    # It's weather-only if it has weather keywords and no image keywords
    return has_weather_keyword and not has_image_keyword


async def process_user_input(user_input: str):
    """Process user input through the agent graph."""
    app = st.session_state.agent_graph
    thread_id = st.session_state.thread_id
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Get existing messages from session state
    messages = st.session_state.message_history.copy()
    
    # Add new user message
    messages.append(HumanMessage(content=user_input))
    
    # Create state
    current_state = {
        "messages": messages.copy(),
        "thread_id": thread_id,
        "prev_node": "user"
    }
    
    # Run the graph
    final_state = None
    node_updates = []
    
    async for event in app.astream(current_state, config):
        for node_name, node_output in event.items():
            node_updates.append(f"Processing through {node_name}...")
            print(f"[DEBUG] Node: {node_name}")
            if node_name == "query":
                # Check if images are in the query node output
                query_images = node_output.get("image_urls", [])
                query_weather = node_output.get("weather_data", [])
                print(f"[DEBUG] Query Node - Images: {len(query_images)}, Weather: {len(query_weather)}")
            if node_name == "user":
                final_state = node_output
                # Check if images made it to user node output
                user_images = node_output.get("image_urls", [])
                user_weather = node_output.get("weather_data", [])
                print(f"[DEBUG] User Node - Images: {len(user_images)}, Weather: {len(user_weather)}")
    
    if not final_state:
        return None, "Error: No output from agent", [], []
    
    # Update message history
    st.session_state.message_history = final_state.get("messages", messages)
    
    # Extract agent response
    agent_messages = final_state.get("messages", [])
    agent_response = ""
    if agent_messages and isinstance(agent_messages[-1], AIMessage):
        agent_response = agent_messages[-1].content
    else:
        agent_response = "I processed your request."
    
    # Extract structured output
    final_output = final_state.get("final_output")
    
    # Get image_urls and weather_data from state (they're stored at top level)
    image_urls = final_state.get("image_urls", [])
    weather_data = final_state.get("weather_data", [])
    
    # Debug info
    print(f"[DEBUG] State keys: {final_state.keys()}")
    print(f"[DEBUG] Image URLs from state: {image_urls}")
    print(f"[DEBUG] Weather data from state: {len(weather_data)} items")
    
    if final_output:
        # Override with values from final_output if they exist
        image_urls = final_output.get("image_urls", image_urls)
        weather_data = final_output.get("weather_forecast", weather_data)
        print(f"[DEBUG] Image URLs from final_output: {image_urls}")
        print(f"[DEBUG] Weather data from final_output: {len(weather_data)} items")
        
        # Build response text
        city = final_output.get("city", "Unknown")
        summary = final_output.get("city_summary", "")
        source = final_output.get("source", "")
        
        response_text = f"**📍 {city}**\n\n"
        response_text += f"*Source: {source.replace('_', ' ').title()}*\n\n"
        response_text += summary
    else:
        response_text = agent_response
    
    # Determine if we should show images (not for weather-only queries)
    show_images = not is_weather_only_query(user_input) if image_urls else False
    
    return final_output, response_text, image_urls if show_images else [], weather_data


def run_async(coro):
    """Helper to run async functions in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# Sidebar
with st.sidebar:
    st.title("🌍 Travel Assistant")
    st.markdown("---")
    
    st.markdown("""
    ### Features
    - 💬 Natural conversation
    - 🌤️ Weather forecasts
    - 📸 City images
    - 🧠 Session memory
    - 🔍 Vector store + web search
    """)
    
    st.markdown("---")
    
    # Session management
    st.subheader("Session Management")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.message_history = []
        clear_session(st.session_state.thread_id)
        st.rerun()
    
    if st.button("💾 View Session Memory"):
        session_data = SESSION_MEMORY.get_session(st.session_state.thread_id)
        st.json(session_data)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Multi-modal travel assistant powered by:
    - **LangGraph** for agent orchestration
    - **ChromaDB** for vector storage
    - **Groq LLMs** for intelligence
    """)

# Main chat interface
st.title("🌍 Travel Assistant Chat")
st.markdown("Ask me about cities, weather, travel plans, and more!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display images if present (for assistant messages) - smaller size
        if message["role"] == "assistant" and "images" in message and message["images"]:
            display_images(message["images"], small=True)
        
        # Display text content
        st.markdown(message["content"])
        
        # Display weather if present (for assistant messages)
        if message["role"] == "assistant" and "weather" in message and message["weather"]:
            display_weather(message["weather"])

# Chat input
if prompt := st.chat_input("Ask about a city, weather, or travel plans..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                final_output, response_text, image_urls, weather_data = run_async(
                    process_user_input(prompt)
                )
                
                # Debug output
                print(f"[DEBUG DISPLAY] Image URLs: {image_urls}")
                print(f"[DEBUG DISPLAY] Weather data: {len(weather_data) if weather_data else 0} items")
                print(f"[DEBUG DISPLAY] Show images: {len(image_urls) > 0}")
                
                # Display images FIRST if available (top 4, smaller size)
                if image_urls:
                    display_images(image_urls, max_images=4, small=True)
                
                # Then display the text response
                st.markdown(response_text)
                
                # Then display weather data
                if weather_data:
                    display_weather(weather_data)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "images": image_urls[:4] if image_urls else [],
                    "weather": weather_data
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "images": [],
                    "weather": []
                })

# Welcome message if no messages yet
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! Ask me about cities, weather, or travel plans.\n\n"
            "💡 **Tip:** Weather queries show data only (no images). "
            "City queries show compact images + info!")
