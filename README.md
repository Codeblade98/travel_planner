# 🌍 Multi-Modal Travel Assistant

An intelligent AI-powered travel assistant that helps you explore cities, check weather forecasts, and plan your trips through natural conversation.

![Architecture](https://img.shields.io/badge/LangGraph-4--Node_Graph-blue)
![LLM](https://img.shields.io/badge/Groq-Llama_3.3-green)
![UI](https://img.shields.io/badge/Streamlit-Chat_Interface-red)
![Vector Store](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Search](https://img.shields.io/badge/SerpAPI-Real_Search-purple)

---

## ✨ What Can It Do?

### 🗨️ Natural Conversations
Talk to the assistant naturally - it understands context and remembers your conversation:
- **"Tell me about Paris"** → Get city information, weather, and images
- **"What's the weather like?"** → Get 7-day weather forecast
- **"What about next week?"** → It remembers you were asking about Paris

### 📸 Visual Information
- See beautiful images of cities (real Google Images via SerpAPI)
- Compact 3-column image grid for better viewing
- Weather-focused queries show data only (no unnecessary images)

### 🌤️ Weather Forecasts
- 7-day weather forecasts for any city
- Temperature, humidity, wind speed, and conditions
- Beautiful weather cards in the chat interface

### 🧠 Smart Context Understanding
- Remembers your preferences and previous questions
- Asks for clarification when needed
- Maintains conversation context across multiple messages

### 🔍 Multi-Source Information
- Pre-loaded knowledge about major cities (Paris, Tokyo, New York)
- Real-time web search for other cities
- Automatic fallback between sources

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd digialpha_task

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configuration

**Required:**
- `GROQ_API_KEY` - Get from [console.groq.com/keys](https://console.groq.com/keys)

**Optional (for better results):**
- `SERPAPI_API_KEY` - Get from [serpapi.com](https://serpapi.com/) (100 free searches/month)
  - Without it: Uses Unsplash placeholder images and basic web info
  - With it: Real Google Images and rich search results

### 3. Run the App

```bash
# Start the Streamlit chat interface
streamlit run streamlit_app.py

# Or use the helper script
./run_streamlit.sh
```

The app will open in your browser at a port on localhost

---

## 💬 How to Use

### Starting a Conversation

Once the app loads, you'll see a chat interface. Try these queries:

**City Information:**
- "Tell me about Paris"
- "What should I see in Tokyo?"
- "Describe New York for me"

**Weather Queries:**
- "What's the weather in Paris?"
- "How cold is it in Tokyo?"
- "Give me the forecast for London"

**Follow-up Questions:**
- First: "Tell me about Paris"
- Then: "What's the weather like?" *(remembers Paris)*
- Then: "What about next week?" *(still remembers Paris)*

**Multi-city Exploration:**
- "Compare Paris and Tokyo"
- "I want to visit Tokyo"
- "What about the weather there?"

### Features in the Chat Interface

**Images** 📸
- Appear at the top of responses (when relevant)
- Compact 3-column layout

**Weather Cards** 🌤️
- First 3 days shown in cards
- More days available in expandable section
- Shows temperature, humidity, and conditions

**Session Management** 💾
- View current conversation memory
- Clear chat history
- Start fresh conversations

---

## 🎨 Interface Features

### Main Chat
- **Clean message bubbles** for user and assistant
- **Inline images** (compact 250px width, 3-column grid)
- **Weather cards** with icons and metrics
- **Smooth scrolling** conversation history

### Sidebar
- **Session info** - View what the AI remembers
- **Clear history** - Start a fresh conversation
- **About section** - Tech stack information

### Smart Display
- Weather queries: Text only
- City queries: Images + text + weather
- Explicit requests: Always shows what you ask for

---

## 🔧 Configuration Options

### Environment Variables

```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here 
```

### Parallel Execution
```python
async def parallel_fetch_node(state: AgentState):
    weather_task = fetch_weather_forecast(city)
    images_task = fetch_city_images(city)
    
    # Concurrent execution
    weather, images = await asyncio.gather(weather_task, images_task)
```

### Memory Persistence
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user_session_123"}}
result = await app.ainvoke(initial_state, config)
```

---

## 🎨 UI Features

### Weather Visualization
- **Current Metrics**: Temperature, condition, humidity, wind speed
- **Detailed Table**: Expandable forecast with all data points

### Image Gallery
- **Responsive Grid**: 3-column layout
- **High-Quality Images**: Fetched from Unsplash (mock URLs)
- **Error Handling**: Graceful fallback for failed loads

### Conversation Management
- **Thread-Based Memory**: Each session has a unique ID
- **Chat History**: Full conversation display
- **Reset Option**: Clear history and start fresh

---

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key ([Get it here](https://console.groq.com/keys)) | Yes |
| `SERPAPI_API_KEY` | Your SerpAPI key for real image/web search ([Get it here](https://serpapi.com/)) | No |

**Note**: Without `SERPAPI_API_KEY`, the app uses fallback mode (Unsplash images + mock data).

### API Implementations

#### Weather API (Mock)
Simulates realistic weather forecasts for any city.

#### Image Search (SerpAPI + Fallback)
- **With SerpAPI**: Real Google Images results
- **Without SerpAPI**: Unsplash placeholder URLs
- **Automatically falls back** if API key missing or error occurs

#### Web Search (SerpAPI + Fallback)
- **With SerpAPI**: Real Google search results (answer boxes, knowledge graphs, snippets)
- **Without SerpAPI**: Pre-defined city information
- **Automatically falls back** if API key missing or error occurs

See [`SERPAPI_INTEGRATION.md`](./SERPAPI_INTEGRATION.md) for detailed documentation.

---

## 🧩 Extending the System

### Add More Cities to Vector Store
Edit `vector_store.py`:
```python
cities_data = [
    {
        "id": "london",
        "city": "London",
        "content": "Detailed information about London...",
        "metadata": {"country": "UK", ...}
    }
]
```

### Add New Tools
Edit `agent.py`:
```python
TOOLS.append({
    "type": "function",
    "function": {
        "name": "book_hotel",
        "description": "Book a hotel in the city",
        "parameters": {...}
    }
})
```

### Customize UI
Edit `streamlit_app.py`:
- Modify CSS in the `st.markdown()` section
- Add new visualizations (maps, reviews, etc.)
- Change layout and color schemes

---

## 🧪 Testing

### Test SerpAPI Integration
```bash
# Test with/without API key
python test_serpapi.py
```

Expected output:
- ✓ With API key: Real images from Google Images, real search results
- ✓ Without API key: Unsplash fallback images, mock search results

### Test Image Display in Streamlit
```bash
# Run the Streamlit app
streamlit run streamlit_app.py

# Query: "Tell me about Paris"
# Check: Images appear before text response
```

### Test Agent Directly
```bash
# Run CLI interface
python main.py

# Try queries:
# - "Tell me about Tokyo"
# - "What's the weather like?"
# - "Show me Paris images"
```

### Run All Tests
```bash
# Run test suite (if available)
pytest tests/
```

---

## 📊 Performance

- **Vector Search**: < 100ms
- **Weather API (Mock)**: 0.5-1.5s
- **Image API (SerpAPI)**: 0.5-2.0s (or 0.3-1.0s fallback)
- **Web Search (SerpAPI)**: 0.5-2.0s (or instant fallback)
- **Parallel Execution**: ~1-2s total (vs ~3-4s sequential)
- **LLM Response**: 1-3s (depends on Groq load)

**Total Latency**: 
- With SerpAPI: ~4-7 seconds from query to display
- Fallback mode: ~3-5 seconds from query to display

---

## 🐛 Troubleshooting

### SerpAPI Not Working
```bash
# Verify API key is set
cat .env | grep SERPAPI

# Test SerpAPI directly
python test_serpapi.py

# Check console for error messages:
# - "📸 SerpAPI: Retrieved X images" = Working
# - "⚠️ SERPAPI_API_KEY not found" = Key missing
# - "❌ SerpAPI error" = API issue (check quota/key)
```

### Images Not Displaying in Streamlit
```bash
# Check console for debug output
# Should see: "[DEBUG] Image URLs: ['https://...']"

# If using SerpAPI, verify rate limits
# Free tier: 100 searches/month
```

### ChromaDB Issues
```bash
# Clear vector store and reinitialize
rm -rf chroma_db/
python -c "from vector_store import get_vector_store; get_vector_store()"
```

### API Key Errors
```bash
# Verify .env file exists and is loaded
cat .env | grep GROQ_API_KEY
cat .env | grep SERPAPI_API_KEY
```

### Dependency Conflicts
```bash
# Reinstall in fresh venv
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## 🔒 Security Notes

- Never commit `.env` file to version control
- API keys should be stored securely
- Use environment variables in production
- Implement rate limiting for public deployments

---

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Groq API Docs](https://console.groq.com/docs)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [Streamlit API Reference](https://docs.streamlit.io/)

---

## 🤝 Contributing

This is a technical challenge submission. For production use:
1. Replace mock APIs with real integrations
2. Add error handling and retry logic
3. Implement proper logging
4. Add unit and integration tests
5. Set up CI/CD pipeline

---

## 📄 License

This project is created for educational purposes as part of an AI Engineering technical challenge.

---

## 👨‍💻 Author

**Senior AI Engineer Candidate**
- Demonstrates: LangGraph orchestration, vector search, parallel execution, stateful memory
- Tech Stack: Python 3.12, LangChain, Groq, ChromaDB, Streamlit, Plotly

---

## 🌟 Distinctions Implemented

✅ **Manual Tool Calling**: Custom parsing without `create_tool_calling_agent`  
✅ **Parallel Fan-Out**: Concurrent weather + image fetching  
✅ **Human-in-the-Loop**: Memory persistence with checkpointer  
✅ **Structured Output**: JSON schema instead of raw Markdown  
✅ **Intelligent Routing**: Conditional edges based on knowledge availability  
✅ **Rich UI**: Interactive charts, galleries, and metrics

---

**Built with ❤️ using LangGraph and Streamlit**
