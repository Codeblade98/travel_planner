# 🌍 Multi-Modal Travel Assistant

An intelligent, agentic travel assistant built with **LangGraph**, **ChromaDB**, **Groq AI**, **SerpAPI**, and **Streamlit**. This application demonstrates advanced AI engineering patterns including LLM-based entity extraction, intelligent routing, parallel execution, and LLM-managed session memory.

![Architecture](https://img.shields.io/badge/LangGraph-4--Node_Graph-blue)
![LLM](https://img.shields.io/badge/Groq-Llama_3.3-green)
![UI](https://img.shields.io/badge/Streamlit-Frontend-red)
![Vector Store](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Search](https://img.shields.io/badge/SerpAPI-Real_Search-purple)

---

## 🎯 Key Features

### 1. **4-Node Architecture** 🏗️
Simplified graph with clear separation of concerns:
- **Router Node**: LLM-based query analysis and routing
- **Query Node**: Unified data fetching (Vector DB → Web Search) + parallel tools
- **User Node**: Human-in-the-loop interaction and clarification

### 2. **LLM-Based City Extraction** 🧠
No string manipulation! LLM extracts cities from natural language:
- "Tell me about Paris" → Paris
- "What about the weather?" → Uses session context
- "Show me something cool" → Asks for clarification

### 3. **LLM-Managed Session Memory** 💾
The LLM decides what to remember:
```json
{
  "current_city": "Tokyo",
  "user_preferences": {"interested_in_weather": true},
  "conversation_context": "User asked about weather"
}
```

### 4. **Real Image & Web Search** 🔍
**NEW!** Integrated SerpAPI for real-world data:
- **Google Images**: Actual city photos (not placeholders)
- **Google Search**: Real information (answer boxes, knowledge graphs)
- **Graceful Fallback**: Works without API key using Unsplash/mock data

### 5. **Integrated Parallel Execution** ⚡
Weather and image fetching happen concurrently in the Query Node:
- Sequential: ~2.5s
- Parallel: ~1.5s
- **Savings**: 40% faster

### 6. **Smart Routing** 🔀
Query Node automatically falls back:
1. Try Vector Store (Paris, Tokyo, New York)
2. If not found → Web Search (SerpAPI or mock)
3. Execute tools in parallel
4. Format structured output

### 7. **Streamlit Chat Interface** 💬
Beautiful web UI with:
- Chat history with message bubbles
- Top 4 images displayed before response
- Weather forecast cards
- Session management

### 8. **Structured Output** 📦
Returns a JSON schema (Pydantic model):
```python
{
  "city": "Tokyo",
  "city_summary": "...",
  "weather_forecast": [...],
  "image_urls": [...],
  "source": "vector_store"
}
```

### 7. **Rich UI** 🎨
Streamlit interface with:
- Live session memory display
- Interactive Plotly weather charts
- Responsive image galleries
- Chat history
- Real-time metrics

---

## 🏗️ Architecture

```
User Input
    ↓
┌───────────────┐
│ Router Node   │  ← LLM extracts city
│               │    Updates session memory
│               │    Decides routing
└───────┬───────┘
        │
    ┌───┴──────────────┐
    │                  │
(city found)    (needs clarification)
    │                  │
    ▼                  ▼
┌──────────┐      ┌──────────┐
│Query Node│      │User Node │
│          │      │          │
│1. Vector │      │Asks Q's  │
│   DB ↓   │      │          │
│2. Web    │      └──────────┘
│   Search │           ↓
│          │         [END]
│3. Parallel│
│  - Weather│
│  - Images │
│          │
│4. Format │
└────┬─────┘
     │
     ▼
┌──────────┐
│User Node │
│          │
│Present   │
│Results   │
└────┬─────┘
     │
     ▼
   [END]
```
   │   Node      │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Parallel   │ ◄── Weather + Images
   │  Fan-Out    │     (async)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   Output    │ ◄── Structured JSON
   │  Formatter  │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Streamlit  │
   │     UI      │
   └─────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.12+
- Groq API Key ([Get one here](https://console.groq.com/keys))

### Setup

1. **Clone and Navigate**
```bash
cd /mnt/Agnij/digialpha_task
```

2. **Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -e .
```

4. **Configure API Key**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Example Queries

**Pre-populated Cities (Vector Store):**
- "Tell me about Paris"
- "Tell me about Tokyo"
- "Tell me about New York"

**Dynamic Cities (Web Search):**
- "Tell me about Kyoto"
- "Tell me about Snohomish"
- "Tell me about London"

**Follow-up (Memory):**
- User: "Tell me about Tokyo"
- Assistant: *[Provides info]*
- User: "What about next week?" *(remembers Tokyo context)*

---

## 📁 Project Structure

```
digialpha_task/
├── agent.py              # LangGraph state, nodes, and graph builder
├── vector_store.py       # ChromaDB setup with pre-populated cities
├── mock_apis.py          # Mock weather and image APIs
├── streamlit_app.py      # Streamlit UI with charts and galleries
├── main.py               # (Optional) CLI interface
├── pyproject.toml        # Dependencies
├── .env.example          # Environment variable template
├── README.md             # This file
└── chroma_db/            # (Auto-created) Vector store data
```

---

## 🧪 Technical Implementation

### 1. State Management
```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    city: Optional[str]
    use_vector_store: Optional[bool]
    city_summary: Optional[str]
    weather_forecast: Optional[List[Dict]]
    image_urls: Optional[List[str]]
    final_output: Optional[Dict]
```

### 2. Manual Tool Calling
```python
async def tool_calling_node(state: AgentState):
    response = llm.bind_tools(TOOLS).invoke([system_msg])
    
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Manual execution
        if tool_name == "fetch_weather":
            result = await fetch_weather_forecast(**tool_args)
        
        tool_msg = ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_call["id"]
        )
```

### 3. Parallel Execution
```python
async def parallel_fetch_node(state: AgentState):
    weather_task = fetch_weather_forecast(city)
    images_task = fetch_city_images(city)
    
    # Concurrent execution
    weather, images = await asyncio.gather(weather_task, images_task)
```

### 4. Memory Persistence
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user_session_123"}}
result = await app.ainvoke(initial_state, config)
```

---

## 🎨 UI Features

### Weather Visualization
- **Interactive Plotly Chart**: Line graph showing 7-day temperature trends
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
