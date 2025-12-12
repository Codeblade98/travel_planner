# 🌍 Multi-Modal Travel Assistant - Architecture Overview

An intelligent AI-powered travel assistant built with **LangGraph**, featuring a stateful multi-node graph architecture for orchestrating conversational AI, vector search, and parallel API execution.

![Architecture](https://img.shields.io/badge/LangGraph-Stateful_Graph-blue)
![LLM](https://img.shields.io/badge/Groq-Llama_3.3_70B-green)
![UI](https://img.shields.io/badge/Streamlit-Real--time_Chat-red)
![Vector Store](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Search](https://img.shields.io/badge/SerpAPI-Live_Search-purple)

---

## 🏗️ System Architecture

### Architecture Philosophy

This system implements a **state-driven, graph-based conversational agent** that:
- **Decouples concerns** through distinct nodes (routing, data fetching, user interaction)
- **Maintains stateful memory** across conversation turns using LangGraph's checkpointing
- **Optimizes performance** via parallel API execution
- **Ensures reliability** through multi-source data retrieval with automatic fallbacks
- **Scales gracefully** using token-aware message summarization

---

## 📊 High-Level Architecture Diagram

```mermaid
graph TB
    %% User Interfaces
    CLI[CLI Interface]
    WEB[Streamlit Web App]
    
    %% Core Agent System
    subgraph "LangGraph Stateful Workflow"
        START([START])
        ROUTER[Router Node<br/>City Extraction & Routing]
        QUERY[Query Node<br/>Data Aggregation]
        USER[User Node<br/>Response Generation]
        END([END])
    end
    
    %% State Management
    subgraph "State & Memory Layer"
        STATE[AgentState<br/>Messages, Session Data, Images, Weather]
        SESSION[Session Memory<br/>User Preferences, Context Cache]
        CHECKPOINT[MemorySaver<br/>Thread-based Persistence]
    end
    
    %% LLM Services
    subgraph "LLM Intelligence Layer"
        LLM[ChatGroq<br/>3 Fallback Models]
        RETRY[Rate Limit Handler<br/>Auto Model Switching]
        SUMMARIZE[Token Management<br/>Auto-Summarization at 8K]
    end
    
    %% Data Sources
    subgraph "Knowledge Layer"
        VECTOR[(ChromaDB<br/>Pre-loaded Cities)]
        WEB_SEARCH[SerpAPI Web Search<br/>Fallback Source]
    end
    
    %% External APIs
    subgraph "External Data Sources"
        WEATHER[Weather API<br/>Mock Forecast]
        IMAGES[Image Search<br/>SerpAPI/Fallback]
        SEARCH[Web Search<br/>SerpAPI/Fallback]
    end
    
    %% Flows
    CLI --> START
    WEB --> START
    START --> ROUTER
    ROUTER -->|City Found| QUERY
    ROUTER -->|Needs Clarification| USER
    QUERY --> ROUTER
    USER --> END
    
    ROUTER <--> STATE
    QUERY <--> STATE
    USER <--> STATE
    STATE <--> SESSION
    STATE <--> CHECKPOINT
    
    ROUTER --> LLM
    QUERY --> LLM
    USER --> LLM
    
    QUERY --> VECTOR
    QUERY --> WEB_SEARCH
    QUERY ==>|Parallel| WEATHER
    QUERY ==>|Parallel| IMAGES
```

---

## 🔄 Graph Flow Architecture

### Node Structure

The system uses a **4-node LangGraph** with conditional routing:

#### 1️⃣ **Router Node** (Decision Point)
**Responsibility**: Intelligent query analysis and traffic routing

**Process**:
1. Receives user input via `AgentState.messages`
2. Invokes LLM to extract:
   - City name (if present)
   - User preferences (trip type, travel preferences, name)
   - Need for clarification
3. Updates `SessionMemory` with extracted preferences
4. Routes to:
   - **Query Node** if city is clear and sufficient context exists
   - **User Node** if clarification needed or no city found

**Key Features**:
- LLM-driven session management (dynamic key-value updates)
- Token-aware message chain (triggers summarization at 8K tokens)
- Contextual clarification questions

**State Updates**:
```python
{
    "next_node": "query" | "user",
    "clarification_question": JSON serialized query data,
    "session_data": {user_name, city_name, trip_type, preferences}
}
```

---

#### 2️⃣ **Query Node** (Data Orchestration)
**Responsibility**: Multi-source data retrieval with intelligent fallbacks

**Process**:
1. **Parse Query**: Extract city and search query from router output
2. **Session Cache Check**: Look for cached city summary in session
3. **Vector Search** (ChromaDB):
   - Query with semantic similarity
   - Accept only high-confidence matches (similarity > 0.9)
   - Cache results in session for future queries
4. **Web Search Fallback** (SerpAPI):
   - Triggered when vector similarity < 0.9
   - Fetches real-time information
   - Caches in session
5. **Parallel API Execution**:
   ```python
   weather, images, web_results = await asyncio.gather(
       fetch_weather_forecast(city),
       fetch_city_images(city),
       execute_web_search(query)
   )
   ```

**Data Flow**:
```
Query Node
├── Session Cache Hit? → Use cached data
├── Vector Search (similarity > 0.9) → Cache & use
├── Web Search (fallback) → Cache & use
└── Parallel Fetch:
    ├── Weather API
    ├── Image Search
    └── Web Search
```

**State Updates**:
```python
{
    "image_urls": List[str],  # Replaced each query
    "weather_data": List[Dict],  # Replaced each query
    "last_extracted_data": JSON city summary,
    "next_node": "router"  # Loop back for response
}
```

---

#### 3️⃣ **User Node** (Response Synthesis)
**Responsibility**: Natural language response generation

**Process**:
1. Receives full context from `AgentState`
2. Constructs prompt with:
   - Conversation history
   - Session memory (user preferences)
   - Fetched data (city info, weather, images)
3. Invokes LLM for natural response generation
4. Returns formatted response to user

**State Updates**:
```python
{
    "messages": messages + [AIMessage(content=response)],
    "next_node": "__end__"
}
```

---

### Conditional Routing Logic

```python
def route_from_router(state: AgentState) -> str:
    """
    Routing Decision:
    - "query": City extracted, context sufficient
    - "user": Needs clarification or no city found
    """
    return state.get("next_node", "user")
```

**Routing Table**:
| Condition | Route | Reason |
|-----------|-------|--------|
| City found + context sufficient | `router → query` | Fetch data |
| City found + needs clarification | `router → user` | Ask follow-up |
| No city found | `router → user` | Request city |
| Data fetched | `query → router` | Process response |
| Response ready | `user → END` | Return to user |

---

## 💾 State Management Architecture

### AgentState Schema

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]           # Full conversation history
    thread_id: Optional[str]              # Session identifier
    session_data: Optional[Dict]          # LLM-managed preferences
    prev_node: Optional[str]              # Last executed node
    next_node: Optional[str]              # Routing destination
    clarification_question: Optional[str] # Pending user query
    last_extracted_data: Optional[str]    # Cached city info
    image_urls: Optional[List[str]]       # Current query images
    weather_data: Optional[List[Dict]]    # Current query weather
```

### Session Memory Layer

**Two-tier memory system**:

1. **LangGraph Checkpointer** (MemorySaver):
   - Persists entire `AgentState` per thread
   - Enables conversation continuity
   - Thread-scoped isolation

2. **Custom Session Memory**:
   - Stores user preferences extracted by LLM
   - Caches city summaries for performance
   - Managed by `SessionMemory` class
   - In-memory (can be swapped for Redis/DB)

**Memory Update Flow**:
```
User Query → Router Node
    ↓
LLM Extraction → {user_name, city_name, trip_type, preferences}
    ↓
SESSION_MEMORY.update_session(thread_id, extracted_data)
    ↓
Available to all subsequent nodes
```

### Token Management

**Problem**: Long conversations → Token overflow → API errors

**Solution**: Automatic message summarization

```python
def check_and_summarize_messages(messages, llm, max_tokens=8000):
    """
    - Counts tokens in message chain
    - If > max_tokens: Summarize older messages
    - Keep recent 4 messages + summary
    """
    if count_tokens > max_tokens:
        summary = llm.invoke("Summarize conversation: ...")
        return [SystemMessage(summary)] + messages[-4:]
    return messages
```

**Triggers**:
- Router Node: Before processing each user input
- Interactive Loop: Preventive check at 6K tokens

---

## 🔌 Data Sources & APIs

### 1. Vector Store (ChromaDB)

**Architecture**:
```
ChromaDB Persistent Client
├── Collection: "city_knowledge"
├── Pre-loaded: Paris, Tokyo, New York
├── Embedding: Default ChromaDB embeddings
└── Similarity Threshold: 0.9 for high confidence
```

**Query Strategy**:
1. Semantic search with user query
2. Calculate similarity: `1 - distance`
3. Accept if similarity > 0.9
4. Otherwise, fallback to web search

**Advantages**:
- Fast retrieval (< 100ms)
- Offline capability for known cities
- Semantic understanding (e.g., "City of Light" → Paris)

---

### 2. Web Search (SerpAPI + Fallback)

**Primary**: SerpAPI Google Search
- Real-time web results
- Answer boxes, knowledge graphs
- Structured snippets

**Fallback**: Basic text return
- Triggers when API key missing or quota exceeded
- Returns generic message

**Integration**:
```python
def execute_web_search(query: str) -> str:
    if SERPAPI_KEY:
        results = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY
        }).get_dict()
        return parse_answer_box(results)
    else:
        return f"Search query: {query} (fallback mode)"
```

---

### 3. Image Search (SerpAPI + Fallback)

**Primary**: SerpAPI Google Images
- Real Google Images results
- Top 4 images per query
- High-quality URLs

**Fallback**: Unsplash placeholders
- Generates placeholder URLs
- Always returns 4 images

**Architecture Decision**: Graceful degradation ensures UI never breaks

---

### 4. Weather API (Mock Implementation)

**Design**: Realistic mock for demonstration

**Features**:
- 7-day forecast
- Date range: Dec 2025 - Jun 2026
- City-specific temperature profiles
- Seasonal weather patterns
- Async simulation with latency (0.5-1.5s)

**Production Ready**: Easy swap for OpenWeatherMap, WeatherAPI, etc.

---

## 🧠 LLM Intelligence Layer

### Model Strategy

**Multi-Model Fallback Chain**:
```python
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",  # Primary: Best quality
    "mixtral-8x7b-32768",       # Fallback 1: Good balance
    "gemma2-9b-it"              # Fallback 2: Lightweight
]
```

**Auto-Switching Logic**:
1. Start with Llama 3.3 (70B params)
2. On rate limit error → Switch to Mixtral
3. On second rate limit → Switch to Gemma2
4. Exponential backoff between retries

### Rate Limit Handling

**Detection**:
```python
def parse_rate_limit_error(error_message):
    # Detects TPM (tokens/minute) or TPD (tokens/day)
    # Extracts retry_after time
    # Returns limit type and wait duration
```

**Response Strategy**:
- **TPM**: Wait 80 seconds, retry same model
- **TPD**: Switch to next model immediately
- **Max retries**: 3 attempts before failing gracefully

### Retry Mechanism

```python
def invoke_llm_with_retry(llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except RateLimitError as e:
            if e.type == "TPD":
                switch_to_next_model()
            else:
                time.sleep(calculate_backoff(attempt))
```

---
