# Streamlit Travel Assistant App

A beautiful chat interface for the Multi-Modal Travel Assistant, built with Streamlit.

## Features

### Chat Interface
- 💬 Natural conversation flow with persistent chat history
- 🎨 Clean, modern UI with message bubbles
- 🔄 Real-time processing with loading indicators

### Visual Content Display
- 📸 **Image Grid**: Automatically displays top 4 city images before the agent response
- 🌤️ **Weather Cards**: Beautiful weather forecast cards showing temperature, humidity, and conditions
- 📊 **Expandable Content**: Additional weather days shown in expandable sections

### Session Management
- 🧠 Persistent session memory across messages
- 🗑️ Clear chat history button
- 💾 View current session memory (JSON format)
- 🔍 Maintains context across multi-turn conversations

### Agent Logic
Based on the same `run_agent_interactive` logic from `main.py`:
1. User provides input via chat interface
2. Router Node analyzes the query
3. Query Node fetches data (weather, images, vector store)
4. User Node displays formatted output
5. Loop continues with maintained context

## Running the App

### Option 1: Using the Shell Script
```bash
./run_streamlit.sh
```

### Option 2: Manual Run
```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit
streamlit run streamlit_app.py
```

### Option 3: Direct Command
```bash
python -m streamlit run streamlit_app.py
```

## Usage

1. **Start the app** using one of the methods above
2. **Enter your query** in the chat input at the bottom
3. **View results**:
   - Images appear first (top 4 in a 2x2 grid)
   - Agent response text follows
   - Weather forecast cards displayed below (if applicable)
4. **Continue conversation** - the agent maintains context

## Example Queries

- "Tell me about Paris"
- "What's the weather like in Tokyo?"
- "I'm planning a trip next week" (agent will ask for city)
- "Show me images of New York"
- "What should I pack for London in December?"

## UI Components

### Main Chat Area
- **User messages**: Right-aligned with user icon
- **Assistant messages**: Left-aligned with bot icon
- **Images**: 2x2 grid layout, rounded corners, shadows
- **Weather cards**: Color-coded cards with icons

### Sidebar
- **Features overview**: Quick reference
- **Session controls**: Clear history, view memory
- **About section**: Tech stack info

## Architecture

```
User Input (Streamlit chat input)
    ↓
process_user_input()
    ↓
Agent Graph (LangGraph)
    ├── Router Node
    ├── Query Node
    └── User Node
    ↓
Final State
    ├── final_output (structured data)
    ├── image_urls
    └── weather_data
    ↓
Display (Images → Text → Weather)
```

## Key Differences from CLI

1. **Visual Output**: Images and weather displayed inline
2. **Persistent UI**: Chat history visible at all times
3. **Image Priority**: Top 4 images shown BEFORE text response
4. **No Input Callback**: Uses Streamlit's `st.chat_input()` directly
5. **Async Handling**: Custom `run_async()` helper for event loop management

## Styling

Custom CSS provides:
- Rounded message bubbles
- Image grid with responsive layout
- Weather cards with hover effects
- Consistent spacing and typography

## Session State

Streamlit maintains:
- `messages`: Chat display history with images/weather
- `message_history`: LangChain message objects for agent
- `thread_id`: Session identifier
- `agent_graph`: Compiled LangGraph instance

## Error Handling

- Graceful image loading failures
- Error messages displayed in chat
- Session state recovery
- Async/await compatibility checks

## Performance

- Lazy graph compilation (only once per session)
- Efficient message history management
- Cached vector store
- Minimal re-renders with Streamlit's state management

## Dependencies

All dependencies are specified in `pyproject.toml`:
- streamlit >= 1.30.0
- langchain >= 0.1.0
- langgraph >= 0.0.40
- chromadb >= 0.4.0

## Troubleshooting

### Port Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Async Loop Issues
The app includes a `run_async()` helper that handles event loop creation automatically.

### Images Not Loading
Check network connectivity and image URLs. The app shows warnings for failed image loads.

## Future Enhancements

- [ ] Dark mode toggle
- [ ] Export chat history
- [ ] More weather visualizations (charts)
- [ ] Image carousel instead of grid
- [ ] Voice input support
- [ ] Multi-language support
