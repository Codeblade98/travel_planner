# Image Display Troubleshooting Guide

## Changes Made to Fix Image Display

### 1. Fixed Return Value Count
**Problem:** The `process_user_input` function was returning 3 values on error but 4 values on success.
**Fix:** Changed error return from `return None, "Error: No output from agent", []` to `return None, "Error: No output from agent", [], []`

### 2. Extract Images from State Top-Level
**Problem:** Images were only being extracted from `final_output` dict, but they're stored at the state's top level.
**Fix:** Added extraction from state first:
```python
# Get image_urls and weather_data from state (they're stored at top level)
image_urls = final_state.get("image_urls", [])
weather_data = final_state.get("weather_data", [])
```

### 3. Added Debug Output
Added comprehensive debug printing to trace image flow:
- State keys and image URLs from state
- Image URLs from final_output
- Images at each node (query, user)
- Display-time image counts

### 4. Better Display Ordering
Images are displayed in this order in chat history:
1. Images (if present) - TOP 4 in 2×2 grid
2. Text content
3. Weather cards (if present)

### 5. Enhanced Error Messages
Changed from generic warnings to specific error messages with image index.

## How to Test

### Method 1: Run the Streamlit App
```bash
./run_streamlit.sh
```

Then ask: "Tell me about Paris"

Expected behavior:
- Console shows: `[DEBUG] Image URLs: ['https://source.unsplash.com/...', ...]`
- Streamlit shows: "🖼️ Found 5 images"
- 4 images displayed in 2×2 grid
- Text response below
- Weather cards at bottom

### Method 2: Test Image Fetch Directly
```bash
python test_image_fetch.py
```

This should print 5 Unsplash URLs.

### Method 3: Test Agent Directly
```bash
python -c "
import asyncio
from agent import run_agent_interactive

async def test_input(prompt):
    return input(f'{prompt}\\nYou: ')

async def main():
    await run_agent_interactive('test', test_input)

asyncio.run(main())
"
```

Type: "Tell me about Tokyo"
Look for console output showing image URLs.

## Debug Console Output

When you run the Streamlit app, watch the terminal for:

```
[DEBUG] Node: router
[DEBUG] Node: query
[DEBUG] Query Node - Images: 5, Weather: 7
[DEBUG] Node: user  
[DEBUG] User Node - Images: 5, Weather: 7
[DEBUG] State keys: dict_keys(['messages', 'thread_id', 'prev_node', 'last_extracted_data', 'image_urls', 'weather_data', ...])
[DEBUG] Image URLs from state: ['https://source.unsplash.com/800x600/?eiffel-tower,Paris', ...]
[DEBUG] Weather data from state: 7 items
[DEBUG] Image URLs from final_output: ['https://source.unsplash.com/800x600/?eiffel-tower,Paris', ...]
[DEBUG DISPLAY] Image URLs: ['https://source.unsplash.com/800x600/?eiffel-tower,Paris', ...]
[DEBUG DISPLAY] Weather data: 7 items
```

## Common Issues

### Issue 1: "No images available for this query"
**Cause:** The agent routed to clarification instead of query node
**Solution:** Provide a clear city name: "Show me Paris" or "Tell me about Tokyo"

### Issue 2: Images show but don't load
**Cause:** Unsplash's source.unsplash.com API may have rate limits
**Solution:** 
- Wait a moment and refresh
- Check network connectivity
- Unsplash random images may occasionally fail - this is normal

### Issue 3: Empty image list but agent found city
**Cause:** Query node didn't execute or images weren't added to state
**Check:** Look for "Query Node - Images: 5" in console
**Solution:** Check if Query Node is being reached (clarification might have been triggered)

### Issue 4: Images in history don't show on reload
**Cause:** Images are only in first display, not persisted correctly
**Solution:** Should be fixed - images are now stored in message dict with "images" key

## Architecture Flow

```
User: "Tell me about Paris"
    ↓
Router Node: Extracts city → next_node = "query"
    ↓
Query Node: 
    - Calls fetch_city_images("Paris", 5)
    - Stores in state["image_urls"]
    - Returns state with image_urls
    ↓
User Node:
    - Receives state with image_urls
    - Deep copies state (preserves image_urls)
    - Returns state with image_urls
    ↓
Streamlit:
    - final_state.get("image_urls", [])
    - Displays images
    - Saves to message history
```

## Code Locations

**Image fetch:** `mock_apis.py` line 102-127
**Query node stores images:** `agent.py` line 916
**User node preserves images:** `agent.py` line 999 (deepcopy)
**Streamlit extracts images:** `streamlit_app.py` line 156-167
**Streamlit displays images:** `streamlit_app.py` line 278-282

## Quick Fix Checklist

- [x] Return 4 values consistently from process_user_input
- [x] Extract image_urls from state, not just final_output
- [x] Add debug output to trace image flow
- [x] Display images before text in chat
- [x] Store images in message history
- [x] Display images from history on reload
- [x] Handle None vs empty list gracefully
- [x] Show informative messages when no images

## Next Steps if Still Not Working

1. Run `python test_image_fetch.py` - confirms mock API works
2. Check console for `[DEBUG]` lines - confirms data flow
3. Check for "🖼️ Found X images" in UI - confirms detection
4. Check browser console (F12) - confirms Streamlit rendering
5. Try a different city: "Tokyo", "New York", "Kyoto"
