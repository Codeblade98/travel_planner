# SerpAPI Integration Guide

## Overview

The `mock_apis.py` file has been updated to use **SerpAPI** for real-world image and web search capabilities instead of mock data.

## What Changed

### Before (Mock Data)
- **Images**: Unsplash placeholder URLs with generic keywords
- **Web Search**: Hardcoded city descriptions

### After (SerpAPI)
- **Images**: Real Google Images search results via SerpAPI
- **Web Search**: Real Google search results (answer boxes, knowledge graphs, snippets) via SerpAPI
- **Fallback**: Graceful fallback to Unsplash/mock data if SerpAPI unavailable

## Features

### 1. Image Search (`fetch_city_images`)
- Uses Google Images via SerpAPI
- Returns actual image URLs from web
- Prefers original images over thumbnails
- Falls back to Unsplash if API key missing or error occurs

### 2. Web Search (`mock_web_search`)
- Uses Google Search via SerpAPI
- Extracts:
  - Answer boxes (direct answers)
  - Knowledge graph descriptions
  - Top 3 organic result snippets
- Falls back to mock data if API key missing or error occurs

### 3. Weather Forecast (`fetch_weather_forecast`)
- **No change** - still uses mock data
- Simulates realistic weather patterns
- Could be replaced with OpenWeatherMap or similar in future

## Setup

### 1. Get SerpAPI Key
1. Go to [https://serpapi.com/](https://serpapi.com/)
2. Sign up for free account
3. Get your API key from [Dashboard](https://serpapi.com/manage-api-key)
4. Free tier includes 100 searches/month

### 2. Add to .env File
```bash
# Add this line to your .env file
SERPAPI_API_KEY=your_actual_key_here
```

### 3. Install Package
```bash
pip install google-search-results
```

Or using the project:
```bash
pip install -e .
```

## Usage

The integration is transparent - no code changes needed in the agent or Streamlit app!

### With SerpAPI Key (Real Search)
```python
# In your .env
SERPAPI_API_KEY=abc123xyz...

# Agent will automatically use real search
images = await fetch_city_images("Paris", count=5)
# Returns: Real Paris images from Google Images

results = mock_web_search("Tell me about Paris")
# Returns: Real Google search results about Paris
```

### Without SerpAPI Key (Fallback Mode)
```python
# No SERPAPI_API_KEY in .env

# Gracefully falls back to Unsplash
images = await fetch_city_images("Paris", count=5)
# Returns: Unsplash placeholder URLs

# Falls back to mock data
results = mock_web_search("Tell me about Paris")
# Returns: Hardcoded city information
```

## API Response Format

### Image Search Response
```python
[
    "https://example.com/real-image-1.jpg",
    "https://example.com/real-image-2.jpg",
    "https://example.com/real-image-3.jpg",
    "https://example.com/real-image-4.jpg",
    "https://example.com/real-image-5.jpg"
]
```

### Web Search Response
```text
Paris is the capital and most populous city of France...

Paris is known for its art, fashion, gastronomy and culture. 
Its 19th-century cityscape is crisscrossed by wide boulevards 
and the River Seine.

The Eiffel Tower, built in 1889, has become the global cultural 
icon of France and one of the most recognizable structures in the world.
```

## Error Handling

The implementation includes robust error handling:

### Scenarios Handled
1. **No API Key**: Falls back to mock/Unsplash data
2. **API Error**: Catches exceptions and uses fallback
3. **No Results**: Returns fallback data
4. **Network Issues**: Graceful degradation

### Console Output
```
📸 SerpAPI: Retrieved 5 images for Paris
🔍 SerpAPI: Retrieved search results for 'Tell me about Paris'
```

Or with fallback:
```
⚠️  SERPAPI_API_KEY not found, using Unsplash fallback
⚠️  SerpAPI returned no results, using fallback
❌ SerpAPI error: HTTPError 401, using fallback
```

## Testing

### Test with SerpAPI
```bash
# Add your key to .env
echo "SERPAPI_API_KEY=your_key" >> .env

# Run the app
streamlit run streamlit_app.py

# Query: "Tell me about Paris"
# Check console for: "📸 SerpAPI: Retrieved 5 images for Paris"
```

### Test Fallback Mode
```bash
# Comment out the key in .env
# SERPAPI_API_KEY=

# Run the app
streamlit run streamlit_app.py

# Query: "Tell me about Paris"
# Check console for: "⚠️  SERPAPI_API_KEY not found, using Unsplash fallback"
```

### Test Image Fetch Directly
```python
import asyncio
from mock_apis import fetch_city_images, mock_web_search

async def test():
    # Test images
    images = await fetch_city_images("Tokyo", count=5)
    print(f"Images: {images}")
    
    # Test web search
    info = mock_web_search("Tell me about Tokyo")
    print(f"Info: {info}")

asyncio.run(test())
```

## Code Architecture

### Image Search Flow
```
fetch_city_images(city, count)
    ↓
Check SERPAPI_API_KEY in env
    ↓
If key exists:
    → Run _fetch_images_serpapi() in thread pool
    → Parse Google Images results
    → Return image URLs
    ↓
If key missing or error:
    → Fall back to _fetch_unsplash_fallback()
    → Return placeholder URLs
```

### Web Search Flow
```
mock_web_search(query)
    ↓
Check SERPAPI_API_KEY in env
    ↓
If key exists:
    → Call _web_search_serpapi()
    → Extract answer boxes
    → Extract knowledge graph
    → Extract organic results
    → Combine into summary
    ↓
If key missing or error:
    → Fall back to _mock_web_search_fallback()
    → Return hardcoded city info
```

## SerpAPI Parameters

### Image Search Parameters
```python
{
    "engine": "google_images",  # Use Google Images
    "q": "Paris",               # Search query
    "api_key": "...",           # Your API key
    "num": 5,                   # Number of results
    "ijn": 0                    # Page number (0 = first page)
}
```

### Web Search Parameters
```python
{
    "engine": "google",         # Use Google Search
    "q": "Tell me about Paris", # Search query
    "api_key": "...",           # Your API key
    "num": 5                    # Number of results
}
```

## Rate Limits

### Free Tier
- 100 searches per month
- Includes both image and web searches
- Resets monthly

### Paid Tiers
- Starting at $50/month for 5,000 searches
- See [SerpAPI Pricing](https://serpapi.com/pricing)

### Optimization Tips
1. Use fallback mode during development
2. Enable SerpAPI only for production/demos
3. Cache results when possible
4. Combine image + web search in single query node call

## Benefits of SerpAPI Integration

✅ **Real Data**: Actual images and information from Google
✅ **Always Updated**: Fresh results, not stale mock data
✅ **Better UX**: Users see real, relevant images
✅ **Fallback Support**: Works even without API key
✅ **Easy to Use**: Drop-in replacement, no code changes needed
✅ **Reliable**: Well-maintained API with good uptime

## Future Enhancements

Potential improvements:

1. **Image Filtering**: Filter by image type, size, color
2. **Location-Based**: Use `location` parameter for localized results
3. **Language Support**: Use `hl` parameter for different languages
4. **Caching**: Implement Redis/disk cache for repeated queries
5. **Alternative Providers**: Add support for Bing, DuckDuckGo
6. **Weather API**: Replace mock weather with OpenWeatherMap

## Troubleshooting

### Issue: "Import serpapi could not be resolved"
**Solution**: Install the package
```bash
pip install google-search-results
```

### Issue: "HTTPError 401 Unauthorized"
**Solution**: Check your API key in .env file
```bash
# Verify key is set correctly
cat .env | grep SERPAPI
```

### Issue: "No images returned"
**Solution**: 
- Check API quota on SerpAPI dashboard
- Verify city name is spelled correctly
- Check console for error messages

### Issue: "Slow response times"
**Solution**:
- SerpAPI calls run in thread pool (non-blocking)
- Network latency is normal (1-2 seconds)
- Consider caching results for repeated queries

## Environment Variables

```bash
# Required for LLM
GROQ_API_KEY=your_groq_key_here

# Optional for real search (falls back to mock/Unsplash if not set)
SERPAPI_API_KEY=your_serpapi_key_here
```

## Dependencies

Added to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "google-search-results>=2.4.2",
]
```

## Summary

The SerpAPI integration provides:
- 🖼️ **Real images** from Google Images
- 🔍 **Real information** from Google Search
- 🔄 **Graceful fallback** when API unavailable
- 🚀 **Easy setup** with just an API key
- 💰 **Free tier** for testing and development

The agent now has access to the entire web for images and information, while maintaining backward compatibility with the fallback system!
