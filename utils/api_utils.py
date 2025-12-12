"""
API implementations for weather, image search, and web search.
- Weather: Mock API (simulates forecast data)
- Images: SerpAPI Google Images
- Web Search: SerpAPI Google Search
"""
import asyncio
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict
from serpapi import GoogleSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def fetch_weather_forecast(city: str, days: int = 7, start_date: datetime = None) -> List[Dict]:
    """
    Mock weather API that simulates fetching forecast data.
    Returns a list of daily weather data points from late December 2025 to early June 2026.
    
    Args:
        city: City name for weather forecast
        days: Number of days to forecast (default 7)
        start_date: Optional start date (defaults to last week of December 2025)
    """
    # Simulate API latency
    await asyncio.sleep(random.uniform(0.5, 1.5))
    
    # Default start date: December 25, 2025
    if start_date is None:
        start_date = datetime(2025, 12, 25)
    
    # Base temperatures for different cities (winter to spring transition)
    # Temperature ranges adjusted for December-June period
    city_temps = {
        "Paris": {
            "dec": 7, "jan": 5, "feb": 6, "mar": 10, "apr": 14, "may": 17, "jun": 21
        },
        "Tokyo": {
            "dec": 10, "jan": 8, "feb": 9, "mar": 13, "apr": 16, "may": 20, "jun": 24
        },
        "New York": {
            "dec": 4, "jan": 2, "feb": 3, "mar": 8, "apr": 13, "may": 18, "jun": 23
        },
        "Kyoto": {
            "dec": 9, "jan": 7, "feb": 8, "mar": 12, "apr": 16, "may": 21, "jun": 25
        },
        "Snohomish": {
            "dec": 6, "jan": 5, "feb": 7, "mar": 10, "apr": 13, "may": 16, "jun": 19
        },
    }
    
    # Default city temperatures if not in list
    default_temps = {
        "dec": 8, "jan": 6, "feb": 7, "mar": 11, "apr": 15, "may": 19, "jun": 23
    }
    
    city_temp_profile = city_temps.get(city, default_temps)
    
    forecast = []
    current_date = start_date
    
    # Ensure we don't go beyond June 7, 2026
    end_date = datetime(2026, 6, 7)
    
    for i in range(days):
        if current_date > end_date:
            break
            
        date = current_date + timedelta(days=i)
        
        # Get month abbreviation
        month_abbr = date.strftime("%b").lower()
        
        # Get base temperature for the month
        base_temp = city_temp_profile.get(month_abbr, 15)
        
        # Add daily variation
        temp_variation = random.uniform(-4, 6)
        
        # Weather conditions vary by season
        if date.month in [12, 1, 2]:  # Winter
            conditions = ["Cloudy", "Rainy", "Partly Cloudy", "Clear", "Snowy"]
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        elif date.month in [3, 4, 5]:  # Spring
            conditions = ["Partly Cloudy", "Sunny", "Cloudy", "Rainy", "Clear"]
            weights = [0.3, 0.3, 0.2, 0.15, 0.05]
        else:  # June (early summer)
            conditions = ["Sunny", "Clear", "Partly Cloudy", "Cloudy", "Rainy"]
            weights = [0.35, 0.25, 0.2, 0.15, 0.05]
        
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "temperature": round(base_temp + temp_variation, 1),
            "condition": random.choices(conditions, weights=weights)[0],
            "humidity": random.randint(45 if date.month >= 4 else 60, 85),
            "wind_speed": random.randint(8 if date.month <= 3 else 5, 25)
        })
    
    return forecast


async def fetch_city_images(city: str, count: int = 5) -> List[str]:
    """
    Fetch city images using SerpAPI Google Images search.
    Falls back to Unsplash placeholders if SerpAPI fails or API key not available.
    
    Args:
        city: City name to search images for
        count: Number of images to return (default 5)
    
    Returns:
        List of image URLs
    """
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    
    if not serpapi_key:
        print("⚠️  SERPAPI_API_KEY not found, using Unsplash fallback")
        return await _fetch_unsplash_fallback(city, count)
    
    try:
        # Run SerpAPI call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None, 
            _fetch_images_serpapi, 
            city, 
            count, 
            serpapi_key
        )
        
        if images:
            print(f"📸 SerpAPI: Retrieved {len(images)} images for {city}")
            return images
        else:
            print(f"⚠️  SerpAPI returned no images, using fallback")
            return await _fetch_unsplash_fallback(city, count)
            
    except Exception as e:
        print(f"❌ SerpAPI error: {e}, using fallback")
        return await _fetch_unsplash_fallback(city, count)


def _fetch_images_serpapi(city: str, count: int, api_key: str) -> List[str]:
    """
    Helper function to fetch images from SerpAPI (runs in thread pool).
    """
    params = {
        "engine": "google_images",
        "q": city,
        "api_key": api_key,
        "num": count,
        "ijn": 0  # Page number
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    images = []
    if "images_results" in results:
        for img in results["images_results"][:count]:
            # Prefer original > thumbnail
            if "original" in img:
                images.append(img["original"])
            elif "thumbnail" in img:
                images.append(img["thumbnail"])
    
    return images


async def _fetch_unsplash_fallback(city: str, count: int = 5) -> List[str]:
    """
    Fallback image search using Unsplash placeholder images.
    Used when SerpAPI is unavailable or fails.
    """
    # Simulate API latency
    await asyncio.sleep(random.uniform(0.3, 1.0))
    
    # Using Unsplash placeholder images with city-specific keywords
    city_keywords = {
        "Paris": ["eiffel-tower", "louvre", "seine", "arc-de-triomphe", "montmartre"],
        "Tokyo": ["tokyo-tower", "shibuya", "mount-fuji", "cherry-blossom", "senso-ji"],
        "New York": ["statue-of-liberty", "times-square", "central-park", "brooklyn-bridge", "empire-state"],
        "Kyoto": ["fushimi-inari", "kinkaku-ji", "bamboo-forest", "geisha", "zen-garden"],
        "Snohomish": ["mountain", "forest", "lake", "waterfall", "nature"],
    }
    
    keywords = city_keywords.get(city, ["city", "architecture", "skyline", "landmark", "culture"])
    
    # Generate placeholder image URLs
    images = []
    for i in range(min(count, len(keywords))):
        # Using Unsplash's source API for random images by keyword
        images.append(f"https://source.unsplash.com/800x600/?{keywords[i]},{city}")
    
    return images


def execute_web_search(query: str) -> str:
    """
    Web search function using SerpAPI Google Search.
    Falls back to mock data if SerpAPI fails or API key not available.
    
    Args:
        query: Search query string
    
    Returns:
        Search results as text summary
    """
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    
    if not serpapi_key:
        print("⚠️  SERPAPI_API_KEY not found, using mock fallback")
        return _mock_web_search_fallback(query)
    
    try:
        result = _web_search_serpapi(query, serpapi_key)
        if result:
            print(f"🔍 SerpAPI: Retrieved search results for '{query[:50]}...'")
            return result
        else:
            print(f"⚠️  SerpAPI returned no results, using fallback")
            return _mock_web_search_fallback(query)
            
    except Exception as e:
        print(f"❌ SerpAPI web search error: {e}, using fallback")
        return _mock_web_search_fallback(query)


def _web_search_serpapi(query: str, api_key: str) -> str:
    """
    Helper function to perform web search using SerpAPI.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 5  # Number of results
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract information from search results
    summary_parts = []
    
    # Add answer box if available
    if "answer_box" in results:
        answer_box = results["answer_box"]
        if "answer" in answer_box:
            summary_parts.append(answer_box["answer"])
        elif "snippet" in answer_box:
            summary_parts.append(answer_box["snippet"])
    
    # Add knowledge graph if available
    if "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        if "description" in kg:
            summary_parts.append(kg["description"])
    
    # Add organic results snippets
    if "organic_results" in results:
        for result in results["organic_results"][:3]:
            if "snippet" in result:
                summary_parts.append(result["snippet"])
    
    if summary_parts:
        return "\n\n".join(summary_parts)
    
    return ""


def _mock_web_search_fallback(query: str) -> str:
    """
    Fallback web search with mock data.
    Used when SerpAPI is unavailable or fails.
    """
    city_info = {
        "Kyoto": """
        Kyoto is a city located in the central part of the island of Honshu, Japan. 
        It was the capital of Japan for over 1,000 years and is known for its beautiful 
        temples, traditional geisha districts, and stunning gardens. The city has over 
        2,000 temples and shrines, including the famous Fushimi Inari Shrine with its 
        thousands of vermillion torii gates. Kyoto is also renowned for its traditional 
        crafts, tea ceremonies, and seasonal beauty, particularly during cherry blossom 
        season in spring and autumn foliage.
        """,
        "Snohomish": """
        Snohomish is a historic city in Snohomish County, Washington, United States. 
        Known as the "Antique Capital of the Northwest," it features a charming historic 
        downtown with over 100 antique shops and stores. The city sits along the 
        Snohomish River and offers scenic views of the Cascade Mountains. Founded in 
        the 1850s, Snohomish retains much of its 19th-century charm with Victorian-era 
        architecture and tree-lined streets. The area is popular for outdoor activities 
        including hiking, fishing, and exploring the nearby mountain trails.
        """,
        "London": """
        London is the capital and largest city of England and the United Kingdom. 
        With a history spanning over two millennia, it's a global hub for culture, 
        finance, and the arts. Famous landmarks include the Tower of London, 
        Buckingham Palace, Big Ben, and the British Museum. The city is known for 
        its diverse population, world-class theaters in the West End, and iconic 
        red double-decker buses and black cabs.
        """,
    }
    
    for city, info in city_info.items():
        if city.lower() in query.lower():
            return info.strip()
    
    return f"Information about {query}: A fascinating place worth exploring with rich history and culture."
