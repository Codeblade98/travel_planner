"""
Test script to verify SerpAPI integration for images and web search.
Run this to test both with and without API key.
"""
import asyncio
import os
from utils.api_utils import fetch_city_images, mock_web_search
from dotenv import load_dotenv

load_dotenv()

async def test_serpapi_integration():
    """Test SerpAPI image and web search functions."""
    
    print("=" * 80)
    print("SERPAPI INTEGRATION TEST")
    print("=" * 80)
    
    # Check API key status
    api_key = os.getenv("SERPAPI_API_KEY")
    if api_key and api_key != "your_serpapi_key_here" and api_key.strip():
        print(f"✓ SERPAPI_API_KEY found: {api_key[:10]}...{api_key[-5:]}")
        mode = "LIVE MODE (using SerpAPI)"
    else:
        print("⚠️  SERPAPI_API_KEY not set or invalid")
        mode = "FALLBACK MODE (using Unsplash/mock data)"
    
    print(f"\nRunning in: {mode}")
    print("=" * 80)
    
    # Test 1: Image Search
    print("\n📸 TEST 1: Image Search")
    print("-" * 80)
    
    test_cities = ["Paris", "Tokyo", "New York"]
    
    for city in test_cities:
        print(f"\nFetching images for: {city}")
        try:
            images = await fetch_city_images(city, count=5)
            print(f"✓ Retrieved {len(images)} images")
            for i, img in enumerate(images[:3], 1):  # Show first 3
                print(f"  {i}. {img[:80]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 2: Web Search
    print("\n\n🔍 TEST 2: Web Search")
    print("-" * 80)
    
    test_queries = [
        "Tell me about Paris",
        "Tokyo travel information",
        "What is Kyoto known for"
    ]
    
    for query in test_queries:
        print(f"\nSearching: '{query}'")
        try:
            results = mock_web_search(query)
            print(f"✓ Retrieved {len(results)} characters")
            # Show first 200 chars
            preview = results[:200].replace("\n", " ")
            print(f"  Preview: {preview}...")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if api_key and api_key != "your_serpapi_key_here" and api_key.strip():
        print("✓ Tests completed using SerpAPI")
        print("  - Images: Real Google Images results")
        print("  - Web Search: Real Google search results")
        print("\nNote: Check console output above for SerpAPI confirmation messages")
    else:
        print("✓ Tests completed using fallback mode")
        print("  - Images: Unsplash placeholder URLs")
        print("  - Web Search: Hardcoded mock data")
        print("\nTo test with SerpAPI:")
        print("  1. Get API key from https://serpapi.com/")
        print("  2. Add to .env file: SERPAPI_API_KEY=your_key_here")
        print("  3. Run this test again")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_serpapi_integration())
