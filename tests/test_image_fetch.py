"""
Quick test script to verify image URLs are being generated correctly.
"""
import asyncio
from utils.api_utils import fetch_city_images

async def test_images():
    print("Testing image fetch...")
    images = await fetch_city_images("Paris", count=5)
    print(f"Got {len(images)} images:")
    for i, url in enumerate(images, 1):
        print(f"  {i}. {url}")

if __name__ == "__main__":
    asyncio.run(test_images())
