#!/usr/bin/env python3
"""
Test script to verify weather forecast date range (Dec 2025 - June 2026)
"""
import asyncio
from datetime import datetime
from utils.api_utils import fetch_weather_forecast

async def test_weather_dates():
    print("🌤️  Testing Weather Forecast Date Range")
    print("=" * 60)
    
    # Test 1: Short forecast from default start date
    print("\n1️⃣  Test: 7-day forecast (default start date)")
    forecast_7 = await fetch_weather_forecast("Paris", days=7)
    print(f"   Start: {forecast_7[0]['date']}")
    print(f"   End: {forecast_7[-1]['date']}")
    print(f"   Conditions: {[f['condition'] for f in forecast_7]}")
    
    # Test 2: Extended forecast through spring
    print("\n2️⃣  Test: 30-day forecast into January 2026")
    forecast_30 = await fetch_weather_forecast("Tokyo", days=30)
    print(f"   Start: {forecast_30[0]['date']}")
    print(f"   End: {forecast_30[-1]['date']}")
    print(f"   Temperature range: {min([f['temperature'] for f in forecast_30]):.1f}°C to {max([f['temperature'] for f in forecast_30]):.1f}°C")
    
    # Test 3: Full 6-month forecast (Dec 2025 - June 2026)
    print("\n3️⃣  Test: Full 165-day forecast (Dec 25, 2025 - June 7, 2026)")
    forecast_full = await fetch_weather_forecast("New York", days=165)
    print(f"   Start: {forecast_full[0]['date']}")
    print(f"   End: {forecast_full[-1]['date']}")
    print(f"   Total days: {len(forecast_full)}")
    
    # Show temperature progression by month
    print("\n   📊 Monthly Temperature Averages:")
    months = {}
    for day in forecast_full:
        month = day['date'][:7]  # YYYY-MM
        if month not in months:
            months[month] = []
        months[month].append(day['temperature'])
    
    for month in sorted(months.keys()):
        avg_temp = sum(months[month]) / len(months[month])
        print(f"      {month}: {avg_temp:.1f}°C (avg)")
    
    # Test 4: Custom start date
    print("\n4️⃣  Test: Custom start date (March 1, 2026)")
    custom_start = datetime(2026, 3, 1)
    forecast_march = await fetch_weather_forecast("Kyoto", days=30, start_date=custom_start)
    print(f"   Start: {forecast_march[0]['date']}")
    print(f"   End: {forecast_march[-1]['date']}")
    print(f"   Spring conditions: {set([f['condition'] for f in forecast_march])}")
    
    # Test 5: Seasonal condition distribution
    print("\n5️⃣  Test: Seasonal condition distribution")
    
    # Winter (Dec-Feb)
    winter = await fetch_weather_forecast("Paris", days=70, start_date=datetime(2025, 12, 25))
    winter_conditions = [f['condition'] for f in winter if datetime.strptime(f['date'], "%Y-%m-%d").month in [12, 1, 2]]
    
    # Spring (Mar-May)
    spring = await fetch_weather_forecast("Paris", days=90, start_date=datetime(2026, 3, 1))
    spring_conditions = [f['condition'] for f in spring if datetime.strptime(f['date'], "%Y-%m-%d").month in [3, 4, 5]]
    
    print(f"   Winter conditions: {dict((c, winter_conditions.count(c)) for c in set(winter_conditions))}")
    print(f"   Spring conditions: {dict((c, spring_conditions.count(c)) for c in set(spring_conditions))}")
    
    print("\n✅ Weather forecast date range test completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_weather_dates())
