#!/usr/bin/env python3
"""Test ChromaDB similarity scores"""

from vector_store import get_vector_store

# Initialize vector store
vector_store = get_vector_store()

# Test different queries
test_queries = [
    "Paris",
    "Tell me about Paris",
    "I want to visit Paris",
    "Tokyo",
    "New York",
    "London",  # Not in DB
]

print("="*80)
print("ChromaDB Similarity Testing")
print("="*80)

for query in test_queries:
    result = vector_store.search_city(query, n_results=1)
    
    if result["found"]:
        distance = result.get("distance", 1.0)
        similarity = 1.0 - distance if distance is not None else 0.0
        city = result.get("metadata", {}).get("country", "Unknown")
        
        print(f"\nQuery: '{query}'")
        print(f"  Distance: {distance:.4f}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Match: {result['content'][:100]}...")
    else:
        print(f"\nQuery: '{query}'")
        print(f"  No match found")

print("\n" + "="*80)
