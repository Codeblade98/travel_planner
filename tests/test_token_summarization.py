#!/usr/bin/env python3
"""Test to force message summarization by exceeding 50k tokens"""

from langchain_core.messages import HumanMessage, AIMessage
from agent import count_tokens_in_messages, check_and_summarize_messages, get_llm

print("="*80)
print("MESSAGE SUMMARIZATION DEMO")
print("="*80)

# Create a long conversation that will exceed 50k tokens
messages = []

# Add 110 exchanges with long content to reach 50k+ tokens
for i in range(110):
    # Each human message ~500 characters = ~125 tokens
    human_content = f"""
    Query #{i+1}: I'm planning a trip to Paris and I need very detailed information about 
    the city. Can you tell me about the best museums, restaurants, hotels, transportation 
    options, weather conditions, cultural events, historical landmarks, shopping areas, 
    nightlife, parks, gardens, architectural highlights, local customs, language tips, 
    safety recommendations, day trip options, best neighborhoods to stay in, food markets,
    bakeries, cafes, river cruises, walking tours, bicycle rentals, and any other important
    information that would help me plan a comprehensive and memorable visit to the city?
    I'm particularly interested in family-friendly activities since I'll be traveling with 
    two children aged 7 and 10. We want to experience authentic French culture while also
    ensuring the kids have fun. Budget is around $10,000 for a 10-day trip.
    """
    
    # Each AI response ~600 characters = ~150 tokens
    ai_content = f"""
    Response #{i+1}: Paris is a magnificent city with endless attractions. For museums, I recommend 
    the Louvre, Musée d'Orsay, and Centre Pompidou. The weather in spring is mild with temperatures 
    around 15-20°C. For family accommodations, consider the Marais or Latin Quarter. Transportation 
    is excellent with metro, buses, and taxis. Top restaurants include Le Comptoir du Relais and 
    L'Avant Comptoir. For kids, visit Disneyland Paris, the Luxembourg Gardens playground, and the 
    Cité des Sciences. The Eiffel Tower offers spectacular views and is perfect for families.
    Don't miss a Seine river cruise and a visit to Montmartre. French bakeries like Poilâne and
    Du Pain et des Idées are must-visits. The Paris Museum Pass can save money on attractions.
    Safety is generally good but watch for pickpockets in tourist areas. Day trips to Versailles
    and Giverny are highly recommended. Local markets like Marché des Enfants Rouges offer
    authentic food experiences. The city is very walkable and bike-friendly with Vélib' rentals.
    """
    
    messages.append(HumanMessage(content=human_content))
    messages.append(AIMessage(content=ai_content))

# Calculate initial token count
initial_tokens = count_tokens_in_messages(messages)
print(f"\nInitial conversation:")
print(f"  Messages: {len(messages)}")
print(f"  Estimated tokens: {initial_tokens:,}")
print(f"  Exceeds 50k threshold: {'YES' if initial_tokens > 50000 else 'NO'}")

if initial_tokens > 50000:
    print(f"\n⚠️  Message chain exceeds 50,000 tokens!")
    print(f"   Triggering summarization...\n")
    
    # Initialize LLM
    llm = get_llm()
    
    # Perform summarization
    summarized_messages = check_and_summarize_messages(messages, llm, max_tokens=50000)
    
    # Calculate new token count
    new_tokens = count_tokens_in_messages(summarized_messages)
    reduction = initial_tokens - new_tokens
    reduction_pct = (reduction / initial_tokens) * 100
    
    print(f"\n" + "="*80)
    print("SUMMARIZATION RESULTS")
    print("="*80)
    print(f"Original:")
    print(f"  Messages: {len(messages)}")
    print(f"  Tokens: {initial_tokens:,}")
    print(f"\nAfter summarization:")
    print(f"  Messages: {len(summarized_messages)}")
    print(f"  Tokens: {new_tokens:,}")
    print(f"\nReduction:")
    print(f"  Removed: {reduction:,} tokens ({reduction_pct:.1f}%)")
    print(f"  Compression ratio: {initial_tokens/new_tokens:.1f}x")
    
    # Show summary content
    print(f"\n" + "="*80)
    print("SUMMARY CONTENT")
    print("="*80)
    for i, msg in enumerate(summarized_messages):
        msg_type = type(msg).__name__
        preview = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
        print(f"\nMessage {i+1} ({msg_type}):")
        print(f"  {preview}")
    
else:
    print(f"\n⚠️  Not enough tokens to trigger summarization")
    print(f"   Need at least 50,000 tokens (current: {initial_tokens:,})")

print(f"\n" + "="*80)
print("✓ Test completed")
print("="*80)
