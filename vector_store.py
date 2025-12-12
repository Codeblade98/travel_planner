import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os


class CityVectorStore:
    """Manages vector storage for pre-populated city information."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client and collection."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = "city_knowledge"
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Pre-populated city information"}
            )
            self._populate_cities()
    
    def _populate_cities(self):
        """Pre-populate the vector store with detailed city information."""
        cities_data = [
            {
                "id": "paris",
                "city": "Paris",
                "content": """
                Paris, the capital of France, is one of the world's most iconic cities. 
                Known as the "City of Light," it's famous for the Eiffel Tower, which 
                stands at 330 meters tall and was built for the 1889 World's Fair. 
                The Louvre Museum is the world's largest art museum, housing the Mona Lisa 
                and over 38,000 objects. Notre-Dame Cathedral, a masterpiece of French 
                Gothic architecture from the 12th century, sits on the Île de la Cité. 
                The Champs-Élysées is one of the world's most famous avenues, leading to 
                the Arc de Triomphe. Montmartre, with its Sacré-Cœur Basilica, offers 
                stunning views of the city. Paris has a population of about 2.2 million 
                in the city proper and over 12 million in the metropolitan area. The city 
                is divided into 20 arrondissements (districts) arranged in a spiral pattern. 
                French cuisine originated here with dishes like croissants, macarons, and 
                coq au vin. The Seine River flows through the heart of Paris, spanned by 
                37 bridges. Paris is also a global center for art, fashion, gastronomy, 
                and culture, with numerous theaters, museums, and galleries.
                """,
                "metadata": {
                    "country": "France",
                    "population": "2.2 million",
                    "famous_for": "Eiffel Tower, Louvre, Fashion, Cuisine"
                }
            },
            {
                "id": "tokyo",
                "city": "Tokyo",
                "content": """
                Tokyo, officially the Tokyo Metropolis, is the capital and most populous 
                city of Japan with over 14 million residents in the city proper and 
                nearly 40 million in the Greater Tokyo Area, making it the most populous 
                metropolitan area in the world. The city was originally named Edo and 
                served as the seat of the Tokugawa shogunate from 1603. It became the 
                imperial capital in 1868 when Emperor Meiji moved his seat from Kyoto. 
                Tokyo is known for its blend of traditional and ultra-modern architecture, 
                from ancient temples like Senso-ji (founded in 628 AD) to the futuristic 
                Tokyo Skytree, which stands at 634 meters. Shibuya Crossing is one of 
                the world's busiest pedestrian intersections, with thousands crossing at 
                once. The city has an extensive rail network, including the famous Shinkansen 
                bullet trains. Tsukiji Outer Market and Toyosu Market are world-renowned 
                for fresh seafood. Tokyo is divided into 23 special wards, each with its 
                own character - from the electronics hub of Akihabara to the fashion 
                district of Harajuku. Cherry blossom season (sakura) in spring is a major 
                cultural event. Tokyo hosted the Summer Olympics in 1964 and 2020 (held in 2021). 
                The city is a global financial center and home to numerous Fortune 500 companies.
                """,
                "metadata": {
                    "country": "Japan",
                    "population": "14 million",
                    "famous_for": "Technology, Anime, Sushi, Cherry Blossoms"
                }
            },
            {
                "id": "new_york",
                "city": "New York",
                "content": """
                New York City, often called NYC or simply New York, is the most populous 
                city in the United States with over 8.3 million residents in five boroughs: 
                Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. The city was 
                founded as New Amsterdam by Dutch colonists in 1624 and renamed New York 
                when the English took control in 1664. The Statue of Liberty, a gift from 
                France, has welcomed immigrants since 1886 and stands on Liberty Island. 
                Ellis Island processed over 12 million immigrants between 1892 and 1954. 
                Manhattan's skyline is iconic, featuring the Empire State Building (completed 
                in 1931), One World Trade Center (541 meters tall), and the Chrysler Building. 
                Times Square in Midtown Manhattan is known as "The Crossroads of the World" 
                and is one of the world's most visited tourist attractions. Central Park, 
                an 843-acre urban park designed by Frederick Law Olmsted and Calvert Vaux, 
                provides a green oasis in the heart of Manhattan. Wall Street is the heart 
                of America's financial district and home to the New York Stock Exchange. 
                Broadway is synonymous with world-class theater, with over 40 professional 
                theaters. The Metropolitan Museum of Art (The Met) is the largest art museum 
                in the United States. NYC is incredibly diverse, with over 800 languages 
                spoken and cuisines from every corner of the world. The subway system 
                operates 24/7 with 472 stations across 245 miles of track.
                """,
                "metadata": {
                    "country": "United States",
                    "population": "8.3 million",
                    "famous_for": "Statue of Liberty, Broadway, Wall Street, Diversity"
                }
            }
        ]
        
        # Add documents to collection
        for city_data in cities_data:
            self.collection.add(
                documents=[city_data["content"]],
                metadatas=[city_data["metadata"]],
                ids=[city_data["id"]]
            )
        
        print(f"✓ Pre-populated vector store with {len(cities_data)} cities")
    
    def search_city(self, query: str, n_results: int = 1) -> Dict:
        """
        Search for city information in the vector store.
        Returns the best match or None if not found.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if results["documents"] and len(results["documents"][0]) > 0:
            return {
                "found": True,
                "content": results["documents"][0][0],
                "metadata": results["metadatas"][0][0] if results["metadatas"] else {},
                "distance": results["distances"][0][0] if results["distances"] else None
            }
        
        return {"found": False, "content": None, "metadata": {}}
    
    def has_city_info(self, city: str, threshold: float = 0.5) -> bool:
        """
        Check if the vector store has detailed information about a city.
        Uses similarity threshold to determine if a match is good enough.
        """
        result = self.search_city(city)
        
        # If no results or distance is too high, consider it as not found
        if not result["found"]:
            return False
        
        # Lower distance = better match. Threshold can be adjusted.
        if result["distance"] is not None and result["distance"] > threshold:
            return False
        
        return True


# Initialize the vector store (singleton pattern)
_vector_store = None

def get_vector_store() -> CityVectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = CityVectorStore()
    return _vector_store
