from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class WeatherData(BaseModel):
    """Schema for weather data point."""
    date: str
    temperature: float
    condition: str
    humidity: int
    wind_speed: int


class TravelAssistantOutput(BaseModel):
    """Final structured output from the agent."""
    city: str = Field(description="The city name")
    city_summary: str = Field(description="Detailed summary about the city")
    weather_forecast: List[WeatherData] = Field(description="Weather forecast data")
    image_urls: List[str] = Field(description="List of image URLs for the city")
    source: str = Field(description="Source of information: 'vector_store' or 'web_search'")


class CityExtractionResult(BaseModel):
    """Schema for LLM-based city extraction."""
    city_found: bool = Field(description="Whether a city was found in the query")
    city_name: Optional[str] = Field(description="The extracted city name")
    needs_clarification: bool = Field(description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(description="Question to ask user if clarification needed")
    session_updates: Dict = Field(description="Updates to session memory (LLM decides keys)")
