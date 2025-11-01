from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field


# JSON-RPC Protocol Models
class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request format for A2A protocol"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "weather.get",
                "params": {"city": "Lagos"}
            }
        }


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response format for A2A protocol"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Any] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"city": "Lagos", "temperature_c": 28.5}
            }
        }


# Weather-specific Models
class WeatherParams(BaseModel):
    """Parameters for weather query"""
    city: str = Field(..., description="City name to get weather for", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {"city": "Lagos"}
        }


class WeatherResult(BaseModel):
    """Weather information result"""
    city: str = Field(..., description="City name")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    feels_like_c: Optional[float] = Field(None, description="Feels like temperature")
    humidity: Optional[int] = Field(None, description="Humidity percentage")
    weather: str = Field(..., description="Weather condition description")
    wind_speed: Optional[float] = Field(None, description="Wind speed in m/s")
    source: str = Field(default="open-meteo", description="Data source")
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Lagos",
                "temperature_c": 28.5,
                "feels_like_c": 31.2,
                "humidity": 75,
                "weather": "partly cloudy",
                "wind_speed": 3.5,
                "source": "open-meteo"
            }
        }


# Additional useful models for Telex integration
# class TelexMessage(BaseModel):
#     """Message format for Telex.im"""
#     role: Literal["user", "assistant", "system"] = "user"
#     content: str = Field(..., min_length=1)


# class AgentRequest(BaseModel):
#     """Request from Telex to the agent"""
#     messages: list[TelexMessage] = Field(..., min_length=1)
#     context: Optional[Dict[str, Any]] = None


# class AgentResponse(BaseModel):
#     """Response from agent to Telex"""
#     response: str = Field(..., description="Agent's response message")
#     data: Optional[Dict[str, Any]] = Field(None, description="Additional data")
#     error: Optional[str] = Field(None, description="Error message if any")