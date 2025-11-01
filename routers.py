from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import json

from models import JSONRPCRequest, JSONRPCResponse
from services import (
    weather_agent,
    CityNotFoundError,
    WeatherAPIError,
    fetch_weather_batch
)
from cache import cache
from logger import logger

router = APIRouter()


# JSON-RPC 2.0 Error Codes
class JSONRPCError:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes (application-specific)
    CITY_NOT_FOUND = -32001
    WEATHER_API_ERROR = -32002
    CACHE_ERROR = -32003


def create_error_response(
    request_id: Optional[Any],
    code: int,
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> JSONRPCResponse:
    error = {
        "code": code,
        "message": message
    }
    if data:
        error["data"] = data
    
    return JSONRPCResponse(id=request_id, error=error)

# ============================================================================
# REST Endpoints (for testing and Telex.im integration)
# ============================================================================

@router.post("/a2a/weather")
async def weather_rest_endpoint(request: Request):
    """
    JSON-RPC style endpoint for Telex/Mastra calling this agent.
    Accepts raw request body and returns a JSON-RPC response.
    """
    body = None
    try:
        # Read raw body first so we can return a proper JSON-RPC parse error
        raw = await request.body()
        if not raw or raw.strip() == b"":
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": JSONRPCError.PARSE_ERROR,
                        "message": "Parse error: empty request body"
                    }
                }
            )

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": JSONRPCError.PARSE_ERROR,
                        "message": "Parse error: invalid JSON",
                        "data": {"details": str(e)}
                    }
                }
            )

        # Validate JSON-RPC envelope
        if body.get("jsonrpc") != "2.0" or "id" not in body or "method" not in body:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": JSONRPCError.INVALID_REQUEST,
                        "message": "Invalid Request: jsonrpc must be '2.0', id and method are required"
                    }
                }
            )

        rpc_request = JSONRPCRequest(**body)
        logger.info(f"A2A/Weather Request: method={rpc_request.method}, id={rpc_request.id}")

        # Normalize method and accept common aliases
        raw_method = (rpc_request.method or "").strip()
        method_key = raw_method.lower().replace(" ", "")
        aliases = {
            "getweather": "weather.get",
            "weather.get": "weather.get",
            "weatherget": "weather.get",
        }
        method = aliases.get(method_key, method_key)

        if method != "weather.get":
            resp = JSONRPCResponse(
                id=rpc_request.id,
                error={"code": JSONRPCError.METHOD_NOT_FOUND, "message": f"Method '{raw_method}' not supported on this endpoint"}
            )
            return JSONResponse(status_code=400, content=resp.model_dump())

        # Extract params and optional context for conversation reuse
        params = rpc_request.params or {}
        city = params.get("city")

        channel_id = None
        context = params.get("context") or {}
        if isinstance(context, dict):
            channel_id = context.get("channel_id") or context.get("channel")
        cache_key = f"conversation:last_city:{channel_id}" if channel_id else None

        # Reuse cached city if none provided
        if not city and cache_key:
            try:
                cached_city = await cache.get(cache_key)
                if isinstance(cached_city, str) and cached_city:
                    logger.debug(f"Reusing cached city for channel {channel_id}: {cached_city}")
                    city = cached_city
            except Exception as e:
                logger.warning(f"Failed to read cached city for channel {channel_id}: {e}")

        if not city:
            resp = JSONRPCResponse(
                id=rpc_request.id,
                error={"code": JSONRPCError.INVALID_PARAMS, "message": "Missing required parameter: 'city' (or provide context.channel_id to reuse last city)"}
            )
            return JSONResponse(status_code=400, content=resp.model_dump())

        # Fetch weather
        try:
            weather_data = await weather_agent({"city": city})
        except CityNotFoundError as e:
            resp = JSONRPCResponse(id=rpc_request.id, error={"code": JSONRPCError.CITY_NOT_FOUND, "message": str(e), "data": {"city": city}})
            return JSONResponse(status_code=404, content=resp.model_dump())
        except WeatherAPIError as e:
            resp = JSONRPCResponse(id=rpc_request.id, error={"code": JSONRPCError.WEATHER_API_ERROR, "message": str(e)})
            return JSONResponse(status_code=503, content=resp.model_dump())

        # Persist resolved city for follow-ups
        if cache_key:
            try:
                await cache.set(cache_key, weather_data.get("city", city), ex=3600)
            except Exception as e:
                logger.warning(f"Failed to persist last city for channel {channel_id}: {e}")

        # Build human-friendly response text and return JSON-RPC response
        response_text = format_weather_response(weather_data)
        result_payload = {"response": response_text, "data": weather_data}
        resp = JSONRPCResponse(id=rpc_request.id, result=result_payload)
        return JSONResponse(status_code=200, content=resp.model_dump())

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": body.get("id") if isinstance(body, dict) else None,
                "error": {
                    "code": JSONRPCError.INTERNAL_ERROR,
                    "message": "Internal error",
                    "data": {"details": str(e)}
                }
            }
        )




@router.get("/weather/{city}")
async def weather_get_endpoint(city: str):
    try:
        result = await weather_agent({"city": city})
        return result
    except CityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except WeatherAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error in weather_get_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/weather/batch")
async def weather_batch_endpoint(cities: list[str]):
    try:
        if len(cities) > 10:
            raise HTTPException(
                status_code=400,
                detail="Too many cities. Maximum 10 per request."
            )
        
        results = await fetch_weather_batch(cities)
        return results
    
    except Exception as e:
        logger.error(f"Error in weather_batch_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Helper Functions
# ============================================================================

def extract_city_from_message(message: str) -> Optional[str]:
    import re
    
    message_lower = message.lower().strip()
    
    # Common patterns
    patterns = [
        r"weather (?:in|at|for) ([a-zA-Z\s]+?)[\?\.]?$",
        r"(?:what\'?s?|how\'?s?) (?:the )?weather (?:in|at|for) ([a-zA-Z\s]+?)[\?\.]?$",
        r"forecast (?:in|at|for) ([a-zA-Z\s]+?)[\?\.]?$",
        r"temperature (?:in|at|of) ([a-zA-Z\s]+?)[\?\.]?$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            city = match.group(1).strip()
            # Capitalize properly
            return ' '.join(word.capitalize() for word in city.split())
    
    # If no pattern matches, check if message is just a city name (simple words)
    words = message.split()
    if 1 <= len(words) <= 3 and all(word.replace(',', '').isalpha() for word in words):
        return ' '.join(word.capitalize() for word in words)
    
    return None


def format_weather_response(data: Dict[str, Any]) -> str:
  
    city = data.get("city", "Unknown")
    temp = data.get("temperature_c")
    weather = data.get("weather", "unknown conditions")
    source = data.get("source", "api")
    
    # Create emoji based on weather
    emoji = get_weather_emoji(weather)
    
    response = f"Weather in {city}**\n"
    response += f"Temperature: **{temp}°C**\n"
    response += f"Conditions: **{weather}**\n"
    
    # Add fun suggestions based on weather
    suggestion = get_weather_suggestion(temp, weather)
    if suggestion:
        response += f"\n💡 {suggestion}"
    
    if source == "cache":
        response += "\n\n_ℹ️ (Cached data)_"
    
    return response


def get_weather_emoji(weather: str) -> str:
    """Get appropriate emoji for weather condition"""
    weather_lower = weather.lower()
    
    if "clear" in weather_lower or "sunny" in weather_lower:
        return "☀️"
    elif "cloud" in weather_lower:
        return "☁️"
    elif "rain" in weather_lower or "drizzle" in weather_lower:
        return "🌧️"
    elif "thunder" in weather_lower or "storm" in weather_lower:
        return "⛈️"
    elif "snow" in weather_lower:
        return "❄️"
    elif "fog" in weather_lower:
        return "🌫️"
    else:
        return "🌤️"


def get_weather_suggestion(temp: float, weather: str) -> Optional[str]:
    """Get activity suggestion based on weather"""
    weather_lower = weather.lower()
    
    if "rain" in weather_lower or "drizzle" in weather_lower:
        return "Don't forget your umbrella! ☔"
    elif "thunder" in weather_lower or "storm" in weather_lower:
        return "Stay indoors if possible. It's stormy out there!"
    elif temp and temp > 30:
        return "It's hot! Stay hydrated and wear light clothing. 💧"
    elif temp and temp < 10:
        return "Bundle up! It's quite cold outside. 🧥"
    elif "clear" in weather_lower and temp and 20 <= temp <= 28:
        return "Perfect weather for outdoor activities! 🚶‍♂️"
    
    return None
