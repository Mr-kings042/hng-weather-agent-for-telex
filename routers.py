from __future__ import annotations

import json
import os
import re
import difflib
from typing import Any, Dict, Optional, List

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from models import JSONRPCRequest, JSONRPCResponse
import services as svc 
from services import CityNotFoundError, WeatherAPIError, fetch_weather_batch
from cache import cache
from logger import logger


router = APIRouter()

# ============================================================================
# JSON-RPC Errors and helpers
# ============================================================================

class JSONRPCError:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Application-specific
    CITY_NOT_FOUND = -32001
    WEATHER_API_ERROR = -32002
    CACHE_ERROR = -32003


def create_error_response(
    request_id: Optional[Any],
    code: int,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> JSONRPCResponse:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JSONRPCResponse(id=request_id, error=error)


# ============================================================================
# A2A / Telex Endpoint
# ============================================================================

@router.post("/a2a/weather")
async def weather_rest_endpoint(request: Request) -> JSONResponse:
    """
    Unified JSON-RPC / Telex handler:
    - Accepts 'weather.get' (and alias 'getWeather') with params.city or free-text query.
    - Accepts conversational 'message/send' and 'execute' and extracts city from message-like fields
      (including Telex `message.parts` arrays).
    - Falls back to last city via params.context.channel_id if not provided explicitly.
    - Normalizes city names (abbr/typos) and optionally standardizes via geocoding.
    """
    body: Optional[Dict[str, Any]] = None

    try:
        raw = await request.body()
        if not raw or raw.strip() == b"":
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": JSONRPCError.PARSE_ERROR,
                        "message": "Parse error: empty request body",
                    },
                },
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
                        "data": {"details": str(e)},
                    },
                },
            )

        # JSON-RPC envelope validation
        if body.get("jsonrpc") != "2.0" or "id" not in body or "method" not in body:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": JSONRPCError.INVALID_REQUEST,
                        "message": "Invalid Request: jsonrpc must be '2.0', id and method are required",
                    },
                },
            )

        rpc_request = JSONRPCRequest(**body)
        logger.info(
            f"A2A/Weather Request: method={rpc_request.method}, id={rpc_request.id}"
        )
        logger.info(f"RPC params: {rpc_request.params!r}")

        # Method alias map
        raw_method = (rpc_request.method or "").strip()
        method_key = raw_method.lower().replace(" ", "")
        method_aliases = {
            "getweather": "weather.get",
            "weather.get": "weather.get",
            "weatherget": "weather.get",
            "message/send": "message.send",
            "message.send": "message.send",
            "execute": "execute",
        }
        method = method_aliases.get(method_key, method_key)

        params: Dict[str, Any] = rpc_request.params or {}
        city: Optional[str] = None

        if method == "weather.get":
            # Accept explicit city or free-text query fields
            city = params.get("city")
            if not city:
                q = params.get("query") or params.get("text") or params.get("message")
                if isinstance(q, dict):
                    q = q.get("content") or q.get("text") or ""
                if isinstance(q, str) and q:
                    city = extract_city_from_message(q)

            # Fallback to last-city by channel context
            if not city:
                channel_id = _get_channel_id(params.get("context"))
                if channel_id:
                    try:
                        cached_city = await cache.get(f"conversation:last_city:{channel_id}")
                        if isinstance(cached_city, str) and cached_city:
                            city = cached_city
                            logger.debug(f"Reused cached city for channel {channel_id}: {city}")
                    except Exception as e:
                        logger.warning(f"Failed to read cached city for channel {channel_id}: {e}")

        elif method in ("message.send", "execute"):
            # Collect candidate free-texts from multiple possible fields (Telex-compatible)
            candidates: List[str] = []

            msg = params.get("message")
            if isinstance(msg, dict):
                # Telex message.parts
                parts = msg.get("parts") or []
                if isinstance(parts, list):
                    for p in parts:
                        if not isinstance(p, dict):
                            continue
                        if p.get("kind") == "text" and isinstance(p.get("text"), str):
                            candidates.append(p["text"])
                        elif p.get("kind") == "data":
                            pdata = p.get("data") or []
                            if isinstance(pdata, list):
                                for d in pdata:
                                    if isinstance(d, dict) and d.get("kind") == "text" and isinstance(d.get("text"), str):
                                        candidates.append(d["text"])
                # Legacy fields
                for sub in (msg.get("content"), msg.get("text"), msg.get("message"), msg.get("body")):
                    if isinstance(sub, str) and sub:
                        candidates.append(sub)
            elif isinstance(msg,str):
                candidates.append(msg)

            # Other common fields seen in A2A payloads
            for key in ("input", "text", "query", "body", "data", "event"):
                v = params.get(key)
                if isinstance(v, dict):
                    for sub in (v.get("content"), v.get("text"), v.get("message"), v.get("body")):
                        if isinstance(sub, str) and sub:
                            candidates.append(sub)
                elif isinstance(v, str) and v:
                    candidates.append(v)

            # Execute: messages list
            msgs = params.get("messages") or []
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        for sub in (m.get("content"), m.get("text")):
                            if isinstance(sub, str) and sub:
                                candidates.append(sub)
                    else:
                        candidates.append(str(m))

            # Try extraction in order
            for text in candidates:
                city = extract_city_from_message(text)
                if city:
                    break

            # Explicit override
            if not city:
                city = params.get("city")

            # Context fallback
            if not city:
                channel_id = _get_channel_id(params.get("context"))
                if channel_id:
                    try:
                        cached_city = await cache.get(f"conversation:last_city:{channel_id}")
                        if isinstance(cached_city, str) and cached_city:
                            city = cached_city
                            logger.debug(f"Reused cached city for channel {channel_id}: {city}")
                    except Exception as e:
                        logger.warning(f"Failed to read cached city for channel {channel_id}: {e}")

        else:
            resp = create_error_response(
                rpc_request.id,
                JSONRPCError.METHOD_NOT_FOUND,
                f"Method '{raw_method}' not supported on this endpoint",
            )
            return JSONResponse(status_code=400, content=resp.model_dump())

        # Normalize/resolve city (abbr/typos + API standardization when available)
        if city:
            city = await resolve_city_name(city)

        logger.debug(f"Resolved city: {city!r}")
        if not city:
            resp = create_error_response(
                rpc_request.id,
                JSONRPCError.INVALID_PARAMS,
                "Missing required parameter: 'city' (provide in params, query/text/message, or context.channel_id)",
            )
            return JSONResponse(status_code=400, content=resp.model_dump())

        # Fetch weather
        try:
            weather_data = await svc.weather_agent({"city": city})
        except CityNotFoundError as e:
            resp = create_error_response(
                rpc_request.id, JSONRPCError.CITY_NOT_FOUND, str(e), data={"city": city}
            )
            return JSONResponse(status_code=404, content=resp.model_dump())
        except WeatherAPIError as e:
            resp = create_error_response(
                rpc_request.id, JSONRPCError.WEATHER_API_ERROR, str(e)
            )
            return JSONResponse(status_code=503, content=resp.model_dump())

        # Ensure city in response is the normalized one (useful when tests stub weather_agent)
        if city and weather_data.get("city") != city:
            weather_data["city"] = city

        # Persist last-city for conversation context
        channel_id = _get_channel_id((rpc_request.params or {}).get("context"))
        if channel_id:
            try:
                await cache.set(
                    f"conversation:last_city:{channel_id}",
                    weather_data.get("city", city),
                    ex=3600,
                )
            except Exception as e:
                logger.warning(f"Failed to persist last city for channel {channel_id}: {e}")

        # Build JSON-RPC result
        human_text = format_weather_response(weather_data)
        result_payload = {"response": human_text, "data": weather_data}
        resp = JSONRPCResponse(id=rpc_request.id, result=result_payload)
        return JSONResponse(status_code=200, content=resp.model_dump())

    except Exception as e:
        err = create_error_response(
            body.get("id") if isinstance(body, dict) else None,
            JSONRPCError.INTERNAL_ERROR,
            "Internal error",
            data={"details": str(e)},
        )
        return JSONResponse(status_code=500, content=err.model_dump())


# ============================================================================
# Convenience REST endpoints (manual tests / demos)
# ============================================================================

@router.get("/weather/{city}")
async def weather_get_endpoint(city: str):
    try:
        resolved = await resolve_city_name(city)
        result = await svc.weather_agent({"city": resolved or city})
        return result
    except CityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except WeatherAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error in weather_get_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/weather/batch")
async def weather_batch_endpoint(cities: List[str]):
    try:
        if len(cities) > 10:
            raise HTTPException(
                status_code=400,
                detail="Too many cities. Maximum 10 per request.",
            )
        # Optionally normalize each city before batch fetch to improve hit rate
        normed: List[str] = []
        for c in cities:
            resolved = await resolve_city_name(c)
            normed.append(resolved or c)
        results = await fetch_weather_batch(normed)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in weather_batch_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Helper functions
# ============================================================================

GEO_URL_DEFAULT = "https://geocoding-api.open-meteo.com/v1/search"

STOPWORDS = {
    "hi", "hello", "hey", "please", "thanks", "thank", "ok", "okay", "help",
    "the", "a", "an", "is", "it", "what", "whats", "what's", "how", "hows", "how's",
    "weather", "forecast", "temperature", "in", "at", "for", "on", "about", "tell", "me",
}

ABBREVIATIONS = {
    "nyc": "New York City",
    "la": "Los Angeles",
    "sf": "San Francisco",
    "ph": "Port Harcourt",
    "phc": "Port Harcourt",
    "vegas": "Las Vegas",
    "dc": "Washington",
    "d.c.": "Washington",
    "uk": "London",    # heuristic
    "uae": "Dubai",    # heuristic
    "naija": "Lagos",  # colloquial
}

COMMON_CITIES = {
    "Lagos", "Abuja", "Kano", "Kaduna", "Ibadan", "Port Harcourt", "Enugu",
    "London", "Paris", "Berlin", "Madrid", "Rome", "Amsterdam", "Dublin",
    "New York", "New York City", "Los Angeles", "San Francisco", "Seattle",
    "Chicago", "Houston", "Miami", "Boston", "Washington", "Toronto",
    "Vancouver", "Montreal", "Mexico City", "Rio de Janeiro", "Sao Paulo",
    "Cairo", "Nairobi", "Johannesburg", "Accra", "Kigali", "Addis Ababa",
    "Dubai", "Abu Dhabi", "Doha", "Istanbul", "Mumbai", "Delhi", "Bengaluru",
    "Tokyo", "Osaka", "Seoul", "Beijing", "Shanghai", "Singapore", "Sydney",
    "Melbourne", "Auckland", "Cape Town", "Lisbon", "Zurich", "Stockholm",
}

def _title_case(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split())

def _maybe_from_abbr(token: str) -> Optional[str]:
    k = token.lower().strip(". ")
    return ABBREVIATIONS.get(k)

def _fuzzy_city(token: str, cutoff: float = 0.85) -> Optional[str]:
    candidates = list(COMMON_CITIES) + list(ABBREVIATIONS.values())
    matches = difflib.get_close_matches(_title_case(token), candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def _get_channel_id(context: Any) -> Optional[str]:
    if isinstance(context, dict):
        return context.get("channel_id") or context.get("channel")
    return None

def extract_city_from_message(message: str) -> Optional[str]:
    """
    Heuristic extractor that:
    - Parses common phrasings (with or without 'weather' keywords)
    - Understands prompts like 'Tell me about Paris'
    - Handles abbreviations (NYC -> New York City)
    - Does lightweight typo correction via fuzzy match against COMMON_CITIES
    """
    if not message:
        return None

    txt = message.strip()
    txt_lower = txt.lower()

    # 1) Patterns including weather and non-weather prompts
    patterns = [
        r"(?:weather|forecast|temperature)\s+(?:in|at|for)\s+([a-zA-Z\s,]+)",
        r"(?:what(?:'|’)s|whats|how(?:'|’)s)\s+(?:the\s+)?(?:weather|forecast)\s+(?:in|at|for)\s+([a-zA-Z\s,]+)",
        r"(?:tell\s+me\s+about|about|regarding|info\s+on|information\s+on)\s+([a-zA-Z\s,]+)",
        r"(?:in|at|for)\s+([A-Za-z\s]{2,})$",
    ]
    for p in patterns:
        m = re.search(p, txt_lower)
        if m:
            candidate = m.group(1).strip(" .?;!,")
            ab = _maybe_from_abbr(candidate)
            if ab:
                return ab
            fuzzy = _fuzzy_city(candidate)
            if fuzzy:
                return fuzzy
            return _title_case(candidate)

    # 2) Short tokens (1-3 words) that look like a place (avoid stopwords)
    tokens = [t.strip(".,!?") for t in re.split(r"\s+", txt) if t.strip()]
    filtered = [t for t in tokens if t.lower() not in STOPWORDS]
    if 1 <= len(filtered) <= 3 and all(t.replace(",", "").isalpha() for t in filtered):
        candidate = " ".join(filtered)
        ab = _maybe_from_abbr(candidate)
        if ab:
            return ab
        fuzzy = _fuzzy_city(candidate)
        if fuzzy:
            return fuzzy
        return _title_case(candidate)

    # 3) Capitalized sequence after a preposition in original-cased text
    m = re.search(r"(?:in|at|for|about)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3})", txt)
    if m:
        candidate = m.group(1).strip()
        ab = _maybe_from_abbr(candidate)
        if ab:
            return ab
        fuzzy = _fuzzy_city(candidate)
        if fuzzy:
            return fuzzy
        return _title_case(candidate)

    return None

async def resolve_city_name(candidate: str) -> Optional[str]:
    """
    Resolve and standardize city names:
    - Map abbreviations (NYC -> New York City)
    - Fuzzy-correct common typos (e.g., 'lagoss' -> 'Lagos')
    - Validate/normalize via Open-Meteo geocoding API to return 'Name, Country'
    """
    if not candidate:
        return None

    # Abbreviations
    ab = _maybe_from_abbr(candidate)
    if ab:
        candidate = ab

    # Fuzzy correction
    fuzzy = _fuzzy_city(candidate)
    if fuzzy:
        candidate = fuzzy

    # Geocode to standardize (best effort)
    geo_url = os.environ.get("GEO_URL", GEO_URL_DEFAULT)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(geo_url, params={"name": candidate, "count": 1, "language": "en"})
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") or []
            if results:
                loc = results[0]
                name = loc.get("name") or candidate
                country = loc.get("country")
                return f"{name}, {country}" if country else name
    except Exception:
        # Swallow geocoding errors; fall back to local best guess
        pass

    return _title_case(candidate)

def format_weather_response(data: Dict[str, Any]) -> str:
    """Build a user-friendly weather response."""
    city = data.get("city", "Unknown")
    temp = data.get("temperature_c")
    weather = data.get("weather", "unknown conditions")
    source = data.get("source", "api")

    emoji = get_weather_emoji(weather)
    parts = [f"{emoji} Weather in {city}"]

    if temp is not None:
        parts.append(f"🌡️ Temperature: {temp}°C")
    parts.append(f"☁️ Conditions: {weather}")

    suggestion = get_weather_suggestion(temp, weather)
    if suggestion:
        parts.append(f"💡 {suggestion}")

    if source == "cache":
        parts.append("_ℹ️ (Cached data)_")

    return "\n\n".join(parts)

def get_weather_emoji(weather: str) -> str:
    """Get appropriate emoji for weather condition."""
    wl = (weather or "").lower()
    if "clear" in wl or "sunny" in wl:
        return "☀️"
    if "cloud" in wl:
        return "☁️"
    if "rain" in wl or "drizzle" in wl:
        return "🌧️"
    if "thunder" in wl or "storm" in wl:
        return "⛈️"
    if "snow" in wl:
        return "❄️"
    if "fog" in wl:
        return "🌫️"
    return "🌤️"

def get_weather_suggestion(temp: Optional[float], weather: str) -> Optional[str]:
    """Get activity suggestion based on weather."""
    wl = (weather or "").lower()
    if "rain" in wl or "drizzle" in wl:
        return "Don't forget your umbrella! ☔"
    if "thunder" in wl or "storm" in wl:
        return "Stay indoors if possible. It's stormy out there!"
    if temp is not None and temp > 30:
        return "It's hot! Stay hydrated and wear light clothing. 💧"
    if temp is not None and temp < 10:
        return "Bundle up! It's quite cold outside. 🧥"
    if "clear" in wl and temp is not None and 20 <= temp <= 28:
        return "Perfect weather for outdoor activities! 🚶‍♂️"
    return None