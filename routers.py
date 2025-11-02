from __future__ import annotations

import json
import os
import re
import difflib
import uuid
from datetime import datetime, timezone
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
    - Falls back to last city via params.context.channel_id.
    - Normalizes city names (abbr/typos) and optionally standardizes via geocoding/canonical map.
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
        logger.info(f"A2A/Weather Request: method={rpc_request.method}, id={rpc_request.id}")
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
                # Telex message.parts: prefer current user text, de-prioritize history in 'data'
                parts = msg.get("parts") or []
                first_user_text: Optional[str] = None
                backlog_texts: List[str] = []
                if isinstance(parts, list):
                    for p in parts:
                        if not isinstance(p, dict):
                            continue
                        if p.get("kind") == "text" and isinstance(p.get("text"), str) and not first_user_text:
                            first_user_text = p["text"]
                        elif p.get("kind") == "data":
                            pdata = p.get("data") or []
                            if isinstance(pdata, list):
                                for d in pdata:
                                    if isinstance(d, dict) and d.get("kind") == "text" and isinstance(d.get("text"), str):
                                        backlog_texts.append(d["text"])
                if first_user_text:
                    candidates.append(first_user_text)
                candidates.extend(backlog_texts)  # history as low-priority

                # Legacy fields
                for sub in (msg.get("content"), msg.get("text"), msg.get("message"), msg.get("body")):
                    if isinstance(sub, str) and sub:
                        candidates.append(sub)

            elif isinstance(msg, str):
                # Allow plain string message
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

        # Normalize/resolve city (abbr/typos + API/canonical standardization)
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

        # Build result payloads
        human_text = format_weather_response(weather_data)

        # Telex-style task result for conversational methods
        if method in ("message.send", "execute"):
            task_id = _get_task_id(params)
            context_id = channel_id or _generate_context_id()
            message_id = _new_id()

            telex_result = {
                "id": task_id,
                "contextId": context_id,
                "status": {
                    "state": "succeeded",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": {
                        "messageId": message_id,
                        "role": "agent",
                        "parts": [{"kind": "text", "text": human_text}],
                        "kind": "message",
                        "taskId": task_id,
                    },
                },
                "artifacts": [
                    {
                        "artifactId": _new_id(),
                        "name": "weather",
                        "parts": [
                            {"kind": "text", "text": json.dumps(weather_data, ensure_ascii=False)}
                        ],
                    }
                ],
                "history": [],
                "kind": "task",
            }
            resp = JSONRPCResponse(id=rpc_request.id, result=telex_result)
            return JSONResponse(status_code=200, content=resp.model_dump())

        # Classic result for weather.get (keeps tests passing)
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
    "uk": "London",    
    "uae": "Dubai",    
    "eko": "Lagos",    
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

# Minimal canonical names to avoid network in tests
CANONICAL_CITY = {
    "lagos": "Lagos, Nigeria",
    "london": "London, United Kingdom",
    "paris": "Paris, France",
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

def _get_task_id(params: Dict[str, Any]) -> str:
    msg = params.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("taskId"), str) and msg["taskId"]:
        return msg["taskId"]
    if isinstance(params.get("taskId"), str):
        return params["taskId"]
    return _new_id()

def _generate_context_id() -> str:
    return _new_id()

def _new_id() -> str:
    return str(uuid.uuid4())

def extract_city_from_message(message: str) -> Optional[str]:
    """
    Smarter extractor that:
    - Prefers the latest city if multiple appear in one sentence.
    - Stops matches at punctuation / conjunctions / next 'weather' clause.
    - Handles 'Tell me about Paris' (no weather keyword).
    - Expands abbreviations (NYC -> New York City) and fuzzy-corrects typos.
    """
    if not message:
        return None

    # Strip simple HTML tags Telex sometimes includes (e.g., <p>...</p>)
    txt = re.sub(r"<[^>]+>", " ", message).strip()
    if not txt:
        return None
    lo = txt.lower()

    def _slice_from(lo_match: re.Match) -> str:
        s, e = lo_match.span(1)
        return txt[s:e].strip(" .,!?:;")

    cities: List[str] = []

    # 1) Weather prompts: find all occurrences, non-greedy up to punctuation/conjunction/next keyword
    p_weather_1 = r"(?:weather|forecast|temperature)\s+(?:in|at|for)\s+([a-z][a-z\s,]{1,50}?)(?=$|[?.!]|(?:\s+(?:and|but|or|then|what|how)\b)|\s+weather\b)"
    p_weather_2 = r"(?:what(?:'|’)?s|whats|how(?:'|’)?s)\s+(?:the\s+)?(?:weather|forecast)\s+(?:in|at|for)\s+([a-z][a-z\s,]{1,50}?)(?=$|[?.!]|(?:\s+(?:and|but|or|then|what|how)\b)|\s+weather\b)"
    for m in re.finditer(p_weather_1, lo):
        cities.append(_slice_from(m))
    for m in re.finditer(p_weather_2, lo):
        cities.append(_slice_from(m))

    # 2) Non-weather info prompts (about/regarding ...)
    p_about = r"(?:tell\s+me\s+about|about|regarding|info\s+on|information\s+on)\s+([a-z][a-z\s,]{1,50}?)(?=$|[?.!]|(?:\s+(?:and|but|or)\b))"
    for m in re.finditer(p_about, lo):
        cities.append(_slice_from(m))

    # 3) Generic preposition + capitalized phrase in original text (fallback)
    m = re.search(r"(?:in|at|for|about)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3})", txt)
    if m:
        cities.append(m.group(1).strip())

    # 4) Plain short tokens that look like a place (1-3 words, non-stopwords)
    if not cities:
        tokens = [t.strip(".,!?") for t in re.split(r"\s+", txt) if t.strip()]
        filtered = [t for t in tokens if t.lower() not in STOPWORDS]
        if 1 <= len(filtered) <= 3 and all(t.replace(",", "").isalpha() for t in filtered):
            cities.append(" ".join(filtered))

    # Normalize all candidates, keep only plausible uniques, choose the last
    normalized: List[str] = []
    seen = set()
    for c in cities:
        if not c:
            continue
        cand = c.strip(" .,!?:;")
        # Abbreviations
        ab = _maybe_from_abbr(cand)
        if ab:
            cand = ab
        # Fuzzy correction
        fuzzy = _fuzzy_city(cand)
        if fuzzy:
            cand = fuzzy
        cand = _title_case(cand)
        key = cand.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(cand)

    if not normalized:
        return None
    # Prefer the last mentioned city if multiple were found
    return normalized[-1]

async def resolve_city_name(candidate: str) -> Optional[str]:
    """
    Resolve and standardize city names:
    - Map abbreviations (NYC -> New York City)
    - Fuzzy-correct common typos (e.g., 'lagoss' -> 'Lagos')
    - Use canonical mapping when available (offline-friendly)
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

    # Canonical map (avoid network in tests)
    canon = CANONICAL_CITY.get(candidate.strip().lower())
    if canon:
        return canon

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