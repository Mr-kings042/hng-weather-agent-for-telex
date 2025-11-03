import os
import asyncio
import random
import bisect
from functools import wraps
from typing import Dict, Any, Optional, Tuple, List

import httpx

from models import WeatherParams, WeatherResult
from cache import cache
from logger import logger

# Configuration
CACHE_TTL: int = int(os.environ.get("WEATHER_CACHE_TTL", "300"))
API_TIMEOUT: float = float(os.environ.get("API_TIMEOUT", "10.0"))
GEO_URL = os.getenv("GEO_URL")
WEATHER_URL = os.getenv("WEATHER_URL")


# Errors
class WeatherServiceError(Exception):
    """Base exception for weather service errors."""


class CityNotFoundError(WeatherServiceError):
    """Raised when geocoding cannot find the city."""


class WeatherAPIError(WeatherServiceError):
    """Raised when weather API fails or returns unexpected data."""


# Small utilities
def get_weather_description(code: int) -> str:
    """Map Open-Meteo weather codes to human-friendly description."""
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, f"Unknown weather (code: {code})")


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


# Lightweight async retry decorator (exponential backoff + jitter)
def async_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 8.0,
    exceptions: Tuple = (httpx.HTTPError,),
):
    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            attempt = 1
            delay = initial_delay
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_attempts:
                        logger.error("Retry attempts exhausted", exc_info=True)
                        raise
                    jitter = random.uniform(0, 0.1 * delay)
                    wait = min(delay + jitter, max_delay)
                    logger.debug(f"Retry {attempt}/{max_attempts} after {wait:.2f}s due to: {exc}")
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, max_delay)
                    attempt += 1
        return wrapped
    return decorator


# Network helpers
async def _get_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = await client.get(url, params=params, timeout=API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@async_retry(exceptions=(httpx.HTTPError,))
async def geocode_city(city: str, client: httpx.AsyncClient) -> Tuple[float, float, str]:
    """
    Geocode using Open-Meteo geocoding API.
    Returns (lat, lon, resolved_name).
    """
    logger.info(f"Geocoding city: {city}")
    data = await _get_json(client, GEO_URL, {"name": city, "count": 1, "language": "en"})
    results = data.get("results") or []
    if not results:
        logger.warning(f"Geocode: no results for '{city}'")
        raise CityNotFoundError(f"Could not find city: {city}")
    loc = results[0]
    lat = float(loc["latitude"])
    lon = float(loc["longitude"])
    name = loc.get("name", city)
    country = loc.get("country")
    resolved = f"{name}, {country}" if country else name
    logger.debug(f"Geocoded '{city}' -> {resolved} ({lat},{lon})")
    return lat, lon, resolved


@async_retry(exceptions=(httpx.HTTPError,))
async def fetch_weather_data(lat: float, lon: float, client: httpx.AsyncClient) -> Dict[str, Any]:
    """
    Fetch current weather and hourly series (humidity/apparent temp) aligned to current_weather.time.
    Returns normalized dictionary with numeric values where possible.
    """
    logger.info(f"Fetching weather for coordinates: ({lat}, {lon})")
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "relativehumidity_2m,apparent_temperature",
        "timezone": "auto",
    }
    data = await _get_json(client, WEATHER_URL, params)
    current = data.get("current_weather")
    if not current:
        logger.error("Weather response missing 'current_weather'")
        raise WeatherAPIError("No current weather available from provider")

    # align hourly series to current time
    hourly = data.get("hourly", {}) or {}
    times: List[str] = hourly.get("time", []) or []
    humidity = None
    apparent = None
    curr_time = current.get("time")
    if curr_time and times:
        try:
            idx = times.index(curr_time)
        except ValueError:
            idx = bisect.bisect_left(times, curr_time)
            if idx >= len(times):
                idx = len(times) - 1
        if 0 <= idx < len(times):
            rh = hourly.get("relativehumidity_2m") or []
            at = hourly.get("apparent_temperature") or []
            if isinstance(rh, list) and len(rh) > idx:
                humidity = _coerce_int(rh[idx])
            if isinstance(at, list) and len(at) > idx:
                apparent = _coerce_float(at[idx])

    result = {
        "temperature": _coerce_float(current.get("temperature")),
        "windspeed": _coerce_float(current.get("windspeed")),
        "winddirection": current.get("winddirection"),
        "weathercode": _coerce_int(current.get("weathercode")),
        "time": curr_time,
        "relativehumidity_2m": humidity,
        "apparent_temperature": apparent,
    }
    logger.debug("Fetched weather data: %s", result)
    return result


async def fetch_weather_open_meteo(city: str) -> Dict[str, Any]:
    """
    High-level fetch: geocode -> fetch_weather_data -> normalize payload returned to callers.
    """
    async with httpx.AsyncClient() as client:
        lat, lon, resolved = await geocode_city(city, client)
        raw = await fetch_weather_data(lat, lon, client)

    temp = raw.get("temperature")
    wind_speed = raw.get("windspeed")
    weather_code = raw.get("weathercode")
    humidity = raw.get("relativehumidity_2m")
    feels_like = raw.get("apparent_temperature")

    description = get_weather_description(weather_code if weather_code is not None else -1)

    return {
        "city": resolved,
        "temperature_c": temp,
        "feels_like_c": feels_like,
        "humidity": humidity,
        "weather": description,
        "wind_speed": wind_speed,
        "wind_direction": raw.get("winddirection"),
        "weather_code": weather_code,
        "source": "open-meteo",
    }


# Cache helpers (small, used by weather_agent)
async def _read_cache(key: str) -> Optional[Dict[str, Any]]:
    try:
        return await cache.get(key)
    except Exception as e:
        logger.warning("Cache read error for %s: %s", key, e)
        return None


async def _write_cache(key: str, value: Dict[str, Any], ttl: int = CACHE_TTL) -> None:
    try:
        await cache.set(key, value, ex=ttl)
    except Exception as e:
        logger.warning("Cache write error for %s: %s", key, e)


# Public agent API
async def weather_agent(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry used by routers. Validates params, uses cache, and fetches from provider when needed.
    Returns dict matching WeatherResult.
    """
    weather_params = WeatherParams(**params)
    normalized = weather_params.city.strip().lower()
    cache_key = f"weather:{normalized}"

    cached = await _read_cache(cache_key)
    if cached:
        logger.info("Serving weather from cache for %s", weather_params.city)
        return WeatherResult(
            city=cached.get("city", weather_params.city),
            temperature_c=cached["temperature_c"],
            feels_like_c=cached.get("feels_like_c"),
            humidity=cached.get("humidity"),
            weather=cached["weather"],
            wind_speed=cached.get("wind_speed"),
            source="cache",
        ).model_dump()

    logger.info("Fetching weather for %s from provider", weather_params.city)
    try:
        api = await fetch_weather_open_meteo(weather_params.city)
    except CityNotFoundError:
        logger.error("City not found: %s", weather_params.city)
        raise
    except WeatherAPIError:
        logger.error("Weather API error for: %s", weather_params.city)
        raise
    except Exception as e:
        logger.exception("Unexpected error fetching%s: %s", weather_params.city, e)
        raise WeatherAPIError(f"Failed to fetch weather: {e}") from e

    # Cache normalized payload
    cache_payload = {
        "city": api["city"],
        "temperature_c": api["temperature_c"],
        "feels_like_c": api.get("feels_like_c"),
        "humidity": api.get("humidity"),
        "weather": api["weather"],
        "wind_speed": api.get("wind_speed"),
        "source": api.get("source", "open-meteo"),
    }
    await _write_cache(cache_key, cache_payload)

    return WeatherResult(
        city=api["city"],
        temperature_c=api["temperature_c"],
        feels_like_c=api.get("feels_like_c"),
        humidity=api.get("humidity"),
        weather=api["weather"],
        wind_speed=api.get("wind_speed"),
        source=api.get("source", "open-meteo"),
    ).model_dump()


async def fetch_weather_batch(cities: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parallel batch fetch wrapper that returns a dict[city -> result|error]."""
    async def _one(city: str):
        try:
            res = await weather_agent({"city": city})
            return city, res
        except Exception as e:
            logger.error("Batch fetch failed for %s: %s", city, e)
            return city, {"error": str(e), "city": city, "source": "error"}

    tasks = [_one(c) for c in cities]
    results = await asyncio.gather(*tasks)
    return dict(results)
