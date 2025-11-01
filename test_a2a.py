import pytest
from fastapi.testclient import TestClient
from main import app
from routers import JSONRPCError
from services import CityNotFoundError
import routers
client = TestClient(app)

def test_invalid_jsonrpc_envelope():
    # missing "jsonrpc" field -> INVALID_REQUEST
    payload = {"id": 1, "method": "weather.get", "params": {"city": "Lagos"}}
    resp = client.post("/a2a/weather", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == JSONRPCError.INVALID_REQUEST


def test_method_not_supported():
    payload = {"jsonrpc": "2.0", "id": 2, "method": "unknown.method", "params": {}}
    resp = client.post("/a2a/weather", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == JSONRPCError.METHOD_NOT_FOUND


def test_reuse_cached_city(monkeypatch):
    # Simulate cache returning a last-city for the channel, and stub weather_agent
    async def fake_cache_get(key):
        return "Lagos, Nigeria"

    async def fake_weather_agent(params):
        return {
            "city": "Lagos, Nigeria",
            "temperature_c": 28.0,
            "feels_like_c": 30.0,
            "humidity": 70,
            "weather": "Partly cloudy",
            "wind_speed": 3.2,
            "source": "open-meteo",
        }

    monkeypatch.setattr("cache.cache.get", fake_cache_get)
    monkeypatch.setattr("services.weather_agent", fake_weather_agent)

    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "weather.get",
        "params": {"context": {"channel_id": "chan-123"}}
    }
    resp = client.post("/a2a/weather", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["data"]["city"] == "Lagos, Nigeria"
    assert "response" in body["result"]


def test_city_not_found_returns_jsonrpc_error(monkeypatch):
    async def fake_weather_agent(params):
        raise CityNotFoundError("Could not find city: Atlantis")

    monkeypatch.setattr("services.weather_agent", fake_weather_agent)

    payload = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "weather.get",
        "params": {"city": "Atlantis"}
    }
    resp = client.post("/a2a/weather", json=payload)
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == JSONRPCError.CITY_NOT_FOUND
    assert "Atlantis" in body["error"]["data"]["city"] or "Atlantis" in body["error"]["message"]
def test_a2a_weather_missing_body():
    resp = client.post("/a2a/weather", json={})
    assert resp.status_code == 400

def test_a2a_weather_success(monkeypatch):
    # Stub weather_agent to avoid external calls
    async def fake_weather_agent(params):
        return {
            "city": "Lagos",
            "temperature_c": 28.0,
            "weather": "Partly cloudy",
            "source": "open-meteo"
        }
    monkeypatch.setattr("services.weather_agent", fake_weather_agent)
    payload = {"jsonrpc":"2.0","id":1,"method":"weather.get","params":{"city":"Lagos"}}
    resp = client.post("/a2a/weather", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["data"]["city"] == "Lagos, Nigeria"