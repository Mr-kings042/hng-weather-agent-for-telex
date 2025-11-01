# A2A Weather Agent

Small JSON‑RPC / HTTP agent that returns current weather for a requested city. Designed for Telex/Mastra A2A integration and local testing.

Key features
- JSON‑RPC endpoint for A2A: POST /a2a/weather
- Simple GET and batch endpoints for manual testing
- Caching (Redis if configured, otherwise in‑memory) to reuse recent lookups
- Follow‑up conversation support: stores last city per channel_id so clients can omit the city on follow-ups
- Dependency‑free async retry helper for network calls

Important files / symbols
- Router / endpoints: [`routers.weather_rest_endpoint`](routers.py) — [routers.py](routers.py)  
- Weather service logic: [`services.weather_agent`](services.py) — [services.py](services.py)  
- Cache implementation: [`cache.Cache`](cache.py) and module instance [`cache`](cache.py) — [cache.py](cache.py)  
- Data models: [`models.JSONRPCRequest`](models.py), [`models.JSONRPCResponse`](models.py), [`models.WeatherParams`](models.py), [`models.WeatherResult`](models.py) — [models.py](models.py)  
- App entrypoint / startup: [main.app](main.py) — [main.py](main.py)  
- Tests: [test_a2a.py](test_a2a.py) — [test_a2a.py](test_a2a.py)

Requirements
- Python 3.11+ (your environment shows 3.13)
- Install project dependencies:
```bash
pip install -r requirements.txt
```

Configuration
- Environment variables live in `.env` (examples):
  - REDIS_URL — optional; if unset, the app uses an in‑memory cache. See [.env](.env).
  - WEATHER_URL, GEO_URL — provider endpoints (defaults point to Open‑Meteo).
  - WEATHER_CACHE_TTL — cache TTL in seconds.
- The app loads `.env` in [main.py](main.py).

Run locally
1. (Optional) start Redis if you want persistent caching:
```bash
docker run --name redis -p 6379:6379 -d redis:7
```

2. Start the FastAPI app:
```bash
uvicorn main:app --reload --port 8000
```
(or use your preferred port; `main.py` uses the .env loader on startup).

JSON‑RPC usage examples
- Request weather for Lagos:
```bash
curl -X POST http://127.0.0.1:8000/a2a/weather \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"weather.get","params":{"city":"Lagos"}}'
```

- Store last city for a channel and then follow up without city:
```bash
# initial request (stores last city for channel_id "chan-123")
curl -X POST http://127.0.0.1:8000/a2a/weather \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":10,"method":"weather.get","params":{"city":"Lagos","context":{"channel_id":"chan-123"}}}'

# follow-up (omits city; router reuses cached city for the channel)
curl -X POST http://127.0.0.1:8000/a2a/weather \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":11,"method":"weather.get","params":{"context":{"channel_id":"chan-123"}}}'
```

HTTP convenience endpoints
- Single city (GET):
```bash
curl http://127.0.0.1:8000/weather/Lagos
```
- Batch:
```bash
curl -X POST http://127.0.0.1:8000/weather/batch \
  -H "Content-Type: application/json" \
  -d '["Lagos","London"]'
```

Testing
- Run unit tests:
```bash
pytest -q
```
- Tests exercise the JSON‑RPC handler and can monkeypatch `routers.weather_agent` and `routers.cache` to avoid external calls. See [test_a2a.py](test_a2a.py).

Troubleshooting
- 500 / JSON parse errors: ensure request body is valid JSON and includes JSON‑RPC envelope fields (`jsonrpc`, `id`, `method`).
- Missing feels_like/humidity: the Open‑Meteo response may not include hourly series for that location/time; the fetcher aligns hourly variables to the reported `current_weather.time`. See [`services.fetch_weather_data`](services.py) — [services.py](services.py).
- Redis not connecting: check `REDIS_URL` in [.env](.env) and ensure Docker/Redis is running. See [`cache.Cache.connect`](cache.py) — [cache.py](cache.py).

Extending / Integrating with Telex
- Telex/Mastra should send a JSON‑RPC POST with `params.context.channel_id` so the agent can persist per‑conversation state (last city).
- The primary A2A route is `/a2a` or `/a2a/weather`. The router normalizes method aliases (e.g., `getWeather` → `weather.get`). See [`routers.weather_rest_endpoint`](routers.py) — [routers.py](routers.py).

Developer notes
- The service uses a small dependency‑free `async_retry` helper instead of `tenacity`. Adjust retry settings in [`services.py`](services.py).
- Data models are defined in [`models.py`](models.py) and used to validate inputs and structure outputs.

If you want, I can:
- Add a short example Mastra/Telex workflow JSON for deploying the agent.
- Add a quickstart script that runs Redis (docker) and starts the app.
Which would you like?// filepath: README.md
# A2A Weather Agent

Small JSON‑RPC / HTTP agent that returns current weather for a requested city. Designed for Telex/Mastra A2A integration and local testing.

Key features
- JSON‑RPC endpoint for A2A: POST /a2a/weather
- Simple GET and batch endpoints for manual testing
- Caching (Redis if configured, otherwise in‑memory) to reuse recent lookups
- Follow‑up conversation support: stores last city per channel_id so clients can omit the city on follow-ups
- Dependency‑free async retry helper for network calls

Important files / symbols
- Router / endpoints: [`routers.weather_rest_endpoint`](routers.py) — [routers.py](routers.py)  
- Weather service logic: [`services.weather_agent`](services.py) — [services.py](services.py)  
- Cache implementation: [`cache.Cache`](cache.py) and module instance [`cache`](cache.py) — [cache.py](cache.py)  
- Data models: [`models.JSONRPCRequest`](models.py), [`models.JSONRPCResponse`](models.py), [`models.WeatherParams`](models.py), [`models.WeatherResult`](models.py) — [models.py](models.py)  
- App entrypoint / startup: [main.app](main.py) — [main.py](main.py)  
- Tests: [test_a2a.py](test_a2a.py) — [test_a2a.py](test_a2a.py)

Requirements
- Python 3.11+ (your environment shows 3.13)
- Install project dependencies:
```bash
pip install -r requirements.txt
```

Configuration
- Environment variables live in `.env` (examples):
  - REDIS_URL — optional; if unset, the app uses an in‑memory cache. See [.env](.env).
  - WEATHER_URL, GEO_URL — provider endpoints (defaults point to Open‑Meteo).
  - WEATHER_CACHE_TTL — cache TTL in seconds.
- The app loads `.env` in [main.py](main.py).

Run locally
1. (Optional) start Redis if you want persistent caching:
```bash
docker run --name redis -p 6379:6379 -d redis:7
```

2. Start the FastAPI app:
```bash
uvicorn main:app --reload --port 8000
```
(or use your preferred port; `main.py` uses the .env loader on startup).

JSON‑RPC usage examples
- Request weather for Lagos:
```bash
curl -X POST http://127.0.0.1:8000/a2a/weather \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"weather.get","params":{"city":"Lagos"}}'
```

- Store last city for a channel and then follow up without city:
```bash
# initial request (stores last city for channel_id "chan-123")
curl -X POST http://127.0.0.1:8000/a2a/weather \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":10,"method":"weather.get","params":{"city":"Lagos","context":{"channel_id":"chan-123"}}}'

# follow-up (omits city; router reuses cached city for the channel)
curl -X POST http://127.0.0.1:8000/a2a/weather \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":11,"method":"weather.get","params":{"context":{"channel_id":"chan-123"}}}'
```

HTTP convenience endpoints
- Single city (GET):
```bash
curl http://127.0.0.1:8000/weather/Lagos
```
- Batch:
```bash
curl -X POST http://127.0.0.1:8000/weather/batch \
  -H "Content-Type: application/json" \
  -d '["Lagos","London"]'
```

Testing
- Run unit tests:
```bash
pytest -q
```
- Tests exercise the JSON‑RPC handler and can monkeypatch `routers.weather_agent` and `routers.cache` to avoid external calls. See [test_a2a.py](test_a2a.py).

Troubleshooting
- 500 / JSON parse errors: ensure request body is valid JSON and includes JSON‑RPC envelope fields (`jsonrpc`, `id`, `method`).
- Missing feels_like/humidity: the Open‑Meteo response may not include hourly series for that location/time; the fetcher aligns hourly variables to the reported `current_weather.time`. See [`services.fetch_weather_data`](services.py) — [services.py](services.py).
- Redis not connecting: check `REDIS_URL` in [.env](.env) and ensure Docker/Redis is running. See [`cache.Cache.connect`](cache.py) — [cache.py](cache.py).

Extending / Integrating with Telex
- Telex should send a JSON‑RPC POST with `params.context.channel_id` so the agent can persist per‑conversation state (last city).
- The primary A2A route is `/a2a` or `/a2a/weather`. The router normalizes method aliases (e.g., `getWeather` → `weather.get`). See [`routers.weather_rest_endpoint`](routers.py) — [routers.py](routers.py).

Developer notes
- The service uses a small dependency‑free `async_retry` helper instead of `tenacity`. Adjust retry settings in [`services.py`](services.py).
- Data models are defined in [`models.py`](models.py) and used to validate inputs and structure outputs.
