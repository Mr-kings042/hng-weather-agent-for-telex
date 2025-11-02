from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from routers import router
from cache import cache
from logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	# startup
	await cache.connect()
	try:
		yield
	finally:
		# shutdown
		await cache.close()


app = FastAPI(title="A2A Weather Agent", lifespan=lifespan)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} {request.url}")
    try:
        raw = await request.body()
        logger.debug(f"Request body: {raw!r}")
    except Exception:
        logger.debug("Failed to read request body for logging")
    response = await call_next(request)
    logger.info(f"Response {response.status_code} for {request.url}")
    return response
app.include_router(router,tags=["A2A Weather Agent"])
@app.get("/")
async def root():
    return {"status": "success", "message": "A2A Weather Agent is running."}
