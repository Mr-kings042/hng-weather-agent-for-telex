from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import router
from cache import cache


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
app.include_router(router,tags=["A2A Weather Agent"])
@app.get("/")
async def root():
    return {"status": "success", "message": "A2A Weather Agent is running."}
