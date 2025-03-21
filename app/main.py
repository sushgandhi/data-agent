from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

from .api.endpoints import data, query
from .core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for processing and analyzing data through natural language queries",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix=f"{settings.API_V1_STR}/data", tags=["data"])
app.include_router(query.router, prefix=f"{settings.API_V1_STR}/query", tags=["query"])

@app.middleware("http")
async def log_requests(request, call_next):
    logger.debug(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 