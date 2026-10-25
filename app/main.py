from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.agentcore import router as agentcore_router
from app.api.auth import router as auth_router

app = FastAPI(
    title="LLM Agent Orchestrator",
    description="A Gemini tool-calling agent API with semantic memory and background execution.",
    version="0.1.0",
)
app.include_router(auth_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")
app.include_router(agentcore_router, prefix="/api/v1")


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
