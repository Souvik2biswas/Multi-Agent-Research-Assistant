import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Import the LangGraph app we built
from backend.agent import app as wf_app

app = FastAPI(title="Multi-Agent Research API")

# Allow CORS for our Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.post("/api/research")
async def start_research(req: ResearchRequest):
    # This acts as a streaming endpoint using EventSourceResponse
    
    async def event_generator():
        initial_state = {
            "topic": req.topic,
            "raw_data": "",
            "draft_summary": "",
            "final_report": "",
            "revision_feedback": "",
            "iteration_count": 0
        }
        
        try:
            # We use astream to stream updates from each node
            async for chunk in wf_app.astream(initial_state, stream_mode="updates"):
                # chunk is a dict like {"node_name": {"key": "value"}}
                for node, state_update in chunk.items():
                    # We send a standard SSE payload JSON string
                    payload = {
                        "node": node,
                        "iteration": state_update.get("iteration_count", 0),
                        "feedback": state_update.get("revision_feedback", "")
                    }
                    
                    if "final_report" in state_update and state_update["final_report"]:
                        payload["status"] = "completed"
                        payload["report"] = state_update["final_report"]
                    elif "draft_summary" in state_update and state_update["draft_summary"]:
                        payload["status"] = "drafting"
                        payload["draft"] = state_update["draft_summary"]
                    elif "raw_data" in state_update and state_update["raw_data"]:
                        payload["status"] = "researching"
                        
                    yield json.dumps(payload)
                    
        except Exception as e:
            yield json.dumps({"error": str(e)})

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
