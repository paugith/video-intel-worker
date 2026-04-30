import os
import uuid
import threading
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from pipeline import process_video

app = FastAPI()

WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
jobs: dict = {}


class ExtractRequest(BaseModel):
    url: str
    user_id: str = "default"


def auth(key: str):
    if not WORKER_API_KEY or key != WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract")
def extract(req: ExtractRequest, x_api_key: str = Header(...)):
    auth(x_api_key)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "stage": "queued", "progress": 0, "slug": None, "error": None}

    def run():
        def progress(stage: str, pct: int):
            jobs[job_id].update({"status": "running", "stage": stage, "progress": pct})
        try:
            slug = process_video(req.url, req.user_id, progress)
            jobs[job_id].update({"status": "done", "stage": "done", "progress": 100, "slug": slug})
        except Exception as e:
            jobs[job_id].update({"status": "failed", "error": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str = Header(...)):
    auth(x_api_key)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
