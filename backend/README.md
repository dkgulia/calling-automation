# Roister Cold-Call Simulation — Backend

FastAPI backend for the conversational cold-call simulation.

## Quick Start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path                           | Description                     |
|--------|--------------------------------|---------------------------------|
| GET    | `/health`                      | Health check                    |
| POST   | `/api/v1/run`                  | Start a simulation              |
| GET    | `/api/v1/outcome/{session_id}` | Poll for outcome                |

## Architecture

```
app/
  domain/       Pure business logic (state, scoring, decisions)
  usecases/     Application workflows (run_simulation, end_call)
  infra/        External adapters (session store, Pipecat, providers)
  schemas/      Pydantic request/response models
  api/          HTTP layer (FastAPI routes)
  core/         Config and logging
  utils/        Helpers (ID generation, time)
```
