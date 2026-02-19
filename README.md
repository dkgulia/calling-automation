# Roister Cold-Call Simulation

A real-time conversational cold-call simulation with live voice interaction. An AI sales agent qualifies prospects using BANT+ methodology through natural voice conversations, producing structured outcome reports with scoring and next-action recommendations.

## Features

- **Human Voice Mode** — Speak to the AI agent using your microphone via real-time WebSocket audio streaming
- **AI Prospect Mode** — Fully automated simulation where an LLM plays the prospect role
- **Text Chat Mode** — Type-based interaction for testing and development
- **BANT+ Qualification** — Tracks Budget, Authority, Need, Timeline, and Pain across conversation turns
- **Deterministic Decision Engine** — Rule-based action selection (ASK_SLOT, HANDLE_OBJECTION, CLOSE, END) ensures explainable behavior
- **LLM-Powered Wording** — DeepSeek R1 generates natural conversational responses with template fallback
- **Live Transcript (optional/debug)** — Real-time text display of both agent and prospect speech during voice calls
- **Opportunity Scoring** — 0-10 scoring with Weak/Medium/Strong labels and recommended next actions

## Architecture

Clean architecture with three layers — domain logic is pure and testable, with no I/O or framework dependencies.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│   Voice Mode: Pipecat protobuf frames (PCM16 @ 24kHz) over WS  │
│   Text Mode:  REST API calls                                    │
│   AI Mode:    REST API calls (auto-play)                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    WebSocket /ws?session_id=<id>
                    REST      /api/v1/*
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      Backend (FastAPI)                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Pipecat Voice Pipeline                  │   │
│  │  Deepgram STT → BrainProcessor → Cartesia TTS            │   │
│  │       ↓              ↓                ↓                   │   │
│  │  TranscriptSender  process_input  TranscriptSender        │   │
│  │  (user text)       (domain)       (agent text)            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │   Domain     │  │  Use Cases   │  │   Infrastructure   │     │
│  │  state.py    │  │ process_input│  │  deepseek_r1.py    │     │
│  │  decision_   │  │ end_call     │  │  session_store.py  │     │
│  │   engine.py  │  │ run_sim      │  │  pipeline.py       │     │
│  │  qualifi-    │  │ generate_    │  │  brain.py          │     │
│  │   cation.py  │  │  prospect    │  │  events.py         │     │
│  │  outcome.py  │  │              │  │                    │     │
│  └─────────────┘  └──────────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid LLM Architecture

The LLM is used for signal extraction and response wording only. The decision engine remains fully deterministic.

```
User text
    │
    ▼
┌────────────────────────┐       ┌──────────────────────┐
│ LLM Extraction         │─fail─▶│ Rule-based Extractor │
│ (DeepSeek R1)          │       │ (keyword heuristics)  │
└───────────┬────────────┘       └──────────┬───────────┘
            │                               │
            ▼                               ▼
    ExtractedSignals ◄──────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Deterministic Decision Engine                 │
│ state + signals → Action                      │
│ (ASK_SLOT / HANDLE_OBJECTION / CLOSE / END)  │
└───────────┬──────────────────────────────────┘
            │
            ▼
┌────────────────────────┐       ┌──────────────────────┐
│ LLM Wording            │─fail─▶│ Template Replies     │
│ (DeepSeek R1)          │       │ (canned responses)    │
└───────────┬────────────┘       └──────────┬───────────┘
            │                               │
            ▼                               ▼
       agent_text ◄─────────────────────────┘
```

## Pipecat Under the Hood (Voice Mode)

This project uses [Pipecat](https://github.com/pipecat-ai/pipecat) as a real-time streaming pipeline. Pipecat passes protobuf-serialized frames over a WebSocket transport; the audio payload is PCM 16-bit @ 24kHz.

**1) WebSocket transport**
- The browser connects to `WS /ws?session_id=<id>`.
- Audio is sent as Pipecat frames (protobuf), not as ad-hoc JSON.

**2) Streaming STT (Deepgram)**
- Pipecat streams incoming audio frames to Deepgram STT.
- Deepgram emits partial and final transcription events.
- The app only triggers business logic on **finalized** transcriptions to avoid responding to partial speech.

**3) BrainProcessor bridges STT → business logic**
- When a finalized transcript arrives, `BrainProcessor` calls `process_input(session_id, user_text)`.
- `process_input`:
  - extracts signals (DeepSeek R1, or rule-based fallback)
  - updates ProspectState
  - recomputes score and chooses the next Action deterministically
  - generates agent text (DeepSeek R1, or template fallback)
  - appends a trace turn for explainability
  - ends the session if Action is END/CLOSE

**4) Streaming TTS (Cartesia)**
- The agent text is pushed downstream to Cartesia TTS.
- Cartesia returns audio frames which Pipecat streams back to the browser over the same WebSocket connection.

**5) Session lifecycle**
- `POST /api/v1/run` creates the session and returns a `session_id`.
- The voice WebSocket uses that `session_id` for stateful turns.
- When the call ends, the frontend fetches `GET /api/v1/outcome/{session_id}` to render the final report.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11+, FastAPI, Pydantic |
| Voice Pipeline | Pipecat (WebSocket transport) |
| STT | Deepgram (real-time streaming) |
| TTS | Cartesia (low-latency voice synthesis) |
| LLM | DeepSeek R1 (extraction + wording) |
| Frontend | React 18, TypeScript, Vite |
| Audio | Pipecat protobuf frames carrying PCM16 @ 24kHz over WebSocket |

## Project Structure

```
roister-coldcall/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + WebSocket endpoint
│   │   ├── api/v1/routes.py     # REST API routes
│   │   ├── domain/              # Pure business logic (no I/O)
│   │   │   ├── state.py         # ProspectState dataclass
│   │   │   ├── decision_engine.py  # Deterministic action selection
│   │   │   ├── qualification.py # BANT+ scoring (0-10)
│   │   │   └── outcome.py       # Outcome generation
│   │   ├── usecases/            # Application workflows
│   │   │   ├── process_input.py # Main turn orchestration
│   │   │   ├── end_call.py      # Call finalization + scoring
│   │   │   ├── run_simulation.py    # Session creation
│   │   │   └── generate_prospect_turn.py  # AI prospect mode
│   │   ├── infra/               # External adapters
│   │   │   ├── pipecat/
│   │   │   │   ├── pipeline.py  # Voice pipeline factory
│   │   │   │   ├── brain.py     # STT → Domain → TTS bridge
│   │   │   │   └── events.py    # Pipeline event handlers
│   │   │   ├── providers/
│   │   │   │   └── deepseek_r1.py  # LLM extraction + wording
│   │   │   └── session_store.py # In-memory session storage
│   │   ├── schemas/             # Pydantic request/response models
│   │   ├── evals/               # Evaluation framework
│   │   │   ├── scenarios.py     # Scripted test scenarios
│   │   │   └── run_evals.py     # Eval runner with assertions
│   │   └── core/                # Config & logging
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── pages/Home.tsx       # Main UI (voice, text, AI modes)
│   │   ├── components/OutcomeView.tsx  # Outcome display
│   │   ├── api/client.ts        # API client functions
│   │   └── types/outcome.ts     # TypeScript types
│   ├── vite.config.ts           # Vite config with API proxy
│   └── package.json
├── README.md
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- API keys for Deepgram, Cartesia, and DeepSeek

### Environment Setup

```bash
cd backend
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

```
DEEPGRAM_API_KEY=...      # Deepgram STT
CARTESIA_API_KEY=...      # Cartesia TTS
CARTESIA_VOICE_ID=...     # Cartesia voice ID
DEEPSEEK_API_KEY=...      # DeepSeek R1 LLM
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat          # or deepseek-reasoner
```

### Running the Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend runs at http://localhost:8000. Swagger docs at http://localhost:8000/docs.

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:5173. Vite proxies `/api` and `/ws` requests to the backend.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/run` | Start a new session. Body: `{ "prospect_mode": "human" \| "ai" }` |
| `POST` | `/api/v1/input/{session_id}` | Send user text in text/AI mode. Body: `{ "user_text": "..." }` |
| `POST` | `/api/v1/prospect/{session_id}` | Generate AI prospect turn (AI mode only) |
| `GET`  | `/api/v1/outcome/{session_id}` | Get call outcome and scoring |
| `WS`   | `/ws?session_id=<id>` | Real-time voice pipeline (Pipecat protobuf frames) |

## Interaction Modes

### Voice Mode (Human)
User speaks into microphone. Audio is carried as Pipecat protobuf frames (PCM16 @ 24kHz) over `WS /ws?session_id=<id>` to the Pipecat pipeline (Deepgram STT → BrainProcessor → Cartesia TTS). Agent audio streams back over the same connection. An optional live transcript is shown in the UI for debugging.

### Text Chat Mode
User types messages. REST API processes each turn through the same domain logic (extraction → decision engine → wording). No voice pipeline involved.

### AI Prospect Mode
LLM generates prospect responses based on a persona. Both sides are automated — agent and prospect take turns via REST API. Useful for testing and evaluation.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Clean architecture (3 layers)** | Domain logic is pure and testable without HTTP or I/O concerns |
| **Deterministic decision engine** | Actions are explainable and predictable; LLM only handles extraction + wording |
| **Confidence gating (configurable via `LLM_MIN_CONFIDENCE`, default 0.35)** | Low-confidence LLM extractions fall back to rule-based keyword matching |
| **Question alignment gating** | Slot fills are only accepted for the slot the agent just asked, unless confidence >= 0.85 — prevents "yeah" from filling budget |
| **Safe correction overwrites** | Already-filled slots can be corrected when user says "actually..." or new confidence substantially exceeds previous |
| **Diminishing objection penalties** | Repeating the same objection applies 100%/50%/0% penalty (1st/2nd/3rd+), capped at -2 total |
| **Template fallback for wording** | System works fully without LLM API keys using canned responses |
| **In-memory session store** | Simple for MVP; swap for Redis/Postgres when scaling |
| **Pipecat protobuf over WebSocket** | Browser sends/receives Pipecat protobuf frames directly instead of using the Pipecat client SDK, for finer control over audio buffering |
| **AudioBufferSourceNode playback** | Batched PCM playback with 500ms buffering for smooth audio output |

## Tradeoffs

- **Deterministic engine vs LLM-only agent**
  - Pros: predictable, explainable decisions; traceable scoring; easier to test and regress.
  - Cons: less flexible than a pure LLM conversation; new objection types require adding rules.

- **LLM extraction/wording with fallback**
  - Pros: natural language quality improves, but the system remains functional if the LLM fails (rule-based + templates).
  - Cons: quality depends on STT accuracy; low-confidence gating may miss nuanced info.

- **In-memory session store**
  - Pros: simplest for an assignment/MVP; easy to reason about.
  - Cons: not persistent and not horizontally scalable; in production this would be Redis or Postgres.

## Evaluation System

Deterministic eval framework with scripted scenarios. The runner auto-sets `FORCE_RULE_BASED=1` for reproducible results.

```bash
cd backend
python -m app.evals.run_evals
```

Scenarios cover strong leads, weak leads, objection handling, and edge cases. Each scenario asserts expected label (Weak/Medium/Strong), score range, and filled BANT+ slots.
