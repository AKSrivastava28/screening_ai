# AI Voice Screening Agent (Scripted, Groq + Exotel)

A backend-only service that places an outbound screening call to an Indian phone number via Exotel, asks a **fixed, pre-scripted set of 9 questions**, records and transcribes candidate spoken answers, and generates a structured candidate evaluation report at the end of the call.

---

## System Architecture

```
                                    +------------------------------+
                                    |    Recruiter / Admin         |
                                    |   POST /call/trigger         |
                                    +--------------+---------------+
                                                   |
                                                   v
+------------------+     PSTN      +---------------+--------------+   WebSocket (wss://)  +------------------------+
| Candidate Phone  |<=============>|  Exotel Telephony Gateway    |<=====================>| FastAPI Server (/media)|
+------------------+               +------------------------------+                       +-----------+------------+
                                                                                              |       ^
                                                     1. Stream 8kHz question PCM              |       | 2. Candidate PCM
                                                        (audio/q1.wav .. q9.wav)              v       |
                                                                                    +---------+-------+--------+
                                                                                    | Local Silero VAD (8kHz)  |
                                                                                    | Silence & Turn Detection |
                                                                                    +--------------------------+
                                                                                              |
                                                                                  Turn Done   v
                                                                                    +--------------------------+
                                                                                    | Groq Whisper STT         |
                                                                                    | (whisper-large-v3-turbo) |
                                                                                    +--------------------------+
                                                                                              |
                                                                                  Transcript  v
                                                                                    +--------------------------+
                                                                                    | Groq LLaMA 3.3 70B       |
                                                                                    | 5-Criteria Report & Cost |
                                                                                    +--------------------------+
                                                                                              |
                                                                                              v
                                                                                    reports/{call_sid}.json
                                                                                    reports/{call_sid}.md
```

### Key Principles
- **Scripted IVR-Style**: Fixed questions, no improvising, no freeform conversation.
- **Zero Real-time TTS Cost**: Question audio is pre-generated **once** at setup time using Groq Orpheus TTS, resampled to 8kHz mono WAV, and served statically.
- **Local VAD**: Silero VAD runs in-process on raw 8kHz linear PCM frames without API fees.
- **End-of-Call Evaluation**: Groq LLaMA 3.3 70B is invoked exactly once after the call concludes.
- **Cost & Safety Guardrails**: Server-enforced per-answer timeout (`MAX_ANSWER_SECONDS`), silence timeout (`MAX_SILENCE_SECONDS`), and whole-call safety net (`TOTAL_CALL_TIMEOUT_SECONDS`).

---

## Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Telephony** | Exotel Voicebot Applet / AgentStream | Outbound PSTN calling & bidirectional WebSocket audio streaming |
| **LLM** | Groq `llama-3.3-70b-versatile` | Candidate evaluation against 5 criteria at end of call |
| **STT** | Groq `whisper-large-v3-turbo` | Speech-to-text per candidate answer |
| **TTS** | Groq Orpheus (`canopylabs/orpheus-v1-english`) | Setup-time audio generation (voice: `hannah`), resampled to 8kHz |
| **VAD** | Silero VAD (v4/v5 local PyTorch) | In-process silence and end-of-turn detection (no API cost) |
| **Backend** | Python 3.12, FastAPI, Uvicorn, WebSockets | Webhook & WebSocket streaming service |
| **Deployment**| Render (or Docker container) | Single web service hosting HTTPS + WSS |

---

## Directory Structure

```
Screening_AI/
├── .env.example                  # Environment configuration template
├── questions.json                # Fixed 9-question screening script
├── generate_question_audio.py    # Setup script to pre-render static 8kHz audio
├── generate_report.py            # Standalone CLI evaluation report generator
├── render.yaml                   # Render service configuration
├── Dockerfile                    # Container definition
├── requirements.txt              # Pinned Python dependencies
├── app/
│   ├── config.py                 # Configuration and settings loader
│   ├── audio_utils.py            # WAV/PCM manipulation & Exotel 3200-byte chunking
│   ├── vad.py                    # Silero VAD turn detection state machine
│   ├── stt.py                    # Groq Whisper STT client
│   ├── evaluator.py              # Candidate evaluation report & cost calculation
│   ├── telephony.py              # Exotel Call Connect API integration
│   └── server.py                 # FastAPI server & WebSocket /media handler
├── audio/                        # Pre-generated 8kHz WAV files (q1.wav .. q9.wav)
├── reports/                      # Output JSON & Markdown evaluation reports
└── tests/                        # Comprehensive test suite (20 unit & integration tests)
```

---

## Setup & Quickstart

### 1. Prerequisites
- Python 3.12+
- `ffmpeg` installed and available in system `PATH` (for audio resampling)
- An Exotel account with ExoPhone (virtual number)
- A Groq API key (from [console.groq.com](https://console.groq.com))

### 2. Install Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your credentials:
```ini
# Exotel
EXOTEL_ACCOUNT_SID=your_exotel_account_sid
EXOTEL_API_KEY=your_exotel_api_key
EXOTEL_API_TOKEN=your_exotel_api_token
EXOTEL_SUBDOMAIN=api.exotel.com
EXOTEL_EXOPHONE=080XXXXXXXX
EXOTEL_CALLER_NUMBER=+919876543210
EXOTEL_APP_ID=                 # Optional: Exotel flow ID if using Call Flow

# Groq
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_LLM_MODEL=llama-3.3-70b-versatile
GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_TTS_MODEL=canopylabs/orpheus-v1-english
GROQ_TTS_VOICE=hannah

# Server
PUBLIC_BASE_URL=https://your-domain.ngrok-free.app
PORT=8000

# Call Behavior & Guardrails
MAX_SILENCE_SECONDS=3
MAX_ANSWER_SECONDS=45
TOTAL_CALL_TIMEOUT_SECONDS=480

# Cost Estimation Constants
TELEPHONY_RATE_PER_MINUTE=0.015
GROQ_WHISPER_RATE_PER_MINUTE=0.00111
GROQ_LLAMA_INPUT_PER_1M=0.59
GROQ_LLAMA_OUTPUT_PER_1M=0.79
```

---

## Step 1: Pre-Generate Question Audio

This script loops through `questions.json`, calls Groq Orpheus TTS, resamples output from 48kHz to 8kHz mono WAV (Exotel format), and saves each file to `audio/q{id}.wav`. **This runs once at setup time, never during live calls.**

```bash
# With live Groq credentials:
python generate_question_audio.py

# Or generate synthetic audio for offline testing without API calls:
python generate_question_audio.py --mock --force
```

Verify the files:
```bash
dir audio\*.wav
```

---

## Step 2: Run Server Locally with ngrok

Exotel requires a publicly accessible HTTPS/WSS URL to deliver media streams and webhooks.

### Terminal 1 — Start the FastAPI Server
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```
Verify the health endpoint:
```bash
curl http://localhost:8000/health
```

### Terminal 2 — Start ngrok Tunnel
```bash
ngrok http 8000
```
Copy your forwarding URL (e.g. `https://xyz123.ngrok-free.app`) and update `.env`:
```ini
PUBLIC_BASE_URL=https://xyz123.ngrok-free.app
```

---

## Step 3: Configure Exotel Applet (Dashboard)

You can route calls via **Direct Voicebot Stream** or **Call Flow**:

### Option A: Direct Stream Connection (AgentStream)
No flow configuration needed. The server automatically tells Exotel to stream directly to:
```
wss://xyz123.ngrok-free.app/media
```

### Option B: Flow-Based Connection (Voicebot Applet)
1. In Exotel Dashboard, go to **App Bazaar > Create App**.
2. Add a **VoiceBot Applet**:
   - **WebSocket URL**: `wss://xyz123.ngrok-free.app/media`
   - **Sample Rate**: `8000`
3. Add a **Passthru Applet** immediately after:
   - **URL**: `https://xyz123.ngrok-free.app/call/status`
   - **Method**: `POST`
4. Save and note the `App ID` (e.g., `123456`).
5. Set `EXOTEL_APP_ID=123456` in your `.env`.

---

## Step 4: Trigger the Screening Call

Trigger the call to `EXOTEL_CALLER_NUMBER`:
```bash
curl -X POST http://localhost:8000/call/trigger
```

Or pass a custom destination number and ExoPhone:
```bash
curl -X POST http://localhost:8000/call/trigger \
  -H "Content-Type: application/json" \
  -d '{"to_number": "+919876543210", "caller_id": "08047491899"}'
```

### What Happens Next:
1. Exotel dials the candidate phone number.
2. When the candidate answers, Exotel connects to `wss:///media`.
3. The bot immediately streams `q1.wav` (disclosure).
4. Candidate speaks their answer; Silero VAD detects end-of-turn when silence >= `MAX_SILENCE_SECONDS` (or `MAX_ANSWER_SECONDS` cap).
5. Groq Whisper STT transcribes the answer.
6. The bot streams `q2.wav`, repeating through `q9.wav`.
7. Call terminates; Exotel sends `/call/status` callback.
8. Groq LLaMA 3.3 70B scores the interview and saves:
   - `reports/{call_sid}.json` (machine-readable data & scores)
   - `reports/{call_sid}.md` (human-readable report with cost breakdown)

---

## Step 5: View Evaluation Reports

Reports are stored in the `reports/` folder:
```bash
# View the Markdown summary:
cat reports/DEMO_CALL_001.md
```

### Test Evaluation Without a Live Call:
You can test the evaluation pipeline with the included demo transcript:
```bash
python generate_report.py --demo
```

### Evaluation Criteria (1–5 scale):
1. **Communication Clarity**: Articulation, structure, responsiveness.
2. **Experience Relevance**: Relevance of background to the job requirements.
3. **Availability / Notice Fit**: Timeline to join vs hiring needs.
4. **Compensation Fit**: Current/expected CTC alignment with role budget.
5. **Overall Recommendation**: `PROCEED`, `HOLD`, or `REJECT` with one-line justification.

---

## Cost & Safety Guardrails

- **Silence Cutoff**: `MAX_SILENCE_SECONDS=3` prevents dead air from burning telephony minutes.
- **Answer Hard Cap**: `MAX_ANSWER_SECONDS=45` prevents rambling answers.
- **Call Safety Net**: `TOTAL_CALL_TIMEOUT_SECONDS=480` (8 minutes) automatically terminates silent or hanging calls.
- **Cost Calculation**: Every report logs duration and estimated costs based on configurable rates:
  $$\text{Total Cost} = (\text{call\_min} \times \text{rate}_{\text{telephony}}) + (\text{audio\_min} \times \text{rate}_{\text{whisper}}) + (\text{tokens} \times \text{rate}_{\text{llama}})$$

### Cost Breakdown in Indian Rupees (INR @ ₹85/USD)

All costs are calculated dynamically at the end of every screening call and converted to Indian Rupees:

| Component | Provider / Engine | Unit Rate (USD) | Unit Rate (INR) | Measurement Basis |
| :--- | :--- | :--- | :--- | :--- |
| **Telephony** | Exotel Outbound PSTN | $0.015 / minute | **₹1.275 / min** (~₹1.28) | Total connected call duration |
| **STT (Speech-to-Text)** | Groq Whisper Large v3 Turbo | $0.00111 / minute ($0.067/hr) | **₹0.094 / min** (< 10 paise/min) | Candidate speech audio duration only |
| **LLM Evaluation** | Groq LLaMA 3.3 70B | In: $0.59 / 1M tokens<br>Out: $0.79 / 1M tokens | In: **₹0.050 / 1K tokens**<br>Out: **₹0.067 / 1K tokens** | Exact tokens from Groq API response |
| **TTS (Text-to-Speech)** | Pre-rendered 8kHz mono WAV | $0.000 / call | **₹0.00** | Static assets, zero live API cost |

#### Typical Per-Call Cost (1-Minute Screening Session):

| Item | Usage in Call | Cost (USD) | Cost (INR / Rupees) |
| :--- | :--- | :--- | :--- |
| **Exotel Telephony** | 60 seconds (1.0 min) | $0.01500 | **₹1.28** |
| **Groq Whisper STT** | ~15 seconds speech | $0.00028 | **₹0.02** |
| **Groq LLaMA 3.3 LLM** | ~1,100 prompt + 300 output tokens | $0.00089 | **₹0.08** |
| **TOTAL PER CANDIDATE** | **1 min call + evaluation + PDF** | **~$0.0162** | **~₹1.38 INR** |

> **Note:** The downloadable PDF candidate report automatically outputs all cost estimates in Indian Rupees (`Rs.`).

---

## Deployment to Render

This service is pre-configured for **Render Web Services**:

1. Push your repository to GitHub / GitLab.
2. In [Render Dashboard](https://dashboard.render.com), click **New > Blueprint** and select your repository (it automatically reads `render.yaml`).
3. Set your environment variables in the Render Dashboard (`EXOTEL_ACCOUNT_SID`, `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_EXOPHONE`, `GROQ_API_KEY`).
4. Set `PUBLIC_BASE_URL` to your Render service URL (e.g. `https://screening-ai.onrender.com`).
5. Render deploys the service with HTTPS + WSS support enabled by default.

---

## Running Automated Tests

Run the complete test suite:
```bash
pytest -v tests/
```

Test coverage includes:
- **`test_audio_utils.py`**: RIFF headers, PCM extraction, and Exotel 3,200-byte chunking & 320-byte alignment.
- **`test_vad.py`**: Real Silero VAD neural network inference and turn detector state machine transitions.
- **`test_evaluator.py`**: Cost calculation, Markdown formatting, and JSON evaluation schema.
- **`test_telephony.py`**: Exotel API request construction and error handling.
- **`test_server.py`**: FastAPI endpoints (`/health`, `/call/trigger`, `/call/status`) and full WebSocket `/media` streaming session simulation.
