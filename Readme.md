# 🎙️ Audio Meeting Summariser

> A lightweight backend service that transcribes and summarises your meeting recordings using extractive NLP — no heavy models, no cloud AI, runs entirely on your machine.

---

## What It Does

It takes an audio recording of any meeting — Zoom, Google Meet, Teams, or any other platform — captured via a Chrome extension, and returns a clean, concise summary of what was discussed.

No fluff. No hallucinations. Just the key points extracted directly from what was actually said.

---

## How It Works

```
Chrome Extension
      │
      │  audio file (mp3/wav)
      ▼
FastAPI Backend
      │
      ├── 1. Transcribe audio → faster-whisper / mlx-whisper
      │
      ├── 2. Clean transcript → remove fillers, repeated words, noise
      │
      ├── 3. Extractive NLP summarisation
      │         ├── TF-IDF scoring
      │         ├── NER boosting
      │         └── Noun chunk density
      │
      └── 4. Return summary as JSON
```

---

## API Endpoints

### `GET /`
Health check. Returns `{"hello": 1}` if the server is running.

---

### `POST /upload-audio`
Main endpoint. Accepts an audio file, transcribes it, and returns a summary.

**Request:**
- `file` — audio file (mp3, wav, m4a), max 100MB

**Response:**
```json
{
  "filename": "meeting.mp3",
  "filesize": 34803389,
  "Summary": "The engineering key review at GitLab discussed..."
}
```

**Errors:**
```json
{ "error": "File too large" }
```
---

## Summarisation Approach

The summariser is fully extractive — it picks real sentences from the transcript rather than generating new ones. This means:

- No hallucinations — every sentence in the summary was actually said
- No large language model required
- Works on any domain or topic without fine-tuning

### Scoring pipeline
Each sentence is scored using a combination of signals:

| Signal | What it does |
|---|---|
| TF-IDF | Rewards sentences with words that are frequent in this document but rare across sentences |
| NER boost | Gives extra weight to sentences containing named entities like organisations, people, and products |
| Noun chunk density | Rewards sentences packed with meaningful noun phrases over sentences heavy with pronouns and filler |

### Quality filters
Before scoring, sentences are filtered by:
- Minimum length — removes fragments
- Punctuation ratio — removes garbled speech
- Short word ratio — removes low information sentences

---

## Configuration

All tunable parameters are defined as constants at the top of `processing.py`:

| Constant | Default | What it controls |
|---|---|---|
| `ACCEPTABLE_SENTENCE_LEN` | `10` | Minimum words for a sentence to be considered |
| `SUMMARY_LEN` | `0.2` | Fraction of sentences to include in summary |
| `NER_THRESHOLD` | `1.9` | Score multiplier for named entity tokens |
| `NOUN_CHUNK_WEIGHT` | `0.45` | Weight of noun chunk density in final score |

---

## Project Structure

```
├── server.py          — FastAPI app, endpoints, transcription
├── processing.py      — NLP pipeline, summarisation functions
├── contents/          — Temporary storage for uploaded files
├── summary.txt        — Input file for local testing
├── Trans.txt          — Output file for local testing
└── requirements.txt   — Dependencies
```

---

## Dependencies

- `fastapi` — web framework
- `faster-whisper` / `mlx-whisper` — audio transcription
- `spacy` + `en_core_web_md` — NLP processing
- `uvicorn` — ASGI server

---

## Local Development

```bash
# install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

# run server
uvicorn server:app --reload
```

---

## Future Updates

### MMR — Maximal Marginal Relevance
Currently the summariser picks the top N highest scoring sentences which can result in redundant sentences covering the same topic. MMR would iteratively select sentences that are both relevant and different from already selected ones — guaranteeing better topic diversity across the summary.

### Audio Streaming
Currently the Chrome extension sends the complete audio file after the meeting ends. A future version would stream audio chunks to the backend in real time as the meeting progresses — allowing the summary to be built incrementally and delivered immediately when the meeting ends rather than waiting for transcription of the full recording.

---
**NOTE:**
> This README was generated using AI.
