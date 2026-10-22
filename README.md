# Meeting Q&A: Speech-to-Text RAG API

Record a meeting, transcribe it with Whisper, and ask questions about it. The app retrieves the relevant parts of the transcript and uses a local LLM to answer, served through a FastAPI endpoint.

## Pipeline
Audio (WAV) → transcript (Whisper) → chunks (LangChain) → embeddings (Ollama, embeddinggemma) → vector store (Chroma) → answer (Ollama, Gemma) → API (FastAPI + Pydantic)

## Tools
- Python
- OpenAI Whisper (speech-to-text)
- LangChain (loading, splitting, retrieval chain)
- Chroma (vector database)
- Ollama with embeddinggemma and gemma:7b (runs locally, no API key)
- FastAPI, Pydantic, Uvicorn (API)

## How to run
1. Install Ollama, then run `ollama pull embeddinggemma` and `ollama pull gemma:7b`
2. Install ffmpeg (required by Whisper)
3. Create a virtual environment and install packages: `pip install -r requirements.txt`
4. Put your WAV recordings in `audio/` and create an empty `transcripts/` folder
5. Run `speech_rag.ipynb` from top to bottom to create the transcripts
6. Start the API: `uvicorn app:app --reload`
7. Open http://127.0.0.1:8000/docs and send a question to `POST /ask`

## Example
**Question:** Who owns the frontend work, and when is the login screen due?

**Answer:** The frontend work is owned by Sahara, and the login screen is due by March 12th

## Project structure
- `app.py`: the FastAPI app
- `speech_rag.ipynb`: step-by-step notebook version
