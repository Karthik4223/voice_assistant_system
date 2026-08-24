import os
import asyncio
import json
import logging
import uuid
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.orchestration.event_bus import EventBus, EventType, SystemEvent
from src.orchestration.sentence_demuxer import IndicSentenceDemuxer
from src.services.llm_service import LLMService
from src.services.asr_service import StreamingASRService
from src.services.tts_service import StreamingTTSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VoiceGateway")

app = FastAPI(title="Multilingual AI Voice Assistant Gateway", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../static"))
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Multilingual AI Voice Assistant Gateway Running"}

# Core Component Singleton Instances
event_bus = EventBus()
llm_service = LLMService(endpoint_url="http://localhost:8000/v1", model_name="Qwen/Qwen3.6-27B")
asr_service = StreamingASRService()
tts_service = StreamingTTSService()

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing API Gateway & Model Manager resources...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API Gateway resources...")
    await llm_service.close()

@app.websocket("/v1/audio/stream")
async def audio_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket session established: {session_id}")
    
    event_bus.register_session(session_id)
    demuxer = IndicSentenceDemuxer(min_chars_per_clause=25)
    conversation_history = []
    
    active_tts_task: asyncio.Task = None

    try:
        while True:
            # Receive binary audio PCM frame or JSON message from client
            try:
                message = await websocket.receive()
            except (RuntimeError, WebSocketDisconnect):
                logger.info(f"WebSocket session disconnected: {session_id}")
                break
            
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                pcm_data = np.frombuffer(raw_bytes, dtype=np.float32)
                
                # Check for client speech during active TTS output (Barge-in Interrupt)
                if event_bus.is_interrupted(session_id) or (active_tts_task and not active_tts_task.done()):
                    # User interrupted - cancel ongoing speech generation immediately
                    if active_tts_task and not active_tts_task.done():
                        logger.info(f"[{session_id}] Client Barge-in detected! Cancelling active TTS Task.")
                        active_tts_task.cancel()
                    
                    await event_bus.publish(SystemEvent(EventType.CLIENT_INTERRUPT, session_id))
                    event_bus.clear_interrupt(session_id)
                    await websocket.send_text(json.dumps({"type": "INTERRUPT_ACK"}))

                # Process streaming audio chunk through ASR
                partial_text = asr_service.process_audio_chunk(session_id, pcm_data)
                if partial_text:
                    await websocket.send_text(json.dumps({"type": "ASR_PARTIAL", "text": partial_text}))

            elif "text" in message and message["text"]:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")

                if msg_type == "INTERRUPT":
                    logger.info(f"[{session_id}] Explicit INTERRUPT requested by client.")
                    if active_tts_task and not active_tts_task.done():
                        active_tts_task.cancel()
                    event_bus.set_interrupt(session_id)
                    await websocket.send_text(json.dumps({"type": "INTERRUPT_ACK"}))

                elif msg_type == "SPEECH_END" or msg_type == "TEXT_PROMPT":
                    text_input = payload.get("text")
                    if text_input:
                        asr_service.update_transcript(session_id, text_input)
                    user_prompt = text_input or asr_service.finalize_session(session_id) or "Hello"
                    logger.info(f"[{session_id}] Final User Input: '{user_prompt}'")
                    
                    conversation_history.append({"role": "user", "content": user_prompt})
                    await websocket.send_text(json.dumps({"type": "ASR_FINAL", "text": user_prompt}))

                    # Trigger streaming pipeline (LLM -> Sentence Demuxer -> TTS -> Client)
                    async def run_pipeline():
                        full_response_text = ""
                        system_prompt = (
                            "You are a helpful, fluent multilingual AI voice assistant. "
                            "Automatically detect the language of the user's input (English, Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi, etc.) "
                            "and reply fluently in that exact same language. Keep your responses natural, conversational, and concise for spoken voice interaction."
                        )
                        
                        token_stream = llm_service.stream_chat(
                            messages=conversation_history,
                            system_prompt=system_prompt
                        )

                        await websocket.send_text(json.dumps({"type": "LLM_START"}))

                        async for token in token_stream:
                            if event_bus.is_interrupted(session_id):
                                logger.info(f"[{session_id}] Pipeline interrupted during LLM generation.")
                                break

                            full_response_text += token
                            await websocket.send_text(json.dumps({"type": "LLM_TOKEN", "token": token}))

                            # Demux tokens into clauses for immediate sentence TTS synthesis
                            clauses = demuxer.push_token(token)
                            for clause in clauses:
                                if event_bus.is_interrupted(session_id):
                                    break
                                logger.info(f"[{session_id}] Demuxed Clause for TTS: '{clause}'")
                                await websocket.send_text(json.dumps({"type": "LLM_SENTENCE", "clause": clause}))
                                
                                # Synthesize and stream audio frames
                                async for audio_chunk in tts_service.synthesize_clause_stream(clause):
                                    if event_bus.is_interrupted(session_id):
                                        break
                                    await websocket.send_bytes(audio_chunk)

                        # Flush any remaining buffer in demuxer
                        final_clause = demuxer.flush()
                        if final_clause and not event_bus.is_interrupted(session_id):
                            await websocket.send_text(json.dumps({"type": "LLM_SENTENCE", "clause": final_clause}))
                            async for audio_chunk in tts_service.synthesize_clause_stream(final_clause):
                                await websocket.send_bytes(audio_chunk)

                        if full_response_text:
                            conversation_history.append({"role": "assistant", "content": full_response_text})
                            await websocket.send_text(json.dumps({"type": "RESPONSE_DONE"}))

                    active_tts_task = asyncio.create_task(run_pipeline())

    except WebSocketDisconnect:
        logger.info(f"WebSocket session disconnected: {session_id}")
    finally:
        event_bus.unregister_session(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
