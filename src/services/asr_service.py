import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger("ASRService")

class StreamingASRService:
    """
    Streaming ASR Service managing acoustic decoding state & voice activity detection (VAD).
    Processes 80ms Float32 PCM audio strides and emits smooth partial transcripts.
    """
    def __init__(self, sample_rate: int = 16000, stride_ms: int = 80):
        self.sample_rate = sample_rate
        self.stride_samples = int(sample_rate * (stride_ms / 1000.0))
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def init_session(self, session_id: str, target_language: str = "te"):
        """Initialize streaming state for a new audio session."""
        self.active_sessions[session_id] = {
            "language": target_language,
            "buffer": np.array([], dtype=np.float32),
            "transcript_tokens": [],
            "frame_count": 0,
            "speech_detected": False,
            "last_transcript": ""
        }

    def update_transcript(self, session_id: str, text: str):
        """Update active session with live transcribed text."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_transcript"] = text

    def process_audio_chunk(self, session_id: str, pcm_chunk: np.ndarray) -> Optional[str]:
        """
        Process an 80ms PCM chunk (float32, normalized [-1.0, 1.0]).
        Returns smooth partial status or transcript update.
        """
        if session_id not in self.active_sessions:
            self.init_session(session_id)

        session = self.active_sessions[session_id]
        session["buffer"] = np.append(session["buffer"], pcm_chunk)
        session["frame_count"] += 1

        # Calculate RMS energy to detect voice activity
        if len(pcm_chunk) > 0:
            energy = float(np.sqrt(np.mean(pcm_chunk ** 2)))
            if energy > 0.01:
                session["speech_detected"] = True

        if len(session["buffer"]) >= self.stride_samples:
            max_window = self.sample_rate * 2  # 2 sec window
            if len(session["buffer"]) > max_window:
                session["buffer"] = session["buffer"][-max_window:]

            # Return actual live transcript if available
            if session["last_transcript"]:
                return session["last_transcript"]

            # Otherwise return clean voice active status periodically
            if session["speech_detected"] and session["frame_count"] % 10 == 0:
                return "Listening... (Voice Active)"

        return None

    def finalize_session(self, session_id: str) -> str:
        """Finalize session state and return final ASR transcript."""
        session = self.active_sessions.pop(session_id, None)
        if session:
            return session.get("last_transcript", "").strip()
        return ""
