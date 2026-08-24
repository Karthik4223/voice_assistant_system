import asyncio
import logging
from enum import Enum, auto
from typing import Dict, List, Callable, Awaitable, Any

logger = logging.getLogger("EventBus")

class EventType(Enum):
    ASR_PARTIAL = auto()
    ASR_FINAL = auto()
    LLM_START = auto()
    LLM_TOKEN = auto()
    LLM_SENTENCE = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    TTS_START = auto()
    TTS_CHUNK = auto()
    PLAY_AUDIO = auto()
    CLIENT_INTERRUPT = auto()

class SystemEvent:
    def __init__(self, event_type: EventType, session_id: str, payload: Any = None):
        self.event_type = event_type
        self.session_id = session_id
        self.payload = payload

EventHandler = Callable[[SystemEvent], Awaitable[None]]

class EventBus:
    """
    High-Performance In-Memory Async Event Bus for real-time streaming pipelines.
    Supports session-scoped event isolation and zero-latency interrupt handling.
    """
    def __init__(self):
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._session_cancel_tokens: Dict[str, asyncio.Event] = {}

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def publish(self, event: SystemEvent) -> None:
        # Check for interrupt signal
        if event.session_id in self._session_cancel_tokens:
            if self._session_cancel_tokens[event.session_id].is_set() and event.event_type != EventType.CLIENT_INTERRUPT:
                # Drop non-interrupt events if session is currently interrupted
                return

        if event.event_type == EventType.CLIENT_INTERRUPT:
            self.set_interrupt(event.session_id)

        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                asyncio.create_task(handler(event))
            except Exception as e:
                logger.error(f"Error executing event handler for {event.event_type.name}: {e}")

    def register_session(self, session_id: str) -> None:
        self._session_cancel_tokens[session_id] = asyncio.Event()

    def unregister_session(self, session_id: str) -> None:
        self._session_cancel_tokens.pop(session_id, None)

    def set_interrupt(self, session_id: str) -> None:
        if session_id in self._session_cancel_tokens:
            self._session_cancel_tokens[session_id].set()

    def clear_interrupt(self, session_id: str) -> None:
        if session_id in self._session_cancel_tokens:
            self._session_cancel_tokens[session_id].clear()

    def is_interrupted(self, session_id: str) -> bool:
        if session_id in self._session_cancel_tokens:
            return self._session_cancel_tokens[session_id].is_set()
        return False
