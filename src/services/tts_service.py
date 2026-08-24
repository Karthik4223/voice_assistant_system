import asyncio
import io
import logging
import re
from typing import AsyncGenerator, Optional, Tuple
from gtts import gTTS

logger = logging.getLogger("TTSService")

def clean_text_for_tts(text: str) -> str:
    """
    Strips Markdown formatting, emojis, and symbols so TTS synthesizes clean spoken text.
    """
    if not text:
        return ""
    # Remove Markdown bold, italic, headers, bullet points, code ticks
    text = re.sub(r'[\*\#\`\_\~\-\>]', ' ', text)
    # Remove emojis and miscellaneous symbols
    text = re.sub(r'[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\u2000-\u206F]', '', text)
    # Normalize extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text: str) -> Tuple[str, Optional[str]]:
    """
    Detects script/language for TTS to ensure perfect regional pronunciation.
    Supports Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, and Indian English.
    """
    if not text:
        return 'en', 'co.in'

    for char in text:
        code = ord(char)
        # Telugu: U+0C00 to U+0C7F
        if 0x0C00 <= code <= 0x0C7F:
            return 'te', None
        # Devanagari (Hindi / Marathi): U+0900 to U+097F
        if 0x0900 <= code <= 0x097F:
            return 'hi', None
        # Tamil: U+0B80 to U+0BFF
        if 0x0B80 <= code <= 0x0BFF:
            return 'ta', None
        # Kannada: U+0C80 to U+0CFF
        if 0x0C80 <= code <= 0x0CFF:
            return 'kn', None
        # Malayalam: U+0D00 to U+0D7F
        if 0x0D00 <= code <= 0x0D7F:
            return 'ml', None
        # Bengali: U+0980 to U+09FF
        if 0x0980 <= code <= 0x09FF:
            return 'bn', None
        # Gujarati: U+0A80 to U+0AFF
        if 0x0A80 <= code <= 0x0AFF:
            return 'gu', None
        # Gurmukhi (Punjabi): U+0A00 to U+0A7F
        if 0x0A00 <= code <= 0x0A7F:
            return 'pa', None

    # Default to Indian English voice tuning ('co.in') for English text
    return 'en', 'co.in'

def _generate_gtts_bytes(text: str, lang: str, tld: Optional[str]) -> bytes:
    """
    Synchronous helper to run gTTS network synthesis in thread pool.
    """
    fp = io.BytesIO()
    if tld:
        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
    else:
        tts = gTTS(text=text, lang=lang, slow=False)
    tts.write_to_fp(fp)
    return fp.getvalue()

class StreamingTTSService:
    """
    Streaming TTS Service powered by gTTS & Indic-Parler-TTS.
    Synthesizes natural human voice audio with non-blocking thread-pool pre-buffering.
    """
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.default_description = "A clear female voice from South India with natural pace and clear articulation."

    async def synthesize_clause_stream(
        self,
        text_clause: str,
        voice_description: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesizes a text clause into real human voice audio streaming chunks.
        """
        cleaned_clause = clean_text_for_tts(text_clause)
        if not cleaned_clause:
            logger.info(f"Skipping symbol-only/emoji clause: '{text_clause}'")
            return

        lang_code, tld_code = detect_language(cleaned_clause)
        logger.info(f"Synthesizing speech clause (lang='{lang_code}', tld='{tld_code}'): '{cleaned_clause[:40]}...'")

        try:
            # Run blocking gTTS call in thread pool to prevent event loop blocking
            audio_bytes = await asyncio.to_thread(_generate_gtts_bytes, cleaned_clause, lang_code, tld_code)
            if audio_bytes:
                yield audio_bytes
        except Exception as e:
            logger.error(f"TTS Speech Synthesis Error for '{cleaned_clause}': {e}")
