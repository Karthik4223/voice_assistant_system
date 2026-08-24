import re
import logging
from typing import List

logger = logging.getLogger("SentenceDemuxer")

class IndicSentenceDemuxer:
    """
    Streaming Sentence & Clause Demuxer for multilingual AI voice assistant.
    Aggregates incoming LLM tokens and yields complete, natural speech sentences (English + Indic delimiters).
    """
    # Sentence termination delimiters (Purna Viram '।', period, question mark, exclamation, newline)
    SENTENCE_DELIMITERS = re.compile(r'([।?!\n])')

    def __init__(self, min_chars_per_clause: int = 80, **kwargs):
        self.buffer = ""
        self.min_chars_per_clause = min_chars_per_clause

    def push_token(self, token: str) -> List[str]:
        """
        Synchronously push a token and return complete, natural speech sentences.
        """
        self.buffer += token
        clauses = []

        # Check for strong sentence end boundaries (. ! ? । \n)
        matches = self.SENTENCE_DELIMITERS.split(self.buffer)
        if len(matches) > 1:
            clause = "".join(matches[:-1]).strip()
            self.buffer = matches[-1]
            if clause and len(clause) > 1:
                clauses.append(clause)
            return clauses

        # If buffer is long (>80 chars), check for natural clause boundary (comma/semicolon followed by space)
        if len(self.buffer) >= self.min_chars_per_clause:
            last_comma = self.buffer.rfind(', ')
            if last_comma > 40:
                clause = self.buffer[:last_comma + 1].strip()
                self.buffer = self.buffer[last_comma + 2:]
                if clause:
                    clauses.append(clause)

        return clauses

    def flush(self) -> str:
        """
        Flush remaining buffer at the end of token generation stream.
        """
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining
