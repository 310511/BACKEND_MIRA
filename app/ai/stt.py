"""
AI-Powered Speech-to-Text Module
Uses OpenAI Whisper for multilingual transcription (Hindi/English)
"""
import whisper
import tempfile
import os
from typing import Tuple

class WhisperSTT:
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model
        model_size: tiny, base, small, medium, large
        """
        self.model = whisper.load_model(model_size)
    
    def transcribe(self, audio_file_path: str, language: str = None) -> Tuple[str, str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (en, hi) or None for auto-detect
        
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        result = self.model.transcribe(
            audio_file_path,
            language=language,
            task="transcribe"
        )
        
        return result["text"], result.get("language", language or "unknown")

# Singleton instance
_whisper_instance = None

def get_whisper_stt() -> WhisperSTT:
    """Get or create Whisper STT instance"""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperSTT(model_size="base")
    return _whisper_instance
