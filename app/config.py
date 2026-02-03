"""
AI Service Configuration
Central place to configure which AI providers to use for STT, NLP, and TTS
Change providers here without modifying service code
"""
from enum import Enum
from pydantic_settings import BaseSettings
from typing import Optional

class STTProvider(str, Enum):
    """Speech-to-Text Provider Options"""
    WHISPER_LOCAL = "whisper_local"      # OpenAI Whisper (local model)
    WHISPER_API = "whisper_api"          # OpenAI Whisper (API)
    GOOGLE = "google"                     # Google Cloud Speech-to-Text
    AZURE = "azure"                       # Azure Speech Services
    ASSEMBLYAI = "assemblyai"            # AssemblyAI

class NLPProvider(str, Enum):
    """NLP/Intent Processing Provider Options"""
    OPENAI_GPT4 = "openai_gpt4"          # OpenAI GPT-4
    OPENAI_GPT35 = "openai_gpt35"        # OpenAI GPT-3.5
    GOOGLE_GEMINI = "google_gemini"      # Google Gemini
    ANTHROPIC_CLAUDE = "anthropic_claude" # Anthropic Claude
    GROQ = "groq"                        # Groq (OpenAI-compatible Chat Completions)
    LOCAL_LLM = "local_llm"              # Local LLM (Llama, Mistral)

class TTSProvider(str, Enum):
    """Text-to-Speech Provider Options"""
    SWARA = "swara"                       # Swara TTS (Hugging Face) - Primary for Hindi
    GOOGLE = "google"                     # Google Cloud TTS
    AZURE = "azure"                       # Azure TTS
    ELEVENLABS = "elevenlabs"            # ElevenLabs
    OPENAI = "openai"                     # OpenAI TTS
    PYTTSSX3 = "pyttsx3"                   # pyttsx3 (offline TTS) - Most reliable
    WEB_SPEECH = "web_speech"            # Browser Web Speech API (client-side)

class Settings(BaseSettings):
    """Application Settings"""
    
    # ============================================
    # AI PROVIDER SELECTION - CHANGE MODELS HERE
    # ============================================
    
    # Speech-to-Text Provider (Whisper supports Hindi, Tamil, Telugu, and 90+ languages)
    STT_PROVIDER: STTProvider = STTProvider.ASSEMBLYAI
    
    # NLP/Intent Processing Provider
    NLP_PROVIDER: NLPProvider = NLPProvider.GROQ
    
    # Text-to-Speech Provider (pyttsx3 is more reliable for immediate use)
    TTS_PROVIDER: TTSProvider = TTSProvider.PYTTSSX3
    
    # ============================================
    # API KEYS (Set via environment variables)
    # ============================================
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    
    # Google Cloud
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    
    # Azure
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    
    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None

    # Groq (OpenAI-compatible)
    GROQ_API_KEY: Optional[str] = None
    
    # AssemblyAI
    ASSEMBLYAI_API_KEY: Optional[str] = None
    
    # ElevenLabs
    ELEVENLABS_API_KEY: Optional[str] = None
    
    # ============================================
    # SHOPIFY CONFIGURATION
    # ============================================
    
    SHOPIFY_STORE_DOMAIN: str = "your-store.myshopify.com"
    SHOPIFY_STOREFRONT_ACCESS_TOKEN: Optional[str] = None
    SHOPIFY_ADMIN_ACCESS_TOKEN: Optional[str] = None
    
    # ============================================
    # MODEL-SPECIFIC SETTINGS
    # ============================================
    
    # Whisper Model Size (tiny, base, small, medium, large)
    # Use "tiny" for fastest loading (39MB), "base" for better accuracy (142MB)
    # For Indian languages, "tiny" provides good speed with reasonable accuracy
    WHISPER_MODEL_SIZE: str = "tiny"

    # Optional path to ffmpeg/ffprobe binaries (Windows reliability)
    FFMPEG_BIN: Optional[str] = None
    
    # GPT Model Version
    GPT_MODEL: str = "gpt-4"
    
    # Gemini Model Version
    GEMINI_MODEL: str = "gemini-pro"
    
    # Local LLM Configuration (for offline/low-cost operation)
    LOCAL_LLM_API_URL: Optional[str] = "http://localhost:11434"  # Ollama default
    LOCAL_LLM_MODEL: str = "llama3.2:3b"  # Lightweight model for low-resource environments

    # Groq model configuration
    GROQ_API_BASE_URL: str = "https://api.groq.com/openai/v1"
    # Updated to current supported model
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Swara TTS Configuration
    SWARA_MODEL_NAME: str = "ai4bharat/indic-tts"  # Default Indic TTS model
    SWARA_SPEAKER_ID: int = 0  # Hindi speaker ID
    
    # Offline/Resilience Settings
    ENABLE_OFFLINE_MODE: bool = True  # Prefer local models when available
    FALLBACK_TO_CLOUD: bool = False  # Only use cloud if explicitly enabled
    
    # Language Support (Indian languages first for underserved communities)
    # Whisper supports: hi (Hindi), ta (Tamil), te (Telugu), kn (Kannada), 
    # ml (Malayalam), mr (Marathi), gu (Gujarati), pa (Punjabi), bn (Bengali), en (English)
    SUPPORTED_LANGUAGES: list[str] = ["hi", "ta", "te", "en"]  # Hindi, Tamil, Telugu, English
    DEFAULT_LANGUAGE: str = "hi"  # Hindi as default
    
    # ============================================
    # APPLICATION SETTINGS
    # ============================================
    
    APP_NAME: str = "MIRA Voice Commerce"
    DEBUG: bool = True
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000", 
        "http://localhost:8000",
        "http://localhost:8080",
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8501",  # Streamlit alternative
        "http://127.0.0.1:8002",  # Backend alternative
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Helper function to get current provider info
def get_provider_info() -> dict:
    """Get current AI provider configuration"""
    return {
        "stt_provider": settings.STT_PROVIDER.value,
        "nlp_provider": settings.NLP_PROVIDER.value,
        "tts_provider": settings.TTS_PROVIDER.value,
        "shopify_store": settings.SHOPIFY_STORE_DOMAIN,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
    }
