"""
MIRA Voice Commerce - FastAPI Backend
AI-Powered Voice Commerce with Shopify Integration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from app.config import settings, get_provider_info
from app.api.voice import router as voice_router
from app.api.products import router as products_router
from app.api.listing import router as listing_router

app = FastAPI(
    title="MIRA Voice Commerce API",
    description="AI-Powered Voice Commerce for Tribal Areas with Shopify Integration",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mira-comm.myshopify.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(voice_router)
app.include_router(products_router)
app.include_router(listing_router)

@app.get("/")
async def root():
    """API root endpoint"""
    try:
        providers = get_provider_info()
    except Exception:
        providers = {"error": "Could not load provider info"}
    
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "status": "ready",
        "features": [
            "Multi-language Voice Input (Hindi/English)",
            "AI-Powered Intent Recognition",
            "Shopify Product Integration",
            "Text-to-Speech Responses"
        ],
        "providers": providers
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - simple and fast"""
    try:
        return {
            "status": "healthy",
            "ai_services": {
                "stt": settings.STT_PROVIDER.value,
                "nlp": settings.NLP_PROVIDER.value,
                "tts": settings.TTS_PROVIDER.value
            },
            "shopify": {
                "store": settings.SHOPIFY_STORE_DOMAIN,
                "connected": bool(settings.SHOPIFY_STOREFRONT_ACCESS_TOKEN)
            }
        }
    except Exception as e:
        # Return minimal response if settings fail
        return {
            "status": "healthy",
            "error": str(e)
        }

@app.get("/config")
async def get_config():
    """
    Get current AI provider configuration
    
    This endpoint shows which AI models are currently active.
    To change providers, edit backend/app/config.py
    """
    return {
        "message": "To change AI providers, edit backend/app/config.py",
        "current_providers": get_provider_info(),
        "available_providers": {
            "stt": ["whisper_local", "whisper_api", "google", "azure", "assemblyai"],
            "nlp": ["openai_gpt4", "openai_gpt35", "google_gemini", "anthropic_claude", "local_llm"],
            "tts": ["swara", "google", "azure", "openai", "elevenlabs", "web_speech"]
        },
        "configuration_file": "backend/app/config.py",
        "instructions": {
            "step_1": "Open backend/app/config.py",
            "step_2": "Change STT_PROVIDER, NLP_PROVIDER, or TTS_PROVIDER",
            "step_3": "Set corresponding API keys in .env file",
            "step_4": "Restart the server"
        }
    }

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files"""
    import tempfile
    import urllib.parse
    
    # Decode filename in case it's URL encoded
    filename = urllib.parse.unquote(filename)
    
    # Remove any path components - we only want the filename
    filename = os.path.basename(filename)
    
    if settings.DEBUG:
        print(f"[DEBUG] Audio request for filename: '{filename}'")
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(tempfile.gettempdir(), filename),  # Most common location
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", filename),  # Windows temp
        f"/tmp/{filename}",  # Linux/Mac
        filename,  # Direct path (in case it's already a full path)
    ]
    
    # Also try to find the file by searching temp directory (but limit search depth)
    temp_dir = tempfile.gettempdir()
    try:
        # Only search first level to avoid performance issues
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                if item == filename:
                    possible_paths.insert(0, os.path.join(temp_dir, item))
                    break
    except Exception:
        pass  # If search fails, continue with other paths
    
    if settings.DEBUG:
        print(f"[DEBUG] Searching for audio in paths: {possible_paths}")
    
    for audio_path in possible_paths:
        try:
            if os.path.exists(audio_path) and os.path.isfile(audio_path):
                file_size = os.path.getsize(audio_path)
                if settings.DEBUG:
                    print(f"[DEBUG] Found audio file: {audio_path}, size: {file_size} bytes")
                
                # Check if file is too small (likely failed TTS)
                if file_size < 100:
                    if settings.DEBUG:
                        print(f"[DEBUG] Audio file too small ({file_size} bytes), may be corrupted")
                    continue
                
                # Determine media type from extension
                if filename.lower().endswith('.mp3') or audio_path.lower().endswith('.mp3'):
                    media_type = "audio/mpeg"
                elif filename.lower().endswith('.wav') or audio_path.lower().endswith('.wav'):
                    media_type = "audio/wav"
                else:
                    media_type = "audio/mpeg"
                
                if settings.DEBUG:
                    print(f"[DEBUG] Serving audio with media type: {media_type}")
                
                return FileResponse(
                    audio_path, 
                    media_type=media_type,
                    headers={
                        "Cache-Control": "no-cache",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] Error accessing {audio_path}: {e}")
            continue  # Try next path
    
    if settings.DEBUG:
        print(f"[DEBUG] Audio file not found: {filename}")
    
    return {"error": f"Audio file not found: {filename}", "searched_paths": possible_paths}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
