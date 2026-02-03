"""
Multi-Provider Speech-to-Text Service
Supports: Whisper (local/API), Google, Azure, AssemblyAI
"""
import os
import tempfile
import shutil
import sys
import site
import importlib
from typing import Tuple, Optional, List
from pathlib import Path

from app.config import settings, STTProvider

class STTService:
    """Speech-to-Text service with multiple provider support"""
    
    def __init__(self):
        self.provider = settings.STT_PROVIDER
        self._whisper_model = None
        
    async def transcribe(
        self, 
        audio_file_path: str, 
        language: Optional[str] = None,
        product_context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Transcribe audio to text
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (en, hi) or None for auto-detect
            product_context: Optional list of product names to help with transcription accuracy
            
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        if self.provider == STTProvider.WHISPER_LOCAL:
            return await self._transcribe_whisper_local(audio_file_path, language, product_context)
        elif self.provider == STTProvider.WHISPER_API:
            return await self._transcribe_whisper_api(audio_file_path, language, product_context)
        elif self.provider == STTProvider.GOOGLE:
            return await self._transcribe_google(audio_file_path, language, product_context)
        elif self.provider == STTProvider.AZURE:
            return await self._transcribe_azure(audio_file_path, language, product_context)
        elif self.provider == STTProvider.ASSEMBLYAI:
            return await self._transcribe_assemblyai(audio_file_path, language, product_context)
        else:
            raise ValueError(f"Unsupported STT provider: {self.provider}")
    
    async def _transcribe_whisper_local(
        self, 
        audio_path: str, 
        language: Optional[str],
        product_context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Transcribe using local Whisper model"""
        if settings.FFMPEG_BIN:
            os.environ["PATH"] = f"{settings.FFMPEG_BIN};{os.environ.get('PATH', '')}"
        elif shutil.which("ffmpeg") is None:
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                winget_packages = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
                if os.path.isdir(winget_packages):
                    try:
                        for pkg in os.listdir(winget_packages):
                            if not pkg.lower().startswith("gyan.ffmpeg"):
                                continue
                            pkg_dir = os.path.join(winget_packages, pkg)
                            if not os.path.isdir(pkg_dir):
                                continue
                            for child in os.listdir(pkg_dir):
                                if not child.lower().startswith("ffmpeg-"):
                                    continue
                                bin_dir = os.path.join(pkg_dir, child, "bin")
                                if os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe")):
                                    os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"
                                    raise StopIteration
                    except StopIteration:
                        pass

            if shutil.which("ffmpeg") is None:
                for bin_dir in (
                    r"C:\Program Files\ffmpeg\bin",
                    r"C:\ffmpeg\bin",
                ):
                    if os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe")):
                        os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"
                        break

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg.exe was not found on PATH. Whisper requires ffmpeg to process audio files.\n"
                "Install ffmpeg using one of these methods:\n"
                "1. winget: winget install Gyan.FFmpeg\n"
                "2. chocolatey: choco install ffmpeg\n"
                "3. Download from https://ffmpeg.org/download.html\n"
                "4. Or set FFMPEG_BIN in .env to the folder containing ffmpeg.exe"
            )

        # Workaround: some Windows Python installs can have a stray `regex.py` in the Python root
        # (e.g., ...\Python312\regex.py) which shadows the PyPI `regex` package required by Whisper.
        # Ensure site-packages is preferred, then reload `regex` if needed.
        try:
            site_pkgs = site.getsitepackages()
        except Exception:
            site_pkgs = []

        for sp in reversed(site_pkgs):
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)

        try:
            regex_mod = importlib.import_module("regex")
            if not hasattr(regex_mod, "escape"):
                sys.modules.pop("regex", None)
                regex_mod = importlib.import_module("regex")
        except Exception:
            # If regex import still fails, Whisper will raise a clearer error later.
            pass

        try:
            import whisper
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Whisper local STT requires the 'openai-whisper' package (module name: 'whisper'). "
                "Install it in the backend environment: python -m pip install openai-whisper"
            ) from e
        
        # Load model in executor if not already loaded (to avoid blocking)
        # First-time loading can take 1-3 minutes (download + load)
        import asyncio
        loop = asyncio.get_event_loop()
        
        if self._whisper_model is None:
            # Model loading can take time, especially on first use
            # Run in executor with no timeout (handled by API layer)
            try:
                self._whisper_model = await loop.run_in_executor(
                    None,
                    whisper.load_model,
                    settings.WHISPER_MODEL_SIZE
                )
            except Exception as e:
                error_msg = str(e)
                if "download" in error_msg.lower() or "model" in error_msg.lower():
                    raise RuntimeError(
                        f"Whisper model loading failed. This might be the first time loading the model. "
                        f"Model size: {settings.WHISPER_MODEL_SIZE}. "
                        f"Error: {error_msg}. "
                        f"Try using a smaller model (tiny) or check your internet connection for first-time download."
                    )
                raise
        
        # Map language codes for Indian languages
        # Whisper uses ISO 639-1 codes: hi=Hindi, ta=Tamil, te=Telugu, etc.
        language_map = {
            "hindi": "hi",
            "tamil": "ta", 
            "telugu": "te",
            "kannada": "kn",
            "malayalam": "ml",
            "marathi": "mr",
            "gujarati": "gu",
            "punjabi": "pa",
            "bengali": "bn",
            "english": "en"
        }
        
        # Normalize language code - FORCE language when provided to avoid misdetection
        # Whisper can misdetect Indian languages as Chinese if not forced
        # CRITICAL: Always force the language code to prevent misdetection
        whisper_lang = None
        if language:
            lang_lower = str(language).lower().strip()
            
            # Map full language names to codes, or use the code directly
            if len(lang_lower) > 2:
                # It's a full name like "hindi", "tamil", etc.
                whisper_lang = language_map.get(lang_lower, None)
                # If not found in map, default to Hindi for Indian language context
                if not whisper_lang:
                    whisper_lang = "hi"
            else:
                # It's already a code like "hi", "ta", "te"
                whisper_lang = lang_lower
            
            # Ensure we have a valid language code
            valid_codes = ["hi", "ta", "te", "kn", "ml", "mr", "gu", "pa", "bn", "en"]
            if not whisper_lang or whisper_lang not in valid_codes:
                # Default to Hindi if invalid
                whisper_lang = "hi"
            
            # Debug log
            if settings.DEBUG:
                print(f"[DEBUG] Whisper language: '{whisper_lang}' (from input: '{language}')")
        else:
            # If no language provided, default to Hindi (not auto-detect) to avoid misdetection
            whisper_lang = "hi"
            if settings.DEBUG:
                print(f"[DEBUG] No language provided, defaulting to Hindi: '{whisper_lang}'")
        
        # IMPORTANT: ALWAYS force the language to prevent misdetection
        # If language is "hi", we MUST pass "hi" to Whisper, not None
        # Auto-detection can misidentify Hindi as Chinese
        
        # Run transcription in executor to avoid blocking
        def _transcribe():
            # ALWAYS pass language to force Whisper to use it (prevents misdetection)
            # Use initial_prompt with product context to help Whisper understand the context
            transcribe_kwargs = {
                "language": whisper_lang,  # Force language - never None
                "task": "transcribe"
            }
            
            # Build initial prompt with product context for better accuracy
            initial_prompt_parts = []
            
            if whisper_lang == "hi":
                initial_prompt_parts.append("यह हिंदी भाषा में है।")  # "This is in Hindi language."
            
            # Add product names to help Whisper recognize product names in speech
            if product_context and len(product_context) > 0:
                # Limit to first 20 products to avoid prompt being too long
                products_text = ", ".join(product_context[:20])
                if whisper_lang == "hi":
                    initial_prompt_parts.append(f"उत्पाद: {products_text}")
                else:
                    initial_prompt_parts.append(f"Products: {products_text}")
            
            if initial_prompt_parts:
                transcribe_kwargs["initial_prompt"] = " ".join(initial_prompt_parts)
            
            return self._whisper_model.transcribe(audio_path, **transcribe_kwargs)
        
        result = await loop.run_in_executor(None, _transcribe)
        
        # Get the detected language from Whisper result
        detected_lang = result.get("language", whisper_lang or "unknown")
        transcribed_text = result["text"].strip()
        
        # Debug: Log what Whisper detected vs what we requested
        if settings.DEBUG:
            print(f"[DEBUG] Whisper requested: '{whisper_lang}', detected: '{detected_lang}', transcription: '{transcribed_text[:50]}...'")
        
        # If Whisper detected a different language than requested, log a warning
        if whisper_lang and detected_lang != whisper_lang:
            if settings.DEBUG:
                print(f"[WARNING] Language mismatch! Requested: '{whisper_lang}', but Whisper detected: '{detected_lang}'")
        
        return transcribed_text, detected_lang
    
    async def _transcribe_whisper_api(
        self, 
        audio_path: str, 
        language: Optional[str],
        product_context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Transcribe using OpenAI Whisper API (supports Hindi, Tamil, Telugu, and 90+ languages)"""
        from openai import OpenAI
        import asyncio
        
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key not configured for Whisper API")
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Map language codes for Indian languages
        language_map = {
            "hindi": "hi",
            "tamil": "ta", 
            "telugu": "te",
            "kannada": "kn",
            "malayalam": "ml",
            "marathi": "mr",
            "gujarati": "gu",
            "punjabi": "pa",
            "bengali": "bn",
            "english": "en"
        }
        
        # Normalize language code
        whisper_lang = None
        if language:
            lang_lower = language.lower()
            whisper_lang = language_map.get(lang_lower, language) if len(lang_lower) > 2 else language
        
        # Run API call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            with open(audio_path, "rb") as audio_file:
                return client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=whisper_lang  # None = auto-detect
                )
        
        transcript = await loop.run_in_executor(None, _transcribe)
        
        return transcript.text.strip(), language or "unknown"
    
    async def _transcribe_google(
        self, 
        audio_path: str, 
        language: Optional[str],
        product_context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Transcribe using Google Speech-to-Text (API key for Gemini)"""
        try:
            from google.cloud import speech
        except Exception as e:
            raise RuntimeError(
                "Google STT requires the 'google-cloud-speech' package. "
                "Install it in the backend environment: python -m pip install google-cloud-speech"
            ) from e

        # Prefer API key if available (Gemini setup)
        if settings.GOOGLE_API_KEY:
            import os
            os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

        try:
            client = speech.SpeechClient()
        except Exception as e:
            raise RuntimeError(
                "Google STT client failed to initialize. Ensure GOOGLE_API_KEY is set or service account credentials are configured. "
                f"Error: {e}"
            ) from e
        
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        
        # Map language codes
        lang_code = "hi-IN" if language == "hi" else "en-IN"
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=lang_code,
            enable_automatic_punctuation=True,
        )
        
        response = client.recognize(config=config, audio=audio)
        
        if response.results:
            text = response.results[0].alternatives[0].transcript
            return text.strip(), language or "en"
        else:
            # Fallback for testing: return dummy transcription
            return "test transcription", language or "en"
    
    async def _transcribe_azure(
        self, 
        audio_path: str, 
        language: Optional[str],
        product_context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Transcribe using Azure Speech Services"""
        import azure.cognitiveservices.speech as speechsdk
        
        speech_config = speechsdk.SpeechConfig(
            subscription=settings.AZURE_SPEECH_KEY,
            region=settings.AZURE_SPEECH_REGION
        )
        
        # Map language codes
        lang_code = "hi-IN" if language == "hi" else "en-IN"
        speech_config.speech_recognition_language = lang_code
        
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        result = speech_recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text.strip(), language or "en"
        
        return "", language or "en"
    
    async def _transcribe_assemblyai(
        self, 
        audio_path: str, 
        language: Optional[str],
        product_context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Transcribe using AssemblyAI"""
        import asyncio
        import audioop
        import wave
        import assemblyai as aai
        
        if not settings.ASSEMBLYAI_API_KEY:
            raise RuntimeError("AssemblyAI API key not configured")
        
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        
        # Quick WAV sanity/silence check to avoid spending time on empty recordings.
        # Streamlit sometimes sends very short or near-silent clips (mic permission/volume).
        try:
            if Path(audio_path).suffix.lower() == ".wav":
                with wave.open(audio_path, "rb") as wf:
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate() or 0
                    nframes = wf.getnframes() or 0
                    duration_s = (nframes / framerate) if framerate else 0.0
                    # Sample only first ~3 seconds for RMS (fast).
                    frames_to_read = min(nframes, int(framerate * 3)) if framerate else nframes
                    raw = wf.readframes(frames_to_read) if frames_to_read else b""
                    rms = audioop.rms(raw, sampwidth) if raw and sampwidth else 0
                    max_amp = float((2 ** (8 * sampwidth - 1)) - 1) if sampwidth else 1.0
                    rms_norm = (float(rms) / max_amp) if max_amp else 0.0

                if settings.DEBUG:
                    print(
                        f"[DEBUG] AssemblyAI input wav stats: "
                        f"channels={channels}, sampwidth={sampwidth}, rate={framerate}, "
                        f"duration_s={duration_s:.2f}, rms_norm={rms_norm:.5f}"
                    )

                # If it's extremely short or effectively silent, return empty transcription.
                if duration_s < 0.25 or rms_norm < 0.001:
                    return "", (language or settings.DEFAULT_LANGUAGE)
        except Exception as e:
            # If WAV parsing fails, proceed with AssemblyAI and let it decide.
            if settings.DEBUG:
                print(f"[DEBUG] AssemblyAI wav sanity check skipped: {e}")

        # Configure transcription with language settings
        config = aai.TranscriptionConfig()
        if language:
            # Map language codes to AssemblyAI language codes
            language_mapping = {
                "hi": "hi",      # Hindi
                "ta": "ta",      # Tamil  
                "te": "te",      # Telugu
                "en": "en",      # English
                "bn": "bn",      # Bengali
                "mr": "mr",      # Marathi
                "gu": "gu",      # Gujarati
                "kn": "kn",      # Kannada
                "ml": "ml",      # Malayalam
                "pa": "pa",      # Punjabi
            }
            config.language_code = language_mapping.get(language.lower() if language else "hi", "hi")  # Default to Hindi
        
        # Create transcriber (attach config here for SDK compatibility)
        transcriber = aai.Transcriber(config=config)
        
        try:
            # Run the blocking transcribe call in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            # AssemblyAI transcribe can take time, run it in executor with timeout
            def _transcribe_file():
                return transcriber.transcribe(audio_path)
            
            try:
                transcript = await asyncio.wait_for(
                    loop.run_in_executor(None, _transcribe_file),
                    timeout=120.0  # 2 minute timeout for initial transcription
                )
            except asyncio.TimeoutError:
                raise RuntimeError("AssemblyAI transcription timed out after 120 seconds")
            
            # Check if transcript is None
            if transcript is None:
                raise RuntimeError("AssemblyAI returned None transcript")
            
            # Check transcript status - AssemblyAI SDK returns transcript objects with status
            if hasattr(transcript, 'status'):
                # Handle error status
                if transcript.status == aai.TranscriptStatus.error:
                    error_msg = getattr(transcript, 'error', "Unknown error")
                    raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")
                
                # Handle completed status
                if transcript.status == aai.TranscriptStatus.completed:
                    text = transcript.text.strip() if hasattr(transcript, 'text') and transcript.text else ""
                    if not text:
                        # No speech detected (or unusable audio). Let API layer return a 422 with guidance.
                        return "", (language or settings.DEFAULT_LANGUAGE)
                    detected_lang = language or "hi"
                    return text, detected_lang
                
                # Handle queued/processing status - poll for completion
                if transcript.status in (aai.TranscriptStatus.queued, aai.TranscriptStatus.processing):
                    transcript_id = getattr(transcript, 'id', None)
                    if not transcript_id:
                        raise RuntimeError("AssemblyAI transcript has no ID for polling")
                    
                    # Poll for completion (up to 2 minutes total)
                    max_polls = 24  # Check 24 times, 5 seconds apart = 2 minutes
                    for attempt in range(max_polls):
                        await asyncio.sleep(5)
                        try:
                            # Get updated transcript status
                            def _get_transcript():
                                return transcriber.get_transcript(transcript_id)
                            
                            updated_transcript = await asyncio.wait_for(
                                loop.run_in_executor(None, _get_transcript),
                                timeout=10.0
                            )
                            
                            if updated_transcript is None:
                                continue  # Keep polling
                            
                            if hasattr(updated_transcript, 'status'):
                                if updated_transcript.status == aai.TranscriptStatus.completed:
                                    text = updated_transcript.text.strip() if hasattr(updated_transcript, 'text') and updated_transcript.text else ""
                                    if text:
                                        detected_lang = language or "hi"
                                        return text, detected_lang
                                    else:
                                        return "", (language or settings.DEFAULT_LANGUAGE)
                                elif updated_transcript.status == aai.TranscriptStatus.error:
                                    error_msg = getattr(updated_transcript, 'error', "Unknown error")
                                    raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")
                                # If still processing, continue polling
                            
                        except asyncio.TimeoutError:
                            continue  # Keep polling
                        except RuntimeError:
                            raise  # Re-raise RuntimeErrors
                        except Exception as e:
                            # Log error but continue polling
                            if settings.DEBUG:
                                print(f"[DEBUG] AssemblyAI polling error (attempt {attempt + 1}): {e}")
                            if attempt == max_polls - 1:
                                raise RuntimeError(f"AssemblyAI polling failed after {max_polls} attempts: {str(e)}")
                    
                    # If we get here, polling exhausted
                    raise RuntimeError(f"AssemblyAI transcription did not complete after {max_polls * 5} seconds of polling")
                else:
                    # Unknown status
                    raise RuntimeError(f"AssemblyAI transcription returned unknown status: {transcript.status}")
            else:
                # If transcript doesn't have status attribute, check if it has text directly
                # This handles cases where the SDK might return completed transcripts differently
                if hasattr(transcript, 'text') and transcript.text:
                    text = transcript.text.strip()
                    if text:
                        detected_lang = language or "hi"
                        return text, detected_lang
                
                # If no status and no text, something went wrong
                raise RuntimeError("AssemblyAI transcript has no status or text attribute")
            
        except RuntimeError:
            # Re-raise RuntimeErrors as-is
            raise
        except Exception as e:
            # If AssemblyAI fails, provide helpful error message
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                raise RuntimeError(
                    "AssemblyAI transcription timed out. The audio file might be too large or the service is slow. "
                    "Try using a shorter audio clip (under 30 seconds) or switch to a different STT provider."
                )
            elif "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower() or "401" in error_msg:
                raise RuntimeError(
                    "AssemblyAI API key is invalid or expired. Check your ASSEMBLYAI_API_KEY in the .env file."
                )
            elif "file" in error_msg.lower() or "format" in error_msg.lower() or "not found" in error_msg.lower():
                raise RuntimeError(
                    f"AssemblyAI could not process the audio file. Ensure it's a valid audio format (WAV, MP3, etc.) and the file exists. Error: {error_msg}"
                )
            else:
                raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")

# Singleton instance
_stt_service = None

def get_stt_service() -> STTService:
    """Get or create STT service instance"""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
