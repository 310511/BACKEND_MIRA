"""
Multi-Provider Text-to-Speech Service
Supports: Google Cloud TTS, Azure TTS, OpenAI TTS, ElevenLabs
Note: Web Speech API is client-side only
"""
from typing import Optional
import tempfile
from pathlib import Path
import os

from app.config import settings, TTSProvider

class TTSService:
    """Text-to-Speech service with multiple provider support"""
    
    def __init__(self):
        self.provider = settings.TTS_PROVIDER
        self._swara_pipeline = None
        self._swara_lock = None  # lazy asyncio.Lock (created on first event loop)

    async def _get_swara_pipeline(self):
        """
        Lazily load and cache the HF text-to-speech pipeline.
        IMPORTANT: Loading the model can take time; do it once and reuse.
        """
        import asyncio

        if self._swara_pipeline is not None:
            return self._swara_pipeline

        if self._swara_lock is None:
            self._swara_lock = asyncio.Lock()

        async with self._swara_lock:
            if self._swara_pipeline is not None:
                return self._swara_pipeline

            # Import inside to avoid import cost if not used
            from transformers import pipeline
            import torch

            model_name = settings.SWARA_MODEL_NAME

            if settings.DEBUG:
                print(f"[DEBUG] Loading Swara TTS pipeline: model='{model_name}'")

            loop = asyncio.get_event_loop()

            def _load():
                return pipeline(
                    "text-to-speech",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                )

            self._swara_pipeline = await loop.run_in_executor(None, _load)
            return self._swara_pipeline
        
    async def synthesize(
        self, 
        text: str, 
        language: str = "en",
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            language: Language code (en, hi)
            output_path: Optional output file path (creates temp file if None)
            
        Returns:
            Path to generated audio file
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")

        # Providers that output WAV (not MP3): use .wav path so file is valid
        wav_only_providers = (TTSProvider.SWARA, TTSProvider.PYTTSSX3)
        if self.provider in wav_only_providers and output_path.lower().endswith(".mp3"):
            output_path = (output_path[:-4] if output_path.endswith(".mp3") else output_path) + ".wav"

        # Normalize language code
        lang_code = language.strip().lower() if language else "en"
        if not lang_code:
            lang_code = "en"
        
        # Debug log
        if settings.DEBUG:
            print(f"[DEBUG] TTS synthesize: language='{lang_code}' (input: '{language}'), provider={self.provider}")
            print(f"[DEBUG] TTS output_path: {output_path}")

        try:
            if self.provider == TTSProvider.SWARA:
                return await self._synthesize_swara(text, lang_code, output_path)
            elif self.provider == TTSProvider.GOOGLE:
                return await self._synthesize_google(text, lang_code, output_path)
            elif self.provider == TTSProvider.AZURE:
                return await self._synthesize_azure(text, lang_code, output_path)
            elif self.provider == TTSProvider.OPENAI:
                if not settings.OPENAI_API_KEY:
                    if settings.DEBUG:
                        print("[DEBUG] OpenAI API key not found, falling back to pyttsx3")
                    return await self._synthesize_pyttsx3(text, lang_code, output_path)
                return await self._synthesize_openai(text, lang_code, output_path)
            elif self.provider == TTSProvider.ELEVENLABS:
                return await self._synthesize_elevenlabs(text, lang_code, output_path)
            elif self.provider == TTSProvider.PYTTSSX3:
                return await self._synthesize_pyttsx3(text, lang_code, output_path)
            elif self.provider == TTSProvider.WEB_SPEECH:
                return text
            else:
                if settings.DEBUG:
                    print(f"[DEBUG] Unsupported TTS provider: {self.provider}, falling back to pyttsx3")
                return await self._synthesize_pyttsx3(text, lang_code, output_path)
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] TTS synthesis failed with {self.provider}: {e}")
            try:
                if settings.DEBUG:
                    print("[DEBUG] Falling back to pyttsx3 TTS")
                return await self._synthesize_pyttsx3(text, lang_code, output_path)
            except Exception as e2:
                if settings.DEBUG:
                    print(f"[DEBUG] pyttsx3 TTS also failed: {e2}")
                try:
                    if settings.DEBUG:
                        print("[DEBUG] Falling back to Swara TTS")
                    return await self._synthesize_swara(text, lang_code, output_path)
                except Exception as e3:
                    if settings.DEBUG:
                        print(f"[DEBUG] All TTS methods failed: {e3}")
                    try:
                        if settings.DEBUG:
                            print("[DEBUG] Falling back to gTTS")
                        return await self._synthesize_gtts(text, lang_code, output_path)
                    except Exception as e4:
                        if settings.DEBUG:
                            print(f"[DEBUG] gTTS fallback also failed: {e4}")
                        return self._synthesize_dummy(output_path)

    async def _synthesize_gtts(self, text: str, language: str, output_path: str) -> str:
        """Fallback TTS using gTTS (Google Text-to-Speech). Output is MP3."""
        import asyncio

        if output_path.lower().endswith(".wav"):
            output_path = output_path[:-4] + ".mp3"
        if not output_path.lower().endswith(".mp3"):
            output_path = output_path + ".mp3"

        lang_map = {
            "en": "en",
            "hi": "hi",
            "ta": "ta",
            "te": "te",
            "kn": "kn",
            "ml": "ml",
            "mr": "mr",
            "gu": "gu",
            "pa": "pa",
            "bn": "bn",
        }
        gtts_lang = lang_map.get((language or "en").lower(), "en")

        def _run():
            from gtts import gTTS

            tts = gTTS(text=text if text and text.strip() else " ", lang=gtts_lang)
            tts.save(output_path)
            return output_path

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run)

        if not os.path.exists(output_path):
            raise RuntimeError("gTTS did not create output file")
        if os.path.getsize(output_path) < 500:
            raise RuntimeError("gTTS output file too small")
        return output_path

    def _synthesize_dummy(self, output_path: str) -> str:
        """Create a silent MP3 file as last resort"""
        try:
            # Create a proper silent MP3 file (3 seconds of silence)
            from pydub import AudioSegment
            silence = AudioSegment.silent(duration=3000)  # 3 seconds silence
            silence.export(output_path, format="mp3")
            
            # Verify the file was created properly
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if settings.DEBUG:
                    print(f"[DEBUG] Created silent MP3: {output_path}, size: {file_size} bytes")
                if file_size > 1000:
                    return output_path
            
            if settings.DEBUG:
                print(f"[DEBUG] Silent MP3 creation failed or too small: {file_size} bytes")
        except ImportError:
            if settings.DEBUG:
                print("[DEBUG] pydub not available, creating minimal dummy audio")
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] Error creating silent MP3: {e}")
        
        # If pydub not available or failed, create a larger minimal file
        Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            # Write a larger MP3 header with silent audio data
            # This creates a more valid MP3 file that browsers can play
            f.write(b'\xff\xfb\x90\x00' + b'\x00' * 1000)  # MP3 header + padding
        if settings.DEBUG:
            print(f"[DEBUG] Created minimal dummy audio: {output_path}")
        return output_path
    
    def _get_dummy_audio_bytes(self) -> tuple:
        """Return (audio_bytes, audio_format) for playable fallback when TTS fails. Never returns empty."""
        dummy_path = tempfile.mktemp(suffix=".mp3")
        try:
            self._synthesize_dummy(dummy_path)
            if os.path.exists(dummy_path):
                with open(dummy_path, "rb") as f:
                    data = f.read()
                if data and len(data) > 100:
                    return (data, "audio/mp3")
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] Dummy audio creation failed: {e}")
        finally:
            if os.path.exists(dummy_path):
                try:
                    os.remove(dummy_path)
                except Exception:
                    pass
        # Last resort: minimal valid MP3 bytes so browser doesn't show 0:00
        return (b'\xff\xfb\x90\x00' + b'\x00' * 2000, "audio/mp3")
    
    async def synthesize_bytes(self, text: str, language: str = "en") -> dict:
        """
        Synthesize speech and return raw audio bytes
        Returns dict with audio_bytes, audio_format, and file_path
        """
        import tempfile
        
        # Use .wav for providers that output WAV (pyttsx3, Swara); .mp3 for others
        wav_only_providers = (TTSProvider.SWARA, TTSProvider.PYTTSSX3)
        suffix = ".wav" if self.provider in wav_only_providers else ".mp3"
        output_path = tempfile.mktemp(suffix=suffix)
        
        try:
            # Generate audio file (synthesize may override path to .wav for WAV providers)
            file_path = await self.synthesize(text, language, output_path)
            
            if not file_path or not os.path.exists(file_path):
                if settings.DEBUG:
                    print("[DEBUG] TTS file missing; returning dummy audio")
                dummy_bytes, dummy_fmt = self._get_dummy_audio_bytes()
                return {"audio_bytes": dummy_bytes, "audio_format": dummy_fmt, "file_path": None, "language": language}
            
            # Read the audio file as raw bytes
            with open(file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Only use dummy when clearly empty/corrupt (WAV header is 44 bytes); keep short real TTS
            if not audio_bytes or len(audio_bytes) < 100:
                if settings.DEBUG:
                    print(f"[DEBUG] TTS audio too small ({len(audio_bytes) if audio_bytes else 0} bytes); returning dummy audio")
                dummy_bytes, dummy_fmt = self._get_dummy_audio_bytes()
                return {"audio_bytes": dummy_bytes, "audio_format": dummy_fmt, "file_path": file_path, "language": language}
            actual_format = "audio/wav" if file_path.lower().endswith(".wav") else "audio/mp3"
            
            if settings.DEBUG:
                print(f"[DEBUG] TTS raw bytes: {len(audio_bytes)} bytes")
                print(f"[DEBUG] Audio format: {actual_format}")
            
            return {
                "audio_bytes": audio_bytes,
                "audio_format": actual_format,
                "file_path": file_path,
                "language": language
            }
            
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] TTS bytes synthesis failed: {e}")
            # Always return playable audio so UI never shows "not available"
            dummy_bytes, dummy_fmt = self._get_dummy_audio_bytes()
            return {
                "audio_bytes": dummy_bytes,
                "audio_format": dummy_fmt,
                "file_path": None,
                "language": language
            }
        finally:
            # Clean up temporary file
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

    async def synthesize_base64(self, text: str, language: str = "en") -> dict:
        """
        Synthesize speech and return base64 encoded audio data
        Returns dict with audio_data, audio_format, and file_path
        """
        import tempfile
        import base64
        
        # Create temporary file
        output_path = tempfile.mktemp(suffix=".mp3")
        
        try:
            # Generate audio file
            file_path = await self.synthesize(text, language, output_path)
            
            # Read the audio file and encode as base64
            with open(file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Determine audio format
            audio_format = "audio/mp3"
            
            if settings.DEBUG:
                print(f"[DEBUG] TTS base64 encoded: {len(audio_base64)} characters")
                print(f"[DEBUG] Original file size: {len(audio_bytes)} bytes")
            
            return {
                "audio_data": audio_base64,
                "audio_format": audio_format,
                "file_path": file_path,
                "language": language
            }
            
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] TTS base64 synthesis failed: {e}")
            # Return empty audio data on failure
            return {
                "audio_data": "",
                "audio_format": "audio/mp3",
                "file_path": None,
                "language": language
            }
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

    async def _synthesize_pyttsx3(self, text: str, language: str, output_path: str) -> str:
        """Fallback TTS using pyttsx3 (offline, no internet needed). Output is always WAV."""
        import asyncio
        # pyttsx3 on Windows typically outputs WAV; using .mp3 path can produce invalid file
        if output_path.lower().endswith(".mp3"):
            output_path = output_path[:-4] + ".wav"
        
        def _run_pyttsx3():
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            
            # For Hindi and other Indian languages, try to find appropriate voice
            # If not available, use English voice but keep the original text
            voices = engine.getProperty('voices')
            hindi_voice_found = False
            
            if language in ["hi", "ta", "te", "kn", "ml", "mr", "gu", "pa", "bn"]:
                # Try to find Indian language voice
                for voice in voices:
                    if any(indic_lang in voice.name.lower() for indic_lang in ['hindi', 'india', 'hi-', 'ta-', 'te-', 'kn-', 'ml-', 'mr-', 'gu-', 'pa-', 'bn-']):
                        engine.setProperty('voice', voice.id)
                        hindi_voice_found = True
                        if settings.DEBUG:
                            print(f"[DEBUG] Found Indian voice: {voice.name}")
                        break
                
                # If no Indian voice found, use first available voice
                if not hindi_voice_found and voices:
                    engine.setProperty('voice', voices[0].id)
                    if settings.DEBUG:
                        print(f"[DEBUG] No Indian voice found, using: {voices[0].name}")
            
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            return output_path
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_pyttsx3)
        
        # Verify file was created and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if settings.DEBUG:
                print(f"[DEBUG] pyttsx3 TTS file size: {file_size} bytes")
            
            # Accept short clips (300+ bytes = header + some audio); only reject clearly empty
            if file_size >= 300:
                if settings.DEBUG:
                    print(f"[DEBUG] pyttsx3 TTS successful: {output_path} ({file_size} bytes)")
                return output_path
            else:
                if settings.DEBUG:
                    print(f"[DEBUG] pyttsx3 TTS file too small: {file_size} bytes")
                
                # For Indian languages with tiny files, try English fallback
                if language in ["hi", "ta", "te", "kn", "ml", "mr", "gu", "pa", "bn"] and file_size < 100:
                    if settings.DEBUG:
                        print(f"[DEBUG] Trying English fallback for {language} text")
                    try:
                        fallback_text = f"Response in {language}"
                        return await self._synthesize_pyttsx3(fallback_text, "en", output_path)
                    except Exception as e:
                        if settings.DEBUG:
                            print(f"[DEBUG] English fallback also failed: {e}")
                
                raise Exception(f"pyttsx3 generated file too small: {file_size} bytes")
        else:
            if settings.DEBUG:
                print(f"[DEBUG] pyttsx3 TTS failed to create file: {output_path}")
            raise Exception("pyttsx3 failed to generate audio")
    
    async def _synthesize_swara(
        self, 
        text: str, 
        language: str, 
        output_path: str
    ) -> str:
        """
        Synthesize using Swara TTS (Hugging Face) - Primary for Hindi
        Uses AI4Bharat IndicTTS or similar open-source models
        """
        try:
            from transformers import pipeline
            import torch
            import soundfile as sf
            import numpy as np
        except ImportError:
            raise ImportError(
                "Swara TTS requires transformers, torch, and soundfile. "
                "Install with: pip install transformers torch soundfile"
            )
        
        # Use IndicTTS pipeline for Hindi/English
        # Model can be configured via settings.SWARA_MODEL_NAME
        # soundfile writes WAV reliably; .mp3 can be unsupported or produce invalid file
        if output_path.lower().endswith(".mp3"):
            output_path = output_path[:-4] + ".wav"
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()

            # Load/cached pipeline once (avoids huge per-request delays)
            tts_pipeline = await self._get_swara_pipeline()
            
            # Generate speech
            # IndicTTS models typically expect language code
            # Support multiple Indian languages
            lang_code_map = {
                "hi": "hi",  # Hindi
                "ta": "ta",  # Tamil
                "te": "te",  # Telugu
                "kn": "kn",  # Kannada
                "ml": "ml",  # Malayalam
                "mr": "mr",  # Marathi
                "gu": "gu",  # Gujarati
                "pa": "pa",  # Punjabi
                "bn": "bn",  # Bengali
                "en": "en"   # English
            }
            lang_code = lang_code_map.get(language.lower() if language else "en", "en")
            
            # Generate audio in executor (HF pipeline is blocking/CPU heavy)
            def _run_pipeline():
                return tts_pipeline(text, language=lang_code)

            audio_output = await loop.run_in_executor(None, _run_pipeline)
            
            # Extract audio array and sample rate
            if isinstance(audio_output, dict):
                audio_array = audio_output.get("audio", audio_output.get("raw", None))
                sample_rate = audio_output.get("sampling_rate", 22050)
            else:
                # Fallback: assume it's a numpy array
                audio_array = audio_output
                sample_rate = 22050
            
            # Ensure audio is numpy array
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            
            # Normalize audio
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Save to file (also can block)
            def _write():
                sf.write(output_path, audio_array, sample_rate)
                return output_path

            await loop.run_in_executor(None, _write)
            
            return output_path
            
        except Exception as e:
            # Fallback to simpler approach if pipeline fails
            # Try using Coqui TTS or other open-source alternatives
            try:
                from TTS.api import TTS
                tts = TTS(model_name="tts_models/hi/fastpitch/v1", progress_bar=False)
                tts.tts_to_file(text=text, file_path=output_path, language="hi")
                return output_path
            except:
                # Final fallback: use pyttsx3 for basic TTS (offline, no internet needed)
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                if language == "hi":
                    # Try to set Hindi voice if available
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if 'hindi' in voice.name.lower() or 'hi' in voice.id.lower():
                            engine.setProperty('voice', voice.id)
                            break
                engine.save_to_file(text, output_path)
                engine.runAndWait()
                return output_path
    
    async def _synthesize_google(
        self, 
        text: str, 
        language: str, 
        output_path: str
    ) -> str:
        """Synthesize using Google Cloud TTS"""
        from google.cloud import texttospeech
        if not settings.GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
            raise RuntimeError("Google credentials not configured")
        
        client = texttospeech.TextToSpeechClient()
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Select voice based on language
        if language == "hi":
            voice = texttospeech.VoiceSelectionParams(
                language_code="hi-IN",
                name="hi-IN-Wavenet-A",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
        else:
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                name="en-IN-Wavenet-D",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        
        return output_path
    
    async def _synthesize_azure(
        self, 
        text: str, 
        language: str, 
        output_path: str
    ) -> str:
        """Synthesize using Azure TTS"""
        import azure.cognitiveservices.speech as speechsdk
        
        speech_config = speechsdk.SpeechConfig(
            subscription=settings.AZURE_SPEECH_KEY,
            region=settings.AZURE_SPEECH_REGION
        )
        
        # Select voice based on language
        if language == "hi":
            speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
        else:
            speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_path
        else:
            raise Exception(f"TTS failed: {result.reason}")
    
    async def _synthesize_openai(
        self, 
        text: str, 
        language: str, 
        output_path: str
    ) -> str:
        """Synthesize using OpenAI TTS"""
        from openai import OpenAI
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        
        response.stream_to_file(output_path)
        return output_path
    
    async def _synthesize_elevenlabs(
        self, 
        text: str, 
        language: str, 
        output_path: str
    ) -> str:
        """Synthesize using ElevenLabs"""
        from elevenlabs import generate, save
        
        audio = generate(
            text=text,
            voice="Bella",  # Choose appropriate voice
            api_key=settings.ELEVENLABS_API_KEY
        )
        
        save(audio, output_path)
        return output_path

# Singleton instance
_tts_service = None

def get_tts_service() -> TTSService:
    """Get or create TTS service instance"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service

def reset_tts_service():
    """Reset TTS service singleton to pick up new configuration"""
    global _tts_service
    _tts_service = None
