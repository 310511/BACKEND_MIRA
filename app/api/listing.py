"""
Product Listing API Endpoints
Handles voice-based product listing with conversation flow
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, Dict, List
import tempfile
import os
from pathlib import Path
import json

from app.ai.stt_service import get_stt_service
from app.ai.nlp_service import get_nlp_service
from app.ai.tts_service import get_tts_service
from app.commerce.factory import get_commerce_backend
from app.config import settings, STTProvider

router = APIRouter(prefix="/api/listing", tags=["listing"])

class ListingQuestion(BaseModel):
    """Question to ask the seller"""
    question: str
    field: str  # Field name: title, description, price, category, vendor
    audio_url: Optional[str] = None  # TTS audio URL for the question

class ListingResponse(BaseModel):
    """Response from listing endpoint"""
    question: Optional[ListingQuestion] = None
    collected_data: Dict
    status: str  # "collecting", "complete", "error"
    message: str
    audio_response: Optional[str] = None

# Required fields for product listing
REQUIRED_FIELDS = ["title", "price"]
OPTIONAL_FIELDS = ["description", "category", "vendor", "tags"]

# Questions in multiple languages
QUESTIONS = {
    "hi": {
        "title": "कृपया अपने उत्पाद का नाम बताएं।",
        "description": "कृपया उत्पाद का विवरण बताएं।",
        "price": "कृपया उत्पाद की कीमत बताएं।",
        "category": "कृपया उत्पाद की श्रेणी बताएं।",
        "vendor": "कृपया विक्रेता का नाम बताएं।",
        "tags": "कृपया उत्पाद के टैग बताएं (वैकल्पिक)।"
    },
    "en": {
        "title": "Please tell me the product name.",
        "description": "Please describe the product.",
        "price": "Please tell me the product price.",
        "category": "Please tell me the product category.",
        "vendor": "Please tell me the vendor name.",
        "tags": "Please tell me product tags (optional)."
    },
    "ta": {
        "title": "தயவுசெய்து உங்கள் தயாரிப்பின் பெயரைச் சொல்லுங்கள்.",
        "description": "தயவுசெய்து தயாரிப்பின் விளக்கத்தைச் சொல்லுங்கள்.",
        "price": "தயவுசெய்து தயாரிப்பின் விலையைச் சொல்லுங்கள்.",
        "category": "தயவுசெய்து தயாரிப்பின் வகையைச் சொல்லுங்கள்.",
        "vendor": "தயவுசெய்து விற்பனையாளரின் பெயரைச் சொல்லுங்கள்.",
        "tags": "தயவுசெய்து தயாரிப்பு குறிச்சொற்களைச் சொல்லுங்கள் (விருப்பமானது)."
    },
    "te": {
        "title": "దయచేసి మీ ఉత్పత్తి పేరు చెప్పండి.",
        "description": "దయచేసి ఉత్పత్తి వివరణ చెప్పండి.",
        "price": "దయచేసి ఉత్పత్తి ధర చెప్పండి.",
        "category": "దయచేసి ఉత్పత్తి వర్గం చెప్పండి.",
        "vendor": "దయచేసి విక్రేత పేరు చెప్పండి.",
        "tags": "దయచేసి ఉత్పత్తి ట్యాగ్‌లు చెప్పండి (ఐచ్ఛికం)."
    }
}

def get_question(field: str, language: str = "hi") -> str:
    """Get question text for a field in the specified language"""
    lang_questions = QUESTIONS.get(language, QUESTIONS["en"])
    return lang_questions.get(field, lang_questions.get("title", "Please provide information."))

def get_next_field(collected: Dict) -> Optional[str]:
    """Get the next field that needs to be collected"""
    # First collect required fields
    for field in REQUIRED_FIELDS:
        if field not in collected or not collected[field]:
            return field
    
    # Then collect optional fields
    for field in OPTIONAL_FIELDS:
        if field not in collected or not collected[field]:
            return field
    
    return None  # All fields collected

async def extract_product_info(text: str, field: str, language: str) -> Optional[str]:
    """
    Extract product information from voice response using NLP
    """
    try:
        nlp_service = get_nlp_service()
        extracted = await nlp_service.extract_field_value(text, field)
        return extracted if extracted else None
    except Exception as e:
        print(f"Error extracting {field}: {e}")
    
    # Fallback: return the text as-is if extraction fails
    return text.strip() if text.strip() else None

@router.post("/start", response_model=ListingResponse)
async def start_listing(
    language: str = Form("hi"),
    output_language: str = Form("hi")
):
    """
    Start a new product listing session
    Returns the first question to ask
    """
    collected_data = {}
    
    # Get first field to collect
    first_field = get_next_field(collected_data)
    if not first_field:
        return ListingResponse(
            question=None,
            collected_data=collected_data,
            status="error",
            message="No fields to collect",
            audio_response=None
        )
    
    question_text = get_question(first_field, output_language)
    
    # Generate TTS for the question
    tts_service = get_tts_service()
    import asyncio
    audio_path = None
    try:
        audio_path = await asyncio.wait_for(
            tts_service.synthesize(question_text, output_language),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        if settings.DEBUG:
            print("[DEBUG] Listing start TTS timed out (60s); returning text only.")
    except Exception as e:
        if settings.DEBUG:
            print(f"[DEBUG] Listing start TTS failed; returning text only. Error: {e}")
    
    # Convert to URL path
    if audio_path and os.path.exists(audio_path):
        filename = os.path.basename(audio_path)
        audio_url = f"/audio/{filename}"
    else:
        audio_url = None
    
    return ListingResponse(
        question=ListingQuestion(
            question=question_text,
            field=first_field,
            audio_url=audio_url
        ),
        collected_data=collected_data,
        status="collecting",
        message=f"Please provide {first_field}",
        audio_response=audio_url
    )

@router.post("/answer", response_model=ListingResponse)
async def answer_question(
    file: UploadFile = File(...),
    field: str = Form(...),  # Field being answered
    collected_data: str = Form("{}"),  # JSON string of collected data so far
    language: str = Form("hi"),  # Input language for STT
    output_language: str = Form("hi")  # Output language for TTS
):
    """
    Process seller's voice answer to a question
    Updates collected_data and returns next question or completion
    """
    import json
    
    # Parse collected data
    try:
        collected = json.loads(collected_data) if collected_data else {}
    except:
        collected = {}
    
    # Save uploaded audio file (preserve correct format/extension)
    try:
        content = await file.read()
    except Exception:
        content = b""

    content_type = (getattr(file, "content_type", None) or "").lower()
    filename = getattr(file, "filename", None) or ""

    suffix = ""
    try:
        suffix = Path(filename).suffix or ""
    except Exception:
        suffix = ""

    if not suffix:
        if "webm" in content_type:
            suffix = ".webm"
        elif "ogg" in content_type:
            suffix = ".ogg"
        elif "mpeg" in content_type or "mp3" in content_type:
            suffix = ".mp3"
        elif "wav" in content_type:
            suffix = ".wav"
        else:
            if content.startswith(b"RIFF") and b"WAVE" in content[:16]:
                suffix = ".wav"
            elif content.startswith(b"OggS"):
                suffix = ".ogg"
            elif content.startswith(b"ID3") or content[:2] == b"\xFF\xFB":
                suffix = ".mp3"
            elif content[:4] == b"\x1A\x45\xDF\xA3":
                suffix = ".webm"
            else:
                suffix = ".wav"

    temp_audio = tempfile.mktemp(suffix=suffix)
    try:
        with open(temp_audio, "wb") as f:
            f.write(content)

        if settings.DEBUG:
            try:
                print(
                    f"[DEBUG] Listing answer upload: field='{field}', "
                    f"filename='{getattr(file, 'filename', None)}', "
                    f"content_type='{getattr(file, 'content_type', None)}', "
                    f"suffix='{suffix}', bytes={len(content)}, "
                    f"language='{language}', output_language='{output_language}'"
                )
            except Exception:
                pass
        
        # Transcribe audio with timeout
        import asyncio
        
        stt_service = get_stt_service()
        
        # Set appropriate timeout based on STT provider
        if settings.STT_PROVIDER == STTProvider.WHISPER_LOCAL:
            timeout_seconds = 300.0  # 5 min for Whisper
        elif settings.STT_PROVIDER == STTProvider.ASSEMBLYAI:
            timeout_seconds = 180.0  # 3 min for AssemblyAI
        else:
            timeout_seconds = 120.0  # 2 min for others
        
        try:
            transcription, detected_lang = await asyncio.wait_for(
                stt_service.transcribe(temp_audio, language=language or None),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"STT transcription timed out after {timeout_seconds} seconds. Try using a shorter audio clip."
            )
        except Exception as e:
            error_msg = str(e)
            raise HTTPException(
                status_code=400,
                detail=f"STT transcription failed: {error_msg}"
            )

        if not transcription or not str(transcription).strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No speech detected in the recording. "
                    "Please allow microphone access, speak clearly for 2–5 seconds, and try again."
                ),
            )
        
        # Extract field value from transcription
        field_value = await extract_product_info(transcription, field, detected_lang or language)
        
        # Store the collected value
        if field_value:
            collected[field] = field_value
        
        # Check if we need to collect more fields
        next_field = get_next_field(collected)
        
        if next_field:
            # More fields to collect
            question_text = get_question(next_field, output_language)
            
            # Generate TTS for next question
            tts_service = get_tts_service()
            audio_path = None
            try:
                audio_path = await asyncio.wait_for(
                    tts_service.synthesize(question_text, output_language),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                if settings.DEBUG:
                    print("[DEBUG] Listing next-question TTS timed out (60s); returning text only.")
            except Exception as e:
                if settings.DEBUG:
                    print(f"[DEBUG] Listing next-question TTS failed; returning text only. Error: {e}")
            
            if audio_path and os.path.exists(audio_path):
                filename = os.path.basename(audio_path)
                audio_url = f"/audio/{filename}"
            else:
                audio_url = None
            
            return ListingResponse(
                question=ListingQuestion(
                    question=question_text,
                    field=next_field,
                    audio_url=audio_url
                ),
                collected_data=collected,
                status="collecting",
                message=f"Collected {field}. Please provide {next_field}.",
                audio_response=audio_url
            )
        else:
            # All fields collected, confirm and create product
            if output_language == "hi":
                confirmation_text = "सभी जानकारी एकत्र हो गई है। उत्पाद बनाने के लिए तैयार है।"
            elif output_language == "ta":
                confirmation_text = "அனைத்து தகவல்களும் சேகரிக்கப்பட்டுள்ளன. தயாரிப்பை உருவாக்க தயாராக உள்ளது."
            elif output_language == "te":
                confirmation_text = "అన్ని సమాచారం సేకరించబడింది. ఉత్పత్తిని సృష్టించడానికి సిద్ధంగా ఉంది."
            else:
                confirmation_text = "All information collected. Ready to create product."
            
            # Generate confirmation TTS
            tts_service = get_tts_service()
            audio_path = None
            try:
                audio_path = await asyncio.wait_for(
                    tts_service.synthesize(confirmation_text, output_language),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                if settings.DEBUG:
                    print("[DEBUG] Listing confirmation TTS timed out (60s); returning text only.")
            except Exception as e:
                if settings.DEBUG:
                    print(f"[DEBUG] Listing confirmation TTS failed; returning text only. Error: {e}")
            
            if audio_path and os.path.exists(audio_path):
                filename = os.path.basename(audio_path)
                audio_url = f"/audio/{filename}"
            else:
                audio_url = None
            
            return ListingResponse(
                question=None,
                collected_data=collected,
                status="complete",
                message="All information collected. Ready to create product.",
                audio_response=audio_url
            )
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except:
                pass

@router.post("/create")
async def create_product_from_listing(
    collected_data: str = Form(...)  # JSON string of collected data
):
    """
    Create the product in Shopify using collected data
    """
    import json
    
    try:
        collected = json.loads(collected_data) if collected_data else {}
    except:
        raise HTTPException(status_code=400, detail="Invalid collected_data JSON")
    
    # Validate required fields
    for field in REQUIRED_FIELDS:
        if field not in collected or not collected[field]:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    # Parse price (handle various formats)
    price_str = str(collected.get("price", "0")).strip()
    # Remove currency symbols and extract number
    import re
    price_match = re.search(r'[\d.]+', price_str.replace(",", ""))
    if price_match:
        price = float(price_match.group())
    else:
        raise HTTPException(status_code=400, detail="Invalid price format")
    
    # Create product in Shopify
    commerce_backend = get_commerce_backend()
    
    # Parse tags if provided
    tags = None
    if collected.get("tags"):
        tags_str = str(collected.get("tags", ""))
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    
    product = await commerce_backend.create_product(
        title=collected.get("title", ""),
        description=collected.get("description"),
        price=price,
        currency="INR",
        product_type=collected.get("category"),
        vendor=collected.get("vendor"),
        tags=tags,
        images=None  # Images can be added later
    )
    
    return {
        "status": "success",
        "message": "Product created successfully",
        "product": product
    }
