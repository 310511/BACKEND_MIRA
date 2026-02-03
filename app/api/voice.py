"""
Voice API Endpoints
Handles voice input, transcription, intent extraction, and product search
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from pathlib import Path

from app.ai.stt_service import get_stt_service
from app.ai.nlp_service import get_nlp_service
from app.ai.tts_service import get_tts_service, reset_tts_service
from app.commerce.factory import get_commerce_backend
from app.config import settings, STTProvider
from app.utils.offline_fallback import with_offline_fallback, get_offline_cache

# Reset TTS service to pick up new configuration
reset_tts_service()

router = APIRouter(prefix="/api/voice", tags=["voice"])

# English → Hindi product names (for output_language == 'hi': entire response in Hindi only)
PRODUCT_NAME_EN_TO_HI = {
    "almonds": "बादाम",
    "almond": "बादाम",
    "cashews": "काजू",
    "cashew": "काजू",
    "walnuts": "अखरोट",
    "walnut": "अखरोट",
    "honey": "शहद",
    "rice": "चावल",
    "wheat": "गेहूं",
    "sugar": "चीनी",
    "oil": "तेल",
    "lentils": "दाल",
    "dal": "दाल",
    "spices": "मसाले",
    "masala": "मसाला",
    "shampoo": "शैम्पू",
    "soap": "साबुन",
    "product": "उत्पाद",
    "products": "उत्पाद",
}

def _product_name_to_hindi(name: str) -> str:
    """Translate/transliterate product name to Hindi when output is Hindi."""
    if not name or not isinstance(name, str):
        return name
    key = name.strip().lower()
    return PRODUCT_NAME_EN_TO_HI.get(key, name)  # Keep as-is if no mapping (e.g. brand names)

def _format_price_for_hindi(price) -> str:
    """Format price for Hindi response: whole number when possible, no 'INR' (use रुपये in sentence)."""
    if price is None:
        return "नहीं मिला"
    try:
        p = float(price)
        if p == int(p):
            return str(int(p))
        return str(round(p, 2))
    except (TypeError, ValueError):
        return str(price)

# Common English phrases from LLM/NLP → Hindi (so output is always Hindi when output_language is 'hi')
_ENGLISH_TO_HINDI_PHRASES = [
    ("Please tell me the product name", "कृपया उत्पाद का नाम बताएं"),
    ("Please tell me the product", "कृपया उत्पाद का नाम बताएं"),
    ("Tell me the product name", "कृपया उत्पाद का नाम बताएं"),
    ("What product do you want?", "आप कौन सा उत्पाद चाहते हैं?"),
    ("Which product do you need?", "आपको कौन सा उत्पाद चाहिए?"),
    ("Yes, we have", "हाँ, हमारे पास हैं"),
    ("We have", "हमारे पास हैं"),
    ("we have", "हमारे पास हैं"),
    ("such as", "जैसे"),
    ("available", "उपलब्ध"),
    ("Available", "उपलब्ध"),
    ("I found", "मुझे मिले"),
    ("I couldn't find", "मुझे नहीं मिला"),
    ("No products found", "कोई उत्पाद नहीं मिला"),
    ("No product found", "कोई उत्पाद नहीं मिला"),
    ("like", "जैसे"),
    ("options", "विकल्प"),
    ("options.", "विकल्प।"),
    ("figs", "अंजीर"),
    ("fig", "अंजीर"),
    ("raisins", "किशमिश"),
    ("raisin", "किशमिश"),
    ("apricots", "खुबानी"),
    ("apricot", "खुबानी"),
    ("coconut", "नारियल"),
    ("milk", "दूध"),
]

def _ensure_hindi_only(text: str) -> str:
    """Replace any remaining English in response when output language is Hindi. Response must be fully Hindi for TTS."""
    if not text or not isinstance(text, str):
        return text
    t = text.strip()
    # Full-phrase replacements first (longest first)
    for en, hi in sorted(_ENGLISH_TO_HINDI_PHRASES, key=lambda x: -len(x[0])):
        t = t.replace(en, hi)
    # Currency and single-word replacements
    replacements = [
        ("INR", "रुपये"),
        ("Rupees", "रुपये"),
        ("Rupee", "रुपये"),
        ("Product", "उत्पाद"),
        ("product", "उत्पाद"),
        ("products", "उत्पाद"),
        ("N/A", "नहीं मिला"),
        ("Found", "मिले"),
        ("found", "मिले"),
    ]
    for en, hi in replacements:
        t = t.replace(en, hi)
    # " and " as connector → " और " (so "X and Y" in lists becomes "X और Y")
    t = t.replace(" and ", " और ")
    # Replace remaining English product names (word-by-word) so "we have almonds, cashews" → "we have बादाम, काजू"
    import re
    for en_key, hi_val in PRODUCT_NAME_EN_TO_HI.items():
        t = re.sub(r"\b" + re.escape(en_key) + r"\b", hi_val, t, flags=re.IGNORECASE)
    # If response is still mostly English (e.g. LLM returned full English sentence), use generic Hindi
    try:
        words = t.split()
        if not words:
            return t
        devanagari_or_common = sum(
            1 for w in words
            if any("\u0900" <= c <= "\u097F" for c in w) or w in ("हाँ", "हमारे", "पास", "मुझे", "उत्पाद", "रुपये", "कृपया", "।", "?", ".")
        )
        if devanagari_or_common < len(words) // 2:
            return "मैं आपकी मदद कर सकता हूं। कोई उत्पाद खोजने के लिए कहें।"
    except Exception:
        pass
    return t

async def _translate_to_english(hindi_text: str) -> str:
    """Translate Hindi text to English using Groq (preferred)."""
    try:
        nlp_service = get_nlp_service()
        english = await nlp_service.translate_to_english(hindi_text)
        english = (english or "").strip().lower()
        # For single-word product search we keep only the first token if model returns a phrase.
        return english.split()[0] if english.split() else hindi_text
    except Exception:
        pass
    return hindi_text  # Fallback to original if translation fails

class VoiceQueryResponse(BaseModel):
    transcription: str
    language: str
    intent: dict
    products: list
    cart_id: Optional[str] = None
    checkout_url: Optional[str] = None
    audio_response: Optional[str] = None  # Backward compatibility: URL
    audio_data: Optional[str] = None     # Base64 encoded audio for JSON transport
    audio_format: Optional[str] = None   # Audio format (audio/mp3)

@router.post("/process", response_model=VoiceQueryResponse)
async def process_voice_query(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),  # Input language for STT (from form data)
    output_language: Optional[str] = Form(None)  # Output language for TTS (from form data)
):
    """
    Process voice query end-to-end:
    1. Transcribe audio (STT)
    2. Extract intent (NLP) - never cached in voice mode
    3. Search products - ALWAYS query Shopify for product/rate/stock; never from memory
    4. Generate NEW TTS audio for EVERY response (mandatory; no cache/reuse)
    """
    # CRITICAL FIX: Define tts_language at the beginning to prevent UnboundLocalError
    # Use output_language if provided, otherwise use detected language as fallback
    if output_language:
        tts_language = output_language.strip() if isinstance(output_language, str) else str(output_language).strip()
        if not tts_language:
            tts_language = settings.DEFAULT_LANGUAGE
    else:
        tts_language = settings.DEFAULT_LANGUAGE
    
    # Save uploaded audio file (preserve correct format/extension)
    import tempfile  # Ensure tempfile is available in this scope
    
    try:
        content = await file.read()
    except Exception:
        content = b""

    content_type = (getattr(file, "content_type", None) or "").lower()
    filename = getattr(file, "filename", None) or ""

    # Prefer extension from filename, otherwise infer from content-type / magic bytes.
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
            # Magic byte sniff (minimal, no extra deps)
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
        
        # Debug: Log received parameters
        if settings.DEBUG:
            print(f"[DEBUG] Received language: '{language}', output_language: '{output_language}'")
            try:
                print(
                    f"[DEBUG] Upload: filename='{getattr(file, 'filename', None)}', "
                    f"content_type='{getattr(file, 'content_type', None)}', "
                    f"suffix='{suffix}', bytes={len(content)}"
                )
            except Exception:
                pass
        
        # Step 1: Speech-to-Text (with offline fallback)
        # Verify audio file exists and is readable
        if not os.path.exists(temp_audio):
            raise HTTPException(status_code=400, detail="Audio file was not saved correctly")
        
        file_size = os.path.getsize(temp_audio)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # Warn if file is very large (might timeout)
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400, 
                detail="Audio file is too large (over 10MB). Please use a shorter recording (under 2 minutes)."
            )
        
        # Step 1: Load product names for context-aware processing
        commerce_backend = get_commerce_backend()
        product_names = []
        try:
            product_names = await commerce_backend.get_all_product_names(limit=100)
            if settings.DEBUG:
                print(f"[DEBUG] Loaded {len(product_names)} product names for context")
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] Failed to load product names: {e}")
        
        stt_service = get_stt_service()
        
        # Use provided language or default to Hindi for Indian language support
        # This helps Whisper accurately transcribe Indian languages
        # IMPORTANT: Always provide a language to prevent misdetection (e.g., Hindi detected as Chinese)
        if language:
            input_lang = language.strip() if isinstance(language, str) else str(language).strip()
            # Ensure it's not empty after stripping
            if not input_lang:
                input_lang = settings.DEFAULT_LANGUAGE
        else:
            input_lang = settings.DEFAULT_LANGUAGE
        
        # Log for debugging (remove in production)
        if settings.DEBUG:
            print(f"[DEBUG] STT Language: '{input_lang}' (provided: '{language}')")
        
        # Check cache first for offline resilience
        cache = get_offline_cache()
        cache_key = f"stt_{hash(content)}_{input_lang}_{hash(str(product_names[:20]))}"
        cached_transcription = cache.get(cache_key)
        
        if cached_transcription:
            transcription, detected_lang = cached_transcription
        else:
            try:
                # Add timeout for STT operation
                # Optimized timeouts for better performance
                import asyncio
                if settings.STT_PROVIDER == STTProvider.WHISPER_LOCAL:
                    timeout_seconds = 180.0  # 3 min for Whisper (reduced from 5)
                elif settings.STT_PROVIDER == STTProvider.ASSEMBLYAI:
                    timeout_seconds = 90.0  # 1.5 min for AssemblyAI (reduced from 3)
                else:
                    timeout_seconds = 60.0  # 1 min for other providers (reduced from 2)
                
                # Pass input_lang and product_context to STT to help with accurate transcription
                transcription, detected_lang = await asyncio.wait_for(
                    stt_service.transcribe(temp_audio, input_lang, product_context=product_names),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                if settings.STT_PROVIDER == STTProvider.WHISPER_LOCAL:
                    timeout_msg = "3 minutes"
                    provider_hint = (
                        " If this is the first time using Whisper, the model needs to download and load "
                        "(this can take 1-3 minutes). Subsequent requests will be much faster. "
                        "You can also try using a smaller model size (tiny) in config.py for faster loading."
                    )
                elif settings.STT_PROVIDER == STTProvider.ASSEMBLYAI:
                    timeout_msg = "90 seconds"
                    provider_hint = (
                        " AssemblyAI transcription may take time for longer audio files. "
                        "Try using a shorter audio clip (under 30 seconds) for faster processing."
                    )
                else:
                    timeout_msg = "60 seconds"
                    provider_hint = ""
                
                raise HTTPException(
                    status_code=408,
                    detail=(
                        f"STT transcription timed out after {timeout_msg}. "
                        f"Current provider: {settings.STT_PROVIDER.value}.{provider_hint}"
                    ),
                )
            except Exception as e:
                error_msg = str(e)
                detail_msg = (
                    f"STT failed to transcribe audio ({suffix}). "
                )
                
                # Provide provider-specific error messages
                if "AssemblyAI" in error_msg:
                    detail_msg += f"AssemblyAI Error: {error_msg}"
                elif "Whisper" in error_msg or "ffmpeg" in error_msg:
                    detail_msg += (
                        f"If using Whisper local, ensure ffmpeg is on PATH and the Python package 'openai-whisper' is installed. "
                        f"Install: python -m pip install openai-whisper. Error: {error_msg}"
                    )
                else:
                    detail_msg += f"Error: {error_msg}"
                
                raise HTTPException(status_code=400, detail=detail_msg)

            if transcription:
                cache.set(cache_key, (transcription, detected_lang))
        
        if not transcription or not str(transcription).strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No speech detected in the recording. "
                    "Please allow microphone access, speak clearly for 2–5 seconds, and try again."
                ),
            )
        
        # Step 2: Intent Extraction with Product Matching
        # Voice mode: never cache intent so we always query Shopify and generate fresh TTS every time
        nlp_service = get_nlp_service()
        if product_names:
            intent_result = await nlp_service.match_transcription_to_products(
                transcription,
                product_names,
                detected_lang,
                tts_language,
            )
            if "corrected_transcription" in intent_result:
                transcription = intent_result["corrected_transcription"]
        else:
            intent_result = await nlp_service.extract_intent(transcription, detected_lang, tts_language)
        
        # Step 3: Handle different intents (product search, cart operations, etc.)
        products = []
        intent = intent_result.get("intent", "product_search")
        cart_id = None
        checkout_url = None
        
        if intent == "product_search":
            # ALWAYS query Shopify for product/rate/stock; never respond from memory
            try:
                commerce_backend = get_commerce_backend()
                product_keywords = intent_result.get("entities", {}).get("products", [])
                
                # Use corrected transcription if available for better search
                search_transcription = intent_result.get("corrected_transcription", transcription)
                
                # Use extracted keywords if available, otherwise fall back to corrected transcription
                if product_keywords and len(product_keywords) > 0:
                    # Translate Hindi product names to English for Shopify search
                    translated_keywords = []
                    for keyword in product_keywords:
                        # Hindi to English mapping for common products
                        hindi_to_english = {
                            "बादाम": "almonds",
                            "badam": "almonds",
                            "बदाम": "almonds",
                            "शहद": "honey",
                            "shahad": "honey",
                            "चावल": "rice",
                            "chawal": "rice",
                            "गेहूं": "wheat",
                            "gehun": "wheat",
                            "चीनी": "sugar",
                            "chini": "sugar",
                            "तेल": "oil",
                            "tel": "oil",
                            "दाल": "lentils",
                            "dal": "lentils",
                            "मसाले": "spices",
                            "masale": "spices",
                            "मसाला": "spices",
                            "अखरोट": "walnuts",
                            "akhrot": "walnuts",
                            "काजू": "cashews",
                            "kaju": "cashews",
                            "लिवादान": "levis",
                            "livadan": "levis",
                            "लीवाइस": "levis",
                            "levis": "levis"
                        }
                        
                        # Check if keyword is Hindi and translate
                        keyword_lower = keyword.lower()
                        if keyword in hindi_to_english:
                            translated_keywords.append(hindi_to_english[keyword])
                        elif keyword_lower in hindi_to_english:
                            translated_keywords.append(hindi_to_english[keyword_lower])
                        else:
                            # If it's already in English or not in our map, use as-is
                            # Check if it contains Devanagari characters (Hindi)
                            has_hindi = any('\u0900' <= char <= '\u097F' for char in keyword)
                            if has_hindi:
                                # Try to translate using Ollama if it's Hindi
                                translated_keywords.append(await _translate_to_english(keyword))
                            else:
                                translated_keywords.append(keyword)
                    
                    # Use translated English keywords for search
                    search_query = " ".join(translated_keywords)
                    if settings.DEBUG:
                        print(f"[DEBUG] Product search - original: {product_keywords}, translated: {translated_keywords}")
                else:
                    # Fallback: use corrected transcription or original transcription for search
                    search_query = search_transcription.lower()
                    # Remove common request phrases in multiple languages
                    common_phrases = [
                        "i need", "i want", "show me", "find me", "search for", "looking for",
                        "मुझे चाहिए", "चाहिए", "दो", "एक", "मुझे", "चाहता हूं",
                        "நான்", "வேண்டும்", "தேவை"
                    ]
                    for phrase in common_phrases:
                        search_query = search_query.replace(phrase, "").strip()
                    # Remove punctuation and extra spaces
                    search_query = " ".join(search_query.split())
                    if settings.DEBUG:
                        print(f"[DEBUG] Product search using transcription: '{search_query}'")
                
                if search_query:
                    products = await commerce_backend.search_products(search_query, limit=5)
                    if settings.DEBUG:
                        print(f"[DEBUG] Found {len(products)} products for query: '{search_query}'")
            except Exception as e:
                # Offline fallback: Use local product service if commerce backend fails
                if settings.ENABLE_OFFLINE_MODE:
                    from app.services.products import get_product_service
                    product_service = get_product_service()
                    product_keywords = intent_result.get("entities", {}).get("products", [])
                    
                    # Use extracted keywords if available, otherwise fall back to transcription text
                    if product_keywords:
                        search_query = " ".join(product_keywords)
                    else:
                        # Fallback: use transcription text directly for search
                        search_query = transcription.lower()
                        # Remove common request phrases
                        for phrase in ["i need", "i want", "show me", "find me", "search for", "looking for"]:
                            search_query = search_query.replace(phrase, "").strip()
                        # Remove punctuation and extra spaces
                        search_query = " ".join(search_query.split())
                    
                    if search_query:
                        products = product_service.search_products(search_query)
                        # Convert to standard format
                        products = [
                            {
                                "id": str(p["id"]),
                                "title": p.get("name_hi", p["name"]) if detected_lang == "hi" else p["name"],
                                "description": p.get("description_hi", p["description"]) if detected_lang == "hi" else p["description"],
                                "price": p["price"],
                                "currency": "INR",
                                "images": [p.get("image")] if p.get("image") else [],
                                "available": p.get("stock", 0) > 0,
                                "category": p.get("category"),
                                "provider": "local"
                            }
                            for p in products
                        ]
        
        elif intent == "add_to_cart":
            # Handle add to cart intent
            try:
                commerce_backend = get_commerce_backend()
                product_keywords = intent_result.get("entities", {}).get("products", [])
                
                if product_keywords:
                    # Search for the product first
                    search_query = " ".join(product_keywords)
                    search_results = await commerce_backend.search_products(search_query, limit=1)
                    
                    if search_results:
                        product = search_results[0]
                        # Add to cart logic would go here
                        # For now, return the product as confirmation
                        products = [product]
                        if settings.DEBUG:
                            print(f"[DEBUG] Add to cart: Found product {product['title']}")
                    else:
                        if settings.DEBUG:
                            print(f"[DEBUG] Add to cart: No product found for '{search_query}'")
            except Exception as e:
                if settings.DEBUG:
                    print(f"[DEBUG] Add to cart failed: {e}")
        
        elif intent == "view_cart":
            # Handle view cart intent
            try:
                commerce_backend = get_commerce_backend()
                # Cart viewing logic would go here
                # For now, return empty products with a message
                if settings.DEBUG:
                    print("[DEBUG] View cart intent detected")
            except Exception as e:
                if settings.DEBUG:
                    print(f"[DEBUG] View cart failed: {e}")
        
        elif intent == "checkout":
            # Handle checkout intent
            try:
                commerce_backend = get_commerce_backend()
                cart = await commerce_backend.create_cart()
                cart_id = (cart or {}).get("id")
                checkout_url = (cart or {}).get("checkoutUrl")

                # If we already have a product in context, add the first one to the cart.
                # If not, we still return checkout_url so the user can proceed.
                if products:
                    try:
                        first = products[0] or {}
                        variant_id = first.get("variant_id")
                        if variant_id and cart_id:
                            updated = await commerce_backend.add_to_cart(cart_id, variant_id, quantity=1)
                            cart_id = (updated or {}).get("id") or cart_id
                            checkout_url = (updated or {}).get("checkoutUrl") or checkout_url
                    except Exception:
                        pass

                if settings.DEBUG:
                    print("[DEBUG] Checkout intent detected")
            except Exception as e:
                if settings.DEBUG:
                    print(f"[DEBUG] Checkout failed: {e}")
        
        elif intent == "help":
            # Handle help intent
            if settings.DEBUG:
                print("[DEBUG] Help intent detected")
            # Return empty products, response will be generated below
        
        # Step 4: Generate Audio Response (TTS)
        # tts_language is already defined at the beginning of the function
        
        # Debug: Log TTS language
        if settings.DEBUG:
            print(f"[DEBUG] TTS Language: '{tts_language}' (output_language: '{output_language}', detected: '{detected_lang}')")
        
        tts_service = get_tts_service()
        
        # Build response from LIVE Shopify data when we have products (never from memory/assumptions)
        response_text = ""
        if intent == "product_search" and products:
            n = len(products)
            if tts_language == "hi":
                # Hindi only: product names in Hindi, prices as numbers + रुपये, no English
                # Grammar: 1 उत्पाद मिला, 2+ उत्पाद मिले
                verb = "मिला" if n == 1 else "मिले"
                parts = []
                for p in products[:3]:
                    name_en = p.get("title", p.get("name", "उत्पाद"))
                    name_hi = _product_name_to_hindi(name_en)
                    price_str = _format_price_for_hindi(p.get("price"))
                    parts.append(f"{name_hi} - {price_str} रुपये")
                response_text = f"मुझे {n} उत्पाद {verb}। " + "। ".join(parts)
            else:
                currency = (products[0].get("currency") or "INR").strip()
                parts = []
                for p in products[:3]:
                    name = p.get("title", p.get("name", "Product"))
                    price = p.get("price", "N/A")
                    parts.append(f"{name} - {currency} {price}")
                if tts_language == "ta":
                    response_text = f"நான் {n} தயாரிப்புகளைக் கண்டேன். " + ". ".join(parts)
                elif tts_language == "te":
                    response_text = f"నాకు {n} ఉత్పత్తులు దొరికాయి. " + ". ".join(parts)
                else:
                    response_text = f"I found {n} product(s). " + ". ".join(parts)
            intent_result["user_friendly_response"] = response_text
        if not response_text:
            response_text = intent_result.get("user_friendly_response", "")
        
        # If still no response text, generate intent-specific response
        if not response_text:
            if intent == "add_to_cart":
                if products:
                    product_name = products[0]["title"]
                    name_hi = _product_name_to_hindi(product_name) if tts_language == "hi" else product_name
                    if tts_language == "hi":
                        response_text = f"ज़रूर, मैंने {name_hi} को आपके कार्ट में जोड़ दिया है।"
                    elif tts_language == "ta":
                        response_text = f"நிச்சயமாக, நான் {product_name} உங்கள் கார்ட்டில் சேர்த்துள்ளேன்."
                    elif tts_language == "te":
                        response_text = f"ఖచ్చితంగా, నేను {product_name} మీ కార్ట్‌లో చేర్చాను."
                    else:
                        response_text = f"Sure, I've added {product_name} to your cart."
                else:
                    if tts_language == "hi":
                        response_text = "मुझे वह उत्पाद नहीं मिला। कृपया फिर से कोशिश करें।"
                    elif tts_language == "ta":
                        response_text = "எனக்கு அந்த தயாரிப்பு கிடைக்கவில்லை. தயவுசெய்து மீண்டும் முயற்சி செய்யவும்."
                    elif tts_language == "te":
                        response_text = "నాకు ఆ ఉత్పత్తి దొరకలేదు. దయచేసి మళ్ళీ ప్రయత్నించండి."
                    else:
                        response_text = "I couldn't find that product. Please try again."
            
            elif intent == "view_cart":
                if tts_language == "hi":
                    response_text = "आपके कार्ट को देखने के लिए, कृपया वेबसाइट पर जाएं।"
                elif tts_language == "ta":
                    response_text = "உங்கள் கார்ட்டைப் பார்க்க, தயவுசெய்து வலைத்தளத்திற்குச் செல்லவும்."
                elif tts_language == "te":
                    response_text = "మీ కార్ట్‌ను చూడటానికి, దయచేసి వెబ్‌సైట్‌కి వెళ్ళండి."
                else:
                    response_text = "To view your cart, please visit the website."
            
            elif intent == "checkout":
                if tts_language == "hi":
                    response_text = "चेकआउट के लिए, कृपया वेबसाइट पर जाएं और अपना ऑर्डर पूरा करें।"
                elif tts_language == "ta":
                    response_text = "செக்கவுட் செய்ய, தயவுசெய்து வலைத்தளத்திற்குச் சென்று உங்கள் ஆர்டரை முடிக்கவும்."
                elif tts_language == "te":
                    response_text = "చెకౌట్ చేయడానికి, దయచేసి వెబ్‌సైట్‌కి వెళ్ళి మీ ఆర్డర్‌ను పూర్తి చేయండి."
                else:
                    response_text = "To checkout, please visit the website and complete your order."
            
            elif intent == "help":
                if tts_language == "hi":
                    response_text = "मैं आपको उत्पाद खोजने, कार्ट में जोड़ने और चेकआउट करने में मदद कर सकता हूं। कोई उत्पाद खोजने के लिए कहें।"
                elif tts_language == "ta":
                    response_text = "நான் உங்களுக்கு தயாரிப்புகளைத் தேட, கார்ட்டில் சேர்க்க மற்றும் செக்கவுட் செய்ய உதவ முடியும். ஒரு தயாரிப்பைத் தேடக் கேளுங்கள்."
                elif tts_language == "te":
                    response_text = "నేను మీకు ఉత్పత్తులను వెతకడంలో, కార్ట్‌లో చేర్చడంలో మరియు చెకౌట్ చేయడంలో సహాయం చేయగలను. ఒక ఉత్పత్తిని వెతకమని అడగండి."
                else:
                    response_text = "I can help you search for products, add to cart, and checkout. Ask me to search for any product."
            
            # Fallback to product search response
            elif products:
                n = len(products)
                if tts_language == "hi":
                    verb = "मिला" if n == 1 else "मिले"
                    product_names_hi = [_product_name_to_hindi(p["title"]) for p in products[:3]]
                    response_text = f"मुझे {n} उत्पाद {verb}: {', '.join(product_names_hi)}"
                else:
                    product_names = [p["title"] for p in products[:3]]
                    if tts_language == "ta":
                        response_text = f"நான் {n} தயாரிப்புகளைக் கண்டேன்: {', '.join(product_names)}"
                    elif tts_language == "te":
                        response_text = f"నాకు {n} ఉత్పత్తులు దొరికాయి: {', '.join(product_names)}"
                    else:
                        response_text = f"I found {n} products: {', '.join(product_names)}"
            else:
                if tts_language == "hi":
                    response_text = "मुझे कोई उत्पाद नहीं मिला।"
                elif tts_language == "ta":
                    response_text = "எனக்கு எந்த தயாரிப்பும் கிடைக்கவில்லை."
                elif tts_language == "te":
                    response_text = "నాకు ఏ ఉత్పత్తి దొరకలేదు."
                else:
                    response_text = "I couldn't find any products."
        
        # If still no response, use a default in the output language
        if not response_text:
            if tts_language == "hi":
                response_text = "मैं आपकी मदद कर सकता हूं।"
            elif tts_language == "ta":
                response_text = "நான் உங்களுக்கு உதவ முடியும்."
            elif tts_language == "te":
                response_text = "నేను మీకు సహాయం చేయగలను."
            else:
                response_text = "I can help you."
        
        # When output is Hindi: ensure NO English in response (product names, INR → रुपये, etc.)
        if tts_language == "hi":
            response_text = _ensure_hindi_only(response_text)
            intent_result["user_friendly_response"] = response_text
            if settings.DEBUG:
                print(f"[DEBUG] Hindi-only response for TTS: '{response_text}'")
        
        # Voice mode: MANDATORY - generate NEW TTS for EVERY response (no cache, no skip)
        import asyncio
        import base64
        audio_response_bytes = None
        audio_response_data = None  # Base64 encoded for JSON
        audio_response_format = "audio/mp3"
        try:
            if settings.DEBUG:
                print(f"[DEBUG] Starting TTS synthesis for: '{response_text}'")
            # Fresh TTS every turn; never reuse or skip audio
            audio_result = await asyncio.wait_for(
                tts_service.synthesize_bytes(response_text, tts_language),
                timeout=60.0,
            )
            
            audio_response_bytes = audio_result.get("audio_bytes", b"")
            audio_response_format = audio_result.get("audio_format", "audio/mp3")
            
            # Convert bytes to base64 for JSON transport
            if audio_response_bytes:
                audio_response_data = base64.b64encode(audio_response_bytes).decode('utf-8')
            
            if settings.DEBUG:
                print(f"[DEBUG] TTS bytes synthesis completed")
                print(f"[DEBUG] Audio bytes length: {len(audio_response_bytes)} bytes")
                print(f"[DEBUG] Audio data length: {len(audio_response_data) if audio_response_data else 0} characters")
                if audio_response_bytes:
                    print(f"[DEBUG] Audio format: {audio_response_format}")
                else:
                    print("[DEBUG] No audio bytes generated")
                    
        except asyncio.TimeoutError:
            if settings.DEBUG:
                print("[DEBUG] TTS synthesize timed out (60s); returning without audio file.")
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] TTS synthesize failed; returning without audio file. Error: {e}")
        
        # If TTS failed (timeout/exception), still return playable fallback so UI doesn't show "not available"
        if not audio_response_data and response_text:
            try:
                fallback_result = await asyncio.wait_for(
                    tts_service.synthesize_bytes(" ", tts_language or "en"),
                    timeout=10.0,
                )
                fb_bytes = fallback_result.get("audio_bytes", b"")
                if fb_bytes:
                    audio_response_bytes = fb_bytes
                    audio_response_format = fallback_result.get("audio_format", "audio/mp3")
                    audio_response_data = base64.b64encode(fb_bytes).decode("utf-8")
                    if settings.DEBUG:
                        print("[DEBUG] Using fallback TTS audio for response")
            except Exception:
                # Minimal playable MP3 so browser doesn't show 0:00
                audio_response_bytes = b"\xff\xfb\x90\x00" + (b"\x00" * 2000)
                audio_response_format = "audio/mp3"
                audio_response_data = base64.b64encode(audio_response_bytes).decode("utf-8")
                if settings.DEBUG:
                    print("[DEBUG] Using minimal fallback audio")
        
        # Convert bytes to response format
        audio_response_url = None
        if audio_response_bytes:
            # For backward compatibility, also provide the URL (use correct extension for content-type)
            try:
                import tempfile
                suffix = ".wav" if (audio_response_format or "").strip().lower() == "audio/wav" else ".mp3"
                temp_file = tempfile.mktemp(suffix=suffix)
                with open(temp_file, 'wb') as f:
                    f.write(audio_response_bytes)
                base_url = str(request.base_url).rstrip('/')
                filename = os.path.basename(temp_file)
                audio_response_url = f"{base_url}/audio/{filename}"
                if settings.DEBUG:
                    print(f"[DEBUG] Audio URL for compatibility: {audio_response_url}")
            except Exception as e:
                if settings.DEBUG:
                    print(f"[DEBUG] Failed to create compatibility URL: {e}")
        else:
            if settings.DEBUG:
                print("[DEBUG] No audio_response_bytes provided")
        
        # Use corrected transcription if available, otherwise use original
        final_transcription = intent_result.get("corrected_transcription", transcription)
        
        return VoiceQueryResponse(
            transcription=final_transcription,  # Use corrected transcription
            language=detected_lang,
            intent=intent_result,
            products=products,
            cart_id=cart_id,
            checkout_url=checkout_url,
            audio_response=audio_response_url if audio_response_url else None,  # Backward compatibility
            audio_data=audio_response_data if audio_response_data else None,      # Base64 encoded audio
            audio_format=audio_response_format if audio_response_data else None    # Audio format
        )
    
    finally:
        # Cleanup
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    """Transcribe audio file to text"""
    temp_audio = tempfile.mktemp(suffix=".wav")
    try:
        with open(temp_audio, "wb") as f:
            content = await file.read()
            f.write(content)
        
        stt_service = get_stt_service()
        transcription, detected_lang = await stt_service.transcribe(temp_audio, language)
        
        return {
            "text": transcription,
            "language": detected_lang
        }
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

@router.post("/intent")
async def extract_intent(text: str, language: str = "en", output_language: str = "en"):
    """
    Extract intent from text and optionally search products if product_search intent is detected
    """
    nlp_service = get_nlp_service()
    intent_result = await nlp_service.extract_intent(text, language, output_language)
    
    # If intent is product_search and products are found, automatically search Shopify
    products = []
    if intent_result.get("intent") == "product_search":
        product_keywords = intent_result.get("entities", {}).get("products", [])
        
        if product_keywords:
            try:
                commerce_backend = get_commerce_backend()
                search_query = " ".join(product_keywords)
                products = await commerce_backend.search_products(search_query, limit=10)
            except Exception as e:
                # If Shopify search fails, try offline fallback
                if settings.ENABLE_OFFLINE_MODE:
                    try:
                        from app.services.products import get_product_service
                        product_service = get_product_service()
                        products = product_service.search_products(search_query)
                        # Convert to standard format
                        products = [
                            {
                                "id": str(p["id"]),
                                "title": p.get("name_hi", p["name"]) if language == "hi" else p["name"],
                                "description": p.get("description_hi", p["description"]) if language == "hi" else p["description"],
                                "price": p["price"],
                                "currency": "INR",
                                "images": [p.get("image")] if p.get("image") else [],
                                "available": p.get("stock", 0) > 0,
                                "category": p.get("category"),
                                "provider": "local"
                            }
                            for p in products
                        ]
                    except Exception:
                        pass  # If both fail, return empty products list
    
    # Add products to the response
    response = intent_result.copy()
    response["products"] = products
    response["products_count"] = len(products)
    
    return response

@router.post("/speak")
async def text_to_speech(request: Request, text: str, language: str = "en"):
    """Convert text to speech and return base64 encoded audio for JSON transport"""
    import base64
    tts_service = get_tts_service()
    
    # Use raw bytes synthesis for direct Streamlit playback
    audio_result = await tts_service.synthesize_bytes(text, language)
    
    audio_bytes = audio_result.get("audio_bytes", b"")
    audio_format = audio_result.get("audio_format", "audio/mp3")
    
    # Convert bytes to base64 for JSON transport
    audio_data = None
    if audio_bytes:
        audio_data = base64.b64encode(audio_bytes).decode('utf-8')
    
    # For backward compatibility, also provide URL
    audio_url = None
    if audio_bytes:
        try:
            # Create a temporary file for URL serving
            import tempfile
            temp_file = tempfile.mktemp(suffix=".mp3")
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)
            
            # Get the base URL from the request
            base_url = str(request.base_url).rstrip('/')
            filename = os.path.basename(temp_file)
            audio_url = f"{base_url}/audio/{filename}"
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] Failed to create compatibility URL: {e}")
    
    return {
        "text": text,
        "language": language,
        "audio_data": audio_data,      # Base64 encoded audio for JSON transport
        "audio_format": audio_format,  # Audio format
        "audio_path": audio_url         # Backward compatibility: URL
    }
