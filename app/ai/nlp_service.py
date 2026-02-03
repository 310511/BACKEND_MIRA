"""
Multi-Provider NLP Intent Processing Service
Supports: OpenAI GPT, Google Gemini, Anthropic Claude, Local LLM
"""
from typing import Dict, Optional, List
from app.config import settings, NLPProvider

class NLPService:
    """NLP service for intent extraction and query understanding"""
    
    def __init__(self):
        self.provider = settings.NLP_PROVIDER
        
        # Localized response templates for user_friendly_response
        self.response_templates = {
            "en": {
                "product_search": "Looking for {products}...",
                "product_found": "Found {count} {products} options",
                "add_to_cart": "Adding {products} to your cart...",
                "view_cart": "Opening your cart...",
                "checkout": "Proceeding to checkout...",
                "help": "I can help you search for products, add to cart, and checkout. What would you like?",
                "order_status": "Checking your order status...",
                "general_query": "I'm here to help with your shopping needs."
            },
            "hi": {
                "product_search": "{products} खोज रहे हैं...",
                "product_found": "{products} के {count} विकल्प मिले",
                "add_to_cart": "{products} को आपके कार्ट में जोड़ रहे हैं...",
                "view_cart": "आपका कार्ट खोल रहे हैं...",
                "checkout": "चेकआउट पर जा रहे हैं...",
                "help": "मैं आपको उत्पाद खोजने, कार्ट में जोड़ने और चेकआउट करने में मदद कर सकता हूं। आप क्या चाहते हैं?",
                "order_status": "आपके ऑर्डर की स्थिति जांच रहे हैं...",
                "general_query": "मैं आपकी खरीदारी की जरूरतों में मदद के लिए यहां हूं।"
            },
            "ta": {
                "product_search": "{products} தேடுகிறோம்...",
                "product_found": "{products} க்கு {count} விருப்பங்கள் கிடைத்தன",
                "add_to_cart": "{products} உங்கள் கார்ட்டில் சேர்க்கிறோம்...",
                "view_cart": "உங்கள் கார்ட்டை திறக்கிறோம்...",
                "checkout": "செக்கவுட்டிற்கு செல்கிறோம்...",
                "help": "தயாரிப்புகளைத் தேட, கார்ட்டில் சேர்க்க, மற்றும் செக்கவுட் செய்ய நான் உதவ முடியும். நீங்கள் என்ன விரும்புகிறீர்கள்?",
                "order_status": "உங்கள் ஆர்டர் நிலையைச் சரிபார்க்கிறோம்...",
                "general_query": "உங்கள் ஷாப்பிங் தேவைகளுக்கு நான் இங்கே உதவ உள்ளேன்."
            },
            "te": {
                "product_search": "{products} వెతుకుతున్నాను...",
                "product_found": "{products} కోసం {count} ఎంపికలు దొరికాయి",
                "add_to_cart": "{products} మీ కార్ట్‌లో చేరుస్తున్నాను...",
                "view_cart": "మీ కార్ట్‌ను తెరుస్తున్నాను...",
                "checkout": "చెకౌట్‌కి వెళుతున్నాను...",
                "help": "ఉత్పత్తులను వెతికి, కార్ట్‌లో చేర్చి, చెకౌట్ చేయడానికి నేను సహాయం చేయగలను. మీరు ఏమి కోరుకుంటున్నారు?",
                "order_status": "మీ ఆర్డర్ స్థితిని తనిఖీ చేస్తున్నాను...",
                "general_query": "మీ షాపింగ్ అవసరాలకు నేను ఇక్కడ ఉన్నాను."
            }
        }
    
    def _get_localized_response(self, intent: str, products: List[str], language: str, count: int = None) -> str:
        """Generate localized response based on intent and language"""
        # Default to English if language not supported
        if language not in self.response_templates:
            language = "en"
        
        templates = self.response_templates[language]
        
        # Format product names for display
        if products:
            if len(products) == 1:
                product_str = products[0]
            else:
                conj = {
                    "en": "and",
                    "hi": "और",
                    "ta": "மற்றும்",
                    "te": "మరియు",
                }.get(language, "and")
                product_str = f"{', '.join(products[:-1])} {conj} {products[-1]}" if len(products) > 1 else products[0]
        else:
            product_str = "products" if language == "en" else ("उत्पाद" if language == "hi" else ("தயாரிப்புகள்" if language == "ta" else "ఉత్పత్తులు"))
        
        # Get appropriate template
        if intent in templates:
            template = templates[intent]
            if count is not None and "product_found" in intent:
                return template.format(products=product_str, count=count)
            else:
                return template.format(products=product_str)
        else:
            # Fallback to product_search template
            return templates.get("product_search", templates.get("general_query", "I'm here to help.")).format(products=product_str)
        
    async def match_transcription_to_products(
        self,
        transcription: str,
        product_names: List[str],
        language: str = "en",
        output_language: str = "en"
    ) -> Dict:
        """
        Match transcription against product list and correct transcription.
        This helps improve STT accuracy by using product context.
        
        Args:
            transcription: Raw transcription from STT
            product_names: List of available product names
            language: Language of the transcription
            output_language: Language for the user-facing response
            
        Returns:
            Dict with intent, entities, corrected_transcription, and confidence
        """
        # Prefer Groq when configured; otherwise fall back to local LLM / heuristics.
        if self.provider == NLPProvider.GROQ:
            return await self._match_with_groq(transcription, product_names, language, output_language)
        if self.provider == NLPProvider.LOCAL_LLM:
            return await self._match_with_ollama(transcription, product_names, language, output_language)
        # If provider isn't explicitly supported for matching, still try Groq if key exists.
        if settings.GROQ_API_KEY:
            return await self._match_with_groq(transcription, product_names, language, output_language)
        return self._fallback_intent(transcription, language, output_language)

    async def translate_to_english(self, text: str) -> str:
        """Translate a (likely non-English) product word/phrase to English (Groq)."""
        # If no Groq key, don't block the pipeline.
        if not settings.GROQ_API_KEY:
            return text
        prompt = (
            "Translate the following text to English.\n"
            "Return ONLY the English translation, nothing else.\n\n"
            f'Text: "{text}"\n'
            "English:"
        )
        translated = await self._groq_text(prompt, temperature=0.1, timeout_seconds=10.0)
        translated = (translated or "").strip()
        return translated if translated else text

    async def extract_field_value(self, text: str, field: str) -> Optional[str]:
        """Extract a single field value from seller text (Groq)."""
        if not settings.GROQ_API_KEY:
            return text.strip() if text and text.strip() else None
        prompt = f"""Extract the {field} information from this text. Return ONLY the extracted value, nothing else.

Text: "{text}"
Field: {field}

Examples:
- If field is "title" and text is "मुझे बादाम बेचना है", return "बादाम"
- If field is "price" and text is "कीमत 500 रुपये है", return "500"
- If field is "description" and text is "यह ताजा शहद है", return "ताजा शहद"

Return only the extracted value:"""
        extracted = await self._groq_text(prompt, temperature=0.1, timeout_seconds=10.0)
        if not extracted:
            return text.strip() if text and text.strip() else None
        # Clean first line only (avoid extra text if model misbehaves)
        extracted = extracted.strip().split("\n")[0].strip()
        return extracted if extracted else (text.strip() if text and text.strip() else None)
    
    async def extract_intent(
        self, 
        text: str, 
        language: str = "en",
        output_language: str = "en"
    ) -> Dict:
        """
        Extract user intent from transcribed text
        
        Args:
            text: Transcribed user query
            language: Language of the query
            output_language: Language for the user-facing response
            
        Returns:
            Dict with intent, entities, and confidence
        """
        try:
            if self.provider == NLPProvider.OPENAI_GPT4:
                if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"dummy_key_for_testing", "your_openai_api_key_here"}:
                    return self._fallback_intent(text, language, output_language)
                return await self._extract_openai(text, language, "gpt-4", output_language)
            elif self.provider == NLPProvider.OPENAI_GPT35:
                if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"dummy_key_for_testing", "your_openai_api_key_here"}:
                    return self._fallback_intent(text, language, output_language)
                return await self._extract_openai(text, language, "gpt-3.5-turbo", output_language)
            elif self.provider == NLPProvider.GOOGLE_GEMINI:
                return await self._extract_gemini(text, language, output_language)
            elif self.provider == NLPProvider.ANTHROPIC_CLAUDE:
                if not settings.ANTHROPIC_API_KEY:
                    return self._fallback_intent(text, language, output_language)
                return await self._extract_claude(text, language, output_language)
            elif self.provider == NLPProvider.LOCAL_LLM:
                return await self._extract_local_llm(text, language, output_language)
            else:
                return self._fallback_intent(text, language, output_language)
        except Exception:
            return self._fallback_intent(text, language, output_language)
    
    def _build_prompt(self, text: str, language: str) -> str:
        """Build intent extraction prompt"""
        return f"""You are an AI assistant for a voice-based e-commerce platform serving tribal areas.

User Query ({language}): "{text}"

Extract the following information:
1. Intent: What does the user want? (product_search, order_status, help, general_query, add_to_cart, checkout)
2. Product Keywords: What products are they interested in?
3. Quantity: How many items (if mentioned)?
4. Price Range: Any price preferences?

Respond in JSON format:
{{
  "intent": "product_search",
  "entities": {{
    "products": ["honey", "basket"],
    "quantity": 2,
    "price_range": null
  }},
  "confidence": 0.95,
  "user_friendly_response": "I'll help you find honey and baskets."
}}"""
    
    async def _extract_openai(
        self, 
        text: str, 
        language: str,
        model: str,
        output_language: str = "en"
    ) -> Dict:
        """Extract intent using OpenAI GPT"""
        from openai import OpenAI
        import json
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful e-commerce assistant."},
                {"role": "user", "content": self._build_prompt(text, language)}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    async def _match_with_ollama(
        self,
        transcription: str,
        product_names: List[str],
        language: str,
        output_language: str = "en"
    ) -> Dict:
        """Use Ollama (local LLM) to match transcription against products and correct it"""
        import httpx
        import json
        import re
        
        # Build product context - limit to avoid token limits
        products_list = product_names[:30]  # Limit to 30 products
        products_text = "\n".join([f"- {p}" for p in products_list])
        
        # Use the simplified, focused multilingual NLP prompt
        prompt = f"""You are an intelligent multilingual NLP engine for a voice-based shopping assistant.

The input is raw speech-to-text output and may contain spelling mistakes,
phonetic errors, or incorrectly recognized Hindi/English words.

Available products in store:
{products_text}

Your responsibilities:

1. Correct transcription mistakes.
   Examples:
   - "मूजे पडदाम जाही" → "मुझे बादाम चाहिए" (I need almonds)
   - "पडदाम" → "बादाम" (almonds)
   - "मूजे" → "मुझे" (I/me)
   - "जाही" → "चाहिए" (need/want)

2. Convert incorrect Hindi words to proper Hindi.

3. Understand the user's intent.

4. Identify requested products or services (match to products from list above).

5. Handle Hindi, English, and Hinglish.

6. Normalize informal speech.

7. Generate user_friendly_response in {output_language} language.

Supported intents:
- product_search
- place_order
- add_to_cart
- remove_from_cart
- track_order
- cancel_order
- complaint
- greeting
- help
- unknown

Supported languages:
- hi
- en
- mixed

If you are unsure, set intent = "product_search" and confidence < 0.7.

Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include extra text.

---

User input:
"{transcription}"

---

Return in this exact format:

{{
  "original_text": "{transcription}",
  "corrected_text": "corrected Hindi/English text",
  "language": "{language}",
  "intent": "product_search",
  "product": "product_name_from_list_or_null",
  "confidence": 0.9,
  "corrected_transcription": "same as corrected_text",
  "entities": {{
    "products": ["product1", "product2"],
    "quantity": null,
    "price_range": null,
    "language": "{language}"
  }},
  "user_friendly_response": "Response in {output_language} language"
}}

CRITICAL:
- corrected_text MUST be properly corrected (e.g., "मुझे बादाम चाहिए" not "मूजे पडदाम जाही")
- product MUST be in ENGLISH (e.g., "almonds" not "बादाम") - Shopify search requires English
- If user says "बादाम" → product = "almonds"
- If user says "शहद" → product = "honey"
- If user says "चावल" → product = "rice"
- entities.products should be an array of ENGLISH product names from the list above
- user_friendly_response MUST be in {output_language} language
- If output_language="hi", respond in Hindi script
- If output_language="ta", respond in Tamil script  
- If output_language="te", respond in Telugu script
- If output_language="en", respond in English
- Always return valid JSON only"""
        
        try:
            if settings.DEBUG:
                print(f"[DEBUG] Ollama matching - transcription: '{transcription}', language: '{language}', products: {len(product_names)}")
            
            # Use Ollama API
            api_url = settings.LOCAL_LLM_API_URL
            model = settings.LOCAL_LLM_MODEL
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{api_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,  # Lower temperature for more accurate, consistent results
                            "top_p": 0.95,
                            "top_k": 40,
                        }
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
                result_data = response.json()
                response_text = result_data.get("response", "").strip()
            
            # Try to extract JSON if wrapped in markdown or text
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Normalize and validate response structure
            # Ensure backward compatibility with existing code
            if "entities" not in result:
                result["entities"] = {}
            
            # Ensure products array exists
            if "products" not in result.get("entities", {}):
                result["entities"]["products"] = []
            
            # Map product field to entities.products array for backward compatibility
            if "product" in result and result["product"]:
                # Add product to entities.products if not already there
                if result["product"] not in result["entities"]["products"]:
                    result["entities"]["products"].insert(0, result["product"])
            
            # Map other fields
            if "quantity" in result and result["quantity"] is not None:
                result["entities"]["quantity"] = result["quantity"]
            elif "quantity" not in result.get("entities", {}):
                result["entities"]["quantity"] = None
            
            if "price_range" in result:
                result["entities"]["price_range"] = result["price_range"]
            elif "price_range" not in result.get("entities", {}):
                result["entities"]["price_range"] = None
            
            # Ensure language in entities
            if "language" not in result.get("entities", {}):
                result["entities"]["language"] = result.get("language", language)
            
            # Handle both corrected_text and corrected_transcription
            if "corrected_text" in result and "corrected_transcription" not in result:
                result["corrected_transcription"] = result["corrected_text"]
            elif "corrected_transcription" not in result:
                result["corrected_transcription"] = transcription
            
            # Ensure original_text exists
            if "original_text" not in result:
                result["original_text"] = transcription
            
            # Ensure intent is always product_search for e-commerce context
            if result.get("intent") not in ["product_search", "place_order", "track_order"]:
                result["intent"] = "product_search"
            
            # Ensure confidence exists
            if "confidence" not in result:
                result["confidence"] = 0.75
            
            if settings.DEBUG:
                print(f"[DEBUG] Ollama result - intent: '{result.get('intent')}', product: '{result.get('product')}', "
                      f"corrected: '{result.get('corrected_transcription')}', products: {result.get('entities', {}).get('products', [])}")
            
            return result
            
        except json.JSONDecodeError as e:
            if settings.DEBUG:
                print(f"[DEBUG] Ollama JSON parse error: {e}, response: {response_text[:500] if 'response_text' in locals() else 'No response'}")
            # Fallback: use regular intent extraction
            result = self._fallback_intent(transcription, language, output_language)
            result["intent"] = "product_search"
            result["corrected_transcription"] = transcription
            return result

    async def _match_with_groq(
        self,
        transcription: str,
        product_names: List[str],
        language: str,
        output_language: str = "en"
    ) -> Dict:
        """Use Groq (OpenAI-compatible) to match transcription against products and correct it."""
        import json
        import re

        # Build product context - limit to avoid token limits
        products_list = product_names[:30]
        products_text = "\n".join([f"- {p}" for p in products_list])

        prompt = f"""You are an intelligent multilingual NLP engine for a voice-based shopping assistant.

The input is raw speech-to-text output and may contain spelling mistakes,
phonetic errors, or incorrectly recognized Hindi/English words.

Available products in store:
{products_text}

Your responsibilities:
1. Correct transcription mistakes (Hindi/English/Hinglish).
2. Identify the requested product(s) and return product names in ENGLISH for Shopify search.
3. Always set intent="product_search" for this use case.
4. Generate user_friendly_response in {output_language} language.

Return ONLY valid JSON (no markdown, no extra text) in this exact format:
{{
  "original_text": "{transcription}",
  "corrected_text": "corrected Hindi/English text",
  "language": "{language}",
  "intent": "product_search",
  "product": "english_product_or_null",
  "confidence": 0.9,
  "corrected_transcription": "same as corrected_text",
  "entities": {{
    "products": ["english_product1", "english_product2"],
    "quantity": null,
    "price_range": null,
    "language": "{language}"
  }},
  "user_friendly_response": "short response in {output_language} language"
}}

CRITICAL:
- product MUST be in ENGLISH (e.g., "almonds" not "बादाम")
- entities.products MUST be ENGLISH product names (for Shopify search)
- user_friendly_response MUST be in {output_language} language
- If output_language="hi", respond in Hindi script
- If output_language="ta", respond in Tamil script
- If output_language="te", respond in Telugu script
- If output_language="en", respond in English
"""

        try:
            if settings.DEBUG:
                print(f"[DEBUG] Groq matching - transcription: '{transcription}', language: '{language}', products: {len(product_names)}")

            response_text = await self._groq_chat_json(prompt, temperature=0.2, timeout_seconds=60.0)

            # Extract JSON if wrapped
            json_match = re.search(r'\{[\s\S]*\}', response_text or "", re.DOTALL)
            if json_match:
                response_text = json_match.group()

            result = json.loads(response_text)

            # Normalize and validate response structure (same as Ollama path)
            if "entities" not in result:
                result["entities"] = {}
            if "products" not in result.get("entities", {}):
                result["entities"]["products"] = []
            if "product" in result and result["product"]:
                if result["product"] not in result["entities"]["products"]:
                    result["entities"]["products"].insert(0, result["product"])
            if "quantity" in result and result["quantity"] is not None:
                result["entities"]["quantity"] = result["quantity"]
            elif "quantity" not in result.get("entities", {}):
                result["entities"]["quantity"] = None
            if "price_range" in result:
                result["entities"]["price_range"] = result["price_range"]
            elif "price_range" not in result.get("entities", {}):
                result["entities"]["price_range"] = None
            if "language" not in result.get("entities", {}):
                result["entities"]["language"] = result.get("language", language)
            if "corrected_text" in result and "corrected_transcription" not in result:
                result["corrected_transcription"] = result["corrected_text"]
            elif "corrected_transcription" not in result:
                result["corrected_transcription"] = transcription
            if "original_text" not in result:
                result["original_text"] = transcription
            if result.get("intent") not in ["product_search", "place_order", "track_order"]:
                result["intent"] = "product_search"
            if "confidence" not in result:
                result["confidence"] = 0.75

            if settings.DEBUG:
                print(f"[DEBUG] Groq result - intent: '{result.get('intent')}', product: '{result.get('product')}', "
                      f"corrected: '{result.get('corrected_transcription')}', products: {result.get('entities', {}).get('products', [])}")

            return result
        except Exception as e:
            if settings.DEBUG:
                print(f"[DEBUG] Groq matching failed: {e}, type: {type(e).__name__}")
            result = self._fallback_intent(transcription, language, output_language)
            result["intent"] = "product_search"
            result["corrected_transcription"] = transcription
            return result

    async def _groq_chat_json(self, user_prompt: str, temperature: float = 0.2, timeout_seconds: float = 60.0) -> str:
        """Call Groq Chat Completions expecting JSON text back."""
        import httpx

        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")

        url = f"{settings.GROQ_API_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}

        payload = {
            "model": settings.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise assistant. Output must follow instructions exactly."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
            data = resp.json()
            return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""

    async def _groq_text(self, prompt: str, temperature: float = 0.1, timeout_seconds: float = 10.0) -> str:
        """Call Groq for short plain-text outputs."""
        return (await self._groq_chat_json(prompt, temperature=temperature, timeout_seconds=timeout_seconds)).strip()
    
    async def _extract_gemini(self, text: str, language: str, output_language: str = "en") -> Dict:
        """Extract intent using Google Gemini"""
        import google.generativeai as genai
        import json
        
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        response = model.generate_content(self._build_prompt(text, language))
        
        try:
            result = json.loads(response.text)
            # Override with localized response if available
            if "entities" in result and "products" in result["entities"]:
                intent = result.get("intent", "product_search")
                products = result["entities"]["products"]
                result["user_friendly_response"] = self._get_localized_response(intent, products, output_language)
            return result
        except:
            # Fallback to simple parsing
            return self._fallback_intent(text, language, output_language)
    
    async def _extract_claude(self, text: str, language: str, output_language: str = "en") -> Dict:
        """Extract intent using Anthropic Claude"""
        from anthropic import Anthropic
        import json
        
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": self._build_prompt(text, language)}
            ]
        )
        
        try:
            result = json.loads(message.content[0].text)
            # Override with localized response if available
            if "entities" in result and "products" in result["entities"]:
                intent = result.get("intent", "product_search")
                products = result["entities"]["products"]
                result["user_friendly_response"] = self._get_localized_response(intent, products, output_language)
            return result
        except:
            return self._fallback_intent(text, language, output_language)
    
    async def _extract_local_llm(self, text: str, language: str, output_language: str = "en") -> Dict:
        """
        Extract intent using local LLM (Ollama/LlamaCpp)
        Offline-capable, no API keys required
        """
        try:
            import httpx
            
            # Try Ollama API first (most common local LLM setup)
            api_url = settings.LOCAL_LLM_API_URL
            model = settings.LOCAL_LLM_MODEL
            
            # Build prompt with Hindi support
            prompt = self._build_prompt(text, language)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{api_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    
                    # Parse JSON response
                    import json
                    import re
                    
                    # Try to extract JSON from response
                    json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            return parsed
                        except:
                            pass
                    
                    # If JSON parsing fails, use fallback
                    return self._fallback_intent(text, language, output_language)
                else:
                    # Ollama not available, use fallback
                    return self._fallback_intent(text, language, output_language)
                    
        except Exception as e:
            # If local LLM fails, use keyword-based fallback (offline-safe)
            return self._fallback_intent(text, language, output_language)
    
    def _fallback_intent(self, text: str, language: str, output_language: str = "en") -> Dict:
        """Simple keyword-based fallback intent extraction with improved product name extraction and localized responses"""
        import re
        
        text_lower = text.lower().strip()
        original_text = text
        
        # Intent classification
        if any(word in text_lower for word in ["help", "support", "madad", "मदद"]):
            intent = "help"
        elif any(word in text_lower for word in ["order", "status", "track", "ऑर्डर", "स्थिति"]):
            intent = "order_status"
        elif any(word in text_lower for word in ["cart", "add", "कार्ट", "जोड़ो"]):
            intent = "add_to_cart"
        elif any(word in text_lower for word in ["checkout", "pay", "भुगतान"]):
            intent = "checkout"
        elif any(word in text_lower for word in ["buy", "purchase", "want", "need", "show", "find", "search", "looking", "khareed", "chahiye", "khareedna", "चाहिए", "दिखाओ", "खोज"]):
            intent = "product_search"
        else:
            intent = "general_query"
        
        # Extract product keywords - improved method
        products = []
        
        # Known product keywords (expanded list)
        product_keywords = {
            "honey": ["honey", "shahad", "shehad", "शहद"],
            "basket": ["basket", "टोकरी", "bamboo"],
            "tea": ["tea", "चाय"],
            "jewellry": ["jewellry", "jewelry", "आभूषण", "ornament"],
            "almonds": ["almonds", "almond", "badam", "बादाम", "बदाम"],
            "walnuts": ["walnuts", "walnut", "akhrot", "अखरोट"],
            "cashews": ["cashews", "cashew", "kaju", "काजू"],
            "rice": ["rice", "chawal", "चावल"],
            "wheat": ["wheat", "gehun", "गेहूं"],
            "spices": ["spices", "masala", "मसाला", "मसाले"],
            "levis": ["levis", "levis's", "लिवादान", "livadan", "लीवाइस"],
            "jeans": ["jeans", "जीन्स", "jean"],
            "shirt": ["shirt", "शर्ट", "shirts"],
            "pants": ["pants", "पैंट", "trousers"],
            "dress": ["dress", "ड्रेस", "dresses"],
        }
        
        # First, try known keywords - check for exact matches and partial matches
        for product, keywords in product_keywords.items():
            for kw in keywords:
                # Check for exact word match
                if f" {kw} " in f" {text_lower} " or text_lower.startswith(kw + " ") or text_lower.endswith(" " + kw) or text_lower == kw:
                    products.append(product)
                # Also check for Hindi character matching for Devanagari text
                elif any('\u0900' <= char <= '\u097F' for char in kw) and kw in text:
                    products.append(product)
        
        # If no products found and intent is product_search, extract nouns/words after request phrases
        if not products and intent == "product_search":
            # Common request phrases to remove
            request_phrases = [
                "i need", "i want", "show me", "find me", "search for", 
                "looking for", "i'm looking for", "give me", "get me",
                "mujhe chahiye", "mujhe do", "dikhao", "खोजो", "दिखाओ"
            ]
            
            # Remove request phrases
            cleaned_text = text_lower
            for phrase in request_phrases:
                cleaned_text = cleaned_text.replace(phrase, "").strip()
            
            # Remove common stop words
            stop_words = ["the", "a", "an", "some", "any", "please", "pls", "कृपया"]
            words = cleaned_text.split()
            words = [w for w in words if w not in stop_words and len(w) > 2]
            
            # Extract potential product names (words that are likely products)
            # Remove punctuation and get meaningful words
            for word in words:
                # Clean word (remove punctuation)
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word and len(clean_word) > 2:
                    # Skip common verbs/adjectives
                    if clean_word not in ["need", "want", "show", "find", "search", "looking", "get", "give"]:
                        products.append(clean_word)
            
            # If still no products, use the cleaned text as a single search term
            if not products and cleaned_text:
                products.append(cleaned_text.strip())
        
        # Generate localized user_friendly_response
        user_friendly_response = self._get_localized_response(intent, products, output_language)
        
        return {
            "intent": intent,
            "entities": {
                "products": products,
                "quantity": None,
                "price_range": None,
                "language": language
            },
            "confidence": 0.75,
            "user_friendly_response": user_friendly_response,
            "original_text": original_text
        }

# Singleton instance
_nlp_service = None

def get_nlp_service() -> NLPService:
    """Get or create NLP service instance"""
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService()
    return _nlp_service
