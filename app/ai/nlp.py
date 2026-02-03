"""
AI-Powered NLP Intent Processing
Uses LLM to understand user queries and extract intent
"""
from typing import Dict, Optional
import os

class IntentProcessor:
    def __init__(self):
        """Initialize NLP processor"""
        # TODO: Add OpenAI API key from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
    
    def extract_intent(self, text: str, language: str = "en") -> Dict:
        """
        Extract user intent from transcribed text using LLM
        
        Args:
            text: Transcribed user query
            language: Language of the query
        
        Returns:
            Dict with intent, entities, and confidence
        """
        # TODO: Implement OpenAI GPT integration for intent extraction
        # For now, simple keyword matching
        
        text_lower = text.lower()
        
        # Simple intent classification
        if any(word in text_lower for word in ["buy", "purchase", "want", "need", "खरीदना", "चाहिए"]):
            intent = "product_search"
        elif any(word in text_lower for word in ["order", "status", "track", "ऑर्डर"]):
            intent = "order_status"
        elif any(word in text_lower for word in ["help", "support", "मदद"]):
            intent = "help"
        else:
            intent = "general_query"
        
        # Extract product keywords
        products = []
        product_keywords = {
            "honey": ["honey", "शहद"],
            "basket": ["basket", "टोकरी"],
            "tea": ["tea", "चाय"],
        }
        
        for product, keywords in product_keywords.items():
            if any(kw in text_lower for kw in keywords):
                products.append(product)
        
        return {
            "intent": intent,
            "entities": {
                "products": products,
                "language": language
            },
            "confidence": 0.85,  # Mock confidence
            "original_text": text
        }

# Singleton instance
_intent_processor = None

def get_intent_processor() -> IntentProcessor:
    """Get or create Intent Processor instance"""
    global _intent_processor
    if _intent_processor is None:
        _intent_processor = IntentProcessor()
    return _intent_processor
