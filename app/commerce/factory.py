"""
Commerce Backend Factory
Provides unified interface to commerce backends (Shopify, ONDC, etc.)
Enables easy switching between providers
"""
from typing import Optional
from enum import Enum

from app.config import settings
from app.commerce.interface import CommerceBackend
from app.commerce.shopify_backend import ShopifyBackend
from app.commerce.ondc_backend import ONDCBackend

class CommerceProvider(str, Enum):
    """Commerce backend provider options"""
    SHOPIFY = "shopify"
    ONDC = "ondc"

class CommerceFactory:
    """Factory for creating commerce backend instances"""
    
    @staticmethod
    def create_backend(provider: Optional[CommerceProvider] = None) -> CommerceBackend:
        """
        Create commerce backend instance
        
        Args:
            provider: Commerce provider (defaults to Shopify for now)
            
        Returns:
            CommerceBackend instance
        """
        if provider is None:
            # Default to Shopify for PoC, can be configured via settings
            provider = CommerceProvider.SHOPIFY
        
        if provider == CommerceProvider.SHOPIFY:
            return ShopifyBackend()
        elif provider == CommerceProvider.ONDC:
            return ONDCBackend()
        else:
            raise ValueError(f"Unsupported commerce provider: {provider}")

# Singleton instance
_commerce_backend: Optional[CommerceBackend] = None

def get_commerce_backend() -> CommerceBackend:
    """
    Get or create commerce backend instance
    This is the main entry point for commerce operations
    """
    global _commerce_backend
    if _commerce_backend is None:
        _commerce_backend = CommerceFactory.create_backend()
    return _commerce_backend
