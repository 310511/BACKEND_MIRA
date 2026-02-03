"""
Commerce Backend Interface
Abstract base class for commerce providers (Shopify, ONDC, etc.)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class CommerceBackend(ABC):
    """Abstract interface for commerce backends"""
    
    @abstractmethod
    async def search_products(
        self, 
        query: str = "", 
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search products
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional filters (category, price_range, etc.)
            
        Returns:
            List of product dictionaries with standardized format
        """
        pass
    
    @abstractmethod
    async def get_product(self, product_id: str) -> Optional[Dict]:
        """
        Get product by ID
        
        Returns:
            Product dictionary with standardized format
        """
        pass
    
    @abstractmethod
    async def create_cart(self) -> Dict:
        """
        Create a new shopping cart
        
        Returns:
            Cart dictionary with cart_id and checkout_url
        """
        pass
    
    @abstractmethod
    async def add_to_cart(
        self, 
        cart_id: str, 
        variant_id: str, 
        quantity: int = 1
    ) -> Dict:
        """
        Add item to cart
        
        Returns:
            Updated cart dictionary
        """
        pass
    
    @abstractmethod
    def normalize_product(self, product: Dict) -> Dict:
        """
        Normalize product data to standard format
        Used for converting provider-specific formats to ONDC-compatible schema
        
        Standard format:
        {
            "id": str,
            "title": str,
            "description": str,
            "price": float,
            "currency": str,
            "images": List[str],
            "available": bool,
            "category": Optional[str],
            "vendor": Optional[str]
        }
        """
        pass
    
    async def create_product(
        self,
        title: str,
        description: Optional[str] = None,
        price: float = 0.0,
        currency: str = "INR",
        product_type: Optional[str] = None,
        vendor: Optional[str] = None,
        tags: Optional[List[str]] = None,
        images: Optional[List[str]] = None
    ) -> Dict:
        """
        Create a new product
        
        Args:
            title: Product title
            description: Product description
            price: Product price
            currency: Currency code (default: INR)
            product_type: Product type/category
            vendor: Vendor name
            tags: List of tags
            images: List of image URLs
            
        Returns:
            Created product dictionary with standardized format
        """
        raise NotImplementedError("create_product not implemented for this backend")
    
    async def get_all_product_names(self, limit: int = 100) -> List[str]:
        """
        Get list of all product names for context-aware processing
        Used to improve STT accuracy and NLP matching
        
        Args:
            limit: Maximum number of product names to return
            
        Returns:
            List of product name strings
        """
        # Default implementation: search with empty query to get all products
        products = await self.search_products("", limit=limit)
        return [p.get("title", "") for p in products if p.get("title")]