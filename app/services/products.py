"""
Product service for e-commerce logic
"""
from typing import List, Dict, Optional

class ProductService:
    def __init__(self):
        # Mock product database for tribal commerce
        self.products = [
            {
                "id": 1,
                "name": "Organic Honey",
                "name_hi": "जैविक शहद",
                "price": 400,
                "description": "Pure forest honey from tribal areas",
                "description_hi": "आदिवासी क्षेत्रों से शुद्ध वन शहद",
                "category": "food",
                "image": "/products/honey.jpg",
                "stock": 50
            },
            {
                "id": 2,
                "name": "Bamboo Basket",
                "name_hi": "बांस की टोकरी",
                "price": 250,
                "description": "Handcrafted bamboo basket",
                "description_hi": "हस्तनिर्मित बांस की टोकरी",
                "category": "handicraft",
                "image": "/products/basket.jpg",
                "stock": 30
            },
            {
                "id": 3,
                "name": "Herbal Tea",
                "name_hi": "हर्बल चाय",
                "price": 150,
                "description": "Traditional herbal tea blend",
                "description_hi": "पारंपरिक हर्बल चाय मिश्रण",
                "category": "food",
                "image": "/products/tea.jpg",
                "stock": 100
            },
            {
                "id": 4,
                "name": "Tribal Jewelry",
                "name_hi": "आदिवासी आभूषण",
                "price": 800,
                "description": "Traditional handmade jewelry",
                "description_hi": "पारंपरिक हस्तनिर्मित आभूषण",
                "category": "jewelry",
                "image": "/products/jewelry.jpg",
                "stock": 15
            }
        ]
    
    def search_products(self, query: str = "", category: Optional[str] = None) -> List[Dict]:
        """Search products by query and/or category"""
        results = self.products
        
        if category:
            results = [p for p in results if p["category"] == category]
        
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p["name"].lower() 
                or query_lower in p["name_hi"]
                or query_lower in p["description"].lower()
            ]
        
        return results
    
    def get_product(self, product_id: int) -> Optional[Dict]:
        """Get product by ID"""
        for product in self.products:
            if product["id"] == product_id:
                return product
        return None

# Singleton instance
_product_service = None

def get_product_service() -> ProductService:
    """Get or create Product Service instance"""
    global _product_service
    if _product_service is None:
        _product_service = ProductService()
    return _product_service
