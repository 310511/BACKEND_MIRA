"""
Product API Endpoints
Handles product search and retrieval via commerce abstraction layer
Supports Shopify and ONDC (future-ready)
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from app.commerce.factory import get_commerce_backend

router = APIRouter(prefix="/api/products", tags=["products"])

@router.get("/search")
async def search_products(query: str = "", limit: int = 10):
    """
    Search products using commerce backend (Shopify/ONDC)
    Works with any commerce provider via abstraction layer
    """
    commerce_backend = get_commerce_backend()
    products = await commerce_backend.search_products(query, limit)
    
    return {
        "query": query,
        "count": len(products),
        "products": products,
        "provider": products[0].get("provider", "unknown") if products else None
    }

@router.get("/{product_id}")
async def get_product(product_id: str):
    """Get product details by ID"""
    commerce_backend = get_commerce_backend()
    product = await commerce_backend.get_product(product_id)
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return product

@router.post("/cart/create")
async def create_cart():
    """Create a new shopping cart"""
    commerce_backend = get_commerce_backend()
    cart = await commerce_backend.create_cart()
    return cart

@router.post("/cart/{cart_id}/add")
async def add_to_cart(cart_id: str, variant_id: str, quantity: int = 1):
    """Add item to cart"""
    commerce_backend = get_commerce_backend()
    cart = await commerce_backend.add_to_cart(cart_id, variant_id, quantity)
    return cart
