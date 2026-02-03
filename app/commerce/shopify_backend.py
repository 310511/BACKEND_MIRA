"""
Shopify Commerce Backend Implementation
Implements CommerceBackend interface for Shopify Storefront API
"""
from typing import List, Dict, Optional
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import httpx

from app.config import settings
from app.commerce.interface import CommerceBackend

class ShopifyBackend(CommerceBackend):
    """Shopify implementation of CommerceBackend"""
    
    def __init__(self):
        self.store_domain = settings.SHOPIFY_STORE_DOMAIN
        self.access_token = settings.SHOPIFY_STOREFRONT_ACCESS_TOKEN
        self.admin_token = settings.SHOPIFY_ADMIN_ACCESS_TOKEN
        
        # Setup GraphQL client for Storefront API
        transport = AIOHTTPTransport(
            url=f"https://{self.store_domain}/api/2024-01/graphql.json",
            headers={"X-Shopify-Storefront-Access-Token": self.access_token}
        )
        # Fetching the schema can add a big delay on first request and is not required
        # for executing known queries.
        self.client = Client(transport=transport, fetch_schema_from_transport=False)
    
    async def search_products(
        self, 
        query: str = "", 
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search products in Shopify store"""
        search_query = gql("""
            query SearchProducts($query: String!, $limit: Int!) {
                products(first: $limit, query: $query) {
                    edges {
                        node {
                            id
                            title
                            description
                            priceRange {
                                minVariantPrice {
                                    amount
                                    currencyCode
                                }
                            }
                            images(first: 1) {
                                edges {
                                    node {
                                        url
                                        altText
                                    }
                                }
                            }
                            variants(first: 1) {
                                edges {
                                    node {
                                        id
                                        availableForSale
                                    }
                                }
                            }
                            productType
                            vendor
                        }
                    }
                }
            }
        """)
        
        params = {"query": query, "limit": limit}
        result = await self.client.execute_async(search_query, variable_values=params)
        
        products = []
        for edge in result.get("products", {}).get("edges", []):
            node = edge["node"]
            first_variant = None
            try:
                first_variant = (node.get("variants") or {}).get("edges", [])[0]["node"]
            except Exception:
                first_variant = None
            raw_product = {
                "id": node["id"],
                "title": node["title"],
                "description": node["description"],
                "price": float(node["priceRange"]["minVariantPrice"]["amount"]),
                "currency": node["priceRange"]["minVariantPrice"]["currencyCode"],
                "image": node["images"]["edges"][0]["node"]["url"] if node["images"]["edges"] else None,
                "available": first_variant["availableForSale"] if first_variant else False,
                "variant_id": first_variant["id"] if first_variant else None,
                "category": node.get("productType"),
                "vendor": node.get("vendor")
            }
            # Normalize to standard format
            products.append(self.normalize_product(raw_product))
        
        return products
    
    async def get_product(self, product_id: str) -> Optional[Dict]:
        """Get product by ID"""
        product_query = gql("""
            query GetProduct($id: ID!) {
                product(id: $id) {
                    id
                    title
                    description
                    priceRange {
                        minVariantPrice {
                            amount
                            currencyCode
                        }
                    }
                    images(first: 5) {
                        edges {
                            node {
                                url
                                altText
                            }
                        }
                    }
                    variants(first: 10) {
                        edges {
                            node {
                                id
                                title
                                availableForSale
                                price {
                                    amount
                                    currencyCode
                                }
                            }
                        }
                    }
                    productType
                    vendor
                }
            }
        """)
        
        result = await self.client.execute_async(product_query, variable_values={"id": product_id})
        
        if result.get("product"):
            node = result["product"]
            raw_product = {
                "id": node["id"],
                "title": node["title"],
                "description": node["description"],
                "price": float(node["priceRange"]["minVariantPrice"]["amount"]),
                "currency": node["priceRange"]["minVariantPrice"]["currencyCode"],
                "images": [edge["node"]["url"] for edge in node["images"]["edges"]],
                "available": any(v["node"]["availableForSale"] for v in node["variants"]["edges"]),
                "category": node.get("productType"),
                "vendor": node.get("vendor"),
                "variants": [
                    {
                        "id": edge["node"]["id"],
                        "title": edge["node"]["title"],
                        "price": float(edge["node"]["price"]["amount"]),
                        "available": edge["node"]["availableForSale"]
                    }
                    for edge in node["variants"]["edges"]
                ]
            }
            return self.normalize_product(raw_product)
        
        return None
    
    async def create_cart(self) -> Dict:
        """Create a new cart"""
        create_cart_mutation = gql("""
            mutation CreateCart {
                cartCreate {
                    cart {
                        id
                        checkoutUrl
                    }
                }
            }
        """)
        
        result = await self.client.execute_async(create_cart_mutation)
        return result["cartCreate"]["cart"]
    
    async def add_to_cart(
        self, 
        cart_id: str, 
        variant_id: str, 
        quantity: int = 1
    ) -> Dict:
        """Add item to cart"""
        add_to_cart_mutation = gql("""
            mutation AddToCart($cartId: ID!, $lines: [CartLineInput!]!) {
                cartLinesAdd(cartId: $cartId, lines: $lines) {
                    cart {
                        id
                        checkoutUrl
                        lines(first: 10) {
                            edges {
                                node {
                                    id
                                    quantity
                                    merchandise {
                                        ... on ProductVariant {
                                            id
                                            title
                                            product {
                                                title
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        """)
        
        params = {
            "cartId": cart_id,
            "lines": [{"merchandiseId": variant_id, "quantity": quantity}]
        }
        
        result = await self.client.execute_async(add_to_cart_mutation, variable_values=params)
        return result["cartLinesAdd"]["cart"]
    
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
        Create a new product in Shopify using Admin API
        Requires SHOPIFY_ADMIN_ACCESS_TOKEN
        """
        if not self.admin_token:
            raise ValueError("SHOPIFY_ADMIN_ACCESS_TOKEN is required to create products")
        
        # Prepare product data for Shopify Admin API
        product_data = {
            "product": {
                "title": title,
                "body_html": description or "",
                "vendor": vendor or "",
                "product_type": product_type or "",
                "tags": ", ".join(tags) if tags else "",
                "variants": [
                    {
                        "price": str(price),
                        "currency": currency
                    }
                ]
            }
        }
        
        # Add images if provided
        if images:
            product_data["product"]["images"] = [
                {"src": img_url} for img_url in images
            ]
        
        # Use Admin REST API to create product
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://{self.store_domain}/admin/api/2024-01/products.json",
                headers={
                    "X-Shopify-Access-Token": self.admin_token,
                    "Content-Type": "application/json"
                },
                json=product_data
            )
            
            if response.status_code not in [200, 201]:
                error_msg = response.text
                raise Exception(f"Failed to create product in Shopify: {error_msg}")
            
            result = response.json()
            shopify_product = result.get("product", {})
            
            # Normalize to standard format
            return {
                "id": shopify_product.get("id"),
                "title": shopify_product.get("title"),
                "description": shopify_product.get("body_html"),
                "price": float(shopify_product.get("variants", [{}])[0].get("price", 0)),
                "currency": currency,
                "images": [img.get("src") for img in shopify_product.get("images", [])],
                "available": True,
                "category": shopify_product.get("product_type"),
                "vendor": shopify_product.get("vendor"),
                "provider": "shopify"
            }
    
    def normalize_product(self, product: Dict) -> Dict:
        """
        Normalize Shopify product to standard format
        Compatible with ONDC schema
        """
        # Handle single image vs list of images
        images = product.get("images", [])
        if isinstance(images, str) or (isinstance(images, list) and len(images) == 0):
            images = [product.get("image")] if product.get("image") else []
        elif not isinstance(images, list):
            images = [images] if images else []
        
        return {
            "id": product.get("id", ""),
            "title": product.get("title", ""),
            "description": product.get("description", ""),
            "price": product.get("price", 0.0),
            "currency": product.get("currency", "INR"),
            "images": images,
            "available": product.get("available", False),
            "category": product.get("category"),
            "vendor": product.get("vendor"),
            "variant_id": product.get("variant_id"),
            # Additional fields for ONDC compatibility
            "provider": "shopify",
            "variants": product.get("variants", [])
        }
