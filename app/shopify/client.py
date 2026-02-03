"""
Shopify Storefront API Client
Handles product queries, cart operations, and checkout
"""
from typing import List, Dict, Optional
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

from app.config import settings

class ShopifyClient:
    """Shopify Storefront API client"""
    
    def __init__(self):
        self.store_domain = settings.SHOPIFY_STORE_DOMAIN
        self.access_token = settings.SHOPIFY_STOREFRONT_ACCESS_TOKEN
        
        # Setup GraphQL client
        transport = AIOHTTPTransport(
            url=f"https://{self.store_domain}/api/2024-01/graphql.json",
            headers={"X-Shopify-Storefront-Access-Token": self.access_token}
        )
        # Avoid schema fetching to keep startup fast.
        self.client = Client(transport=transport, fetch_schema_from_transport=False)
    
    async def search_products(
        self, 
        query: str = "", 
        limit: int = 10
    ) -> List[Dict]:
        """
        Search products in Shopify store
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of product dictionaries
        """
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
            products.append({
                "id": node["id"],
                "title": node["title"],
                "description": node["description"],
                "price": float(node["priceRange"]["minVariantPrice"]["amount"]),
                "currency": node["priceRange"]["minVariantPrice"]["currencyCode"],
                "image": node["images"]["edges"][0]["node"]["url"] if node["images"]["edges"] else None,
                "available": node["variants"]["edges"][0]["node"]["availableForSale"] if node["variants"]["edges"] else False
            })
        
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
                }
            }
        """)
        
        result = await self.client.execute_async(product_query, variable_values={"id": product_id})
        
        if result.get("product"):
            node = result["product"]
            return {
                "id": node["id"],
                "title": node["title"],
                "description": node["description"],
                "price": float(node["priceRange"]["minVariantPrice"]["amount"]),
                "currency": node["priceRange"]["minVariantPrice"]["currencyCode"],
                "images": [edge["node"]["url"] for edge in node["images"]["edges"]],
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

# Singleton instance
_shopify_client = None

def get_shopify_client() -> ShopifyClient:
    """Get or create Shopify client instance"""
    global _shopify_client
    if _shopify_client is None:
        _shopify_client = ShopifyClient()
    return _shopify_client
