from typing import List
from langchain_core.tools import tool


class CartItem:

    def __init__(self, name, price):
        self.name = name
        self.price = price


class Cart:

    def __init__(self):
        self.items = []

    def add_item(self, item_name: str, item_price: float):
        cart_item = CartItem(item_name, item_price)
        self.items.append(cart_item)


cart = Cart()


@tool
def add_to_cart(item_name: str, item_price: float) -> str:
    """Add an item to the cart."""
    cart.add_item(item_name, item_price)
    return f"{item_name} has been added to the cart."


@tool
def show_cart() -> List[CartItem]:
    """Show the contents of the cart."""
    return cart.items
