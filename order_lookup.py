
import csv
from pathlib import Path


# Locate the orders.csv file
ORDERS_FILE = Path(__file__).parent / "orders.csv"


def lookup_order(order_id: str) -> str:
    """Find an order using its order ID."""

    order_id = order_id.strip().upper()

    if not ORDERS_FILE.exists():
        return "Error: orders.csv was not found."

    with open(ORDERS_FILE, mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for order in reader:
            if order["order_id"].strip().upper() == order_id:
                return (
                    f"Order ID: {order['order_id']}\n"
                    f"Customer: {order['customer_name']}\n"
                    f"Product: {order['product']}\n"
                    f"Category: {order['category']}\n"
                    f"Order Date: {order['order_date']}\n"
                    f"Status: {order['status']}\n"
                    f"Expected/Delivered Date: "
                    f"{order['expected_or_delivered_date']}\n"
                    f"Amount: ₹{order['amount_inr']}\n"
                    f"Payment Method: {order['payment_method']}"
                )

    return f"Sorry, no order was found with ID {order_id}."


if __name__ == "__main__":
    print("KartEase Order Lookup")
    order_id = input("Enter your order ID: ")

    print()
    print(lookup_order(order_id))