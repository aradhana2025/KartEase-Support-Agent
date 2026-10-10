
from order_lookup import lookup_order
from pathlib import Path


# Test 1: Existing order
def test_existing_order():
    result = lookup_order("KE1002")

    assert "KE1002" in result
    assert "Priya Ramesh" in result
    assert "Shipped" in result


# Test 2: Another existing order
def test_out_for_delivery_order():
    result = lookup_order("KE1004")

    assert "KE1004" in result
    assert "Out for delivery" in result


# Test 3: Nonexistent order
def test_invalid_order():
    result = lookup_order("KE9999")

    assert "no order was found" in result.lower()


# Test 4: Order ID ignores lowercase
def test_order_id_case_insensitive():
    result = lookup_order("ke1002")

    assert "KE1002" in result
    assert "Priya Ramesh" in result


# Test 5: Order ID ignores surrounding spaces
def test_order_id_with_spaces():
    result = lookup_order(" KE1002 ")

    assert "KE1002" in result


# Test 6: CSV file exists
def test_orders_file_exists():
    orders_file = Path(__file__).parent / "orders.csv"

    assert orders_file.exists()