import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


import json
from openai import OpenAI
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# 1. SETUP & TOOL DEFINITIONS
# ---------------------------------------------------------

client = OpenAI()

def get_stock_level_tool(item_name: str):
    """Wrapper for get_stock_level to return a clean string."""
    try:
        # Use a query that calculates net stock (Orders - Sales)
        query = f"""
        SELECT 
            (SELECT COALESCE(SUM(units),0) FROM transactions WHERE item_name='{item_name}' AND transaction_type='stock_orders') -
            (SELECT COALESCE(SUM(units),0) FROM transactions WHERE item_name='{item_name}' AND transaction_type='sales') 
        as current_stock
        """
        df = pd.read_sql(query, db_engine)
        stock = int(df.iloc[0]['current_stock']) if not df.empty and df.iloc[0]['current_stock'] is not None else 0
        return str(stock)
    except Exception as e:
        return f"Error checking stock: {str(e)}"

def get_all_inventory_tool():
    """Wrapper to return all inventory as JSON string."""
    try:
        report = {}
        for item in paper_supplies:
            name = item['item_name']
            stock = get_stock_level_tool(name)
            report[name] = int(stock)
        return json.dumps(report)
    except Exception as e:
        return f"Error getting inventory: {str(e)}"

def get_delivery_date_tool(item_name: str, quantity: int):
    """Wrapper for delivery date."""
    try:
        return get_supplier_delivery_date(datetime.now().isoformat(), quantity)
    except Exception as e:
        return f"Error calculating date: {str(e)}"

def search_history_tool(item_name: str):
    """Wrapper for search_quote_history."""
    try:
        results = search_quote_history([item_name], limit=3)
        if not results:
            return "No past quotes found."
        return json.dumps(results)
    except Exception as e:
        return f"Error searching history: {str(e)}"
        
def create_transaction_tool(item_name: str, transaction_type: str, quantity: int, price: float, date: str):
    """Wrapper for create_transaction."""
    try:
        tid = create_transaction(item_name, transaction_type, quantity, price, date)
        return f"Transaction Successful. ID: {tid}"
    except Exception as e:
        return f"Transaction Failed: {str(e)}"

def get_cash_tool():
    """Wrapper for get_cash_balance."""
    try:
        bal = get_cash_balance(datetime.now())
        return f"{bal:.2f}"
    except Exception as e:
        return "Error retrieving cash balance."

def generate_report_tool():
    """Wrapper for financial report."""
    try:
        rep = generate_financial_report(datetime.now())
        return f"Cash: ${rep['cash_balance']:.2f}, Inventory Value: ${rep['inventory_value']:.2f}"
    except Exception as e:
        return "Error generating report."

# --- Tool Schemas for OpenAI ---

inventory_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_level",
            "description": "Check current stock quantity for an item.",
            "parameters": {"type": "object", "properties": {"item_name": {"type": "string"}}, "required": ["item_name"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_inventory",
            "description": "Get a list of all items and their stock levels.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_supplier_delivery_date",
            "description": "Calculate estimated delivery date for a restocking order.",
            "parameters": {"type": "object", "properties": {"item_name": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["item_name", "quantity"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_transaction",
            "description": "Place a STOCK ORDER (buy from supplier). Transaction type must be 'stock_orders'.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "item_name": {"type": "string"}, 
                    "transaction_type": {"type": "string", "enum": ["stock_orders"]}, 
                    "quantity": {"type": "integer"}, 
                    "price": {"type": "number"}, 
                    "date": {"type": "string"}
                }, 
                "required": ["item_name", "transaction_type", "quantity", "price", "date"]
            }
        }
    }
]

quoting_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_quote_history",
            "description": "Search past quotes for pricing reference.",
            "parameters": {"type": "object", "properties": {"item_name": {"type": "string"}}, "required": ["item_name"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_level",
            "description": "Check if stock exists for a quote.",
            "parameters": {"type": "object", "properties": {"item_name": {"type": "string"}}, "required": ["item_name"]}
        }
    }
]

sales_tools = [
    {
        "type": "function",
        "function": {
            "name": "create_transaction",
            "description": "Finalize a SALE. Transaction type must be 'sales'.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "item_name": {"type": "string"}, 
                    "transaction_type": {"type": "string", "enum": ["sales"]}, 
                    "quantity": {"type": "integer"}, 
                    "price": {"type": "number"}, 
                    "date": {"type": "string"}
                }, 
                "required": ["item_name", "transaction_type", "quantity", "price", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cash_balance",
            "description": "Check available cash.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_financial_report",
            "description": "Get a financial summary.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# ---------------------------------------------------------
# 2. AGENT CLASSES
# ---------------------------------------------------------

class BaseAgent:
    def __init__(self, client, name, instructions, tools):
        self.client = client
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self.model = "gpt-4o"

    def execute(self, message):
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": message}
        ]
        
        # 1. Initial Call
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=self.tools, tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # 2. Tool Execution Loop
        if tool_calls:
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = "Error: Unknown tool"
                
                # Map tool names to wrapper functions
                if function_name == "get_stock_level":
                    result = get_stock_level_tool(args.get('item_name'))
                elif function_name == "get_all_inventory":
                    result = get_all_inventory_tool()
                elif function_name == "get_supplier_delivery_date":
                    result = get_delivery_date_tool(args.get('item_name'), args.get('quantity'))
                elif function_name == "search_quote_history":
                    result = search_history_tool(args.get('item_name'))
                elif function_name == "create_transaction":
                    result = create_transaction_tool(
                        args.get('item_name'), args.get('transaction_type'), 
                        args.get('quantity'), args.get('price'), args.get('date')
                    )
                elif function_name == "get_cash_balance":
                    result = get_cash_tool()
                elif function_name == "generate_financial_report":
                    result = generate_report_tool()
                
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id, 
                    "content": str(result)
                })
            
            # 3. Final Response after tool outputs
            final_response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return final_response.choices[0].message.content
            
        return response_message.content

class ManagerAgent:
    def __init__(self, client):
        self.client = client
        self.model = "gpt-4o"

    def route(self, text):
        prompt = f"""
        You are the Routing Manager. Analyze the request and route to:
        - INVENTORY: For stock checks, buying supplies, or low stock alerts.
        - QUOTING: For price checks, quotes, or history lookups.
        - SALES: For finalizing sales, confirming orders, or financial reports.
        
        Request: "{text}"
        Output only: INVENTORY, QUOTING, or SALES.
        """
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().upper()

# ---------------------------------------------------------
# 3. INITIALIZATION & EXECUTION LOOP
# ---------------------------------------------------------
catalog_text = "\n".join(
    [f"- {item['item_name']}: ${item['unit_price']} per unit" for item in paper_supplies]
)

def run_test_scenarios():
    print("Initializing Database...")
    init_database(db_engine)
    
    # 2. Initialize Agents (Updated with Catalog)

    inventory_agent = BaseAgent(client, "Inventory", 
        "You are the Inventory Manager. Check stock levels. "
        "If needed, buy supplies using 'create_transaction' with type='stock_orders'.", 
        inventory_tools
    )

    # Update Quoting Agent to see the catalog
    quoting_agent = BaseAgent(client, "Quoting", 
        f"You are the Quoting Specialist. Use these standard prices for quotes:\n{catalog_text}\n"
        "Check 'search_quote_history' for past deals. Check stock before quoting.", 
        quoting_tools
    )

    # Update Sales Agent to see the catalog so it can calculate total price
    sales_agent = BaseAgent(client, "Sales", 
        f"You are the Sales Rep. Finalize sales using 'create_transaction' with type='sales'.\n"
        f"CURRENT PRICE LIST:\n{catalog_text}\n"
        "Calculate the total price (quantity * unit_price) automatically. Do not ask the user for prices.", 
        sales_tools
    )

    manager = ManagerAgent(client)

    # Load Requests from DB (populated by init_database)
    requests_df = pd.read_sql("SELECT * FROM quote_requests", db_engine)

    results = []
    print("--- STARTING MULTI-AGENT BATCH PROCESSING ---")

    for index, row in requests_df.iterrows():
        req_id = row['id']
        req_text = row['response'] 
        req_date = "2025-04-05"
        
        print(f"\nProcessing Request {req_id}: {req_text}")

        # 1. Manager Routes
        target = manager.route(req_text)
        print(f" > Manager Routed to: {target}")

        # 2. Agent Executes
        agent_response = "No Action"
        if "INVENTORY" in target:
            agent_response = inventory_agent.execute(f"{req_text} (Date: {req_date})")
        elif "QUOTING" in target:
            agent_response = quoting_agent.execute(f"{req_text} (Date: {req_date})")
        elif "SALES" in target:
            agent_response = sales_agent.execute(f"{req_text} (Date: {req_date})")

        print(f" > Agent Response: {agent_response}")

        # 3. Capture Financial State
        report = generate_financial_report(req_date)
        
        results.append({
            "request_id": req_id,
            "request_date": req_date,
            "cash_balance": report['cash_balance'],
            "inventory_value": report['inventory_value'],
            "response": agent_response
        })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("test_results.csv", index=False)
    print("\nProcessing Complete. Results saved to test_results.csv")

if __name__ == "__main__":
    run_test_scenarios()