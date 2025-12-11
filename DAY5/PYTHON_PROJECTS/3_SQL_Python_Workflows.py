"""
DEMO 3: END-TO-END SQL + PYTHON WORKFLOWS
Integrate Snowflake SQL with Python data processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MOCK DATA GENERATION (Replace with actual Snowflake queries)
# ============================================================================

def create_mock_sales_data():
    """Mock Snowflake sales_orders table"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=300, freq='D')
    
    return pd.DataFrame({
        'order_id': range(1001, 1301),
        'customer_id': np.random.randint(101, 150, 300),
        'order_date': np.random.choice(dates, 300),
        'amount': np.random.uniform(20, 500, 300),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 300),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 300)
    })


def create_mock_customer_data():
    """Mock Snowflake customers table"""
    np.random.seed(42)
    return pd.DataFrame({
        'customer_id': range(101, 150),
        'customer_name': [f'Customer_{i}' for i in range(101, 150)],
        'registration_date': [datetime.now() - timedelta(days=np.random.randint(30, 365)) for _ in range(49)],
        'segment': np.random.choice(['Premium', 'Standard', 'Budget'], 49)
    })


def create_mock_product_data():
    """Mock Snowflake products table"""
    return pd.DataFrame({
        'product_id': range(201, 250),
        'product_name': [f'Product_{i}' for i in range(201, 250)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 49),
        'price': np.random.uniform(10, 500, 49)
    })


# ============================================================================
# PART 1: SQL QUERIES (Example queries)
# ============================================================================

def show_sql_queries():
    """Show SQL query examples for data extraction"""
    print("\n" + "="*70)
    print("PART 1: SQL QUERY TEMPLATES FOR DATA EXTRACTION")
    print("="*70)
    
    queries = {
        'SALES_DATA': """
        SELECT 
            order_id,
            customer_id,
            order_date,
            amount,
            product_category,
            region,
            EXTRACT(YEAR FROM order_date) AS year,
            EXTRACT(MONTH FROM order_date) AS month
        FROM ANALYTICS_DB.PUBLIC.SALES_ORDERS
        WHERE order_date >= CURRENT_DATE - 90
        ORDER BY order_date DESC;
        """,
        
        'CUSTOMER_DATA': """
        SELECT 
            c.customer_id,
            c.customer_name,
            c.registration_date,
            c.segment,
            COUNT(so.order_id) AS num_orders,
            SUM(so.amount) AS lifetime_value
        FROM ANALYTICS_DB.PUBLIC.CUSTOMERS c
        LEFT JOIN ANALYTICS_DB.PUBLIC.SALES_ORDERS so ON c.customer_id = so.customer_id
        GROUP BY c.customer_id, c.customer_name, c.registration_date, c.segment;
        """,
        
        'PRODUCT_DATA': """
        SELECT 
            product_id,
            product_name,
            category,
            price,
            COUNT(DISTINCT order_id) AS times_sold,
            SUM(amount) AS total_revenue
        FROM ANALYTICS_DB.PUBLIC.PRODUCTS p
        LEFT JOIN ANALYTICS_DB.PUBLIC.SALES_ORDERS so ON p.product_id = so.product_id
        GROUP BY product_id, product_name, category, price;
        """,
        
        'SUMMARY_STATS': """
        SELECT 
            DATE_TRUNC('DAY', order_date) AS date,
            COUNT(*) AS total_orders,
            SUM(amount) AS daily_revenue,
            AVG(amount) AS avg_order_value,
            COUNT(DISTINCT customer_id) AS unique_customers
        FROM ANALYTICS_DB.PUBLIC.SALES_ORDERS
        GROUP BY DATE_TRUNC('DAY', order_date)
        ORDER BY date DESC;
        """
    }
    
    for query_name, query in queries.items():
        print(f"\n--- {query_name} ---")
        print(query)
    
    return queries


# ============================================================================
# PART 2: DATA EXTRACTION FROM SQL
# ============================================================================

def extract_sales_data():
    """Extract sales data from Snowflake (mocked)"""
    print("\n" + "="*70)
    print("PART 2: DATA EXTRACTION - Loading from SQL Sources")
    print("="*70)
    
    # In real scenario: 
    # cursor = connection.cursor(DictCursor)
    # cursor.execute(SALES_QUERY)
    # sales_df = pd.DataFrame(cursor.fetchall())
    
    print("\n--- Extracting Sales Data ---")
    sales_df = create_mock_sales_data()
    print(f"✓ Loaded {len(sales_df)} sales records")
    print(f"  Columns: {', '.join(sales_df.columns)}")
    print(f"\n{sales_df.head()}")
    
    print("\n--- Extracting Customer Data ---")
    customer_df = create_mock_customer_data()
    print(f"✓ Loaded {len(customer_df)} customer records")
    print(f"  Columns: {', '.join(customer_df.columns)}")
    print(f"\n{customer_df.head()}")
    
    print("\n--- Extracting Product Data ---")
    product_df = create_mock_product_data()
    print(f"✓ Loaded {len(product_df)} product records")
    print(f"  Columns: {', '.join(product_df.columns)}")
    print(f"\n{product_df.head()}")
    
    return sales_df, customer_df, product_df


# ============================================================================
# PART 3: DATA TRANSFORMATION & ENRICHMENT
# ============================================================================

def transform_data(sales_df, customer_df, product_df):
    """Transform and enrich data with Python"""
    print("\n" + "="*70)
    print("PART 3: DATA TRANSFORMATION & ENRICHMENT")
    print("="*70)
    
    # Step 1: Merge Sales with Customer Data
    print("\n--- Step 1: Merging Sales with Customer Data ---")
    merged_df = sales_df.merge(customer_df, on='customer_id', how='left')
    print(f"✓ Merged shape: {merged_df.shape}")
    print(f"  New columns: {', '.join(customer_df.columns[1:])}")
    
    # Step 2: Merge with Product Data
    print("\n--- Step 2: Merging with Product Data ---")
    # (In this example, we'll add product price separately)
    product_price_map = dict(zip(product_df['product_id'], product_df['price']))
    merged_df['product_price'] = merged_df['product_id'] = np.random.randint(201, 250, len(merged_df))
    merged_df['product_price'] = merged_df['product_id'].map(product_price_map)
    print(f"✓ Added product pricing information")
    
    # Step 3: Data Quality & Validation
    print("\n--- Step 3: Data Quality Checks ---")
    print(f"  Total records: {len(merged_df)}")
    print(f"  Null values in amount: {merged_df['amount'].isnull().sum()}")
    print(f"  Null values in customer_name: {merged_df['customer_name'].isnull().sum()}")
    
    # Drop rows with critical nulls
    merged_df = merged_df.dropna(subset=['amount', 'customer_id'])
    print(f"  After cleaning: {len(merged_df)} records")
    
    # Step 4: Feature Engineering
    print("\n--- Step 4: Feature Engineering ---")
    
    # Convert date to datetime
    merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
    
    # Extract date features
    merged_df['year'] = merged_df['order_date'].dt.year
    merged_df['month'] = merged_df['order_date'].dt.month
    merged_df['day_of_week'] = merged_df['order_date'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6])
    
    # Calculate metrics
    merged_df['profit_margin'] = merged_df['amount'] * 0.20  # Assume 20% margin
    
    # Categorize order size
    def categorize_order(amount):
        if amount < 50:
            return 'Small'
        elif amount < 200:
            return 'Medium'
        else:
            return 'Large'
    
    merged_df['order_size'] = merged_df['amount'].apply(categorize_order)
    
    print(f"  ✓ Added temporal features (year, month, day_of_week, is_weekend)")
    print(f"  ✓ Added profit_margin column")
    print(f"  ✓ Added order_size categorization")
    
    # Step 5: Create Aggregated Features
    print("\n--- Step 5: Create Customer-level Aggregates ---")
    
    customer_agg = merged_df.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'profit_margin': 'sum',
        'customer_name': 'first',
        'segment': 'first'
    }).reset_index()
    
    customer_agg.columns = ['customer_id', 'total_spent', 'avg_order_value', 'num_orders', 'total_profit', 'customer_name', 'segment']
    
    print(f"✓ Created customer aggregates: {len(customer_agg)} unique customers")
    print(customer_agg.head())
    
    return merged_df, customer_agg


# ============================================================================
# PART 4: DATA ANALYSIS & INSIGHTS
# ============================================================================

def analyze_enriched_data(merged_df, customer_agg):
    """Perform analysis on transformed data"""
    print("\n" + "="*70)
    print("PART 4: DATA ANALYSIS & BUSINESS INSIGHTS")
    print("="*70)
    
    # Analysis 1: Revenue by Region & Category
    print("\n--- Analysis 1: Revenue Distribution ---")
    region_category = pd.pivot_table(
        merged_df,
        values='amount',
        index='region',
        columns='product_category',
        aggfunc='sum'
    ).round(2)
    print(region_category)
    
    # Analysis 2: Customer Segment Performance
    print("\n--- Analysis 2: Customer Segment Performance ---")
    segment_performance = customer_agg.groupby('segment').agg({
        'total_spent': ['sum', 'mean'],
        'num_orders': 'mean',
        'customer_id': 'count'
    }).round(2)
    segment_performance.columns = ['Total_Revenue', 'Avg_Spend', 'Avg_Orders', 'Num_Customers']
    print(segment_performance)
    
    # Analysis 3: Order Size Distribution
    print("\n--- Analysis 3: Order Size Distribution ---")
    order_size_dist = merged_df['order_size'].value_counts().sort_index()
    print(order_size_dist)
    
    # Analysis 4: Weekend vs Weekday
    print("\n--- Analysis 4: Weekend vs Weekday Sales ---")
    weekend_analysis = merged_df.groupby('is_weekend').agg({
        'amount': ['sum', 'mean', 'count']
    }).round(2)
    weekend_analysis.index = ['Weekday', 'Weekend']
    weekend_analysis.columns = ['Total_Sales', 'Avg_Sale', 'Num_Transactions']
    print(weekend_analysis)
    
    return region_category, segment_performance, weekend_analysis


# ============================================================================
# PART 5: DATA AGGREGATION & SUMMARIZATION
# ============================================================================

def summarize_for_reporting(merged_df, customer_agg):
    """Create summary tables for reporting"""
    print("\n" + "="*70)
    print("PART 5: SUMMARIZATION FOR REPORTING")
    print("="*70)
    
    # Summary 1: Daily Summary
    print("\n--- Summary 1: Daily Sales Summary ---")
    daily_summary = merged_df.groupby(merged_df['order_date'].dt.date).agg({
        'amount': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).reset_index()
    daily_summary.columns = ['date', 'total_sales', 'num_orders', 'avg_order', 'unique_customers']
    print(daily_summary.head(10))
    
    # Summary 2: Customer Segment Summary
    print("\n--- Summary 2: Top Customers by Spending ---")
    top_customers = customer_agg.nlargest(10, 'total_spent')[['customer_name', 'segment', 'total_spent', 'num_orders', 'avg_order_value']]
    print(top_customers.to_string(index=False))
    
    # Summary 3: Region Performance
    print("\n--- Summary 3: Regional Performance Summary ---")
    region_summary = merged_df.groupby('region').agg({
        'amount': ['sum', 'count', 'mean'],
        'profit_margin': 'sum',
        'customer_id': 'nunique'
    }).round(2)
    region_summary.columns = ['Total_Revenue', 'Num_Orders', 'Avg_Order', 'Total_Profit', 'Unique_Customers']
    print(region_summary)
    
    return daily_summary, top_customers, region_summary


# ============================================================================
# PART 6: LOAD BACK TO WAREHOUSE
# ============================================================================

def prepare_for_warehouse_load(merged_df, customer_agg, daily_summary):
    """Prepare data for loading back to Snowflake"""
    print("\n" + "="*70)
    print("PART 6: PREPARE FOR WAREHOUSE LOADING")
    print("="*70)
    
    print("\n--- Data to be loaded to Snowflake ---")
    
    # Table 1: Enriched Sales Data
    print("\n--- Table 1: ENRICHED_SALES_ORDERS ---")
    enriched_sales = merged_df[[
        'order_id', 'customer_id', 'order_date', 'amount', 'product_category', 
        'region', 'customer_name', 'segment', 'order_size', 'profit_margin',
        'year', 'month', 'day_of_week', 'is_weekend'
    ]]
    print(f"Shape: {enriched_sales.shape}")
    print(f"Columns: {', '.join(enriched_sales.columns)}")
    print("\nSQL to create table:")
    print("""
    CREATE OR REPLACE TABLE ANALYTICS_DB.PUBLIC.ENRICHED_SALES_ORDERS AS
    SELECT * FROM ENRICHED_SALES_ORDERS_STAGING;
    """)
    
    # Table 2: Customer Aggregates
    print("\n--- Table 2: CUSTOMER_AGGREGATES ---")
    print(f"Shape: {customer_agg.shape}")
    print(f"Columns: {', '.join(customer_agg.columns)}")
    print(customer_agg.head(5))
    print("\nSQL to create table:")
    print("""
    CREATE OR REPLACE TABLE ANALYTICS_DB.PUBLIC.CUSTOMER_AGGREGATES AS
    SELECT * FROM CUSTOMER_AGGREGATES_STAGING;
    """)
    
    # Table 3: Daily Summary
    print("\n--- Table 3: DAILY_SALES_SUMMARY ---")
    print(f"Shape: {daily_summary.shape}")
    print(f"Columns: {', '.join(daily_summary.columns)}")
    print(daily_summary.head(5))
    print("\nSQL to create table:")
    print("""
    CREATE OR REPLACE TABLE ANALYTICS_DB.PUBLIC.DAILY_SALES_SUMMARY AS
    SELECT * FROM DAILY_SUMMARY_STAGING;
    """)
    
    print("\n--- Load Instructions ---")
    print("""
    # Python Code to Load to Snowflake:
    
    import snowflake.connector
    from io import StringIO
    
    # Create connection
    conn = snowflake.connector.connect(
        user='YOUR_USER',
        password='YOUR_PASSWORD',
        account='YOUR_ACCOUNT',
        warehouse='COMPUTE_WH',
        database='ANALYTICS_DB',
        schema='PUBLIC'
    )
    
    cursor = conn.cursor()
    
    # Method 1: Using pandas write_pandas (recommended)
    from snowflake.connector.pandas_tools import write_pandas
    
    success, nchunks, nrows, output = write_pandas(
        conn,
        enriched_sales,
        'ENRICHED_SALES_ORDERS_STAGING',
        auto_create_table=True,
        overwrite=True
    )
    
    # Method 2: Using CSV staging area
    csv_buffer = StringIO()
    enriched_sales.to_csv(csv_buffer, index=False)
    
    cursor.execute("PUT 'file:///tmp/enriched_sales.csv' @~/staged_files/")
    cursor.execute(\"\"\"COPY INTO ENRICHED_SALES_ORDERS_STAGING 
                      FROM @~/staged_files/enriched_sales.csv\"\"\")
    
    conn.close()
    """)
    
    return enriched_sales, customer_agg, daily_summary


# ============================================================================
# PART 7: COMPLETE WORKFLOW PIPELINE
# ============================================================================

def show_complete_workflow():
    """Display the complete ETL pipeline"""
    print("\n" + "="*70)
    print("COMPLETE END-TO-END WORKFLOW PIPELINE")
    print("="*70)
    
    workflow = """
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    DATA EXTRACTION                          │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │  SELECT * FROM sales_orders (90 days)              │  │
    │  │  SELECT * FROM customers                           │  │
    │  │  SELECT * FROM products                            │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                           ↓                                 │
    │         Read into Pandas DataFrames in Python              │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │              DATA TRANSFORMATION & ENRICHMENT               │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │  1. Merge tables on customer_id, product_id         │  │
    │  │  2. Data quality checks (nulls, validation)         │  │
    │  │  3. Feature engineering:                            │  │
    │  │     - Date extraction (year, month, day_of_week)    │  │
    │  │     - Derived columns (profit, order_size)          │  │
    │  │     - Categorization & encoding                     │  │
    │  │  4. Create aggregates (customer-level stats)        │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                           ↓                                 │
    │              Processed DataFrames with features             │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   DATA ANALYSIS                             │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │  - Revenue distribution analysis                    │  │
    │  │  - Customer segment performance                     │  │
    │  │  - Trend identification                             │  │
    │  │  - Business metrics calculation                     │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                           ↓                                 │
    │                 Business Insights & Reports                │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │              LOAD BACK TO WAREHOUSE                         │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │  write_pandas() → ENRICHED_SALES_ORDERS             │  │
    │  │  write_pandas() → CUSTOMER_AGGREGATES               │  │
    │  │  write_pandas() → DAILY_SALES_SUMMARY               │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                           ↓                                 │
    │              Updated Snowflake Warehouse                    │
    └─────────────────────────────────────────────────────────────┘
    """
    
    print(workflow)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DEMO 3: END-TO-END SQL + PYTHON WORKFLOWS")
    print("="*70)
    
    # Part 1: Show SQL Queries
    show_sql_queries()
    
    # Part 2: Extract Data
    sales_df, customer_df, product_df = extract_sales_data()
    
    # Part 3: Transform Data
    merged_df, customer_agg = transform_data(sales_df, customer_df, product_df)
    
    # Part 4: Analysis
    region_category, segment_performance, weekend_analysis = analyze_enriched_data(merged_df, customer_agg)
    
    # Part 5: Summarization
    daily_summary, top_customers, region_summary = summarize_for_reporting(merged_df, customer_agg)
    
    # Part 6: Prepare for Load
    enriched_sales, customer_agg_final, daily_summary_final = prepare_for_warehouse_load(merged_df, customer_agg, daily_summary)
    
    # Part 7: Show Complete Workflow
    show_complete_workflow()
    
    print("\n" + "="*70)
    print("END-TO-END WORKFLOW DEMONSTRATION COMPLETED!")
    print("="*70)
