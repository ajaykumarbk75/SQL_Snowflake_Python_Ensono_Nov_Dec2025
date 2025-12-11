"""
Day 7: Python for Data Handling (Part 2) - 2.30 hrs
Topics:
1. Python for Feature Engineering
2. Hands-on: Build Summary Reports
3. Building End-to-End SQL + Python Workflows
4. Error Handling & Optimization
5. Hands-on: Automating Weekly Reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import snowflake.connector
from snowflake.connector import DictCursor
import logging
import time
from functools import wraps
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SnowflakeConnection:
    """Manage Snowflake connections with context manager"""
    def __init__(self, user, password, account):
        self.user = user
        self.password = password
        self.account = account
        self.conn = None
    
    def __enter__(self):
        try:
            self.conn = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse='COMPUTE_WH',
                database='ANALYTICS_DB',
                schema='PUBLIC'
            )
            logger.info("Connected to Snowflake successfully")
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            logger.info("Snowflake connection closed")


# ============================================================================
# 1. PYTHON FOR FEATURE ENGINEERING
# ============================================================================

class FeatureEngineering:
    """Feature engineering techniques for data preparation"""
    
    @staticmethod
    def create_sample_data():
        """Create sample dataset for feature engineering"""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'customer_id': np.random.randint(1001, 1050, 100),
            'transaction_amount': np.random.uniform(10, 500, 100),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'age': np.random.randint(18, 80, 100)
        })
        return df
    
    @staticmethod
    def numerical_transformations(df):
        """Apply numerical transformations"""
        logger.info("Applying numerical transformations...")
        
        df_transformed = df.copy()
        
        # 1. Scaling (Min-Max Normalization)
        df_transformed['transaction_amount_scaled'] = (
            (df['transaction_amount'] - df['transaction_amount'].min()) / 
            (df['transaction_amount'].max() - df['transaction_amount'].min())
        )
        
        # 2. Standardization (Z-score)
        df_transformed['age_standardized'] = (
            (df['age'] - df['age'].mean()) / df['age'].std()
        )
        
        # 3. Logarithmic transformation (handling skewness)
        df_transformed['amount_log'] = np.log1p(df['transaction_amount'])
        
        # 4. Binning/Bucketing
        df_transformed['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 50, 65, 100],
            labels=['Teen', 'Young', 'Middle', 'Senior', 'Elderly']
        )
        
        # 5. Polynomial features
        df_transformed['transaction_amount_squared'] = df['transaction_amount'] ** 2
        
        return df_transformed
    
    @staticmethod
    def categorical_encodings(df):
        """Apply categorical encodings"""
        logger.info("Applying categorical encodings...")
        
        df_encoded = df.copy()
        
        # 1. One-Hot Encoding
        category_dummies = pd.get_dummies(
            df['product_category'], 
            prefix='category', 
            drop_first=True
        )
        df_encoded = pd.concat([df_encoded, category_dummies], axis=1)
        
        # 2. Label Encoding for ordinal data
        region_mapping = {'North': 1, 'South': 2, 'East': 3, 'West': 4}
        df_encoded['region_encoded'] = df['region'].map(region_mapping)
        
        # 3. Frequency Encoding
        category_freq = df['product_category'].value_counts(normalize=True).to_dict()
        df_encoded['category_frequency'] = df['product_category'].map(category_freq)
        
        return df_encoded
    
    @staticmethod
    def temporal_features(df):
        """Extract temporal features"""
        logger.info("Extracting temporal features...")
        
        df_temporal = df.copy()
        
        # Extract date components
        df_temporal['year'] = df['date'].dt.year
        df_temporal['month'] = df['date'].dt.month
        df_temporal['day'] = df['date'].dt.day
        df_temporal['quarter'] = df['date'].dt.quarter
        df_temporal['day_of_week'] = df['date'].dt.dayofweek
        df_temporal['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Create cyclic features for day of week (to capture circular nature)
        df_temporal['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df_temporal['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Business day indicator
        df_temporal['is_business_day'] = df['date'].dt.dayofweek < 5
        
        return df_temporal
    
    @staticmethod
    def statistical_features(df, group_col='customer_id'):
        """Create aggregate statistical features"""
        logger.info("Creating statistical features...")
        
        df_stats = df.copy()
        
        # Aggregations by customer
        agg_dict = {
            'transaction_amount': ['sum', 'mean', 'std', 'count'],
            'age': 'first'
        }
        
        customer_stats = df.groupby(group_col).agg(agg_dict).reset_index()
        customer_stats.columns = ['_'.join(col).strip('_') for col in customer_stats.columns]
        
        # Merge back
        df_stats = df_stats.merge(customer_stats, on=group_col, how='left')
        
        # Calculate derived statistics
        df_stats['transaction_ratio'] = (
            df_stats['transaction_amount'] / (df_stats['transaction_amount_sum'] + 1)
        )
        
        return df_stats


# ============================================================================
# 2. HANDS-ON: BUILD SUMMARY REPORTS
# ============================================================================

class SummaryReports:
    """Build comprehensive summary reports"""
    
    @staticmethod
    def sales_summary(df):
        """Generate sales summary report"""
        logger.info("Generating sales summary report...")
        
        summary = {
            'Total_Sales': df['transaction_amount'].sum(),
            'Average_Sale': df['transaction_amount'].mean(),
            'Median_Sale': df['transaction_amount'].median(),
            'Std_Dev': df['transaction_amount'].std(),
            'Min_Sale': df['transaction_amount'].min(),
            'Max_Sale': df['transaction_amount'].max(),
            'Total_Transactions': len(df)
        }
        
        summary_df = pd.DataFrame([summary])
        logger.info(f"\n{summary_df.to_string()}")
        return summary_df
    
    @staticmethod
    def category_analysis(df):
        """Analyze sales by category"""
        logger.info("Analyzing sales by category...")
        
        category_summary = df.groupby('product_category').agg({
            'transaction_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        category_summary.columns = ['Total_Sales', 'Avg_Sale', 'Transactions', 'Unique_Customers']
        category_summary = category_summary.sort_values('Total_Sales', ascending=False)
        
        logger.info(f"\n{category_summary.to_string()}")
        return category_summary
    
    @staticmethod
    def regional_analysis(df):
        """Analyze sales by region"""
        logger.info("Analyzing sales by region...")
        
        region_summary = df.groupby('region').agg({
            'transaction_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'age': 'mean'
        }).round(2)
        
        region_summary.columns = ['Total_Sales', 'Avg_Sale', 'Transactions', 'Unique_Customers', 'Avg_Age']
        region_summary = region_summary.sort_values('Total_Sales', ascending=False)
        
        logger.info(f"\n{region_summary.to_string()}")
        return region_summary
    
    @staticmethod
    def customer_segmentation(df):
        """Segment customers by spending"""
        logger.info("Performing customer segmentation...")
        
        customer_spending = df.groupby('customer_id').agg({
            'transaction_amount': ['sum', 'count', 'mean'],
            'date': ['min', 'max']
        }).round(2)
        
        customer_spending.columns = ['Total_Spent', 'Num_Transactions', 'Avg_Transaction', 'First_Purchase', 'Last_Purchase']
        
        # Segment customers
        def segment_customer(spending):
            if spending > customer_spending['Total_Spent'].quantile(0.75):
                return 'Premium'
            elif spending > customer_spending['Total_Spent'].quantile(0.50):
                return 'Gold'
            elif spending > customer_spending['Total_Spent'].quantile(0.25):
                return 'Silver'
            else:
                return 'Bronze'
        
        customer_spending['Segment'] = customer_spending['Total_Spent'].apply(segment_customer)
        
        segment_summary = customer_spending.groupby('Segment').size()
        logger.info(f"\nCustomer Segments:\n{segment_summary.to_string()}")
        
        return customer_spending
    
    @staticmethod
    def export_reports(sales_summary, category_analysis, region_analysis, output_dir='./reports'):
        """Export reports to CSV"""
        logger.info(f"Exporting reports to {output_dir}...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        sales_summary.to_csv(f'{output_dir}/sales_summary.csv', index=False)
        category_analysis.to_csv(f'{output_dir}/category_analysis.csv')
        region_analysis.to_csv(f'{output_dir}/region_analysis.csv')
        
        logger.info("Reports exported successfully")


# ============================================================================
# 3. BUILDING END-TO-END SQL + PYTHON WORKFLOWS
# ============================================================================

class SQLPythonWorkflow:
    """Integrate SQL queries with Python processing"""
    
    @staticmethod
    def execute_sql_query(connection, query):
        """Execute SQL query and return DataFrame"""
        try:
            logger.info(f"Executing SQL query...")
            cursor = connection.cursor(DictCursor)
            cursor.execute(query)
            results = cursor.fetchall()
            df = pd.DataFrame(results)
            logger.info(f"Query executed successfully. Rows: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise
    
    @staticmethod
    def data_extraction_pipeline(connection):
        """Extract data from Snowflake using SQL"""
        
        queries = {
            'sales_data': """
                SELECT 
                    order_id,
                    customer_id,
                    order_date,
                    amount,
                    product_category,
                    region
                FROM sales_orders
                WHERE order_date >= CURRENT_DATE - 90
                ORDER BY order_date DESC
            """,
            'customer_data': """
                SELECT 
                    customer_id,
                    customer_name,
                    registration_date,
                    lifetime_value,
                    segment
                FROM customers
            """,
            'product_data': """
                SELECT 
                    product_id,
                    product_name,
                    category,
                    price
                FROM products
            """
        }
        
        # Simulating with sample data instead of actual Snowflake connection
        logger.info("Loading data from SQL sources...")
        
        return queries
    
    @staticmethod
    def data_transformation_pipeline(sales_df, customer_df):
        """Transform and enrich data"""
        logger.info("Transforming and enriching data...")
        
        # Merge datasets
        merged_df = sales_df.merge(customer_df, on='customer_id', how='left')
        
        # Data validation
        merged_df = merged_df.dropna(subset=['amount'])
        
        # Feature engineering
        merged_df['purchase_day_of_week'] = pd.to_datetime(merged_df['order_date']).dt.dayofweek
        
        logger.info(f"Transformed data shape: {merged_df.shape}")
        return merged_df
    
    @staticmethod
    def load_to_warehouse(connection, df, table_name):
        """Load processed data back to Snowflake"""
        try:
            logger.info(f"Loading data to {table_name}...")
            
            cursor = connection.cursor()
            
            # Create staging table
            create_stmt = f"""
                CREATE OR REPLACE TABLE {table_name} (
                    {', '.join([f'{col} VARCHAR' for col in df.columns])}
                )
            """
            cursor.execute(create_stmt)
            
            logger.info(f"Data loaded to {table_name} successfully")
            
        except Exception as e:
            logger.error(f"Load error: {e}")
            raise


# ============================================================================
# 4. ERROR HANDLING & OPTIMIZATION
# ============================================================================

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Completed: {func.__name__} in {duration:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


class ErrorHandling:
    """Error handling patterns and best practices"""
    
    @staticmethod
    @timing_decorator
    def safe_data_loading(file_path, dtype_dict=None):
        """Safely load data with error handling"""
        try:
            logger.info(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path, dtype=dtype_dict)
            
            # Validate data
            if df.empty:
                raise ValueError("Loaded dataframe is empty")
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data loading: {e}")
            raise
    
    @staticmethod
    def validate_data(df, required_columns, data_types=None):
        """Validate data quality"""
        logger.info("Validating data quality...")
        
        errors = []
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check data types
        if data_types:
            for col, dtype in data_types.items():
                if col in df.columns and df[col].dtype != dtype:
                    errors.append(f"Column {col}: expected {dtype}, got {df[col].dtype}")
        
        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
        
        if errors:
            logger.error(f"Data validation failed:\n" + "\n".join(errors))
            return False
        
        logger.info("Data validation passed")
        return True
    
    @staticmethod
    def handle_missing_values(df, strategy='mean'):
        """Handle missing values with different strategies"""
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if strategy == 'mean' and df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == 'mode':
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                
                logger.info(f"Fixed {col}")
        
        return df_clean
    
    @staticmethod
    @timing_decorator
    def optimize_memory_usage(df):
        """Optimize dataframe memory usage"""
        logger.info("Optimizing memory usage...")
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize integers
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                # Optimize floats
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
            
            # Convert objects to category
            elif col_type == 'object':
                num_unique = len(df[col].unique())
                num_total = len(df[col])
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = ((initial_memory - final_memory) / initial_memory) * 100
        
        logger.info(f"Memory usage reduced from {initial_memory:.2f}MB to {final_memory:.2f}MB ({reduction:.1f}% reduction)")
        
        return df


# ============================================================================
# 5. HANDS-ON: AUTOMATING WEEKLY REPORTING
# ============================================================================

class WeeklyReportingAutomation:
    """Automate weekly reporting workflow"""
    
    @staticmethod
    def generate_weekly_report(df, report_date=None):
        """Generate comprehensive weekly report"""
        if report_date is None:
            report_date = datetime.now()
        
        week_start = report_date - timedelta(days=report_date.weekday())
        week_end = week_start + timedelta(days=6)
        
        logger.info(f"Generating weekly report for {week_start.date()} to {week_end.date()}")
        
        # Filter data for the week
        if 'date' in df.columns:
            weekly_df = df[
                (pd.to_datetime(df['date']).dt.date >= week_start.date()) &
                (pd.to_datetime(df['date']).dt.date <= week_end.date())
            ]
        else:
            weekly_df = df
        
        report = {
            'Report_Date': report_date.strftime('%Y-%m-%d'),
            'Week_Start': week_start.strftime('%Y-%m-%d'),
            'Week_End': week_end.strftime('%Y-%m-%d'),
            'Total_Sales': weekly_df['transaction_amount'].sum(),
            'Total_Transactions': len(weekly_df),
            'Avg_Transaction': weekly_df['transaction_amount'].mean(),
            'Unique_Customers': weekly_df['customer_id'].nunique(),
            'Top_Category': weekly_df['product_category'].mode()[0] if len(weekly_df) > 0 else 'N/A',
            'Top_Region': weekly_df['region'].mode()[0] if len(weekly_df) > 0 else 'N/A'
        }
        
        return report
    
    @staticmethod
    def create_email_report(report_dict):
        """Format report for email"""
        logger.info("Formatting email report...")
        
        email_body = f"""
        WEEKLY SALES REPORT
        {'='*50}
        
        Report Date: {report_dict['Report_Date']}
        Period: {report_dict['Week_Start']} to {report_dict['Week_End']}
        
        KEY METRICS
        {'-'*50}
        Total Sales:           ${report_dict['Total_Sales']:,.2f}
        Total Transactions:    {report_dict['Total_Transactions']}
        Average Transaction:   ${report_dict['Avg_Transaction']:,.2f}
        Unique Customers:      {report_dict['Unique_Customers']}
        
        TOP PERFORMERS
        {'-'*50}
        Top Category:          {report_dict['Top_Category']}
        Top Region:            {report_dict['Top_Region']}
        
        {'='*50}
        """
        
        return email_body
    
    @staticmethod
    def schedule_weekly_reports(df, weeks=4):
        """Generate reports for multiple weeks"""
        logger.info(f"Generating {weeks} weeks of reports...")
        
        reports = []
        end_date = datetime.now()
        
        for week in range(weeks):
            report_date = end_date - timedelta(weeks=week)
            report = WeeklyReportingAutomation.generate_weekly_report(df, report_date)
            reports.append(report)
        
        reports_df = pd.DataFrame(reports)
        logger.info(f"\nGenerated Reports:\n{reports_df.to_string()}")
        
        return reports_df
    
    @staticmethod
    def export_weekly_report(report_dict, filename=None):
        """Export report to file"""
        if filename is None:
            filename = f"weekly_report_{report_dict['Report_Date']}.json"
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=4)
        
        logger.info(f"Report exported to {filename}")
        return filename


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute all demonstrations"""
    
    print("\n" + "="*80)
    print("DAY 7: PYTHON FOR DATA HANDLING (PART 2) - COMPREHENSIVE DEMOS")
    print("="*80 + "\n")
    
    # ========== 1. FEATURE ENGINEERING ==========
    print("\n" + "="*80)
    print("1. FEATURE ENGINEERING DEMONSTRATION")
    print("="*80)
    
    fe = FeatureEngineering()
    df = fe.create_sample_data()
    print(f"\nOriginal Data Shape: {df.shape}")
    print(df.head())
    
    # Numerical transformations
    print("\n--- Numerical Transformations ---")
    df_numerical = fe.numerical_transformations(df)
    print(df_numerical[['transaction_amount', 'transaction_amount_scaled', 'age_standardized', 'age_group']].head())
    
    # Categorical encodings
    print("\n--- Categorical Encodings ---")
    df_categorical = fe.categorical_encodings(df)
    print(df_categorical[['product_category', 'region_encoded', 'category_frequency']].head())
    
    # Temporal features
    print("\n--- Temporal Features ---")
    df_temporal = fe.temporal_features(df)
    print(df_temporal[['date', 'month', 'day_of_week', 'is_business_day']].head())
    
    # Statistical features
    print("\n--- Statistical Features ---")
    df_stats = fe.statistical_features(df)
    print(df_stats[['customer_id', 'transaction_amount_sum', 'transaction_amount_mean', 'transaction_ratio']].head())
    
    # ========== 2. SUMMARY REPORTS ==========
    print("\n\n" + "="*80)
    print("2. BUILDING SUMMARY REPORTS")
    print("="*80)
    
    sr = SummaryReports()
    
    print("\n--- Sales Summary ---")
    sales_summary = sr.sales_summary(df)
    
    print("\n--- Category Analysis ---")
    category_analysis = sr.category_analysis(df)
    
    print("\n--- Regional Analysis ---")
    regional_analysis = sr.regional_analysis(df)
    
    print("\n--- Customer Segmentation ---")
    customer_segments = sr.customer_segmentation(df)
    print(f"\nTop 5 Customers by Spending:\n{customer_segments.head()}")
    
    # ========== 3. SQL + PYTHON WORKFLOWS ==========
    print("\n\n" + "="*80)
    print("3. END-TO-END SQL + PYTHON WORKFLOWS")
    print("="*80)
    
    spw = SQLPythonWorkflow()
    
    print("\n--- Data Extraction Queries ---")
    queries = spw.data_extraction_pipeline(None)
    for name, query in queries.items():
        print(f"\n{name.upper()}:")
        print(query)
    
    print("\nData transformation pipeline configured")
    print("Load to warehouse function ready for use")
    
    # ========== 4. ERROR HANDLING & OPTIMIZATION ==========
    print("\n\n" + "="*80)
    print("4. ERROR HANDLING & OPTIMIZATION")
    print("="*80)
    
    eh = ErrorHandling()
    
    # Data validation
    print("\n--- Data Validation ---")
    required_cols = ['date', 'customer_id', 'transaction_amount', 'product_category', 'region', 'age']
    is_valid = eh.validate_data(df, required_cols)
    
    # Handle missing values
    print("\n--- Handling Missing Values ---")
    df_test = df.copy()
    df_test.loc[0:5, 'transaction_amount'] = np.nan
    df_clean = eh.handle_missing_values(df_test, strategy='mean')
    print("Missing values handled successfully")
    
    # Memory optimization
    print("\n--- Memory Optimization ---")
    df_optimized = eh.optimize_memory_usage(df.copy())
    
    # ========== 5. WEEKLY REPORTING AUTOMATION ==========
    print("\n\n" + "="*80)
    print("5. AUTOMATING WEEKLY REPORTING")
    print("="*80)
    
    wra = WeeklyReportingAutomation()
    
    # Generate single weekly report
    print("\n--- Weekly Report ---")
    weekly_report = wra.generate_weekly_report(df)
    print(json.dumps(weekly_report, indent=2, default=str))
    
    # Format for email
    print("\n--- Email Format ---")
    email_body = wra.create_email_report(weekly_report)
    print(email_body)
    
    # Schedule multiple weeks
    print("\n--- Multi-Week Reports ---")
    multi_week_reports = wra.schedule_weekly_reports(df, weeks=4)
    
    print("\n" + "="*80)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
