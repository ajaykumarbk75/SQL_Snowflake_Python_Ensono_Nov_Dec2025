"""
DEMO 2: BUILD SUMMARY REPORTS
Generate comprehensive business reports with Python
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


def create_sample_data():
    """Generate sample sales data"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=500, freq='D')
    
    df = pd.DataFrame({
        'date': np.random.choice(dates, 500),
        'customer_id': np.random.randint(1001, 1100, 500),
        'product_id': np.random.randint(101, 150, 500),
        'transaction_amount': np.random.uniform(10, 500, 500),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], 500),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 500),
        'customer_age': np.random.randint(18, 80, 500),
        'quantity': np.random.randint(1, 10, 500)
    })
    
    return df


# ============================================================================
# 1. BASIC SALES SUMMARY REPORT
# ============================================================================

def generate_sales_summary(df):
    """Generate overall sales summary statistics"""
    print("\n" + "="*70)
    print("REPORT 1: SALES SUMMARY STATISTICS")
    print("="*70)
    
    summary = {
        'Report_Date': datetime.now().strftime('%Y-%m-%d'),
        'Total_Sales': df['transaction_amount'].sum(),
        'Average_Transaction': df['transaction_amount'].mean(),
        'Median_Transaction': df['transaction_amount'].median(),
        'Std_Deviation': df['transaction_amount'].std(),
        'Min_Transaction': df['transaction_amount'].min(),
        'Max_Transaction': df['transaction_amount'].max(),
        'Total_Transactions': len(df),
        'Unique_Customers': df['customer_id'].nunique(),
        'Total_Quantity': df['quantity'].sum(),
        'Average_Quantity_Per_Transaction': df['quantity'].mean()
    }
    
    print("\n--- Key Metrics ---")
    for key, value in summary.items():
        if 'Total_Sales' in key or 'Average' in key or 'Median' in key or 'Std' in key or 'Min' in key or 'Max' in key:
            print(f"{key:.<40} ${value:>15,.2f}")
        else:
            print(f"{key:.<40} {value:>15}")
    
    return pd.DataFrame([summary])


# ============================================================================
# 2. CATEGORY-WISE ANALYSIS
# ============================================================================

def generate_category_report(df):
    """Analyze sales by product category"""
    print("\n" + "="*70)
    print("REPORT 2: SALES BY PRODUCT CATEGORY")
    print("="*70)
    
    category_analysis = df.groupby('product_category').agg({
        'transaction_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    category_analysis.columns = ['Total_Sales', 'Avg_Sale', 'Num_Transactions', 'Unique_Customers', 'Total_Quantity']
    category_analysis = category_analysis.sort_values('Total_Sales', ascending=False)
    category_analysis['% of Total Sales'] = (category_analysis['Total_Sales'] / category_analysis['Total_Sales'].sum() * 100).round(2)
    
    print("\n", category_analysis.to_string())
    
    return category_analysis.reset_index()


# ============================================================================
# 3. REGIONAL PERFORMANCE REPORT
# ============================================================================

def generate_regional_report(df):
    """Analyze sales by region"""
    print("\n" + "="*70)
    print("REPORT 3: REGIONAL PERFORMANCE ANALYSIS")
    print("="*70)
    
    regional_analysis = df.groupby('region').agg({
        'transaction_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique',
        'customer_age': 'mean',
        'quantity': 'sum'
    }).round(2)
    
    regional_analysis.columns = ['Total_Sales', 'Avg_Sale', 'Num_Transactions', 'Unique_Customers', 'Avg_Customer_Age', 'Total_Quantity']
    regional_analysis = regional_analysis.sort_values('Total_Sales', ascending=False)
    regional_analysis['% of Total Sales'] = (regional_analysis['Total_Sales'] / regional_analysis['Total_Sales'].sum() * 100).round(2)
    
    print("\n", regional_analysis.to_string())
    
    return regional_analysis.reset_index()


# ============================================================================
# 4. TOP PRODUCTS REPORT
# ============================================================================

def generate_top_products_report(df, top_n=10):
    """Identify top performing products"""
    print("\n" + "="*70)
    print(f"REPORT 4: TOP {top_n} PRODUCTS BY SALES")
    print("="*70)
    
    top_products = df.groupby('product_id').agg({
        'transaction_amount': ['sum', 'count', 'mean'],
        'quantity': 'sum',
        'product_category': 'first'
    }).round(2)
    
    top_products.columns = ['Total_Sales', 'Num_Purchases', 'Avg_Sale', 'Total_Qty_Sold', 'Category']
    top_products = top_products.sort_values('Total_Sales', ascending=False).head(top_n)
    top_products['Rank'] = range(1, len(top_products) + 1)
    
    print("\n", top_products[['Rank', 'Total_Sales', 'Num_Purchases', 'Avg_Sale', 'Category']].to_string())
    
    return top_products.reset_index()


# ============================================================================
# 5. CUSTOMER SEGMENTATION REPORT
# ============================================================================

def generate_customer_segmentation(df):
    """Segment customers by spending level"""
    print("\n" + "="*70)
    print("REPORT 5: CUSTOMER SEGMENTATION")
    print("="*70)
    
    customer_stats = df.groupby('customer_id').agg({
        'transaction_amount': ['sum', 'count', 'mean'],
        'quantity': 'sum',
        'date': ['min', 'max'],
        'customer_age': 'first'
    }).reset_index()
    
    customer_stats.columns = ['Customer_ID', 'Total_Spent', 'Num_Transactions', 'Avg_Transaction', 'Total_Qty', 'First_Purchase', 'Last_Purchase', 'Age']
    
    # Segment by spending quartiles
    def segment_customer(spending, quartile_25, quartile_50, quartile_75):
        if spending >= quartile_75:
            return 'Platinum'
        elif spending >= quartile_50:
            return 'Gold'
        elif spending >= quartile_25:
            return 'Silver'
        else:
            return 'Bronze'
    
    q25 = customer_stats['Total_Spent'].quantile(0.25)
    q50 = customer_stats['Total_Spent'].quantile(0.50)
    q75 = customer_stats['Total_Spent'].quantile(0.75)
    
    customer_stats['Segment'] = customer_stats['Total_Spent'].apply(
        lambda x: segment_customer(x, q25, q50, q75)
    )
    
    print("\n--- Customer Segments Distribution ---")
    segment_dist = customer_stats['Segment'].value_counts()
    print(segment_dist)
    
    print("\n--- Segment-wise Metrics ---")
    segment_metrics = customer_stats.groupby('Segment').agg({
        'Total_Spent': ['count', 'sum', 'mean'],
        'Num_Transactions': 'mean',
        'Age': 'mean'
    }).round(2)
    
    segment_metrics.columns = ['Num_Customers', 'Total_Revenue', 'Avg_Customer_Value', 'Avg_Transactions', 'Avg_Age']
    segment_metrics = segment_metrics.reindex(['Platinum', 'Gold', 'Silver', 'Bronze'])
    
    print("\n", segment_metrics.to_string())
    
    return customer_stats, segment_metrics


# ============================================================================
# 6. TIME-BASED ANALYSIS REPORT
# ============================================================================

def generate_time_based_report(df):
    """Analyze sales trends over time"""
    print("\n" + "="*70)
    print("REPORT 6: SALES TRENDS BY TIME PERIOD")
    print("="*70)
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Daily analysis
    daily_sales = df.groupby(df['date'].dt.date).agg({
        'transaction_amount': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    daily_sales.columns = ['Daily_Sales', 'Transactions', 'Avg_Transaction', 'Unique_Customers']
    
    print("\n--- Daily Sales (First 10 Days) ---")
    print(daily_sales.head(10).to_string())
    
    # Monthly analysis
    monthly_sales = df.groupby(df['date'].dt.to_period('M')).agg({
        'transaction_amount': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    monthly_sales.columns = ['Monthly_Sales', 'Transactions', 'Avg_Transaction', 'Unique_Customers']
    
    print("\n--- Monthly Sales ---")
    print(monthly_sales.to_string())
    
    return daily_sales, monthly_sales


# ============================================================================
# 7. GEOGRAPHIC HEATMAP REPORT
# ============================================================================

def generate_region_category_heatmap(df):
    """Cross-tabulation: Region vs Category"""
    print("\n" + "="*70)
    print("REPORT 7: SALES HEATMAP (REGION × CATEGORY)")
    print("="*70)
    
    heatmap = pd.pivot_table(
        df,
        values='transaction_amount',
        index='region',
        columns='product_category',
        aggfunc='sum'
    ).round(2)
    
    print("\n--- Total Sales by Region and Category ---")
    print(heatmap)
    
    # Add totals
    heatmap['Region Total'] = heatmap.sum(axis=1)
    heatmap.loc['Category Total'] = heatmap.sum()
    
    print("\n--- With Totals ---")
    print(heatmap)
    
    return heatmap


# ============================================================================
# 8. CUSTOMER RETENTION REPORT
# ============================================================================

def generate_retention_report(df):
    """Analyze customer purchase frequency"""
    print("\n" + "="*70)
    print("REPORT 8: CUSTOMER RETENTION & REPEAT PURCHASE")
    print("="*70)
    
    customer_purchases = df.groupby('customer_id').size().reset_index(name='Purchase_Count')
    
    print("\n--- Purchase Frequency Distribution ---")
    freq_dist = customer_purchases['Purchase_Count'].value_counts().sort_index()
    for count, num_customers in freq_dist.items():
        print(f"Customers with {count:>2} purchases: {num_customers:>4} customers")
    
    # Retention metrics
    one_time_customers = (customer_purchases['Purchase_Count'] == 1).sum()
    repeat_customers = (customer_purchases['Purchase_Count'] > 1).sum()
    high_value_customers = (customer_purchases['Purchase_Count'] >= 5).sum()
    
    retention_metrics = {
        'Total_Customers': len(customer_purchases),
        'One_Time_Customers': one_time_customers,
        'Repeat_Customers': repeat_customers,
        'High_Value_Customers': high_value_customers,
        'One_Time_%': round(one_time_customers / len(customer_purchases) * 100, 2),
        'Repeat_%': round(repeat_customers / len(customer_purchases) * 100, 2),
        'High_Value_%': round(high_value_customers / len(customer_purchases) * 100, 2),
        'Avg_Purchases_Per_Customer': round(customer_purchases['Purchase_Count'].mean(), 2)
    }
    
    print("\n--- Retention Metrics ---")
    for key, value in retention_metrics.items():
        print(f"{key:.<40} {value:>10}")
    
    return customer_purchases, retention_metrics


# ============================================================================
# 9. EXPORT REPORTS TO FILES
# ============================================================================

def export_all_reports(df, output_dir='./reports'):
    """Export all reports to CSV and JSON files"""
    print("\n" + "="*70)
    print("EXPORTING ALL REPORTS TO FILES")
    print("="*70)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Sales Summary
    sales_summary = generate_sales_summary(df)
    sales_summary.to_csv(f'{output_dir}/1_sales_summary.csv', index=False)
    print(f"✓ Exported: 1_sales_summary.csv")
    
    # Category Report
    category_report = generate_category_report(df)
    category_report.to_csv(f'{output_dir}/2_category_analysis.csv', index=False)
    print(f"✓ Exported: 2_category_analysis.csv")
    
    # Regional Report
    regional_report = generate_regional_report(df)
    regional_report.to_csv(f'{output_dir}/3_regional_analysis.csv', index=False)
    print(f"✓ Exported: 3_regional_analysis.csv")
    
    # Top Products
    top_products = generate_top_products_report(df)
    top_products.to_csv(f'{output_dir}/4_top_products.csv', index=False)
    print(f"✓ Exported: 4_top_products.csv")
    
    # Customer Segmentation
    customer_stats, segment_metrics = generate_customer_segmentation(df)
    customer_stats.to_csv(f'{output_dir}/5_customer_segments.csv', index=False)
    segment_metrics.to_csv(f'{output_dir}/5_segment_metrics.csv')
    print(f"✓ Exported: 5_customer_segments.csv")
    print(f"✓ Exported: 5_segment_metrics.csv")
    
    # Time-based
    daily_sales, monthly_sales = generate_time_based_report(df)
    daily_sales.to_csv(f'{output_dir}/6_daily_sales.csv')
    monthly_sales.to_csv(f'{output_dir}/6_monthly_sales.csv')
    print(f"✓ Exported: 6_daily_sales.csv")
    print(f"✓ Exported: 6_monthly_sales.csv")
    
    # Heatmap
    heatmap = generate_region_category_heatmap(df)
    heatmap.to_csv(f'{output_dir}/7_region_category_heatmap.csv')
    print(f"✓ Exported: 7_region_category_heatmap.csv")
    
    # Retention
    customer_purchases, retention_metrics = generate_retention_report(df)
    customer_purchases.to_csv(f'{output_dir}/8_customer_purchases.csv', index=False)
    
    with open(f'{output_dir}/8_retention_metrics.json', 'w') as f:
        json.dump(retention_metrics, f, indent=4)
    
    print(f"✓ Exported: 8_customer_purchases.csv")
    print(f"✓ Exported: 8_retention_metrics.json")
    
    print(f"\n✓ All reports exported to: {os.path.abspath(output_dir)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DEMO 2: BUILD COMPREHENSIVE SUMMARY REPORTS")
    print("="*70)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nDataset loaded: {len(df)} transactions")
    
    # Generate all reports
    print("\n" + "#"*70)
    print("# GENERATING ALL REPORTS")
    print("#"*70)
    
    generate_sales_summary(df)
    generate_category_report(df)
    generate_regional_report(df)
    generate_top_products_report(df)
    generate_customer_segmentation(df)
    generate_time_based_report(df)
    generate_region_category_heatmap(df)
    generate_retention_report(df)
    
    # Export all reports
    print("\n\n" + "#"*70)
    print("# EXPORTING REPORTS")
    print("#"*70)
    
    export_all_reports(df)
    
    print("\n" + "="*70)
    print("ALL REPORTS GENERATED AND EXPORTED SUCCESSFULLY!")
    print("="*70)
