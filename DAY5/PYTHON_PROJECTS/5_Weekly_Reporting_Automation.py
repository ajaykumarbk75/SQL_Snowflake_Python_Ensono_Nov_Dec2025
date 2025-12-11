"""
DEMO 5: AUTOMATING WEEKLY REPORTING
Build automated reporting workflows with Python
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SETUP LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

def create_sales_data(days=30):
    """Generate sample sales data for the period"""
    np.random.seed(42)
    dates = pd.date_range(datetime.now() - timedelta(days=days), datetime.now(), freq='D')
    
    df = pd.DataFrame({
        'date': np.random.choice(dates, 300),
        'customer_id': np.random.randint(101, 150, 300),
        'transaction_amount': np.random.uniform(20, 500, 300),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 300),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 300),
        'quantity': np.random.randint(1, 5, 300)
    })
    
    return df.sort_values('date')


# ============================================================================
# PART 1: WEEKLY REPORT GENERATION
# ============================================================================

def get_week_boundaries(report_date=None):
    """Calculate week start and end dates"""
    if report_date is None:
        report_date = datetime.now()
    
    # Get Monday of the week
    week_start = report_date - timedelta(days=report_date.weekday())
    week_end = week_start + timedelta(days=6)
    
    return week_start, week_end, report_date


def filter_weekly_data(df, week_start, week_end):
    """Filter data for the specific week"""
    df['date'] = pd.to_datetime(df['date'])
    weekly_df = df[
        (df['date'].dt.date >= week_start.date()) & 
        (df['date'].dt.date <= week_end.date())
    ]
    return weekly_df


def calculate_weekly_metrics(df):
    """Calculate key metrics for the week"""
    metrics = {
        'total_sales': df['transaction_amount'].sum(),
        'total_transactions': len(df),
        'avg_transaction': df['transaction_amount'].mean(),
        'median_transaction': df['transaction_amount'].median(),
        'std_transaction': df['transaction_amount'].std(),
        'min_transaction': df['transaction_amount'].min(),
        'max_transaction': df['transaction_amount'].max(),
        'unique_customers': df['customer_id'].nunique(),
        'total_quantity': df['quantity'].sum(),
        'avg_quantity_per_transaction': df['quantity'].mean()
    }
    return metrics


def demo_weekly_report_generation():
    """Generate a single weekly report"""
    print("\n" + "="*70)
    print("DEMO 1: WEEKLY REPORT GENERATION")
    print("="*70)
    
    df = create_sales_data(days=30)
    week_start, week_end, report_date = get_week_boundaries()
    weekly_df = filter_weekly_data(df, week_start, week_end)
    metrics = calculate_weekly_metrics(weekly_df)
    
    print(f"\n--- Report Period: {week_start.date()} to {week_end.date()} ---")
    print(f"Report Generated: {report_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n--- Key Metrics ---")
    
    for metric, value in metrics.items():
        metric_name = metric.replace('_', ' ').title()
        if 'sales' in metric or 'transaction' in metric or 'quantity' in metric:
            print(f"{metric_name:.<40} ${value:>15,.2f}")
        else:
            print(f"{metric_name:.<40} {value:>15}")
    
    return metrics


# ============================================================================
# PART 2: DETAILED BREAKDOWN REPORTS
# ============================================================================

def generate_category_breakdown(df):
    """Breakdown sales by category"""
    print("\n--- Category Performance ---")
    category_sales = df.groupby('product_category').agg({
        'transaction_amount': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    category_sales.columns = ['Total_Sales', 'Transactions', 'Avg_Sale', 'Unique_Customers']
    category_sales = category_sales.sort_values('Total_Sales', ascending=False)
    print(category_sales)
    
    return category_sales


def generate_region_breakdown(df):
    """Breakdown sales by region"""
    print("\n--- Regional Performance ---")
    region_sales = df.groupby('region').agg({
        'transaction_amount': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    region_sales.columns = ['Total_Sales', 'Transactions', 'Avg_Sale', 'Unique_Customers']
    region_sales = region_sales.sort_values('Total_Sales', ascending=False)
    print(region_sales)
    
    return region_sales


def generate_daily_breakdown(df):
    """Breakdown sales by day"""
    print("\n--- Daily Breakdown ---")
    daily_sales = df.groupby(df['date'].dt.date).agg({
        'transaction_amount': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    daily_sales.columns = ['Total_Sales', 'Transactions', 'Avg_Sale', 'Unique_Customers']
    print(daily_sales.to_string())
    
    return daily_sales


def demo_detailed_breakdown():
    """Generate detailed breakdowns"""
    print("\n" + "="*70)
    print("DEMO 2: DETAILED BREAKDOWN REPORTS")
    print("="*70)
    
    df = create_sales_data(days=30)
    week_start, week_end, _ = get_week_boundaries()
    weekly_df = filter_weekly_data(df, week_start, week_end)
    
    category_breakdown = generate_category_breakdown(weekly_df)
    region_breakdown = generate_region_breakdown(weekly_df)
    daily_breakdown = generate_daily_breakdown(weekly_df)
    
    return category_breakdown, region_breakdown, daily_breakdown


# ============================================================================
# PART 3: COMPARISON & TREND ANALYSIS
# ============================================================================

def generate_comparison_report(current_week_metrics, previous_week_metrics):
    """Compare current week with previous week"""
    print("\n" + "="*70)
    print("DEMO 3: WEEK-OVER-WEEK COMPARISON")
    print("="*70)
    
    print("\n--- Comparison Metrics ---")
    print(f"{'Metric':<40} {'This Week':>15} {'Last Week':>15} {'Change':>15}")
    print("-" * 85)
    
    for metric in current_week_metrics.keys():
        current = current_week_metrics[metric]
        previous = previous_week_metrics[metric]
        
        if previous != 0:
            change_pct = ((current - previous) / previous) * 100
        else:
            change_pct = 0
        
        change_str = f"{change_pct:+.1f}%"
        
        if isinstance(current, float):
            print(f"{metric:.<40} ${current:>13,.2f} ${previous:>13,.2f} {change_str:>15}")
        else:
            print(f"{metric:.<40} {current:>15} {previous:>15} {change_str:>15}")
    
    return True


def demo_trend_analysis():
    """Analyze trends across multiple weeks"""
    print("\n" + "="*70)
    print("DEMO 4: MULTI-WEEK TREND ANALYSIS")
    print("="*70)
    
    df = create_sales_data(days=90)  # 3 months of data
    
    # Generate reports for 4 weeks
    print("\n--- Generating 4 Weeks of Reports ---")
    weekly_reports = []
    
    for week_num in range(4):
        report_date = datetime.now() - timedelta(weeks=week_num)
        week_start, week_end, _ = get_week_boundaries(report_date)
        
        weekly_df = filter_weekly_data(df, week_start, week_end)
        metrics = calculate_weekly_metrics(weekly_df)
        
        report = {
            'week': week_num + 1,
            'week_start': week_start.date(),
            'week_end': week_end.date(),
            **metrics
        }
        
        weekly_reports.append(report)
        print(f"✓ Week {week_num + 1}: {week_start.date()} - {week_end.date()}")
    
    # Create trend DataFrame
    trend_df = pd.DataFrame(weekly_reports)
    
    print("\n--- Trend Summary ---")
    print(trend_df[['week', 'week_start', 'week_end', 'total_sales', 'total_transactions']].to_string(index=False))
    
    # Calculate trends
    print("\n--- Trend Analysis ---")
    sales_trend = trend_df['total_sales'].pct_change().mean() * 100
    transaction_trend = trend_df['total_transactions'].pct_change().mean() * 100
    
    print(f"Average Weekly Sales Growth: {sales_trend:+.2f}%")
    print(f"Average Weekly Transaction Growth: {transaction_trend:+.2f}%")
    
    return trend_df


# ============================================================================
# PART 4: EMAIL REPORT FORMATTING
# ============================================================================

def format_html_report(metrics, category_data, region_data, week_start, week_end):
    """Format report as HTML for email"""
    html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #ecf0f1; }}
                .metric {{ margin: 10px 0; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Weekly Sales Report</h1>
            <p><strong>Period:</strong> {week_start.date()} to {week_end.date()}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Key Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Sales</td>
                    <td>${metrics['total_sales']:,.2f}</td>
                </tr>
                <tr>
                    <td>Total Transactions</td>
                    <td>{metrics['total_transactions']}</td>
                </tr>
                <tr>
                    <td>Average Transaction</td>
                    <td>${metrics['avg_transaction']:,.2f}</td>
                </tr>
                <tr>
                    <td>Unique Customers</td>
                    <td>{metrics['unique_customers']}</td>
                </tr>
            </table>
            
            <h2>Category Performance</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Sales</th>
                    <th>Transactions</th>
                    <th>Avg Sale</th>
                </tr>
                {''.join([f'<tr><td>{idx}</td><td>${row["Total_Sales"]:,.2f}</td><td>{row["Transactions"]}</td><td>${row["Avg_Sale"]:,.2f}</td></tr>' 
                          for idx, row in category_data.iterrows()])}
            </table>
            
            <h2>Regional Performance</h2>
            <table>
                <tr>
                    <th>Region</th>
                    <th>Sales</th>
                    <th>Transactions</th>
                    <th>Avg Sale</th>
                </tr>
                {''.join([f'<tr><td>{idx}</td><td>${row["Total_Sales"]:,.2f}</td><td>{row["Transactions"]}</td><td>${row["Avg_Sale"]:,.2f}</td></tr>' 
                          for idx, row in region_data.iterrows()])}
            </table>
        </body>
    </html>
    """
    return html


def format_text_report(metrics, category_data, region_data, week_start, week_end):
    """Format report as plain text"""
    report = f"""
    {'='*70}
    WEEKLY SALES REPORT
    {'='*70}
    
    Report Period: {week_start.date()} to {week_end.date()}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    KEY METRICS
    {'-'*70}
    Total Sales:              ${metrics['total_sales']:>15,.2f}
    Total Transactions:       {metrics['total_transactions']:>15}
    Average Transaction:      ${metrics['avg_transaction']:>15,.2f}
    Median Transaction:       ${metrics['median_transaction']:>15,.2f}
    Unique Customers:         {metrics['unique_customers']:>15}
    Total Quantity:           {metrics['total_quantity']:>15}
    
    CATEGORY PERFORMANCE
    {'-'*70}
    {category_data.to_string()}
    
    REGIONAL PERFORMANCE
    {'-'*70}
    {region_data.to_string()}
    
    {'='*70}
    """
    return report


def demo_email_formatting():
    """Format reports for email"""
    print("\n" + "="*70)
    print("DEMO 5: EMAIL REPORT FORMATTING")
    print("="*70)
    
    df = create_sales_data(days=30)
    week_start, week_end, _ = get_week_boundaries()
    weekly_df = filter_weekly_data(df, week_start, week_end)
    
    metrics = calculate_weekly_metrics(weekly_df)
    category_data = generate_category_breakdown(weekly_df)
    region_data = generate_region_breakdown(weekly_df)
    
    # Text format
    print("\n--- Text Report Preview ---")
    text_report = format_text_report(metrics, category_data, region_data, week_start, week_end)
    print(text_report[:500] + "\n... (truncated for display)")
    
    # HTML format (just show structure)
    print("\n--- HTML Report Created ---")
    html_report = format_html_report(metrics, category_data, region_data, week_start, week_end)
    print("✓ HTML report formatted successfully")
    print(f"  HTML size: {len(html_report)} characters")
    
    return text_report, html_report


# ============================================================================
# PART 5: SENDING EMAIL REPORTS
# ============================================================================

def send_email_report(subject, body, recipient_email, attachment_path=None, use_html=False):
    """Send report via email (mock implementation)"""
    print("\n" + "="*70)
    print("DEMO 6: SENDING EMAIL REPORTS")
    print("="*70)
    
    print(f"\n--- Email Details ---")
    print(f"Subject: {subject}")
    print(f"Recipient: {recipient_email}")
    print(f"Format: {'HTML' if use_html else 'Plain Text'}")
    
    if attachment_path:
        print(f"Attachment: {attachment_path}")
    
    print(f"\n--- Email Body Preview (first 300 chars) ---")
    print(body[:300] + "...\n")
    
    print("✓ Email prepared successfully")
    
    # Mock SMTP sending (commented out for security)
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = 'your_email@gmail.com'
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MIMEText(body, 'html' if use_html else 'plain'))
        
        # Add attachment if provided
        if attachment_path and os.path.exists(attachment_path):
            attachment = open(attachment_path, 'rb')
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(attachment_path)}')
            msg.attach(part)
        
        # Send email (using Gmail SMTP as example)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('your_email@gmail.com', 'your_app_password')
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
    """
    
    print("(In production, actual email would be sent via SMTP)")


# ============================================================================
# PART 6: EXPORT TO FILES
# ============================================================================

def export_report_to_files(metrics, category_data, region_data, daily_data, week_start, week_end, output_dir='./weekly_reports'):
    """Export reports to various file formats"""
    print("\n" + "="*70)
    print("DEMO 7: EXPORTING REPORTS TO FILES")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    report_date = week_start.strftime('%Y-%m-%d')
    
    # Export 1: CSV files
    print(f"\n--- Exporting to CSV ---")
    
    category_data.to_csv(f'{output_dir}/category_performance_{report_date}.csv')
    print(f"✓ Exported: category_performance_{report_date}.csv")
    
    region_data.to_csv(f'{output_dir}/region_performance_{report_date}.csv')
    print(f"✓ Exported: region_performance_{report_date}.csv")
    
    daily_data.to_csv(f'{output_dir}/daily_breakdown_{report_date}.csv')
    print(f"✓ Exported: daily_breakdown_{report_date}.csv")
    
    # Export 2: JSON format
    print(f"\n--- Exporting to JSON ---")
    
    report_json = {
        'report_date': report_date,
        'week_start': str(week_start.date()),
        'week_end': str(week_end.date()),
        'metrics': metrics,
        'generated_at': datetime.now().isoformat()
    }
    
    with open(f'{output_dir}/weekly_report_{report_date}.json', 'w') as f:
        json.dump(report_json, f, indent=4, default=str)
    
    print(f"✓ Exported: weekly_report_{report_date}.json")
    
    # Export 3: Excel file (if openpyxl available)
    print(f"\n--- Exporting to Excel ---")
    
    try:
        excel_file = f'{output_dir}/weekly_report_{report_date}.xlsx'
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame([metrics]).to_excel(writer, sheet_name='Summary', index=False)
            category_data.to_excel(writer, sheet_name='Category')
            region_data.to_excel(writer, sheet_name='Region')
            daily_data.to_excel(writer, sheet_name='Daily')
        
        print(f"✓ Exported: weekly_report_{report_date}.xlsx")
    except ImportError:
        print("⚠ openpyxl not available, skipping Excel export")
    
    print(f"\n✓ All reports exported to: {os.path.abspath(output_dir)}")
    
    return output_dir


# ============================================================================
# PART 7: AUTOMATED SCHEDULING
# ============================================================================

def demo_scheduling_automation():
    """Show how to schedule automated reports"""
    print("\n" + "="*70)
    print("DEMO 8: SCHEDULING AUTOMATED REPORTS")
    print("="*70)
    
    schedule_code = """
    # Option 1: Using APScheduler (recommended)
    from apscheduler.schedulers.background import BackgroundScheduler
    
    def weekly_reporting_job():
        df = load_sales_data()
        week_start, week_end, _ = get_week_boundaries()
        weekly_df = filter_weekly_data(df, week_start, week_end)
        metrics = calculate_weekly_metrics(weekly_df)
        send_email_report(metrics)
        export_report_to_files(metrics)
    
    scheduler = BackgroundScheduler()
    # Run every Monday at 9:00 AM
    scheduler.add_job(weekly_reporting_job, 'cron', day_of_week=0, hour=9)
    scheduler.start()
    
    # Option 2: Using schedule library (simpler)
    import schedule
    import time
    
    schedule.every().monday.at("09:00").do(weekly_reporting_job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
    
    # Option 3: System cron job (Linux/Mac)
    # Add to crontab:
    # 0 9 * * MON python /path/to/weekly_report.py
    
    # Option 4: Windows Task Scheduler
    # Create task to run:
    # python.exe C:\\path\\to\\weekly_report.py
    """
    
    print(schedule_code)


# ============================================================================
# MAIN EXECUTION - COMPLETE WORKFLOW
# ============================================================================

def main():
    """Execute complete weekly reporting workflow"""
    print("\n" + "="*70)
    print("DEMO 5: AUTOMATING WEEKLY REPORTING - COMPLETE WORKFLOW")
    print("="*70)
    
    # Step 1: Generate weekly metrics
    print("\n### STEP 1: GENERATE WEEKLY METRICS ###")
    demo_weekly_report_generation()
    
    # Step 2: Generate detailed breakdowns
    print("\n\n### STEP 2: GENERATE DETAILED BREAKDOWNS ###")
    category_breakdown, region_breakdown, daily_breakdown = demo_detailed_breakdown()
    
    # Step 3: Comparison analysis
    print("\n\n### STEP 3: WEEK-OVER-WEEK COMPARISON ###")
    df = create_sales_data(days=30)
    week_start, week_end, _ = get_week_boundaries()
    weekly_df = filter_weekly_data(df, week_start, week_end)
    
    previous_week_start = week_start - timedelta(days=7)
    previous_week_end = week_end - timedelta(days=7)
    previous_weekly_df = filter_weekly_data(df, previous_week_start, previous_week_end)
    
    current_metrics = calculate_weekly_metrics(weekly_df)
    previous_metrics = calculate_weekly_metrics(previous_weekly_df)
    
    generate_comparison_report(current_metrics, previous_metrics)
    
    # Step 4: Trend analysis
    print("\n\n### STEP 4: MULTI-WEEK TREND ANALYSIS ###")
    trend_df = demo_trend_analysis()
    
    # Step 5: Email formatting
    print("\n\n### STEP 5: FORMAT EMAIL REPORTS ###")
    text_report, html_report = demo_email_formatting()
    
    # Step 6: Send email
    print("\n\n### STEP 6: SEND EMAIL REPORTS ###")
    send_email_report(
        subject=f"Weekly Sales Report - {week_start.date()}",
        body=text_report,
        recipient_email="analytics@company.com",
        use_html=False
    )
    
    # Step 7: Export to files
    print("\n\n### STEP 7: EXPORT REPORTS TO FILES ###")
    output_dir = export_report_to_files(
        current_metrics,
        category_breakdown,
        region_breakdown,
        daily_breakdown,
        week_start,
        week_end
    )
    
    # Step 8: Scheduling
    print("\n\n### STEP 8: SCHEDULE AUTOMATED REPORTS ###")
    demo_scheduling_automation()
    
    print("\n" + "="*70)
    print("WEEKLY REPORTING AUTOMATION WORKFLOW COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
