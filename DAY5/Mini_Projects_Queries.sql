-- Diwali Sales dataset -- 


--  Step 2: Basic Exploration Queries
-- 1️⃣ Total Records
SELECT COUNT(*) AS total_rows FROM diwali_sales;

-- 2️⃣ Missing Amount Records
SELECT COUNT(*) AS missing_amount
FROM diwali_sales
WHERE Amount IS NULL OR Amount = 0;

--  Step 3: Business KPI Queries
3️⃣ Total Revenue
SELECT SUM(Amount) AS total_revenue
FROM diwali_sales;

-- 4️⃣ Orders & Revenue by Gender
SELECT 
    Gender,
    COUNT(*) AS total_customers,
    SUM(Orders) AS total_orders,
    SUM(Amount) AS revenue
FROM diwali_sales
GROUP BY Gender
ORDER BY revenue DESC;

-- 5️⃣ Top 5 States by Revenue
SELECT 
    State,
    SUM(Amount) AS total_revenue
FROM diwali_sales
GROUP BY State
ORDER BY total_revenue DESC
LIMIT 5;

-- 6️⃣ Zone-wise Customer Distribution
SELECT 
    Zone,
    COUNT(User_ID) AS customers
FROM diwali_sales
GROUP BY Zone;

-- 7️⃣ Purchase Behavior by Age Group
SELECT 
    Age_Group,
    SUM(Orders) AS total_orders,
    SUM(Amount) AS total_sales
FROM diwali_sales
GROUP BY Age_Group
ORDER BY total_sales DESC;

-- 8️⃣ Most Popular Occupation Segment
SELECT 
    Occupation,
    COUNT(*) AS customer_count,
    SUM(Amount) AS revenue
FROM diwali_sales
GROUP BY Occupation
ORDER BY revenue DESC;

-- 9️⃣ Repeat Customers (same User buying more than once)
SELECT User_ID, Cust_name, COUNT(*) AS purchase_count
FROM diwali_sales
GROUP BY User_ID, Cust_name
HAVING COUNT(*) > 1;

--  Step 4: Advanced Analytics Queries
1️⃣ Revenue Ranking by Customer (Window Function)
SELECT 
    User_ID, Cust_name,
    SUM(Amount) AS total_spent,
    RANK() OVER (ORDER BY SUM(Amount) DESC) AS spending_rank
FROM diwali_sales
GROUP BY User_ID, Cust_name;

-- 2️⃣ Average Spend by Gender & Age Group
SELECT 
    Gender,
    Age_Group,
    ROUND(AVG(Amount),2) AS avg_spend
FROM diwali_sales
GROUP BY Gender, Age_Group
ORDER BY avg_spend DESC;

-- 3️⃣ Most Purchased Product Category
SELECT 
    Product_Category,
    SUM(Orders) AS total_orders
FROM diwali_sales
GROUP BY Product_Category
ORDER BY total_orders DESC;

--*** Amazon - Products.csv ****

✅ 1️⃣ Basic Exploration Queries
SELECT * 
FROM products
LIMIT 10;

SELECT COUNT(*) AS total_records, 
       COUNT(DISTINCT brand_name) AS unique_brands
FROM products;

✅ 2️⃣ Cleaning / Standardization Examples
--Normalize Availability Flag
SELECT asin, availability,
       CASE 
           WHEN availability ILIKE '%in stock%' THEN 'AVAILABLE'
           ELSE 'NOT AVAILABLE'
       END AS availability_status
FROM products;

✅ 3️⃣ Grouping & Aggregation
-- Top Brands Based on Number of Products
SELECT brand_name, COUNT(*) AS product_count
FROM products
GROUP BY brand_name
ORDER BY product_count DESC
LIMIT 10;

--Average Price Per Brand
SELECT brand_name, ROUND(AVG(price_value),2) AS avg_price
FROM products
WHERE price_value IS NOT NULL
GROUP BY brand_name
ORDER BY avg_price DESC;

✅ 4️⃣ Rating Analysis
Average Rating by Brand
-- SELECT brand_name, 
--        ROUND(AVG(rating_stars),2) AS avg_rating,
--        SUM(rating_count) AS total_ratings
-- FROM products
-- WHERE rating_stars IS NOT NULL
-- GROUP BY brand_name
-- ORDER BY avg_rating DESC;

-- // Identify Worst Rated but High Sales Items
-- SELECT title, brand_name, rating_stars, recent_purchases
-- FROM products
-- WHERE recent_purchases > 100
-- ORDER BY rating_stars ASC;

✅ 5️⃣ Working with JSON / Semi-Structured Data
-- -- Extract first image URL
-- SELECT asin,
--        all_images[0]::string AS primary_image
-- FROM products;

-- Explode the Image Array (Flatten)
-- SELECT p.asin, img.value::string AS image_url
-- FROM products p,
--      LATERAL FLATTEN(input => p.all_images) img;

✅ 6️⃣ Window Functions
-- Top 3 Highest Priced Items per Brand
SELECT asin, brand_name, title, price_value,
       ROW_NUMBER() OVER (PARTITION BY brand_name ORDER BY price_value DESC) AS rn
FROM products
QUALIFY rn <= 3;

-- -- Rank Products by Rating within Category
-- SELECT title, product_category, rating_stars,
--        RANK() OVER (PARTITION BY product_category ORDER BY rating_stars DESC) AS cat_rank
-- FROM products;

✅ 7️⃣ Search / NLP Style Query
-- Find Products Related to "Golf"
SELECT asin, title, brand_name
FROM products
WHERE LOWER(title) LIKE '%golf%' OR LOWER(product_description) LIKE '%golf%';

✅ 8️⃣ Price Bucketing / Segmentation
SELECT 
    CASE 
        WHEN price_value < 20 THEN 'Budget'
        WHEN price_value BETWEEN 20 AND 40 THEN 'Mid Range'
        ELSE 'Premium'
    END AS price_segment,
    COUNT(*) AS products_count
FROM products
GROUP BY price_segment;

-- ✅ 9️⃣ Time-Series Trend (Using scrape_time)
-- SELECT scrape_time::date AS scrape_date, 
--        COUNT(*) AS items_scraped
-- FROM products
-- GROUP BY scrape_time::date
-- ORDER BY scrape_date;

✅ 🔟 Profitability/Value Index Demo

-- (Rating × Recent Purchases ÷ Price — a fictional value score)

-- SELECT asin, brand_name, title, price_value, rating_stars, recent_purchases,
--        ROUND((rating_stars * recent_purchases) / NULLIF(price_value,0),2) AS value_score
-- FROM products
-- ORDER BY value_score DESC
-- LIMIT 10;

-- ======== Aazon - REVIEWS.csv 


✅ 1️⃣ Basic Exploration Queries
SELECT * 
FROM reviews
LIMIT 20;

SELECT COUNT(*) AS total_reviews,
       COUNT(DISTINCT productASIN) AS products_reviewed
FROM reviews;

✅ 2️⃣ Rating & Sentiment Analysis
-- 👉 Average Rating Per Product
SELECT productASIN, ROUND(AVG(rating),2) AS avg_rating, COUNT(*) AS review_count
FROM reviews
GROUP BY productASIN
ORDER BY avg_rating DESC;

-- 👉 Compare Rating with Sentiment Score
SELECT 
  productASIN,
  ROUND(AVG(rating),2) AS avg_rating,
  ROUND(AVG(sentiment_score),3) AS avg_sentiment_score
FROM reviews
GROUP BY productASIN
ORDER BY avg_sentiment_score DESC;

✅ 3️⃣ Keyword Search (Text Analytics)
-- Search for Complaints Using Keywords
SELECT reviewID, rating, cleaned_review_text
FROM reviews
WHERE LOWER(cleaned_review_text) LIKE '%bad%' 
   OR LOWER(cleaned_review_text) LIKE '%poor%' 
   OR LOWER(cleaned_review_text) LIKE '%worst%';

-- Identify Positive Emotion Reviews
SELECT reviewID, rating, cleaned_review_text
FROM reviews
WHERE sentiment_score > 0.5
ORDER BY sentiment_score DESC;

✅ 4️⃣ Helpful Review Insights
Most Helpful Reviews (Based on helpfulVoteCount)
SELECT reviewID, reviewTitle, helpfulVoteCount, rating
FROM reviews
ORDER BY helpfulVoteCount DESC, rating DESC
LIMIT 10;

✅ 5️⃣ Verified Purchase Insights
SELECT verifiedPurchase, COUNT(*) AS review_count
FROM reviews
GROUP BY verifiedPurchase;

-- Compare Ratings of Verified vs Non-Verified Purchases
SELECT verifiedPurchase,
       ROUND(AVG(rating),2) AS avg_rating,
       ROUND(AVG(sentiment_score),2) AS avg_sentiment
FROM reviews
GROUP BY verifiedPurchase;

✅ 6️⃣ Time Series Query (Based on Review Date)
-- SELECT 
--     TO_DATE(reviewMetadata::string) AS review_date,
--     COUNT(*) AS total_reviews,
--     ROUND(AVG(rating),2) AS avg_rating
-- FROM reviews
-- GROUP BY review_date
-- ORDER BY review_date;


-- (If date parsing needed:)

-- SELECT TRY_TO_DATE(reviewMetadata, 'MMM DD, YYYY') AS parsed_date
-- FROM reviews;

✅ 7️⃣ Window Functions
-- Top review per product based on helpful votes
SELECT *
FROM (
    SELECT productASIN, reviewID, reviewTitle, helpfulVoteCount,
           ROW_NUMBER() OVER (PARTITION BY productASIN ORDER BY helpfulVoteCount DESC) AS rank
    FROM reviews
)
WHERE rank = 1;

-- Goldman Sachs dataset   gs.csv *****


-- ✅ 1. View Data with Correct Date Format
SELECT 
    Date AS Trade_Date,
    Open, High, Low, Close, Volume
FROM GS;

-- ✅ 2. Find Highest Closing Price
SELECT MAX(Close) AS Highest_Close
FROM GS;

--✅ 3. Daily Price Change (Close - Open)
SELECT 
    Date AS Trade_Date,
    Open, Close,
    ROUND(Close - Open, 2) AS Price_Change
FROM GS
ORDER BY Date;

-- ✅ 4. Calculate Daily Return %
SELECT
    Date AS Trade_Date,
    Close,
    ROUND((Close - Open) / Open * 100, 2) AS Daily_Return_Percent
FROM GS
ORDER BY Date;

--✅ 5. 5-Day Moving Average of Closing Price
SELECT 
    Date AS Trade_Date,
    Close,
    AVG(Close) OVER (
        ORDER BY Date 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS MA_5_Days
FROM GS;

-- ✅ 6. Find Highest Trading Volume Day
SELECT 
    Date AS Trade_Date,
    Volume
FROM GS
ORDER BY Volume DESC
LIMIT 1;

--✅ 7. Detect Days Where Stock Closed Higher Than It Opened
SELECT *
FROM GS
WHERE Close > Open;

-- ✅ 8. Calculate Cumulative Return Over Time
SELECT
    Date AS Trade_Date,
    Close,
    SUM((Close - Open) / Open * 100) 
        OVER (ORDER BY Date) AS Cumulative_Return_Pct
FROM GS;

-- ✅ 9. Identify Bullish or Bearish Days
SELECT 
    Date AS Trade_Date,
    Open, Close,
    CASE 
        WHEN Close > Open THEN 'BULLISH'
        WHEN Close < Open THEN 'BEARISH'
        ELSE 'NEUTRAL'
    END AS Market_Sentiment
FROM GS;

-- ✅ 10. Price Volatility (High - Low)
SELECT 
    Date AS Trade_Date,
    High, Low,
    (High - Low) AS Daily_Volatility
FROM GS
ORDER BY Date;

-- Bonus: Month-wise Stock Summary
SELECT 
    TO_CHAR(Date, 'YYYY-MM') AS Month,
    ROUND(AVG(Close), 2) AS Avg_Close,
    MAX(High) AS Month_High,
    MIN(Low) AS Month_Low,
    SUM(Volume) AS Total_Volume
FROM GS
GROUP BY Month
ORDER BY Month;

--====HEALTHCARE_DATA - healthcare_dataset.csv ********



--✅ Snowflake Analytics Queries
--1️⃣ Total Patients Count
SELECT COUNT(*) AS Total_Patients 
FROM HEALTHCARE_DATA;

--2️⃣ Patient Count by Gender
SELECT Gender, COUNT(*) AS Total
FROM HEALTHCARE_DATA
GROUP BY Gender
ORDER BY Total DESC;

--3️⃣ Patients by Medical Condition
SELECT Medical_Condition, COUNT(*) AS Total_Cases
FROM HEALTHCARE_DATA
GROUP BY Medical_Condition
ORDER BY Total_Cases DESC;

--4️⃣ Top 5 Most Expensive Billing Cases
SELECT Name, Billing_Amount
FROM HEALTHCARE_DATA
ORDER BY Billing_Amount DESC
LIMIT 5;

--5️⃣ Average Billing Amount per Medical Condition
SELECT Medical_Condition, ROUND(AVG(Billing_Amount), 2) AS Avg_Cost
FROM HEALTHCARE_DATA
GROUP BY Medical_Condition
ORDER BY Avg_Cost DESC;

--6️⃣ Insurance Provider Billing Summary
SELECT Insurance_Provider,
       COUNT(*) AS Patients,
       ROUND(SUM(Billing_Amount),2) AS Total_Billing,
       ROUND(AVG(Billing_Amount),2) AS Avg_Bill
FROM HEALTHCARE_DATA
GROUP BY Insurance_Provider
ORDER BY Total_Billing DESC;


--🔟 Most Common Medication
SELECT Medication, COUNT(*) AS Frequency
FROM HEALTHCARE_DATA
GROUP BY Medication
ORDER BY Frequency DESC;

-- Bonus: Doctor Performance Summary
SELECT 
    Doctor,
    COUNT(*) AS Patients_Treated,
    ROUND(AVG(Billing_Amount),2) AS Avg_Billing,
    ROUND(SUM(Billing_Amount),2) AS Total_Revenue
FROM HEALTHCARE_DATA
GROUP BY Doctor
ORDER BY Total_Revenue DESC;

-- Bonus 2: Flag High-Risk Patients (Age > 60 + Cancer/Diabetes)
SELECT 
    Name,
    Age,
    Medical_Condition,
    Billing_Amount
FROM HEALTHCARE_DATA
WHERE Age > 60 
  AND Medical_Condition IN ('Cancer', 'Diabetes')
ORDER BY Billing_Amount DESC;


-- Python - Load sample Sales dataset into Snowflake -- 

-- Fix: Load file with correct encoding

-- Try this first:

import pandas as pd

df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

print(df.head())

-- Clean the dataset, handling missing values  & upload 

Updated Full Working Script
import pandas as pd

# Load dataset with correct encoding
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

print("\n Raw Data Preview:")
print(df.head())

# Step 1: Remove duplicates
df.drop_duplicates(inplace=True)

# Step 2: Handle missing values (only if columns exist)
fill_values = {}

if "Product" in df.columns:
    fill_values["Product"] = "Unknown"

if "Quantity" in df.columns:
    fill_values["Quantity"] = df["Quantity"].median()

if "Price" in df.columns:
    fill_values["Price"] = df["Price"].median()

df.fillna(fill_values, inplace=True)

# Step 3: Create new calculated field safely
if {"Quantity", "Price"}.issubset(df.columns):
    df["Total Revenue"] = df["Quantity"] * df["Price"]

# Step 4: Convert to datetime if column exists
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors='coerce')

print("\n Cleaned & Prepared Data:")
print(df.head())

# Save new cleaned dataset
df.to_csv("cleaned_sales_data.csv", index=False, encoding="utf-8")

print("\n Clean dataset saved successfully!")

-- Data Wrangling: Missing Values, Type Conversion, Merge

import pandas as pd

# Example dataset with missing values
data = {
    "Customer": ["Rahul", "Sneha", "John", "Kiran"],
    "City": ["Mumbai", "Delhi", None, "Pune"],
    "Age": [29, None, 45, 28],
    "Monthly_Bill": ["300", "500", "1000", None]
}

df = pd.DataFrame(data)

print("\n Original Data:")
print(df)

# Handling missing values
df.fillna({
    "City": "Unknown",
    "Age": df["Age"].mean(),
    "Monthly_Bill": "0"
}, inplace=True)

# Convert string to integer
df["Monthly_Bill"] = df["Monthly_Bill"].astype(int)

print("\n🛠 Cleaned Data:")
print(df)

# Example merge (join)
plan_data = {
    "Monthly_Bill": [300, 500, 1000, 0],
    "Plan_Name": ["Basic", "Premium", "Enterprise", "No Plan"]
}

df_plans = pd.DataFrame(plan_data)

merged_df = pd.merge(df, df_plans, on="Monthly_Bill", how="left")

print("\n Merged Dataset:")
print(merged_df)
