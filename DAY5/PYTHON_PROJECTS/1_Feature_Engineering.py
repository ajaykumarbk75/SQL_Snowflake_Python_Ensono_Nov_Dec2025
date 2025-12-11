"""
DEMO 1: PYTHON FOR FEATURE ENGINEERING
Creates and transforms features for machine learning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CREATE SAMPLE DATA
# ============================================================================

def create_sample_data():
    """Generate sample dataset"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'customer_id': np.random.randint(1001, 1050, 200),
        'transaction_amount': np.random.uniform(10, 500, 200),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 200),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
        'age': np.random.randint(18, 80, 200),
        'purchase_frequency': np.random.randint(1, 10, 200)
    })
    return df


# ============================================================================
# 1. NUMERICAL FEATURE TRANSFORMATIONS
# ============================================================================

def demo_scaling():
    """Min-Max Normalization (0-1 range)"""
    print("\n" + "="*70)
    print("1. SCALING - Min-Max Normalization")
    print("="*70)
    
    df = create_sample_data()
    
    # Original values
    print("\nOriginal transaction_amount:")
    print(f"Min: {df['transaction_amount'].min():.2f}")
    print(f"Max: {df['transaction_amount'].max():.2f}")
    print(f"Mean: {df['transaction_amount'].mean():.2f}")
    print(f"\nSample values:\n{df['transaction_amount'].head()}")
    
    # Scaled values (0-1)
    df['scaled_amount'] = (
        (df['transaction_amount'] - df['transaction_amount'].min()) / 
        (df['transaction_amount'].max() - df['transaction_amount'].min())
    )
    
    print("\n--- After Scaling (0-1 range) ---")
    print(f"Min: {df['scaled_amount'].min():.2f}")
    print(f"Max: {df['scaled_amount'].max():.2f}")
    print(f"Mean: {df['scaled_amount'].mean():.2f}")
    print(f"\nScaled values:\n{df['scaled_amount'].head()}")
    
    return df[['transaction_amount', 'scaled_amount']].head(10)


def demo_standardization():
    """Z-Score Standardization (mean=0, std=1)"""
    print("\n" + "="*70)
    print("2. STANDARDIZATION - Z-Score Normalization")
    print("="*70)
    
    df = create_sample_data()
    
    # Original age
    print("\nOriginal age distribution:")
    print(f"Mean: {df['age'].mean():.2f}")
    print(f"Std Dev: {df['age'].std():.2f}")
    print(f"Min: {df['age'].min()}")
    print(f"Max: {df['age'].max()}")
    print(f"\nSample values:\n{df['age'].head()}")
    
    # Standardized age
    df['age_standardized'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    print("\n--- After Standardization ---")
    print(f"Mean: {df['age_standardized'].mean():.2f}")
    print(f"Std Dev: {df['age_standardized'].std():.2f}")
    print(f"\nStandardized values:\n{df['age_standardized'].head()}")
    
    return df[['age', 'age_standardized']].head(10)


def demo_log_transformation():
    """Logarithmic transformation for skewed data"""
    print("\n" + "="*70)
    print("3. LOG TRANSFORMATION - Handling Skewed Data")
    print("="*70)
    
    df = create_sample_data()
    
    # Create skewed data
    df['skewed_values'] = np.abs(np.random.normal(100, 50, len(df)))
    
    print("\nOriginal skewed data:")
    print(f"Mean: {df['skewed_values'].mean():.2f}")
    print(f"Std Dev: {df['skewed_values'].std():.2f}")
    print(f"Skewness: {df['skewed_values'].skew():.2f}")
    
    # Apply log transformation
    df['log_transformed'] = np.log1p(df['skewed_values'])  # log1p handles zero values
    
    print("\n--- After Log Transformation ---")
    print(f"Mean: {df['log_transformed'].mean():.2f}")
    print(f"Std Dev: {df['log_transformed'].std():.2f}")
    print(f"Skewness: {df['log_transformed'].skew():.2f}")
    
    comparison = df[['skewed_values', 'log_transformed']].head(10)
    print(f"\nComparison:\n{comparison}")
    
    return comparison


def demo_binning():
    """Discretize continuous variables into bins"""
    print("\n" + "="*70)
    print("4. BINNING - Discretizing Continuous Data")
    print("="*70)
    
    df = create_sample_data()
    
    print("\nOriginal age distribution:")
    print(f"Min: {df['age'].min()}, Max: {df['age'].max()}")
    print(df['age'].value_counts().sort_index().head(10))
    
    # Equal-width binning
    df['age_binned_equal'] = pd.cut(
        df['age'],
        bins=5,
        labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly']
    )
    
    print("\n--- Equal-Width Binning (5 bins) ---")
    print(df['age_binned_equal'].value_counts())
    
    # Equal-frequency binning (quantiles)
    df['age_binned_quantile'] = pd.qcut(
        df['age'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    
    print("\n--- Quantile-Based Binning ---")
    print(df['age_binned_quantile'].value_counts())
    
    comparison = df[['age', 'age_binned_equal', 'age_binned_quantile']].head(10)
    print(f"\nComparison:\n{comparison}")
    
    return comparison


def demo_polynomial_features():
    """Create polynomial features"""
    print("\n" + "="*70)
    print("5. POLYNOMIAL FEATURES - Creating Interaction Terms")
    print("="*70)
    
    df = create_sample_data()
    
    print("\nOriginal features:")
    print(f"transaction_amount - Min: {df['transaction_amount'].min():.2f}, Max: {df['transaction_amount'].max():.2f}")
    
    # Polynomial features
    df['amount_squared'] = df['transaction_amount'] ** 2
    df['amount_cubed'] = df['transaction_amount'] ** 3
    df['sqrt_amount'] = np.sqrt(df['transaction_amount'])
    
    print("\n--- Polynomial Features Created ---")
    features = df[['transaction_amount', 'amount_squared', 'amount_cubed', 'sqrt_amount']].head(10)
    print(features)
    
    return features


# ============================================================================
# 2. CATEGORICAL FEATURE ENCODING
# ============================================================================

def demo_one_hot_encoding():
    """One-Hot Encoding for categorical variables"""
    print("\n" + "="*70)
    print("6. ONE-HOT ENCODING - Categorical to Binary")
    print("="*70)
    
    df = create_sample_data()
    
    print("\nOriginal categories:")
    print(df['product_category'].value_counts())
    
    # One-Hot Encoding
    category_dummies = pd.get_dummies(df['product_category'], prefix='category')
    
    print("\n--- One-Hot Encoded Features ---")
    print(category_dummies.head(10))
    
    # Merge back
    df_encoded = pd.concat([df[['product_category']], category_dummies], axis=1)
    
    return df_encoded.head(10)


def demo_label_encoding():
    """Label Encoding for ordinal data"""
    print("\n" + "="*70)
    print("7. LABEL ENCODING - Ordinal Mapping")
    print("="*70)
    
    df = create_sample_data()
    
    print("\nOriginal regions:")
    print(df['region'].value_counts())
    
    # Create mapping
    region_mapping = {'North': 1, 'South': 2, 'East': 3, 'West': 4}
    df['region_encoded'] = df['region'].map(region_mapping)
    
    print("\n--- Label Encoded ---")
    print(f"Mapping: {region_mapping}")
    print(f"\nEncoded values:\n{df[['region', 'region_encoded']].drop_duplicates()}")
    
    return df[['region', 'region_encoded']].head(10)


def demo_frequency_encoding():
    """Frequency Encoding based on occurrence"""
    print("\n" + "="*70)
    print("8. FREQUENCY ENCODING - Based on Occurrence")
    print("="*70)
    
    df = create_sample_data()
    
    print("\nOriginal category distribution:")
    print(df['product_category'].value_counts())
    
    # Frequency encoding
    freq_map = df['product_category'].value_counts(normalize=True).to_dict()
    df['category_frequency'] = df['product_category'].map(freq_map)
    
    print("\n--- Frequency Encoded ---")
    print(f"Frequency mapping:\n{freq_map}")
    print(f"\nSample:\n{df[['product_category', 'category_frequency']].drop_duplicates()}")
    
    return df[['product_category', 'category_frequency']].head(10)


# ============================================================================
# 3. TEMPORAL FEATURE ENGINEERING
# ============================================================================

def demo_temporal_features():
    """Extract time-based features"""
    print("\n" + "="*70)
    print("9. TEMPORAL FEATURES - Date/Time Decomposition")
    print("="*70)
    
    df = create_sample_data()
    
    print("\nOriginal date column:")
    print(df['date'].head())
    
    # Extract temporal components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    df['is_business_day'] = df['date'].dt.dayofweek < 5
    
    print("\n--- Extracted Temporal Features ---")
    temporal_cols = ['date', 'year', 'month', 'day', 'quarter', 'day_of_week', 'day_name', 'week_of_year', 'is_weekend', 'is_business_day']
    print(df[temporal_cols].head(15))
    
    return df[temporal_cols].head(15)


def demo_cyclic_features():
    """Create cyclic features for circular data"""
    print("\n" + "="*70)
    print("10. CYCLIC FEATURES - For Circular Patterns")
    print("="*70)
    
    df = create_sample_data()
    
    # Day of week is circular (0-6 wraps around)
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Create sine and cosine features to capture circularity
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    print("\nOriginal day_of_week (0=Monday, 6=Sunday):")
    print(df['day_of_week'].value_counts().sort_index())
    
    print("\n--- Cyclic Encoded (Sin/Cos) ---")
    cyclic_df = df[['date', 'day_of_week', 'day_sin', 'day_cos']].drop_duplicates().sort_values('day_of_week')
    print(cyclic_df)
    
    print("\nAdvantage: Captures that Sunday (6) is close to Monday (0)")
    
    return cyclic_df


# ============================================================================
# 4. STATISTICAL & AGGREGATE FEATURES
# ============================================================================

def demo_aggregate_features():
    """Create aggregate statistical features"""
    print("\n" + "="*70)
    print("11. AGGREGATE FEATURES - Rolling Statistics")
    print("="*70)
    
    df = create_sample_data().sort_values('date')
    
    print("\nOriginal data (first 10 rows):")
    print(df[['date', 'transaction_amount']].head(10))
    
    # Rolling statistics
    df['rolling_mean_7d'] = df['transaction_amount'].rolling(window=7).mean()
    df['rolling_std_7d'] = df['transaction_amount'].rolling(window=7).std()
    df['rolling_sum_7d'] = df['transaction_amount'].rolling(window=7).sum()
    
    print("\n--- Rolling Statistics (7-day window) ---")
    rolling_df = df[['date', 'transaction_amount', 'rolling_mean_7d', 'rolling_std_7d', 'rolling_sum_7d']].head(15)
    print(rolling_df.to_string())
    
    return rolling_df


def demo_groupby_features():
    """Create group-based aggregate features"""
    print("\n" + "="*70)
    print("12. GROUP-BY FEATURES - Aggregated by Category")
    print("="*70)
    
    df = create_sample_data()
    
    # Group by customer and create aggregate features
    customer_agg = df.groupby('customer_id').agg({
        'transaction_amount': ['sum', 'mean', 'count', 'std'],
        'age': 'first'
    }).reset_index()
    
    customer_agg.columns = ['customer_id', 'total_spent', 'avg_transaction', 'num_transactions', 'std_transaction', 'age']
    
    print("\n--- Aggregated Features by Customer ---")
    print(customer_agg.head(10))
    
    # Group by category
    print("\n--- Aggregated Features by Category ---")
    category_agg = df.groupby('product_category').agg({
        'transaction_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).reset_index()
    
    category_agg.columns = ['product_category', 'total_sales', 'avg_sale', 'num_transactions', 'unique_customers']
    print(category_agg)
    
    return customer_agg, category_agg


# ============================================================================
# 5. INTERACTION FEATURES
# ============================================================================

def demo_interaction_features():
    """Create features from interactions between variables"""
    print("\n" + "="*70)
    print("13. INTERACTION FEATURES - Combining Variables")
    print("="*70)
    
    df = create_sample_data()
    
    print("\nBase features:")
    print(df[['transaction_amount', 'age', 'purchase_frequency']].head(10))
    
    # Create interaction features
    df['amount_x_frequency'] = df['transaction_amount'] * df['purchase_frequency']
    df['amount_x_age'] = df['transaction_amount'] * df['age']
    df['age_x_frequency'] = df['age'] * df['purchase_frequency']
    df['amount_per_age'] = df['transaction_amount'] / (df['age'] + 1)
    
    print("\n--- Interaction Features ---")
    interaction_df = df[['transaction_amount', 'age', 'purchase_frequency', 
                         'amount_x_frequency', 'amount_x_age', 'age_x_frequency', 'amount_per_age']].head(10)
    print(interaction_df)
    
    return interaction_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DEMO 1: PYTHON FOR FEATURE ENGINEERING")
    print("="*70)
    
    # Numerical Transformations
    print("\n" + "#"*70)
    print("# PART A: NUMERICAL TRANSFORMATIONS")
    print("#"*70)
    
    demo_scaling()
    demo_standardization()
    demo_log_transformation()
    demo_binning()
    demo_polynomial_features()
    
    # Categorical Encoding
    print("\n\n" + "#"*70)
    print("# PART B: CATEGORICAL ENCODING")
    print("#"*70)
    
    demo_one_hot_encoding()
    demo_label_encoding()
    demo_frequency_encoding()
    
    # Temporal Features
    print("\n\n" + "#"*70)
    print("# PART C: TEMPORAL FEATURES")
    print("#"*70)
    
    demo_temporal_features()
    demo_cyclic_features()
    
    # Statistical & Aggregates
    print("\n\n" + "#"*70)
    print("# PART D: STATISTICAL & AGGREGATE FEATURES")
    print("#"*70)
    
    demo_aggregate_features()
    demo_groupby_features()
    
    # Interactions
    print("\n\n" + "#"*70)
    print("# PART E: INTERACTION FEATURES")
    print("#"*70)
    
    demo_interaction_features()
    
    print("\n\n" + "="*70)
    print("ALL FEATURE ENGINEERING DEMOS COMPLETED!")
    print("="*70)
