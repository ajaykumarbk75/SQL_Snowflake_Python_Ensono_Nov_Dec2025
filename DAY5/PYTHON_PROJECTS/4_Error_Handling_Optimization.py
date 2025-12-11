"""
DEMO 4: ERROR HANDLING & OPTIMIZATION
Best practices for robust and efficient Python code
"""

import pandas as pd
import numpy as np
import logging
import time
from functools import wraps
import traceback
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging(log_file='data_processing.log'):
    """Configure logging for tracking errors"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# PART 1: ERROR HANDLING PATTERNS
# ============================================================================

def demo_try_except():
    """Basic try-except error handling"""
    print("\n" + "="*70)
    print("DEMO 1: BASIC TRY-EXCEPT ERROR HANDLING")
    print("="*70)
    
    # Example 1: Handling FileNotFoundError
    print("\n--- Example 1: File Not Found ---")
    try:
        df = pd.read_csv('nonexistent_file.csv')
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Creating sample data instead...")
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        print(f"✓ Sample data created")
    
    # Example 2: Handling ValueError
    print("\n--- Example 2: Data Type Conversion Error ---")
    try:
        values = ['10', '20', 'thirty', '40']
        numbers = [int(x) for x in values]
    except ValueError as e:
        print(f"❌ Error: {e}")
        print(f"   Handling conversion failure...")
        numbers = []
        for x in values:
            try:
                numbers.append(int(x))
            except ValueError:
                print(f"   ⚠ Skipping non-numeric value: {x}")
                numbers.append(np.nan)
        print(f"✓ Converted: {numbers}")
    
    # Example 3: Multiple Exception Handling
    print("\n--- Example 3: Multiple Exception Types ---")
    try:
        data = {'a': 1, 'b': 2}
        value = data['c']  # KeyError
    except KeyError as e:
        print(f"❌ KeyError: Key {e} not found")
    except ValueError as e:
        print(f"❌ ValueError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("✓ Finally block executed (cleanup operations)")


def demo_custom_exceptions():
    """Create and use custom exceptions"""
    print("\n" + "="*70)
    print("DEMO 2: CUSTOM EXCEPTIONS")
    print("="*70)
    
    class DataValidationError(Exception):
        """Custom exception for data validation"""
        pass
    
    class EmptyDatasetError(Exception):
        """Custom exception for empty data"""
        pass
    
    def validate_dataset(df):
        """Validate dataset with custom exceptions"""
        if df.empty:
            raise EmptyDatasetError("Dataset is empty!")
        
        required_cols = ['id', 'value']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing columns: {missing_cols}")
        
        if (df['value'] < 0).any():
            raise DataValidationError("Value column contains negative numbers!")
    
    # Test with valid data
    print("\n--- Test 1: Valid Data ---")
    df_valid = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    try:
        validate_dataset(df_valid)
        print("✓ Dataset validation passed")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Test with empty data
    print("\n--- Test 2: Empty Data ---")
    df_empty = pd.DataFrame()
    try:
        validate_dataset(df_empty)
    except EmptyDatasetError as e:
        print(f"❌ Empty Dataset Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test with missing columns
    print("\n--- Test 3: Missing Columns ---")
    df_missing = pd.DataFrame({'id': [1, 2, 3]})
    try:
        validate_dataset(df_missing)
    except DataValidationError as e:
        print(f"❌ Validation Error: {e}")


# ============================================================================
# PART 2: DATA VALIDATION TECHNIQUES
# ============================================================================

def demo_data_validation():
    """Comprehensive data validation"""
    print("\n" + "="*70)
    print("DEMO 3: DATA VALIDATION TECHNIQUES")
    print("="*70)
    
    # Create sample data with issues
    df = pd.DataFrame({
        'id': [1, 2, None, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 'thirty', 35, 40, None],
        'salary': [50000, 60000, -5000, 70000, 80000],
        'email': ['alice@email.com', 'invalid_email', 'charlie@email.com', 'david@email.com', 'eve@email.com']
    })
    
    print("\n--- Original Data ---")
    print(df)
    
    # Validation 1: Check for null values
    print("\n--- Validation 1: Null Values ---")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])
    
    # Validation 2: Check data types
    print("\n--- Validation 2: Data Types ---")
    print("Expected types: id=int, name=str, age=int, salary=float, email=str")
    print(f"Actual types:\n{df.dtypes}")
    
    # Validation 3: Check value ranges
    print("\n--- Validation 3: Value Ranges ---")
    negative_salary = df[df['salary'] < 0]
    if len(negative_salary) > 0:
        print(f"❌ Found {len(negative_salary)} records with negative salary")
        print(negative_salary)
    
    # Validation 4: Business logic checks
    print("\n--- Validation 4: Business Logic ---")
    print("Rule: Age should be between 18 and 70")
    df_invalid_age = df[(df['age'] < 18) | (df['age'] > 70)]
    if len(df_invalid_age) > 0:
        print(f"❌ Found {len(df_invalid_age)} invalid age values")
    
    # Validation 5: Email format
    print("\n--- Validation 5: Email Format ---")
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['valid_email'] = df['email'].apply(lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False)
    print(df[['email', 'valid_email']])


def demo_handling_missing_values():
    """Multiple strategies for handling missing data"""
    print("\n" + "="*70)
    print("DEMO 4: HANDLING MISSING VALUES")
    print("="*70)
    
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, np.nan],
        'B': [10, np.nan, np.nan, 40, 50, 60],
        'C': ['x', 'y', 'z', np.nan, 'x', 'y']
    })
    
    print("\n--- Original Data ---")
    print(df)
    
    # Strategy 1: Delete rows with missing values
    print("\n--- Strategy 1: Delete Rows (dropna) ---")
    df_drop = df.dropna()
    print(f"Rows removed: {len(df) - len(df_drop)}")
    print(df_drop)
    
    # Strategy 2: Delete columns with high missing percentage
    print("\n--- Strategy 2: Delete Columns with >50% Missing ---")
    df_drop_cols = df.dropna(thresh=len(df)*0.5, axis=1)
    print(f"Columns dropped: {set(df.columns) - set(df_drop_cols.columns)}")
    print(df_drop_cols)
    
    # Strategy 3: Forward fill
    print("\n--- Strategy 3: Forward Fill ---")
    df_ffill = df.fillna(method='ffill')
    print(df_ffill)
    
    # Strategy 4: Fill with mean
    print("\n--- Strategy 4: Fill with Mean (Numerical) ---")
    df_mean = df.copy()
    df_mean['A'].fillna(df_mean['A'].mean(), inplace=True)
    df_mean['B'].fillna(df_mean['B'].mean(), inplace=True)
    print(df_mean)
    
    # Strategy 5: Fill with mode
    print("\n--- Strategy 5: Fill with Mode (Categorical) ---")
    df_mode = df.copy()
    df_mode['C'].fillna(df_mode['C'].mode()[0], inplace=True)
    print(df_mode)
    
    # Strategy 6: Interpolation
    print("\n--- Strategy 6: Linear Interpolation ---")
    df_interp = df.copy()
    df_interp['A'] = df_interp['A'].interpolate()
    print(df_interp)


# ============================================================================
# PART 3: PERFORMANCE OPTIMIZATION
# ============================================================================

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting: {func.__name__}")
        print(f"\n▶ Executing: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Completed: {func.__name__} in {duration:.4f} seconds")
            print(f"✓ Completed in {duration:.4f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            print(f"❌ Error: {e}")
            raise
    return wrapper


@timing_decorator
def slow_loop_approach(n=1000000):
    """Inefficient approach using Python loops"""
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result


@timing_decorator
def fast_numpy_approach(n=1000000):
    """Efficient approach using NumPy"""
    result = np.arange(n) ** 2
    return result


def demo_performance_optimization():
    """Compare performance of different approaches"""
    print("\n" + "="*70)
    print("DEMO 5: PERFORMANCE OPTIMIZATION")
    print("="*70)
    
    # Test 1: Loops vs NumPy
    print("\n--- Comparison 1: Loops vs NumPy ---")
    slow_result = slow_loop_approach(1000000)
    fast_result = fast_numpy_approach(1000000)
    
    # Test 2: Pandas operations
    print("\n--- Comparison 2: Inefficient Pandas ---")
    df = pd.DataFrame({
        'A': np.random.rand(100000),
        'B': np.random.rand(100000)
    })
    
    @timing_decorator
    def inefficient_apply():
        return df.apply(lambda row: row['A'] * row['B'], axis=1)
    
    @timing_decorator
    def efficient_vectorized():
        return df['A'] * df['B']
    
    inefficient_apply()
    efficient_vectorized()
    
    # Test 3: Memory optimization
    print("\n--- Comparison 3: Memory Optimization ---")
    
    @timing_decorator
    def before_memory_optimization():
        df = pd.DataFrame({
            'col_int64': np.random.randint(0, 100, 100000),
            'col_float64': np.random.rand(100000),
            'col_object': ['category_' + str(i % 100) for i in range(100000)]
        })
        return df
    
    @timing_decorator
    def after_memory_optimization():
        df = pd.DataFrame({
            'col_int64': np.random.randint(0, 100, 100000),
            'col_float64': np.random.rand(100000),
            'col_object': pd.Categorical(['category_' + str(i % 100) for i in range(100000)])
        })
        return df
    
    df_before = before_memory_optimization()
    df_after = after_memory_optimization()
    
    print(f"\nMemory before optimization: {df_before.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Memory after optimization: {df_after.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def demo_memory_optimization():
    """Optimize dataframe memory usage"""
    print("\n" + "="*70)
    print("DEMO 6: MEMORY OPTIMIZATION")
    print("="*70)
    
    # Create large dataframe
    df = pd.DataFrame({
        'id': np.arange(100000),
        'value': np.random.rand(100000),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000),
        'status': np.random.choice(['Active', 'Inactive'], 100000)
    })
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n--- Initial Memory Usage: {initial_memory:.2f} MB ---")
    print(df.dtypes)
    
    # Optimization 1: Reduce int types
    print("\n--- Optimization 1: Reduce Integer Types ---")
    df['id'] = df['id'].astype(np.int32)
    print(f"Changed 'id' from int64 to int32")
    
    # Optimization 2: Convert to categorical
    print("\n--- Optimization 2: Convert to Categorical ---")
    df['category'] = df['category'].astype('category')
    df['status'] = df['status'].astype('category')
    print(f"Changed 'category' and 'status' to categorical")
    
    # Optimization 3: Reduce float precision
    print("\n--- Optimization 3: Reduce Float Precision ---")
    df['value'] = df['value'].astype(np.float32)
    print(f"Changed 'value' from float64 to float32")
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = ((initial_memory - final_memory) / initial_memory) * 100
    
    print(f"\n--- Final Memory Usage: {final_memory:.2f} MB ---")
    print(f"Memory Reduction: {reduction:.1f}%")
    print(f"\n{df.dtypes}")


# ============================================================================
# PART 4: BATCH PROCESSING & CHUNKING
# ============================================================================

def demo_batch_processing():
    """Process large files in batches"""
    print("\n" + "="*70)
    print("DEMO 7: BATCH PROCESSING & CHUNKING")
    print("="*70)
    
    # Create sample data file
    print("\n--- Simulating Large Data Processing ---")
    
    def process_in_chunks(total_rows=1000000, chunk_size=100000):
        """Process data in chunks"""
        chunks_processed = 0
        total_sum = 0
        
        print(f"Processing {total_rows} rows in chunks of {chunk_size}")
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = pd.DataFrame({
                'value': np.random.rand(end_idx - start_idx)
            })
            
            # Process chunk
            chunk_sum = chunk['value'].sum()
            total_sum += chunk_sum
            chunks_processed += 1
            
            print(f"✓ Processed chunk {chunks_processed}: rows {start_idx}-{end_idx}")
        
        return total_sum, chunks_processed
    
    total_sum, num_chunks = process_in_chunks(total_rows=1000000, chunk_size=100000)
    print(f"\nTotal chunks processed: {num_chunks}")
    print(f"Total sum: {total_sum:.2f}")


# ============================================================================
# PART 5: LOGGING & MONITORING
# ============================================================================

def demo_logging_best_practices():
    """Best practices for logging"""
    print("\n" + "="*70)
    print("DEMO 8: LOGGING & MONITORING BEST PRACTICES")
    print("="*70)
    
    custom_logger = logging.getLogger('data_processing')
    
    print("\n--- Logging Levels ---")
    custom_logger.debug("This is a DEBUG message (detailed info)")
    custom_logger.info("This is an INFO message (confirmation)")
    custom_logger.warning("This is a WARNING message (something unexpected)")
    custom_logger.error("This is an ERROR message (serious problem)")
    
    # Logging with context
    print("\n--- Logging with Context ---")
    
    def process_record(record_id, data):
        try:
            custom_logger.info(f"Processing record_id={record_id}")
            if data is None:
                raise ValueError("Data cannot be None")
            custom_logger.info(f"✓ Record {record_id} processed successfully")
        except Exception as e:
            custom_logger.error(f"Failed to process record_id={record_id}: {e}")
    
    process_record(1, {'value': 100})
    process_record(2, None)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DEMO 4: ERROR HANDLING & OPTIMIZATION")
    print("="*70)
    
    # Part 1: Error Handling
    demo_try_except()
    demo_custom_exceptions()
    
    # Part 2: Data Validation
    demo_data_validation()
    demo_handling_missing_values()
    
    # Part 3: Performance Optimization
    demo_performance_optimization()
    demo_memory_optimization()
    
    # Part 4: Batch Processing
    demo_batch_processing()
    
    # Part 5: Logging
    demo_logging_best_practices()
    
    print("\n" + "="*70)
    print("ERROR HANDLING & OPTIMIZATION DEMONSTRATIONS COMPLETED!")
    print("="*70)
