"""
Test Enhanced Data Processor
"""
import pandas as pd
import numpy as np
from enhanced_data_processor import EnhancedDataProcessor


def test_clean_numeric_column():
    """Test numeric column cleaning with various formats"""
    print("\n=== Testing Numeric Column Cleaning ===")
    
    processor = EnhancedDataProcessor()
    
    # Test data with various formats
    test_data = pd.Series([
        "$1,234.56",
        "€ 2,345.67",
        "(123.45)",  # Negative in accounting format
        "45%",
        "1 234 567",
        "$10,000",
        "¥123,456",
        "invalid"
    ])
    
    cleaned = processor.clean_numeric_column(test_data)
    
    print("Original:", list(test_data))
    print("Cleaned:", list(cleaned))
    print(f"✓ Cleaned {cleaned.notna().sum()} values")
    
    assert cleaned.iloc[0] == 1234.56
    assert cleaned.iloc[1] == 2345.67
    assert cleaned.iloc[2] == -123.45
    assert cleaned.iloc[3] == 45.0
    print("✓ All assertions passed")


def test_date_parsing():
    """Test date parsing with multiple formats"""
    print("\n=== Testing Date Parsing ===")
    
    processor = EnhancedDataProcessor()
    
    test_data = pd.Series([
        "2025-11-12",
        "12/11/2025",
        "11-12-2025",
        "Nov 12, 2025",
        "12 November 2025",
        "invalid date"
    ])
    
    parsed = processor.parse_dates(test_data)
    
    print("Original:", list(test_data))
    print("Parsed:", list(parsed))
    print(f"✓ Parsed {parsed.notna().sum()} dates")


def test_missing_values():
    """Test missing value handling"""
    print("\n=== Testing Missing Value Handling ===")
    
    processor = EnhancedDataProcessor()
    
    df = pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, np.nan, 6],
        'category': ['A', 'B', np.nan, 'A', 'B', np.nan]
    })
    
    print("Before:")
    print(df)
    print(f"Missing values: {df.isna().sum().sum()}")
    
    df_filled = processor.handle_missing_values(df, strategy='auto')
    
    print("\nAfter:")
    print(df_filled)
    print(f"Missing values: {df_filled.isna().sum().sum()}")
    print("✓ Missing values handled")


def test_outlier_detection():
    """Test outlier detection and removal"""
    print("\n=== Testing Outlier Detection ===")
    
    processor = EnhancedDataProcessor()
    
    df = pd.DataFrame({
        'value': [10, 12, 11, 10, 500, 12, 11, 600, 10]
    })
    
    print("Before:")
    print(df)
    print(f"Shape: {df.shape}")
    
    df_clean = processor.detect_and_remove_outliers(df, columns=['value'])
    
    print("\nAfter:")
    print(df_clean)
    print(f"Shape: {df_clean.shape}")
    print(f"✓ Removed {len(df) - len(df_clean)} outliers")


def test_category_standardization():
    """Test category standardization"""
    print("\n=== Testing Category Standardization ===")
    
    processor = EnhancedDataProcessor()
    
    categories = pd.Series([
        "New York",
        "new york",
        "NY",
        "NewYork",
        "Los Angeles",
        "los angeles",
        "LA"
    ])
    
    print("Before:", list(categories))
    
    standardized = processor.standardize_categories(categories)
    
    print("After:", list(standardized))
    print("✓ Categories standardized")


def test_duckdb_integration():
    """Test DuckDB integration"""
    print("\n=== Testing DuckDB Integration ===")
    
    processor = EnhancedDataProcessor()
    
    # Create sample data
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'C'],
        'value': [10, 20, 15, 25, 30],
        'quantity': [1, 2, 1, 3, 2]
    })
    
    print("Sample data:")
    print(df)
    
    # Load to DuckDB
    processor.load_to_duckdb(df, 'test_table')
    print("✓ Loaded to DuckDB")
    
    # Test query
    result = processor.query("SELECT * FROM test_table WHERE value > 15")
    print("\nFiltered result (value > 15):")
    print(result)
    print("✓ Query executed")
    
    # Test aggregation
    agg_result = processor.aggregate(
        'test_table',
        group_by='category',
        aggregations={'value': 'SUM', 'quantity': 'AVG'}
    )
    print("\nAggregation result:")
    print(agg_result)
    print("✓ Aggregation completed")


def test_complete_pipeline():
    """Test complete data pipeline"""
    print("\n=== Testing Complete Pipeline ===")
    
    processor = EnhancedDataProcessor()
    
    # Create messy test data
    df = pd.DataFrame({
        'Product Name ': ['  iPhone  ', '  Samsung  ', '  iPhone ', 'samsung'],
        'Price ($)': ['$1,234.56', '$ 999.99', '1,199', '(50.00)'],
        'Sales Date': ['2025-11-12', '12/11/2025', 'Nov 12, 2025', '2025-11-12'],
        'Category': ['Phone', 'phone', 'PHONE', 'Phone'],
        'Stock': [10, np.nan, 5, 8]
    })
    
    print("Before pipeline:")
    print(df)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Manually run pipeline steps
    df = processor.clean_column_names(df)
    df = processor.remove_extra_spaces(df)
    df = processor.infer_and_convert_types(df)
    df = processor.handle_missing_values(df)
    
    print("\nAfter pipeline:")
    print(df)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("✓ Complete pipeline executed")
    
    # Load to DuckDB and test query
    processor.load_to_duckdb(df, 'products')
    
    result = processor.query("""
        SELECT 
            category,
            COUNT(*) as count,
            AVG(price) as avg_price
        FROM products
        GROUP BY category
    """)
    
    print("\nAnalysis result:")
    print(result)
    print("✓ DuckDB analysis completed")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("RUNNING ENHANCED DATA PROCESSOR TESTS")
    print("="*60)
    
    try:
        test_clean_numeric_column()
        test_date_parsing()
        test_missing_values()
        test_outlier_detection()
        test_category_standardization()
        test_duckdb_integration()
        test_complete_pipeline()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()
