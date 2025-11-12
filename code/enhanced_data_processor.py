"""
Enhanced Data Processor with DuckDB integration
Handles data extraction, cleaning, normalization, and analysis
"""
import duckdb
import pandas as pd
import numpy as np
import requests
import logging
import re
from io import BytesIO
from datetime import datetime
import pdfplumber
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class EnhancedDataProcessor:
    """Advanced data processing with DuckDB for fast analytics"""
    
    def __init__(self):
        # Initialize in-memory DuckDB connection
        self.conn = duckdb.connect(':memory:')
        logger.info("DuckDB connection initialized")
    
    def __del__(self):
        """Clean up DuckDB connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    # ==================== DATA LOADING ====================
    
    @staticmethod
    def download_file(url, headers=None):
        """Download file from URL with retry logic"""
        logger.info(f"Downloading: {url}")
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                logger.info(f"Downloaded {len(response.content)} bytes")
                return response.content
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
    
    def load_pdf(self, content_or_url, page_numbers=None):
        """
        Extract tables and text from PDF
        Args:
            content_or_url: PDF bytes or URL
            page_numbers: List of page numbers (1-indexed) or None for all
        """
        logger.info("Loading PDF")
        
        if isinstance(content_or_url, str) and content_or_url.startswith('http'):
            content = self.download_file(content_or_url)
        else:
            content = content_or_url
        
        tables = []
        all_text = []
        
        with pdfplumber.open(BytesIO(content)) as pdf:
            pages = pdf.pages if not page_numbers else [pdf.pages[p-1] for p in page_numbers]
            
            for page_num, page in enumerate(pages, 1):
                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df['_source_page'] = page_num
                        tables.append(df)
                        logger.info(f"Extracted table from page {page_num}: {df.shape}")
                
                # Extract text
                text = page.extract_text()
                if text:
                    all_text.append(text)
        
        return tables, '\n'.join(all_text)
    
    def load_csv(self, content_or_url, **kwargs):
        """Load CSV with automatic delimiter detection"""
        logger.info("Loading CSV")
        
        if isinstance(content_or_url, str) and content_or_url.startswith('http'):
            content = self.download_file(content_or_url)
        else:
            content = content_or_url
        
        # Try different delimiters
        for delimiter in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(BytesIO(content), delimiter=delimiter, **kwargs)
                if df.shape[1] > 1:  # Valid if more than 1 column
                    logger.info(f"CSV loaded with delimiter '{delimiter}': {df.shape}")
                    return df
            except:
                continue
        
        # Fallback to default
        df = pd.read_csv(BytesIO(content), **kwargs)
        logger.info(f"CSV loaded: {df.shape}")
        return df
    
    def load_excel(self, content_or_url, sheet_name=0, **kwargs):
        """Load Excel file"""
        logger.info(f"Loading Excel, sheet: {sheet_name}")
        
        if isinstance(content_or_url, str) and content_or_url.startswith('http'):
            content = self.download_file(content_or_url)
        else:
            content = content_or_url
        
        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name, **kwargs)
        logger.info(f"Excel loaded: {df.shape}")
        return df
    
    def load_html_tables(self, content_or_url, table_index=None):
        """Extract tables from HTML"""
        logger.info("Loading HTML tables")
        
        if isinstance(content_or_url, str) and content_or_url.startswith('http'):
            content = self.download_file(content_or_url).decode('utf-8')
        else:
            content = content_or_url
        
        tables = pd.read_html(content)
        logger.info(f"Found {len(tables)} tables in HTML")
        
        if table_index is not None:
            return tables[table_index]
        return tables
    
    def load_json(self, content_or_url):
        """Load JSON data"""
        logger.info("Loading JSON")
        
        if isinstance(content_or_url, str) and content_or_url.startswith('http'):
            content = self.download_file(content_or_url)
        else:
            content = content_or_url
        
        import json
        data = json.loads(content)
        
        # Convert to DataFrame if possible
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return data
        
        logger.info(f"JSON loaded: {df.shape}")
        return df
    
    def extract_text_from_image(self, image_content_or_url):
        """OCR text extraction from images"""
        logger.info("Extracting text from image using OCR")
        
        if isinstance(image_content_or_url, str) and image_content_or_url.startswith('http'):
            content = self.download_file(image_content_or_url)
        else:
            content = image_content_or_url
        
        image = Image.open(BytesIO(content))
        text = pytesseract.image_to_string(image)
        logger.info(f"Extracted {len(text)} characters from image")
        return text
    
    # ==================== DATA CLEANING ====================
    
    def clean_column_names(self, df):
        """Standardize column names"""
        logger.info("Cleaning column names")
        
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        
        logger.info(f"Cleaned columns: {list(df.columns)}")
        return df
    
    def remove_extra_spaces(self, df):
        """Remove extra spaces from string columns"""
        logger.info("Removing extra spaces from strings")
        
        df = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        
        return df
    
    def clean_numeric_column(self, series, currency_symbols=None):
        """
        Clean numeric columns with currency, separators, special chars
        Args:
            series: Pandas Series
            currency_symbols: List of currency symbols to remove (default: common ones)
        """
        if currency_symbols is None:
            currency_symbols = ['$', '€', '£', '¥', '₹', '₽', '₩', '¢']
        
        logger.info(f"Cleaning numeric column: {series.name}")
        
        cleaned = series.astype(str).copy()
        
        # Remove currency symbols
        for symbol in currency_symbols:
            cleaned = cleaned.str.replace(symbol, '', regex=False)
        
        # Remove common separators and special chars
        cleaned = cleaned.str.replace(',', '')  # Thousand separator
        cleaned = cleaned.str.replace(' ', '')  # Spaces
        cleaned = cleaned.str.replace('_', '')  # Underscores
        cleaned = cleaned.str.replace('%', '')  # Percentage
        
        # Handle parentheses as negative (accounting format)
        mask = cleaned.str.contains(r'\(.*\)', regex=True, na=False)
        cleaned.loc[mask] = '-' + cleaned.loc[mask].str.replace(r'[()]', '', regex=True)
        
        # Remove any remaining non-numeric chars except decimal point and minus
        cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
        
        # Convert to numeric
        result = pd.to_numeric(cleaned, errors='coerce')
        
        logger.info(f"Converted {result.notna().sum()} values, {result.isna().sum()} NaN")
        return result
    
    def parse_dates(self, series, date_formats=None):
        """
        Parse date columns with multiple format support
        Args:
            series: Pandas Series
            date_formats: List of date formats to try
        """
        logger.info(f"Parsing dates: {series.name}")
        
        if date_formats is None:
            date_formats = [
                '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
                '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
                '%d.%m.%Y', '%Y.%m.%d',
                '%b %d, %Y', '%B %d, %Y',
                '%d %b %Y', '%d %B %Y',
            ]
        
        # Try pandas auto-detection first
        try:
            result = pd.to_datetime(series, errors='coerce')
            if result.notna().sum() > len(series) * 0.8:  # 80% success rate
                logger.info(f"Auto-parsed {result.notna().sum()} dates")
                return result
        except:
            pass
        
        # Try each format
        for fmt in date_formats:
            try:
                result = pd.to_datetime(series, format=fmt, errors='coerce')
                success_rate = result.notna().sum() / len(series)
                if success_rate > 0.8:
                    logger.info(f"Parsed with format {fmt}: {result.notna().sum()} dates")
                    return result
            except:
                continue
        
        # Fallback to coerce
        result = pd.to_datetime(series, errors='coerce')
        logger.info(f"Fallback parsing: {result.notna().sum()} dates")
        return result
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values intelligently
        Args:
            strategy: 'auto', 'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_forward'
        """
        logger.info(f"Handling missing values, strategy: {strategy}")
        
        df = df.copy()
        missing_before = df.isna().sum().sum()
        
        if strategy == 'auto':
            # Numeric: fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Categorical: fill with mode
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if df[col].isna().any():
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
        
        elif strategy == 'drop':
            df.dropna(inplace=True)
        
        elif strategy == 'fill_mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        elif strategy == 'fill_median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        elif strategy == 'fill_mode':
            for col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
        
        elif strategy == 'fill_forward':
            df.fillna(method='ffill', inplace=True)
        
        missing_after = df.isna().sum().sum()
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        return df
    
    def detect_and_remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        Detect and remove outliers
        Args:
            columns: List of columns to check (None = all numeric)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
        """
        logger.info(f"Detecting outliers, method: {method}")
        
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        mask = pd.Series([True] * len(df))
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_mask = (df[col] >= lower) & (df[col] <= upper)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_mask = z_scores < threshold
            
            outliers = (~col_mask).sum()
            if outliers > 0:
                logger.info(f"Column {col}: {outliers} outliers detected")
            
            mask &= col_mask
        
        logger.info(f"Removing {(~mask).sum()} rows with outliers")
        return df[mask].reset_index(drop=True)
    
    def infer_and_convert_types(self, df):
        """Automatically infer and convert column types"""
        logger.info("Inferring and converting column types")
        
        df = df.copy()
        
        for col in df.columns:
            # Skip if already numeric or datetime
            if df[col].dtype in [np.int64, np.float64] or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            # Try numeric conversion
            try:
                numeric = pd.to_numeric(df[col], errors='coerce')
                if numeric.notna().sum() / len(df) > 0.9:  # 90% success
                    df[col] = numeric
                    logger.info(f"Column {col} → numeric")
                    continue
            except:
                pass
            
            # Try date conversion
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                if dates.notna().sum() / len(df) > 0.9:
                    df[col] = dates
                    logger.info(f"Column {col} → datetime")
                    continue
            except:
                pass
            
            # Try boolean
            unique_vals = df[col].str.lower().unique()
            if len(unique_vals) <= 2 and all(v in ['true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n', None, 'nan'] for v in unique_vals):
                bool_map = {'true': True, 'false': False, 'yes': True, 'no': False, 
                           '1': True, '0': False, 't': True, 'f': False, 'y': True, 'n': False}
                df[col] = df[col].str.lower().map(bool_map)
                logger.info(f"Column {col} → boolean")
                continue
            
            # Keep as string but optimize
            df[col] = df[col].astype('string')
        
        return df
    
    # ==================== CATEGORICAL PROCESSING ====================
    
    def standardize_categories(self, series, mappings=None):
        """
        Standardize categorical values using fuzzy matching
        Args:
            mappings: Dict of {canonical_value: [variations]} or None for auto
        """
        logger.info(f"Standardizing categories: {series.name}")
        
        from difflib import get_close_matches
        
        series = series.copy()
        series_str = series.astype(str).str.strip().str.lower()
        
        if mappings:
            # Use provided mappings
            reverse_map = {}
            for canonical, variations in mappings.items():
                for var in variations:
                    reverse_map[var.lower()] = canonical
            
            series = series_str.map(reverse_map).fillna(series)
        
        else:
            # Auto-detect similar categories
            unique_vals = series_str.unique()
            clusters = {}
            processed = set()
            
            for val in unique_vals:
                if val in processed or pd.isna(val):
                    continue
                
                matches = get_close_matches(val, unique_vals, n=10, cutoff=0.85)
                if len(matches) > 1:
                    canonical = min(matches, key=len)  # Shortest as canonical
                    clusters[canonical] = matches
                    processed.update(matches)
            
            # Apply clustering
            for canonical, variations in clusters.items():
                for var in variations:
                    series[series_str == var] = canonical
                    logger.info(f"Unified: {var} → {canonical}")
        
        return series
    
    # ==================== DUCKDB INTEGRATION ====================
    
    def load_to_duckdb(self, df, table_name='data', if_exists='replace'):
        """Load DataFrame into DuckDB"""
        logger.info(f"Loading to DuckDB table: {table_name}")
        
        if if_exists == 'replace':
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info(f"Loaded {row_count} rows to {table_name}")
    
    def query(self, sql):
        """Execute SQL query and return DataFrame"""
        logger.info(f"Executing query: {sql[:100]}...")
        result = self.conn.execute(sql).fetchdf()
        logger.info(f"Query returned {len(result)} rows")
        return result
    
    def aggregate(self, table_name, group_by, aggregations):
        """
        Perform aggregations using DuckDB
        Args:
            table_name: DuckDB table name
            group_by: Column(s) to group by
            aggregations: Dict like {'column': 'function'} or list of SQL expressions
        """
        logger.info(f"Aggregating table: {table_name}")
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        if isinstance(aggregations, dict):
            agg_exprs = [f"{func}({col}) as {col}_{func}" for col, func in aggregations.items()]
        else:
            agg_exprs = aggregations
        
        sql = f"""
        SELECT 
            {', '.join(group_by)},
            {', '.join(agg_exprs)}
        FROM {table_name}
        GROUP BY {', '.join(group_by)}
        """
        
        return self.query(sql)
    
    def filter_data(self, table_name, conditions):
        """
        Filter data using DuckDB
        Args:
            conditions: SQL WHERE clause or dict of {column: value}
        """
        logger.info(f"Filtering table: {table_name}")
        
        if isinstance(conditions, dict):
            where_parts = []
            for col, val in conditions.items():
                if isinstance(val, str):
                    where_parts.append(f"{col} = '{val}'")
                else:
                    where_parts.append(f"{col} = {val}")
            where_clause = ' AND '.join(where_parts)
        else:
            where_clause = conditions
        
        sql = f"SELECT * FROM {table_name} WHERE {where_clause}"
        return self.query(sql)
    
    def join_tables(self, left_table, right_table, on, how='inner'):
        """Join two DuckDB tables"""
        logger.info(f"Joining {left_table} and {right_table}")
        
        if isinstance(on, str):
            join_condition = f"{left_table}.{on} = {right_table}.{on}"
        else:
            join_parts = [f"{left_table}.{col} = {right_table}.{col}" for col in on]
            join_condition = ' AND '.join(join_parts)
        
        sql = f"""
        SELECT * FROM {left_table}
        {how.upper()} JOIN {right_table}
        ON {join_condition}
        """
        
        return self.query(sql)
    
    # ==================== COMPLETE WORKFLOW ====================
    
    def complete_data_pipeline(self, data_source, source_type='auto', **kwargs):
        """
        Complete end-to-end data processing pipeline
        Args:
            data_source: URL or file content
            source_type: 'auto', 'csv', 'excel', 'pdf', 'json', 'html'
        Returns:
            Cleaned DataFrame and DuckDB table name
        """
        logger.info("="*60)
        logger.info("STARTING COMPLETE DATA PIPELINE")
        logger.info("="*60)
        
        # 1. Load data
        if source_type == 'auto':
            source_type = self._detect_source_type(data_source)
        
        if source_type == 'pdf':
            tables, text = self.load_pdf(data_source, **kwargs)
            df = tables[0] if tables else pd.DataFrame()
        elif source_type == 'csv':
            df = self.load_csv(data_source, **kwargs)
        elif source_type == 'excel':
            df = self.load_excel(data_source, **kwargs)
        elif source_type == 'json':
            df = self.load_json(data_source)
        elif source_type == 'html':
            tables = self.load_html_tables(data_source)
            df = tables[0] if tables else pd.DataFrame()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        logger.info(f"Initial data shape: {df.shape}")
        
        # 2. Clean column names
        df = self.clean_column_names(df)
        
        # 3. Remove extra spaces
        df = self.remove_extra_spaces(df)
        
        # 4. Infer and convert types
        df = self.infer_and_convert_types(df)
        
        # 5. Handle missing values
        df = self.handle_missing_values(df, strategy='auto')
        
        # 6. Remove outliers (optional, only on request)
        # df = self.detect_and_remove_outliers(df)
        
        # 7. Load to DuckDB
        table_name = 'cleaned_data'
        self.load_to_duckdb(df, table_name)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"DuckDB table: {table_name}")
        logger.info("="*60)
        
        return df, table_name
    
    def _detect_source_type(self, source):
        """Auto-detect data source type"""
        if isinstance(source, str):
            source_lower = source.lower()
            if '.csv' in source_lower:
                return 'csv'
            elif '.xlsx' in source_lower or '.xls' in source_lower:
                return 'excel'
            elif '.pdf' in source_lower:
                return 'pdf'
            elif '.json' in source_lower:
                return 'json'
            elif '.html' in source_lower or source.startswith('http'):
                return 'html'
        return 'csv'  # Default fallback
