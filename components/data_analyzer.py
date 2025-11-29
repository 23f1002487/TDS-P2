"""
Data Analysis Component
Handles data processing, analysis, and visualization
Unified component containing data processing and analysis logic
"""
import tempfile
import os
import re
import io
import zipfile
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urljoin
from io import BytesIO

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
import duckdb
import pandas as pd
import numpy as np
import pdfplumber
from PIL import Image
import pytesseract

from .visualization import Visualizer
from .fallback_strategies import DataFormatFallback


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
    
    def load_csv(self, content, registry=None, **kwargs):
        """Load CSV with automatic delimiter detection and fallback strategies.
        
        Args:
            content: CSV bytes
            registry: Optional CapabilityRegistry to record parsing strategy
            **kwargs: Additional pandas read_csv arguments
        """
        logger.info("Loading CSV")
        
        # First try centralized fallback strategy
        success, result = DataFormatFallback.parse_csv_with_fallback(content)
        if success:
            if registry:
                registry.record("csv_parsing", "fallback_strategy")
            return result
        
        # Original logic as final fallback
        try:
            first_lines = content.decode('utf-8', errors='ignore').split('\n')[:5]
            first_line = first_lines[0].strip()
            
            # Check if first line looks like data (all numeric) rather than header
            has_header = True  # Default to assuming header
            
            # Check for delimited data
            for delimiter in [',', '\t', ';', '|']:
                if delimiter in first_line:
                    fields = first_line.split(delimiter)
                    # If ALL fields are purely numeric, likely no header
                    if all(field.strip().replace('.', '', 1).replace('-', '', 1).isdigit() for field in fields if field.strip()):
                        has_header = False
                        logger.info(f"CSV appears to have no header (all fields numeric with delimiter '{delimiter}')")
                        break
            
            # Check for single-column numeric data (no delimiters)
            if has_header and ',' not in first_line and '\t' not in first_line:
                if all(line.strip().replace('.','',1).replace('-','',1).isdigit() for line in first_lines if line.strip()):
                    has_header = False
                    logger.info("CSV appears to have no header (single numeric column)")
            
            if has_header:
                logger.info("CSV appears to have header")
        except (UnicodeDecodeError, ValueError, IndexError) as e:
            logger.debug(f"Header detection error: {e}")
            has_header = True
        
        for delimiter in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(
                    BytesIO(content), 
                    delimiter=delimiter, 
                    header=0 if has_header else None,
                    **kwargs
                )
                if df.shape[1] > 1 or not has_header:
                    logger.info(f"CSV loaded with delimiter '{delimiter}', header={has_header}: {df.shape}")
                    if not has_header:
                        df.columns = [f'value_{i}' for i in range(len(df.columns))]
                    if registry:
                        registry.record("csv_parsing", f"manual_delimiter_{delimiter}")
                    return df
            except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
                continue
        
        df = pd.read_csv(BytesIO(content), header=0 if has_header else None, **kwargs)
        if not has_header:
            df.columns = [f'value_{i}' for i in range(len(df.columns))]
        logger.info(f"CSV loaded: {df.shape}")
        if registry:
            registry.record("csv_parsing", "default")
        return df
    
    def load_excel(self, content, sheet_name=0, **kwargs):
        """Load Excel file"""
        logger.info(f"Loading Excel, sheet: {sheet_name}")
        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name, **kwargs)
        logger.info(f"Excel loaded: {df.shape}")
        return df
    
    def load_json(self, content, registry=None):
        """Load JSON data with encoding fallback.
        
        Args:
            content: JSON bytes
            registry: Optional CapabilityRegistry to record parsing strategy
        """
        logger.info("Loading JSON")
        
        # Try fallback strategy first
        success, result = DataFormatFallback.parse_json_with_fallback(content)
        if not success:
            logger.error("JSON parsing failed with all strategies")
            if registry:
                registry.record("json_parsing", "failed")
            raise ValueError(result)
        
        data = result
        if registry:
            registry.record("json_parsing", "success")
        
        # Convert to DataFrame if possible
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return data
        
        logger.info(f"JSON loaded: {df.shape}")
        return df
    
    def load_pdf(self, content, page_numbers=None):
        """Extract tables from PDF"""
        logger.info("Loading PDF")
        
        tables = []
        
        with pdfplumber.open(BytesIO(content)) as pdf:
            pages = pdf.pages if not page_numbers else [pdf.pages[p-1] for p in page_numbers]
            
            for page_num, page in enumerate(pages, 1):
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df['_source_page'] = page_num
                        tables.append(df)
                        logger.info(f"Extracted table from page {page_num}: {df.shape}")
        
        return tables
    
    def clean_column_names(self, df):
        """Standardize column names"""
        if df.empty or len(df.columns) == 0:
            return df
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        return df
    
    def remove_extra_spaces(self, df):
        """Remove extra spaces from string columns"""
        df = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        return df
    
    def infer_and_convert_types(self, df):
        """Automatically infer and convert column types"""
        df = df.copy()
        
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64] or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            # Try numeric
            try:
                numeric = pd.to_numeric(df[col], errors='coerce')
                if numeric.notna().sum() / len(df) > 0.9:
                    df[col] = numeric
                    continue
            except:
                pass
            
            # Try date
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                if dates.notna().sum() / len(df) > 0.9:
                    df[col] = dates
                    continue
            except:
                pass
        
        return df
    
    def handle_missing_values(self, df, strategy='auto'):
        """Handle missing values intelligently"""
        df = df.copy()
        
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
        
        return df
    
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
    
    def complete_data_pipeline(self, file_path, source_type='auto', registry=None, **kwargs):
        """Complete end-to-end data processing pipeline.
        
        Args:
            file_path: Path to data file
            source_type: Type of data source (csv, json, etc)
            registry: Optional CapabilityRegistry
            **kwargs: Additional arguments
        """
        logger.info("Starting data pipeline")
        
        # Read file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Detect type from file extension if auto
        if source_type == 'auto':
            ext = file_path.split('.')[-1].lower()
            type_map = {'csv': 'csv', 'xlsx': 'excel', 'xls': 'excel', 'json': 'json', 'pdf': 'pdf', 'zip': 'zip'}
            source_type = type_map.get(ext, 'csv')
        
        # Load data
        if source_type == 'zip':
            # Extract ZIP and process first data file found
            import zipfile
            import tempfile
            import os
            
            logger.info("Processing ZIP archive")
            temp_dir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    zf.extractall(temp_dir)
                    logger.info(f"Extracted ZIP contents: {zf.namelist()}")
                    
                # Find first data file in extracted contents
                data_file = None
                all_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        all_files.append(file)
                        # Look for common data file extensions, including log files
                        if file.lower().endswith(('.csv', '.json', '.xlsx', '.xls', '.txt', '.log', '.tsv')):
                            data_file = os.path.join(root, file)
                            break
                        # Also accept files without extensions as potential data files
                        elif '.' not in file:
                            data_file = os.path.join(root, file)
                            break
                    if data_file:
                        break
                
                if not data_file:
                    logger.error(f"No data file found in ZIP archive. Files present: {all_files}")
                    df = pd.DataFrame()
                else:
                    logger.info(f"Found data file in ZIP: {os.path.basename(data_file)}")
                    # Recursively process the extracted file
                    df, _ = self.complete_data_pipeline(data_file, source_type='auto', registry=registry, **kwargs)
                    
            finally:
                # Cleanup temp directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        elif source_type == 'pdf':
            tables = self.load_pdf(content, **kwargs)
            df = tables[0] if tables else pd.DataFrame()
        elif source_type == 'csv':
            df = self.load_csv(content, registry=registry, **kwargs)
        elif source_type == 'excel':
            df = self.load_excel(content, **kwargs)
        elif source_type == 'json':
            df = self.load_json(content, registry=registry)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        logger.info(f"Initial data shape: {df.shape}")
        
        # Clean and process
        df = self.clean_column_names(df)
        df = self.remove_extra_spaces(df)
        df = self.infer_and_convert_types(df)
        df = self.handle_missing_values(df, strategy='auto')
        
        # Load to DuckDB
        table_name = 'cleaned_data'
        self.load_to_duckdb(df, table_name)
        
        logger.info(f"Pipeline complete. Final shape: {df.shape}")
        return df, table_name


class DataAnalyzer:
    """
    Manages data downloading, processing, and analysis operations
    """
    
    def __init__(self):
        self.data_processor = EnhancedDataProcessor()
        self.visualizer = Visualizer()
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("DataAnalyzer initialized")
    
    async def cleanup(self):
        """Clean up HTTP client"""
        await self.http_client.aclose()
        logger.info("DataAnalyzer cleanup complete")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def download_data(self, url: str, headers: Dict[str, str] = None, registry=None) -> bytes:
        """Download data file with retry logic and optional custom headers.
        
        Args:
            url: URL to download from
            headers: Optional custom HTTP headers
            registry: Optional CapabilityRegistry to record download strategy
        """
        logger.info(f"Downloading data from: {url}")
        if headers:
            logger.info(f"Using custom headers: {list(headers.keys())}")
            if registry:
                registry.record("api_custom_headers", True)
        
        try:
            response = await self.http_client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.content
            logger.success(f"Downloaded {len(data)} bytes")
            if registry:
                registry.record("download_strategy", "standard")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            if registry:
                registry.record("download_strategy", "failed")
            raise
    
    async def process_data(self, task_info: Dict, quiz_url: str = None, registry=None) -> Tuple[Any, Optional[str]]:
        """Download and process data based on task information.
        
        Args:
            task_info: Dictionary containing task details
            quiz_url: Base URL for resolving relative URLs
            registry: Optional CapabilityRegistry to record processing steps
        """
        logger.info("Processing data")
        
        data_url = task_info.get('data_source_url')
        if not data_url:
            logger.warning("No data source URL found")
            return None, None
        
        # Extract custom headers if provided in task info
        custom_headers = task_info.get('headers') or task_info.get('api_headers')
        if custom_headers:
            logger.info(f"Task specifies custom headers: {list(custom_headers.keys())}")
        
        # Handle relative data URLs
        if not data_url.startswith('http://') and not data_url.startswith('https://'):
            if quiz_url:
                data_url = urljoin(quiz_url, data_url)
                logger.info(f"Converted relative data URL to: {data_url}")
            else:
                logger.warning(f"Data URL appears relative but no quiz_url provided: {data_url}")
        
        # Check if this is a scraping task (before downloading)
        operation = task_info.get('operation', '').lower()
        task_summary = task_info.get('task_summary', '').lower()
        data_source_type = task_info.get('data_source_type', '').lower()
        
        # Detect scraping tasks - these need Playwright (JavaScript execution)
        is_scraping = ('scrape' in operation or 'scrape' in task_summary or 
                      'extract' in task_summary or data_source_type == 'html')
        
        # Check file extension to determine if it's likely a data file
        url_lower = data_url.lower()
        is_data_file = any(ext in url_lower for ext in ['.csv', '.xlsx', '.xls', '.json', '.pdf', '.parquet'])
        
        if is_scraping and not is_data_file:
            # This is a scraping task - mark it for Playwright processing
            logger.info("Detected HTML scraping task - will use Playwright to fetch and extract")
            return ('HTML_SCRAPING', data_url, None), None
        
        # Download data file (with custom headers if provided)
        data_content = await self.download_data(data_url, headers=custom_headers, registry=registry)
        
        # Check if downloaded content is HTML when we expected data
        content_str = data_content.decode('utf-8', errors='ignore')[:1000].strip()
        if content_str.startswith('<') or content_str.startswith('<!DOCTYPE'):
            # Got HTML when expecting data file: attempt to discover linked data resources (e.g., CSV)
            logger.warning("Downloaded content appears to be HTML, not a data file. Attempting to locate linked data resources.")
            logger.info(f"Content preview: {content_str[:200]}")

            # Simple href parser for CSV/JSON/TSV links
            linked_url = None
            try:
                import re
                # Find first href that points to a likely data file
                hrefs = re.findall(r'href\s*=\s*\"([^\"]+)\"', content_str, flags=re.IGNORECASE)
                candidates = []
                for h in hrefs:
                    hl = h.lower()
                    if any(hl.endswith(ext) or ext in hl for ext in [".csv", ".tsv", ".json", ".parquet"]):
                        candidates.append(h)
                if candidates:
                    linked_url = candidates[0]
            except Exception as e:
                logger.error(f"Error scanning HTML for linked data: {e}")

            if linked_url:
                # Resolve relative URL using quiz_url or current data_url as base
                try:
                    base_for_join = quiz_url or data_url
                    resolved_url = urljoin(base_for_join, linked_url)
                    logger.info(f"Found linked data resource: {linked_url} -> {resolved_url}")
                    # Download the linked data
                    data_content = await self.download_data(resolved_url, headers=custom_headers, registry=registry)
                    data_url = resolved_url
                    logger.success(f"Downloaded linked data: {len(data_content)} bytes")
                except Exception as e:
                    logger.error(f"Failed to download linked data resource: {e}")
                    return None, None
            else:
                # No linked data found; nothing to process
                logger.warning("No linked data resources (CSV/JSON/TSV) found in HTML. Skipping data processing.")
                return None, None
        
        # Build kwargs for loading
        kwargs = {}
        if task_info.get('page_number'):
            try:
                page_num = int(task_info['page_number'])
                kwargs['page_numbers'] = [page_num]
            except (ValueError, TypeError, KeyError):
                pass
        
        # Extract file extension safely
        url_parts = data_url.split('/')[-1].split('?')[0]
        valid_extensions = {'csv', 'xlsx', 'xls', 'json', 'pdf', 'parquet', 'html', 'xml', 'tsv', 'txt', 'zip'}
        
        if '.' in url_parts:
            file_ext = url_parts.split('.')[-1].lower()
            if file_ext not in valid_extensions:
                type_map = {'csv': 'csv', 'excel': 'xlsx', 'json': 'json', 'pdf': 'pdf'}
                file_ext = type_map.get(task_info.get('data_source_type', '').lower(), 'dat')
        else:
            type_map = {'csv': 'csv', 'excel': 'xlsx', 'json': 'json', 'pdf': 'pdf'}
            file_ext = type_map.get(task_info.get('data_source_type', '').lower(), 'dat')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
            tmp.write(data_content)
            tmp_path = tmp.name
        
        try:
            # Use the complete data pipeline
            df, table_name = self.data_processor.complete_data_pipeline(
                tmp_path,
                source_type=task_info.get('data_source_type', 'auto'),
                registry=registry,
                **kwargs
            )
            
            logger.success(f"Data processed: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            if registry:
                registry.record("data_format", task_info.get('data_source_type', 'auto'))
            
            return df, table_name
            
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
    
    async def perform_analysis(self, df, table_name: str, task_info: Dict, llm_client=None, quiz_info: Dict = None) -> Any:
        """Perform the required analysis using DuckDB"""
        logger.info("Performing analysis")
        
        operation = task_info.get('operation', '').lower()
        target_column = task_info.get('target_column')
        
        # Find target column (case-insensitive, fuzzy match)
        if target_column:
            target_column_clean = target_column.lower().strip()
            for col in df.columns:
                if col.lower() == target_column_clean or target_column_clean in col.lower():
                    target_column = col
                    break
        
        logger.info(f"Operation: {operation}, Target column: {target_column}")
        
        try:
            if operation == 'sum' and target_column:
                result = self.data_processor.query(
                    f"SELECT SUM({target_column}) as result FROM {table_name}"
                )['result'].iloc[0]
                logger.success(f"Sum result: {result}")
                return result
            
            elif operation == 'average' and target_column:
                result = self.data_processor.query(
                    f"SELECT AVG({target_column}) as result FROM {table_name}"
                )['result'].iloc[0]
                logger.success(f"Average result: {result}")
                return result
            
            elif operation == 'count':
                if target_column:
                    sql = f"SELECT COUNT({target_column}) as result FROM {table_name}"
                else:
                    sql = f"SELECT COUNT(*) as result FROM {table_name}"
                result = self.data_processor.query(sql)['result'].iloc[0]
                logger.success(f"Count result: {result}")
                return result
            
            elif operation in ['max', 'min'] and target_column:
                func = operation.upper()
                result = self.data_processor.query(
                    f"SELECT {func}({target_column}) as result FROM {table_name}"
                )['result'].iloc[0]
                logger.success(f"{func} result: {result}")
                return result
            
            elif operation == 'visualize':
                return await self.create_visualization(df, task_info, llm_client)
            
            else:
                # Use LLM for complex analysis
                if llm_client:
                    return await self.llm_assisted_analysis(df, table_name, task_info, llm_client, quiz_info)
                else:
                    raise ValueError("LLM client required for complex analysis")
        
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            # Fallback to LLM
            if llm_client:
                return await self.llm_assisted_analysis(df, table_name, task_info, llm_client, quiz_info)
            raise
    
    async def llm_assisted_analysis(self, df, table_name: str, task_info: Dict, llm_client, quiz_info: Dict = None) -> Any:
        """Use LLM for complex data analysis"""
        # Include page elements (like cutoff values) in the analysis
        page_elements = {}
        if quiz_info and 'elements_with_ids' in quiz_info:
            page_elements = quiz_info['elements_with_ids']
            logger.info(f"Including page elements in analysis: {list(page_elements.keys())}")
        
        analysis_result = await llm_client.assisted_analysis(df, task_info, page_elements, table_name)
        
        # Execute SQL if provided
        if analysis_result.get('sql_query'):
            try:
                safe_query = self._sanitize_sql(analysis_result['sql_query'], table_name, df.columns)
                result_df = self.data_processor.query(safe_query)
                logger.info(f"SQL query result:\n{result_df}")
                
                if len(result_df) == 1 and len(result_df.columns) == 1:
                    answer = result_df.iloc[0, 0]
                else:
                    answer = result_df.to_dict('records')
            except Exception as e:
                logger.error(f"SQL execution failed: {e}")
                answer = analysis_result.get('answer')
        else:
            answer = analysis_result.get('answer')
        
        logger.success(f"LLM analysis result: {answer}")
        return answer

    def _sanitize_sql(self, query: str, table_name: str, allowed_columns) -> str:
        """Sanitize LLM-provided SQL to prevent wrong table usage and injection.
        - Force single SELECT statement.
        - Replace any table references not matching table_name.
        - Remove disallowed keywords.
        """
        original = query
        q = query.strip().rstrip(';')
        upper = q.upper()
        if ';' in q:
            first = q.split(';')[0]
            logger.warning("Multiple statements detected; using first.")
            q = first
        disallowed = ["DROP", "ALTER", "DELETE", "UPDATE", "INSERT", "CREATE", "ATTACH", "DETACH"]
        if any(word in upper for word in disallowed):
            logger.warning("Disallowed keyword found; falling back to simple SUM.")
            # Fallback simple aggregation attempt
            # Choose first numeric column
            num_cols = [c for c in allowed_columns if c.lower().startswith('value') or c.lower().startswith('num')]
            col = num_cols[0] if num_cols else allowed_columns[0]
            return f"SELECT SUM({col}) as result FROM {table_name}"
        # Replace common wrong table names
        import re
        q = re.sub(r"\bdata\b", table_name, q, flags=re.IGNORECASE)
        q = re.sub(r"\bcleaned_data\b", table_name, q, flags=re.IGNORECASE)  # ensure consistent
        # Whitelist columns in SELECT clause simplistic approach
        # If SELECT * keep it; else ensure columns exist
        try:
            select_part = re.search(r"SELECT(.*)FROM", q, flags=re.IGNORECASE | re.DOTALL)
            if select_part:
                cols_segment = select_part.group(1)
                if '*' not in cols_segment:
                    # Split by commas and validate
                    pieces = [p.strip() for p in cols_segment.split(',')]
                    cleaned = []
                    for p in pieces:
                        # Remove aliases
                        base = p.split()[0]
                        base_clean = re.sub(r'[^A-Za-z0-9_]', '', base)
                        if base_clean.lower() in [c.lower() for c in allowed_columns] or any(func in base_clean.lower() for func in ["sum","avg","count","max","min"]):
                            cleaned.append(p)
                    if cleaned:
                        new_cols = ', '.join(cleaned)
                        q = re.sub(cols_segment, ' ' + new_cols + ' ', q)
        except Exception as e:
            logger.debug(f"Column whitelist failed: {e}")
        logger.info(f"Sanitized SQL from: {original} --> {q}")
        return q
    
    async def create_visualization(self, df, task_info: Dict, llm_client) -> str:
        """Create visualization based on task requirements"""
        logger.info("Creating visualization")
        
        viz_info = await llm_client.get_visualization_config(df, task_info)
        
        chart_type = viz_info.get('chart_type')
        x_col = viz_info.get('x_column')
        y_col = viz_info.get('y_column')
        title = viz_info.get('title', 'Chart')
        
        # Create chart
        if chart_type == 'bar_chart':
            result = self.visualizer.create_bar_chart(df, x_col, y_col, title)
        elif chart_type == 'line_chart':
            result = self.visualizer.create_line_chart(df, x_col, y_col, title)
        elif chart_type == 'scatter_plot':
            result = self.visualizer.create_scatter_plot(df, x_col, y_col, title)
        elif chart_type == 'pie_chart':
            result = self.visualizer.create_pie_chart(df, x_col, y_col, title)
        else:
            result = self.visualizer.create_bar_chart(df, x_col, y_col, title)
        
        logger.success("Visualization created")
        return result
    
    def format_answer(self, answer: Any, expected_type: str) -> Any:
        """Format answer according to expected type"""
        logger.info(f"Formatting answer as: {expected_type}")
        
        if expected_type == 'number':
            try:
                if isinstance(answer, (int, float)):
                    return int(answer) if answer == int(answer) else float(answer)
                return float(answer)
            except:
                return answer
        
        elif expected_type == 'boolean':
            if isinstance(answer, bool):
                return answer
            if isinstance(answer, str):
                return answer.lower() in ['true', 'yes', '1', 't', 'y']
            return bool(answer)
        
        elif expected_type == 'json':
            if isinstance(answer, (dict, list)):
                return answer
            try:
                import json
                return json.loads(answer)
            except:
                return answer
        
        elif expected_type == 'base64_image':
            return answer
        
        else:
            return str(answer)
