"""Fallback Strategies Component
Centralizes fallback logic for missing optional dependencies and runtime failures.
Provides deterministic fallback selection and logging of capability degradation.
"""
from loguru import logger
from typing import Literal, Optional, Callable, Any
from enum import Enum

# ======== OCR Fallback Strategy ========
class OCREngine(str, Enum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    UNAVAILABLE = "unavailable"

class OCRFallbackStrategy:
    """Determines OCR engine to use based on availability and task hints."""
    
    @staticmethod
    def check_availability() -> dict[str, bool]:
        """Check which OCR engines are available."""
        availability = {
            "easyocr": False,
            "tesseract": False
        }
        
        try:
            import easyocr
            availability["easyocr"] = True
        except ImportError:
            pass
        
        try:
            import pytesseract
            availability["tesseract"] = True
        except ImportError:
            pass
        
        return availability
    
    @staticmethod
    def select_engine(prefer_accuracy: bool = True, 
                     availability: Optional[dict] = None) -> OCREngine:
        """Select best available OCR engine.
        
        Args:
            prefer_accuracy: If True, prefer EasyOCR (slower, more accurate).
                           If False, prefer Tesseract (faster, less accurate).
            availability: Optional pre-checked availability dict.
        
        Returns:
            OCREngine enum indicating which engine to use.
        """
        if availability is None:
            availability = OCRFallbackStrategy.check_availability()
        
        if prefer_accuracy and availability["easyocr"]:
            logger.info("OCR fallback: Selected EasyOCR (accuracy mode)")
            return OCREngine.EASYOCR
        
        if availability["tesseract"]:
            logger.info("OCR fallback: Selected Tesseract")
            return OCREngine.TESSERACT
        
        if availability["easyocr"]:
            logger.info("OCR fallback: Selected EasyOCR (fallback)")
            return OCREngine.EASYOCR
        
        logger.warning("OCR fallback: No engines available")
        return OCREngine.UNAVAILABLE


# ======== Visualization Fallback Strategy ========
class VizEngine(str, Enum):
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    UNAVAILABLE = "unavailable"

class VisualizationFallbackStrategy:
    """Determines visualization library to use."""
    
    @staticmethod
    def check_availability() -> dict[str, bool]:
        """Check which visualization libraries are available."""
        availability = {
            "plotly": False,
            "kaleido": False,  # For static Plotly export
            "matplotlib": False
        }
        
        try:
            import plotly.express
            availability["plotly"] = True
        except ImportError:
            pass
        
        try:
            import kaleido
            availability["kaleido"] = True
        except ImportError:
            pass
        
        try:
            import matplotlib
            availability["matplotlib"] = True
        except ImportError:
            pass
        
        return availability
    
    @staticmethod
    def select_engine(prefer_interactive: bool = True,
                     need_static_export: bool = True,
                     availability: Optional[dict] = None) -> VizEngine:
        """Select best visualization engine.
        
        Args:
            prefer_interactive: Prefer Plotly for interactive charts.
            need_static_export: Require static image export capability.
            availability: Optional pre-checked availability dict.
        
        Returns:
            VizEngine enum.
        """
        if availability is None:
            availability = VisualizationFallbackStrategy.check_availability()
        
        # Plotly with kaleido for static export
        if prefer_interactive and availability["plotly"]:
            if need_static_export and not availability["kaleido"]:
                logger.warning("Plotly available but kaleido missing; static export may fail")
            logger.info("Viz fallback: Selected Plotly")
            return VizEngine.PLOTLY
        
        # Matplotlib fallback
        if availability["matplotlib"]:
            logger.info("Viz fallback: Selected Matplotlib")
            return VizEngine.MATPLOTLIB
        
        logger.error("Viz fallback: No visualization libraries available")
        return VizEngine.UNAVAILABLE


# ======== Transcription Fallback Strategy ========
class TranscriptionEngine(str, Enum):
    OPENAI_WHISPER = "openai-whisper"
    UNAVAILABLE = "unavailable"

class TranscriptionFallbackStrategy:
    """Determines transcription capability."""
    
    @staticmethod
    def check_availability(api_key: Optional[str] = None) -> dict[str, bool]:
        """Check if transcription is available."""
        availability = {
            "openai": False
        }
        
        try:
            from openai import AsyncOpenAI
            if api_key:
                availability["openai"] = True
        except ImportError:
            pass
        
        return availability
    
    @staticmethod
    def select_engine(api_key: Optional[str] = None,
                     availability: Optional[dict] = None) -> TranscriptionEngine:
        """Select transcription engine."""
        if availability is None:
            availability = TranscriptionFallbackStrategy.check_availability(api_key)
        
        if availability["openai"]:
            logger.info("Transcription fallback: OpenAI Whisper available")
            return TranscriptionEngine.OPENAI_WHISPER
        
        logger.warning("Transcription fallback: No engines available")
        return TranscriptionEngine.UNAVAILABLE


# ======== Data Format Fallback Strategy ========
class DataFormatFallback:
    """Handle various data format parsing failures with graceful degradation."""
    
    @staticmethod
    def parse_csv_with_fallback(content: bytes) -> tuple[bool, Any]:
        """Try parsing CSV with multiple strategies.
        
        Returns:
            (success: bool, result: DataFrame or error message)
        """
        import pandas as pd
        import io
        
        # First, detect if first line is all numeric (no header)
        has_header = True
        try:
            lines = content.decode('utf-8', errors='ignore').split('\n')[:3]
            first_line = lines[0].strip()
            
            # Check delimited data
            for delimiter in [',', '\t', ';', '|']:
                if delimiter in first_line:
                    fields = first_line.split(delimiter)
                    # If ALL fields are purely numeric, likely no header
                    if all(field.strip().replace('.', '', 1).replace('-', '', 1).isdigit() for field in fields if field.strip()):
                        has_header = False
                        logger.info(f"CSV fallback detected no header (all numeric fields)")
                        break
            
            # Check single-column numeric data
            if has_header and ',' not in first_line and '\t' not in first_line:
                if all(line.strip().replace('.','',1).replace('-','',1).isdigit() for line in lines if line.strip()):
                    has_header = False
                    logger.info("CSV fallback detected no header (single numeric column)")
        except (UnicodeDecodeError, ValueError, IndexError):
            pass
        
        strategies = [
            {"name": "UTF-8 with header detection", "encoding": "utf-8", "header": 0 if has_header else None},
            {"name": "UTF-8 skip errors", "encoding": "utf-8", "on_bad_lines": "skip"},
            {"name": "Latin-1", "encoding": "latin-1", "error_bad_lines": False},
            {"name": "Force no header", "encoding": "utf-8", "header": None}
        ]
        
        for strategy in strategies:
            try:
                name = strategy.pop("name")
                df = pd.read_csv(io.BytesIO(content), **strategy)
                # If no header, assign generic column names
                if df.columns[0] == 0 or (isinstance(df.columns[0], int)):
                    df.columns = [f'value_{i}' for i in range(len(df.columns))]
                logger.success(f"CSV parsed with strategy: {name}")
                return True, df
            except Exception as e:
                logger.debug(f"CSV strategy '{name}' failed: {e}")
                continue
        
        logger.error("All CSV parsing strategies failed")
        return False, "CSV_PARSE_FAILED"
    
    @staticmethod
    def parse_json_with_fallback(content: bytes) -> tuple[bool, Any]:
        """Try parsing JSON with encoding fallbacks."""
        import json
        
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                data = json.loads(text)
                logger.success(f"JSON parsed with encoding: {encoding}")
                return True, data
            except Exception as e:
                logger.debug(f"JSON parse failed with {encoding}: {e}")
                continue
        
        logger.error("All JSON parsing strategies failed")
        return False, "JSON_PARSE_FAILED"


# ======== Network Request Fallback Strategy ========
class NetworkFallback:
    """Handle network failures with retry and timeout strategies."""
    
    @staticmethod
    def get_retry_config(task_type: Literal["critical", "standard", "optional"]) -> dict:
        """Get retry configuration based on task criticality.
        
        Returns dict with: max_attempts, base_delay, max_delay, timeout
        """
        configs = {
            "critical": {
                "max_attempts": 5,
                "base_delay": 2,
                "max_delay": 30,
                "timeout": 60
            },
            "standard": {
                "max_attempts": 3,
                "base_delay": 1,
                "max_delay": 10,
                "timeout": 30
            },
            "optional": {
                "max_attempts": 2,
                "base_delay": 1,
                "max_delay": 5,
                "timeout": 15
            }
        }
        
        config = configs.get(task_type, configs["standard"])
        logger.debug(f"Network fallback config ({task_type}): {config}")
        return config


# ======== Utility: Graceful Degradation Wrapper ========
def with_fallback(primary_fn: Callable, 
                 fallback_fn: Optional[Callable] = None,
                 fallback_value: Any = None,
                 log_prefix: str = "Operation") -> Callable:
    """Decorator/wrapper for graceful degradation.
    
    Tries primary function; if it fails, tries fallback_fn or returns fallback_value.
    """
    async def wrapper(*args, **kwargs):
        try:
            result = await primary_fn(*args, **kwargs) if callable(primary_fn) else primary_fn
            logger.debug(f"{log_prefix}: Primary succeeded")
            return result
        except Exception as e:
            logger.warning(f"{log_prefix}: Primary failed ({e}), using fallback")
            
            if fallback_fn and callable(fallback_fn):
                try:
                    result = await fallback_fn(*args, **kwargs)
                    logger.info(f"{log_prefix}: Fallback succeeded")
                    return result
                except Exception as fe:
                    logger.error(f"{log_prefix}: Fallback also failed ({fe})")
            
            logger.info(f"{log_prefix}: Returning fallback value")
            return fallback_value
    
    return wrapper
