"""Vision / OCR Component
Provides enhanced OCR with optional EasyOCR fallback and preprocessing.
"""
from loguru import logger
from typing import Optional
import io
from PIL import Image, ImageOps, ImageFilter
import base64
from .fallback_strategies import OCRFallbackStrategy, OCREngine

try:
    import easyocr
    _EASY_AVAILABLE = True
except Exception:
    _EASY_AVAILABLE = False

try:
    import pytesseract
    _TESS_AVAILABLE = True
except Exception:
    _TESS_AVAILABLE = False

class VisionExtractor:
    def __init__(self):
        self.easy_reader = None
        self.availability = OCRFallbackStrategy.check_availability()
        self.selected_engine = OCRFallbackStrategy.select_engine(
            prefer_accuracy=True,
            availability=self.availability
        )
        
        if self.selected_engine == OCREngine.EASYOCR and _EASY_AVAILABLE:
            try:
                self.easy_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
                self.selected_engine = OCREngine.TESSERACT

    @staticmethod
    def _preprocess(img: Image.Image) -> Image.Image:
        # Convert to grayscale, increase contrast, sharpen
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        return img

    def ocr_bytes(self, content: bytes, hint: Optional[str] = None, registry=None) -> str:
        """Extract text from image bytes using best available OCR engine.
        
        Args:
            content: Image bytes
            hint: Optional hint about content type
            registry: Optional CapabilityRegistry to record engine used
        """
        try:
            img = Image.open(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to open image bytes: {e}")
            if registry:
                registry.record("ocr_engine", "image_load_failed")
            return "OCR_IMAGE_LOAD_FAILED"
        img = self._preprocess(img)

        # EasyOCR path
        if self.selected_engine == OCREngine.EASYOCR and self.easy_reader:
            try:
                results = self.easy_reader.readtext(content, detail=0)
                text = "\n".join(results).strip()
                logger.success("EasyOCR extraction complete")
                if registry:
                    registry.record("ocr_engine", "easyocr")
                return text or ""
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}; falling back to Tesseract")

        # Tesseract fallback
        if _TESS_AVAILABLE:
            try:
                text = pytesseract.image_to_string(img)
                logger.success("Tesseract OCR complete")
                if registry:
                    registry.record("ocr_engine", "tesseract")
                return text.strip() or ""
            except Exception as e:
                logger.error(f"Tesseract OCR failed: {e}")
                if registry:
                    registry.record("ocr_engine", "tesseract_failed")
                return "OCR_TESSERACT_FAILED"

        logger.warning("No OCR engine available")
        if registry:
            registry.record("ocr_engine", "unavailable")
        return "OCR_UNAVAILABLE"
