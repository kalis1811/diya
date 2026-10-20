import pymupdf as fitz
import numpy as np

_BACKEND = None


def needs_ocr(page_text: str, min_chars: int = 20) -> bool:
    return len(page_text.strip()) < min_chars


def _load_paddle():
    from paddleocr import PaddleOCR

    try:
        engine = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                           use_doc_unwarping=False, use_textline_orientation=False)
    except TypeError:
        engine = PaddleOCR(lang="en")

    def run(np_img) -> list[str]:
        texts: list[str] = []
        for res in engine.predict(np_img):
            if isinstance(res, dict):
                texts.extend(res.get("rec_texts") or [])
            else:
                texts.extend(getattr(res, "rec_texts", None) or [])
        return texts

    return run


def _load_rapid():
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def run(np_img) -> list[str]:
        result, _elapse = engine(np_img)
        return [item[1] for item in (result or [])]

    return run


def get_ocr_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        _BACKEND = _load_paddle()
    except ImportError:
        _BACKEND = _load_rapid()
    return _BACKEND


def ocr_pdf_pages(path, pages: list[int] | None = None,
                  backend=None, dpi: int = 200) -> dict[int, str]:
    fn = backend or get_ocr_backend()
    doc = fitz.open(path)
    try:
        targets = pages if pages is not None else list(range(len(doc)))
        out: dict[int, str] = {}
        for i in targets:
            pix = doc[i].get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            out[i] = "\n".join(fn(img)).strip()
        return out
    finally:
        doc.close()


def ocr_image(path, backend=None, dpi: int = 200) -> str:
    """Recognize text in a standalone raster image (JPEG, PNG, TIFF, etc.)."""
    fn = backend or get_ocr_backend()
    pix = fitz.Pixmap(str(path))
    if pix.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        img = img[:, :, 0]
    texts = fn(img)
    return "\n".join(texts).strip()
