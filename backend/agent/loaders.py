from __future__ import annotations

from pathlib import Path

def load_text(path: str | Path) -> str:
    path = Path(path)
    suf = path.suffix.lower()

    if suf in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suf == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError("Missing dependency for PDFs. Install pypdf.") from e

        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n\n".join(parts)

    if suf in {".docx"}:
        try:
            import docx
        except Exception as e:
            raise RuntimeError("Missing dependency for DOCX. Install python-docx.") from e

        d = docx.Document(str(path))
        return "\n".join([p.text for p in d.paragraphs])

    # fallback: treat as text
    return path.read_text(encoding="utf-8", errors="ignore")
