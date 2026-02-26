from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]


class SmartChunker:
    """
    'Smart' chunker that tries hard not to split paragraphs/ideas.

    Strategy:
    1) Split into paragraphs (blank-line separated).
    2) Merge paragraphs into chunks up to max_chars.
    3) If a paragraph is too long, split it into sentences and merge.
    4) If still too long, fall back to hard character slicing as a last resort.
    """
    def __init__(self, max_chars: int = 1200, overlap: int = 150):
        self.max_chars = int(max_chars)
        self.overlap = int(overlap)

    def chunk(self, text: str, *, source: str = "", section: str = "") -> List[Chunk]:
        raw = (text or "").strip()
        if not raw:
            return []

        paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
        pieces: List[str] = []

        for p in paras:
            if len(p) <= self.max_chars:
                pieces.append(p)
                continue

            # split long paragraph into sentences
            sents = [s.strip() for s in _SENT_SPLIT.split(p) if s.strip()]
            if len(sents) <= 1:
                pieces.extend(self._hard_split(p))
                continue

            buf = ""
            for s in sents:
                if not buf:
                    buf = s
                    continue
                if len(buf) + 1 + len(s) <= self.max_chars:
                    buf = buf + " " + s
                else:
                    pieces.append(buf)
                    buf = s
            if buf:
                pieces.append(buf)

        # merge pieces into chunks, with overlap
        chunks: List[Chunk] = []
        buf = ""
        for piece in pieces:
            if not buf:
                buf = piece
                continue
            if len(buf) + 2 + len(piece) <= self.max_chars:
                buf = buf + "\n\n" + piece
            else:
                chunks.append(Chunk(text=buf, meta={"source": source, "section": section}))
                buf = piece
        if buf:
            chunks.append(Chunk(text=buf, meta={"source": source, "section": section}))

        # apply overlap (simple tail overlap)
        if self.overlap > 0 and len(chunks) > 1:
            overlapped: List[Chunk] = []
            prev_tail = ""
            for c in chunks:
                if prev_tail:
                    c2 = (prev_tail + "\n\n" + c.text).strip()
                    overlapped.append(Chunk(text=c2, meta=c.meta))
                else:
                    overlapped.append(c)
                prev_tail = c.text[-self.overlap:] if len(c.text) > self.overlap else c.text
            chunks = overlapped

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        out = []
        t = text
        while t:
            out.append(t[: self.max_chars])
            t = t[self.max_chars - self.overlap :] if self.overlap > 0 else t[self.max_chars :]
        return [o.strip() for o in out if o.strip()]
