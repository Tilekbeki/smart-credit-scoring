"""Portable digest of CSV content across Git checkout line-ending policies."""

import hashlib
from pathlib import Path


def dataset_sha256(path: Path) -> str:
    # Git may check out the same CSV with LF or CRLF. The model provenance
    # should identify the rows, not the host's line-ending convention.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
