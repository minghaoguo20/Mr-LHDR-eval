"""XOR-with-SHA256-keystream decipher for the dataset's `question` / `answer`
/ `checklist` fields.

Not real security: the password is the `canary` string stored in every row,
so anyone with the file can decrypt it. It exists only to keep the plaintext
off naive scraping, search-engine indexing, and training-corpus grep
matches -- the same convention as BrowseComp / HLE / MM-BrowseComp. See
`scripts/export_to_hf/crypto.py` in the main repo for the writer side (kept
as a separate copy since the two repos share no dependency).
"""
from __future__ import annotations

import base64
import hashlib


def _derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return (digest * (length // len(digest) + 1))[:length]


def decrypt(ciphertext_b64: str, password: str) -> str:
    data = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(data))
    return bytes(a ^ b for a, b in zip(data, key)).decode()


def decrypt_row(row: dict) -> dict:
    """Decrypt `question` / `answer` / `checklist` using the row's own
    `canary` field. Rows without a `canary` are assumed already plaintext
    (older exports, or hand-written test fixtures) and returned unchanged.
    """
    password = row.get("canary")
    if not password:
        return row

    out = dict(row)
    if isinstance(out.get("question"), str):
        out["question"] = decrypt(out["question"], password)
    if isinstance(out.get("answer"), list):
        out["answer"] = [decrypt(line, password) if isinstance(line, str) else line
                         for line in out["answer"]]
    if isinstance(out.get("checklist"), list):
        out["checklist"] = [decrypt(line, password) if isinstance(line, str) else line
                            for line in out["checklist"]]
    return out
