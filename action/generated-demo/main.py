import hashlib
import random
import string
from typing import Dict, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, validator

app = FastAPI()

# In-memory store
url_store: Dict[str, str] = {}

# Base62 characters
BASE62_CHARS = string.ascii_letters + string.digits

class ShortenRequest(BaseModel):
    url: str

    @validator("url")
    def validate_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must start with http or https")
        return v

def generate_code(url: str) -> str:
    # Create a deterministic base for the given URL to avoid collisions
    # Use a hash of the URL + random salt to get a 6-char base62 code
    salt = "".join(random.choices(BASE62_CHARS, k=6))
    hash_input = f"{url}{salt}".encode()
    hash_digest = hashlib.sha256(hash_input).hexdigest()
    # Take first 6 bytes and convert to base62
    hash_int = int(hash_digest[:12], 16)
    code = ""
    for _ in range(6):
        code += BASE62_CHARS[hash_int % 62]
        hash_int //= 62
    return code

@app.post("/api/shorten")
async def shorten(request: ShortenRequest):
    url = request.url
    # Check if URL already has a short code (optional idempotency)
    for code, original_url in url_store.items():
        if original_url == url:
            short_url = f"{app.url_path_for('redirect', code=code)}"
            return {"code": code, "short_url": short_url}
    
    # Generate a unique short code
    while True:
        code = generate_code(url)
        if code not in url_store:
            break
    
    url_store[code] = url
    short_url = f"{app.url_path_for('redirect', code=code)}"
    return {"code": code, "short_url": short_url}

@app.get("/{code}")
async def redirect(code: str, request: Request):
    url = url_store.get(code)
    if not url:
        raise HTTPException(status_code=404, detail="Short code not found")
    return RedirectResponse(url=url, status_code=307)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
