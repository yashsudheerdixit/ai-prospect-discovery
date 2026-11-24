import re
import unicodedata
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "igshid",
    "igsh",
    "ref",
    "ref_src",
    "s",
    "t",
    "trk",
    "originalSubdomain",
}

PLATFORM_SUFFIX_PATTERNS = [
    r"\s*\|\s*LinkedIn.*$",
    r"\s*-\s*LinkedIn.*$",
    r"\s*\|\s*Facebook.*$",
    r"\s*-\s*Facebook.*$",
    r"\s*on Instagram.*$",
    r"\s*•\s*Instagram photos and videos.*$",
    r"\s*\(@[^)]+\)\s*•\s*Instagram.*$",
    r"\s*-\s*YouTube.*$",
    r"\s*on X\b.*$",
    r"\s*\(@[^)]+\)\s*/\s*X.*$",
    r"\s*/\s*X\s*$",
    r"\s*\(@[^)]+\)\s*•\s*Threads.*$",
    r"\s*on Threads.*$",
    r"\s*-\s*Threads.*$",
]

LOCATION_ALIASES = {
    "dc": "Washington DC",
    "d.c.": "Washington DC",
    "washington d.c.": "Washington DC",
    "washington, dc": "Washington DC",
    "nyc": "New York",
    "new york city": "New York",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
}


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_name(name: str) -> str:
    """Normalize a person or business name for comparison while preserving readability."""
    if not name:
        return ""
    cleaned = unicodedata.normalize("NFKC", name)
    for pattern in PLATFORM_SUFFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Remove trailing handle in parentheses like "John Smith (@johnsmith)"
    cleaned = re.sub(r"\s*\(@[A-Za-z0-9._-]+\)\s*$", "", cleaned)
    # Remove emoji / decorative symbols while preserving letters, numbers, spaces, &, ', -, .
    cleaned = re.sub(r"[^\w\s&'.-]", " ", cleaned)
    return normalize_whitespace(cleaned)


def canonical_name_key(name: str) -> str:
    """Create a lowercase alphanumeric key for fuzzy identity matching."""
    cleaned = normalize_name(name).lower()
    cleaned = re.sub(r"\b(mr|mrs|ms|dr|prof|jr|sr|ii|iii|llc|inc|ltd)\b\.?", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    return normalize_whitespace(cleaned)


def normalize_url(url: str) -> str:
    """Canonicalize a public URL by stripping tracking params and normalizing scheme/path."""
    if not url:
        return ""
    raw = url.strip()
    if raw.startswith(("mailto:", "tel:")):
        return raw

    if not re.match(r"^https?://", raw, flags=re.IGNORECASE):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    scheme = "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("m.") and any(
        d in netloc for d in ("facebook.com", "youtube.com", "twitter.com", "x.com")
    ):
        netloc = "www." + netloc[2:]
    if netloc in {"twitter.com", "www.twitter.com"}:
        netloc = "x.com"
    if netloc == "www.x.com":
        netloc = "x.com"
    if netloc == "instagram.com":
        netloc = "www.instagram.com"
    if netloc == "facebook.com":
        netloc = "www.facebook.com"
    if netloc == "linkedin.com":
        netloc = "www.linkedin.com"
    if netloc == "threads.net":
        netloc = "www.threads.net"

    path = re.sub(r"/+", "/", parsed.path or "")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_params = parse_qs(parsed.query, keep_blank_values=False)
    filtered_params = {
        k: v for k, v in query_params.items() if k not in TRACKING_QUERY_PARAMS
    }
    query_str = urlencode(filtered_params, doseq=True) if filtered_params else ""

    return urlunparse((scheme, netloc, path, "", query_str, ""))


def normalize_handle(handle: str) -> str:
    """Normalize a social handle for cross-platform matching."""
    if not handle:
        return ""
    return handle.strip().lstrip("@").lower()


def normalize_location_token(token: str) -> str:
    """Normalize common city/country abbreviations for location matching."""
    low = normalize_whitespace(token).lower()
    return LOCATION_ALIASES.get(low, normalize_whitespace(token))


def normalize_whatsapp_link(raw_link_or_phone: str) -> Optional[str]:
    """Convert a WhatsApp URL or phone number into a canonical https://wa.me/<digits> link."""
    if not raw_link_or_phone:
        return None
    val = raw_link_or_phone.strip()
    if "chat.whatsapp.com/" in val.lower():
        return normalize_url(val)

    digits_match = re.search(r"(?:wa\.me/|phone=)(\+?\d[\d\s-]{6,18}\d)", val, re.IGNORECASE)
    if digits_match:
        digits = re.sub(r"\D", "", digits_match.group(1))
    else:
        digits = re.sub(r"\D", "", val)

    if 7 <= len(digits) <= 15:
        return f"https://wa.me/{digits}"
    return None
