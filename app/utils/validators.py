import re
from typing import Optional
from urllib.parse import urlparse

import phonenumbers
from email_validator import EmailNotValidError, validate_email

from app.utils.normalizers import normalize_url, normalize_whatsapp_link


DISALLOWED_EMAIL_LOCAL_PARTS = {
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "postmaster",
    "abuse",
    "privacy",
    "legal",
    "dmca",
    "security",
    "webmaster",
    "notifications",
    "notification",
}

DISALLOWED_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "test.com",
    "domain.com",
    "email.com",
    "sentry.io",
    "wixpress.com",
    "schema.org",
    "w3.org",
}

IMAGE_OR_ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".css", ".js")

GENERIC_SEARCH_OR_DIRECTORY_DOMAINS = {
    "google.com",
    "www.google.com",
    "bing.com",
    "www.bing.com",
    "duckduckgo.com",
    "search.yahoo.com",
    "yandex.com",
    "en.wikipedia.org",
    "wikipedia.org",
    "wikidata.org",
    "yelp.com",
    "www.yelp.com",
    "tripadvisor.com",
    "www.tripadvisor.com",
    "yellowpages.com",
    "www.yellowpages.com",
    "glassdoor.com",
    "www.glassdoor.com",
    "indeed.com",
    "www.indeed.com",
    "reddit.com",
    "www.reddit.com",
    "quora.com",
    "www.quora.com",
    "pinterest.com",
    "www.pinterest.com",
}


def is_valid_public_url(url: str) -> bool:
    """Validate that a URL is a well-formed HTTP(S) URL and not a generic search page."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc or "." not in parsed.netloc:
        return False

    host = parsed.netloc.lower()
    if host in {"localhost", "127.0.0.1", "example.com", "www.example.com"}:
        return False
    if host in GENERIC_SEARCH_OR_DIRECTORY_DOMAINS:
        return False

    path_low = (parsed.path or "").lower()
    if any(
        seg in path_low
        for seg in ("/search", "/hashtag/", "/explore/", "/login", "/signup", "/policies/")
    ):
        return False

    return True


def is_valid_platform_profile_url(platform: str, url: str) -> bool:
    """Validate that a URL represents an actual profile/page on the given platform."""
    canonical = normalize_url(url)
    if not is_valid_public_url(canonical):
        return False

    parsed = urlparse(canonical)
    host = parsed.netloc.lower()
    path = (parsed.path or "").strip("/")
    segments = [s for s in path.split("/") if s]

    if platform == "LinkedIn":
        if "linkedin.com" not in host:
            return False
        return len(segments) >= 2 and segments[0] in {"in", "company", "pub"}

    if platform == "X":
        if host not in {"x.com", "twitter.com"}:
            return False
        if not segments:
            return False
        reserved = {"home", "explore", "search", "i", "intent", "hashtag", "settings", "compose"}
        if segments[0].lower() in reserved:
            return False
        # Exclude tweet status links (/user/status/12345) unless we extract the user handle separately
        return len(segments) == 1

    if platform == "YouTube":
        if "youtube.com" not in host and host != "youtu.be":
            return False
        if not segments:
            return False
        if segments[0].startswith("@"):
            return True
        return len(segments) >= 2 and segments[0] in {"channel", "c", "user"}

    if platform == "Instagram":
        if "instagram.com" not in host:
            return False
        if not segments:
            return False
        reserved = {"p", "reel", "reels", "stories", "explore", "accounts", "directory", "tv"}
        return segments[0].lower() not in reserved and len(segments) == 1

    if platform == "Facebook":
        if "facebook.com" not in host:
            return False
        if not segments:
            return False
        reserved = {
            "watch",
            "marketplace",
            "groups",
            "events",
            "gaming",
            "share",
            "sharer.php",
            "login",
            "help",
            "policies",
            "photo",
            "reel",
        }
        if segments[0].lower() in reserved:
            return False
        if segments[0].lower() == "profile.php":
            return "id=" in (parsed.query or "")
        return len(segments) in {1, 2} and segments[0].lower() != "pages" or (
            segments[0].lower() == "pages" and len(segments) >= 2
        )

    if platform == "Threads":
        if "threads.net" not in host and "threads.com" not in host:
            return False
        return len(segments) == 1 and segments[0].startswith("@")

    if platform == "WhatsApp":
        return normalize_whatsapp_link(canonical) is not None

    if platform in {"Google", "Website"}:
        # Reject social platforms if they leaked into Website/Google discovery
        social_hosts = (
            "linkedin.com",
            "instagram.com",
            "facebook.com",
            "x.com",
            "twitter.com",
            "youtube.com",
            "threads.net",
            "threads.com",
            "wa.me",
        )
        return not any(sh in host for sh in social_hosts)

    return True


def validate_and_normalize_email(email: str) -> Optional[str]:
    """Validate an email address and filter out system/noreply/asset false positives."""
    if not email:
        return None
    candidate = email.strip().lower().removeprefix("mailto:")
    if any(candidate.endswith(ext) for ext in IMAGE_OR_ASSET_EXTENSIONS):
        return None

    try:
        info = validate_email(candidate, check_deliverability=False)
    except EmailNotValidError:
        return None

    normalized = info.normalized.lower()
    local, _, domain = normalized.partition("@")
    if local in DISALLOWED_EMAIL_LOCAL_PARTS:
        return None
    if domain in DISALLOWED_EMAIL_DOMAINS:
        return None
    if "." not in domain:
        return None
    return normalized


def validate_and_normalize_phone(phone: str, default_region: str = "US") -> Optional[str]:
    """Validate a phone number using libphonenumber and return E.164 format."""
    if not phone:
        return None
    raw = phone.strip().removeprefix("tel:")
    # Quick sanity check to ignore years, timestamps, or short IDs
    digits_only = re.sub(r"\D", "", raw)
    if len(digits_only) < 7 or len(digits_only) > 15:
        return None

    try:
        parsed = phonenumbers.parse(raw, None if raw.startswith("+") else default_region)
        if not phonenumbers.is_possible_number(parsed):
            return None
        if not phonenumbers.is_valid_number(parsed):
            return None
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return None


def infer_platform_from_url(url: str) -> Optional[str]:
    """Map a URL to its canonical assessment platform name."""
    if not url:
        return None
    low = url.lower().strip()
    if low.startswith("mailto:"):
        return "Email"
    if low.startswith("tel:"):
        return "Phone"

    host = urlparse(normalize_url(low)).netloc.lower()
    if "linkedin.com" in host:
        return "LinkedIn"
    if host in {"x.com", "twitter.com"}:
        return "X"
    if "youtube.com" in host or host == "youtu.be":
        return "YouTube"
    if "instagram.com" in host:
        return "Instagram"
    if "facebook.com" in host:
        return "Facebook"
    if "threads.net" in host or "threads.com" in host:
        return "Threads"
    if "wa.me" in host or "whatsapp.com" in host:
        return "WhatsApp"
    if "maps.google." in host or host == "g.page":
        return "Google"
    return "Google"
