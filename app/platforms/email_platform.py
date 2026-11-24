import re

from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent, RawSearchResult
from app.utils.validators import is_valid_public_url, validate_and_normalize_email

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class EmailPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public pages listing valid contact email addresses via authorized search."""

    platform_name = "Email"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "Email")
        return f'("@gmail.com" OR "email" OR "contact") {terms}'

    def is_valid_result(self, canonical_url: str, item: RawSearchResult) -> bool:
        if not is_valid_public_url(canonical_url):
            return False
        text = f"{item.title} {item.snippet}"
        matches = EMAIL_REGEX.findall(text)
        return any(validate_and_normalize_email(m) is not None for m in matches)
