from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent


class LinkedInPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public LinkedIn individual or company profiles via authorized search."""

    platform_name = "LinkedIn"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "LinkedIn")
        if intent.target_type == "business":
            return f"(site:linkedin.com/company/ OR site:linkedin.com/in/) {terms}"
        return f"site:linkedin.com/in/ {terms}"
