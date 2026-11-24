from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent, RawSearchResult
from app.utils.normalizers import normalize_whatsapp_link
from app.utils.validators import is_valid_public_url


class WhatsAppPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers publicly shared WhatsApp contact links via authorized search."""

    platform_name = "WhatsApp"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "WhatsApp")
        return f'("wa.me/" OR "api.whatsapp.com/send") {terms}'

    def normalize_candidate_url(self, url: str) -> str:
        wa_link = normalize_whatsapp_link(url)
        if wa_link:
            return wa_link
        return super().normalize_candidate_url(url)

    def is_valid_result(self, canonical_url: str, item: RawSearchResult) -> bool:
        if normalize_whatsapp_link(canonical_url):
            return True
        # Also allow public landing pages whose snippet explicitly contains a wa.me link
        snippet_text = f"{item.title} {item.snippet}".lower()
        return is_valid_public_url(canonical_url) and (
            "wa.me/" in snippet_text or "api.whatsapp.com" in snippet_text
        )
