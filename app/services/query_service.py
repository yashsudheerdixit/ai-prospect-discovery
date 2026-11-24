import logging
import re
from typing import Any, Dict, List

from cachetools import TTLCache

from app.config import Settings
from app.schemas import LocationInfo, QueryIntent
from app.services.llm_service import LLMService, LLMServiceError
from app.utils.normalizers import normalize_whitespace

logger = logging.getLogger(__name__)

QUERY_SYSTEM_PROMPT = """You extract structured prospect-discovery parameters from a natural-language query.
Return strict JSON with keys:
{
  "interest": "core topic, industry, or niche",
  "audience": "who the target prospects are",
  "location": {
    "city": "string or null",
    "state": "state/province/emirate/region or null",
    "country": "country name or null"
  },
  "target_type": "individual | business | organization | mixed",
  "keywords": ["3 to 6 relevant domain terms or synonyms"],
  "platform_queries": {
    "Google": "concise search terms for personal/business websites or public bios",
    "LinkedIn": "concise query for LinkedIn profiles or company pages",
    "X": "concise query for X profiles",
    "YouTube": "concise query for YouTube creators/channels",
    "Instagram": "concise query for Instagram profiles",
    "Facebook": "concise query for Facebook profiles/pages",
    "Threads": "concise query for Threads profiles",
    "WhatsApp": "concise query for public WhatsApp contact links",
    "Email": "concise query for publicly listed contact emails"
  }
}
Rules:
- Infer state/region if unambiguous (e.g. Washington DC -> District of Columbia, USA; Ahmedabad -> Gujarat, India; New York -> New York, USA; London -> England, UK; Dubai -> Dubai, UAE).
- Do not include 'site:' operators in platform_queries; the platform modules add their own domain filters.
- Choose target_type='business' for owners/agencies/restaurants/companies, 'individual' for fans/lovers/enthusiasts/founders, or 'mixed' when both fit."""

CITY_STATE_COUNTRY_HINTS: Dict[str, tuple[str, str, str]] = {
    "washington dc": ("Washington DC", "District of Columbia", "USA"),
    "washington d.c.": ("Washington DC", "District of Columbia", "USA"),
    "new york": ("New York", "New York", "USA"),
    "london": ("London", "England", "UK"),
    "ahmedabad": ("Ahmedabad", "Gujarat", "India"),
    "dubai": ("Dubai", "Dubai", "UAE"),
}


class QueryService:
    """Understands natural-language queries and produces structured QueryIntent objects."""

    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self.settings = settings
        self.llm_service = llm_service
        self._cache: TTLCache[str, QueryIntent] = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=max(settings.cache_ttl_seconds, 1),
        )

    async def understand_query(self, query: str) -> QueryIntent:
        """Parse a natural-language query into a structured QueryIntent."""
        normalized_query = normalize_whitespace(query)
        cache_key = normalized_query.lower()

        if self.settings.cache_ttl_seconds > 0 and cache_key in self._cache:
            logger.debug("Query understanding cache hit for: %s", normalized_query)
            return self._cache[cache_key]

        intent: QueryIntent
        if self.llm_service.has_configured_provider():
            try:
                raw = await self.llm_service.generate_json(
                    system_prompt=QUERY_SYSTEM_PROMPT,
                    user_prompt=f"Query: {normalized_query}",
                    max_tokens=700,
                )
                intent = self._build_intent_from_llm(normalized_query, raw)
            except LLMServiceError as exc:
                logger.warning("Falling back to heuristic query parser: %s", exc)
                intent = self._heuristic_parse(normalized_query)
        else:
            intent = self._heuristic_parse(normalized_query)

        if self.settings.cache_ttl_seconds > 0:
            self._cache[cache_key] = intent
        return intent

    def _build_intent_from_llm(self, original_query: str, raw: Dict[str, Any]) -> QueryIntent:
        loc_raw = raw.get("location") or {}
        location = LocationInfo(
            city=loc_raw.get("city") or None,
            state=loc_raw.get("state") or None,
            country=loc_raw.get("country") or None,
        )
        target_type = raw.get("target_type", "individual")
        if target_type not in {"individual", "business", "organization", "mixed"}:
            target_type = "individual"

        interest = normalize_whitespace(str(raw.get("interest") or original_query))
        audience = normalize_whitespace(str(raw.get("audience") or original_query))
        keywords = [
            normalize_whitespace(str(k))
            for k in (raw.get("keywords") or [])
            if normalize_whitespace(str(k))
        ]
        if interest and interest.lower() not in {k.lower() for k in keywords}:
            keywords.insert(0, interest)

        platform_queries = {
            str(k): normalize_whitespace(str(v))
            for k, v in (raw.get("platform_queries") or {}).items()
            if normalize_whitespace(str(v))
        }

        intent = QueryIntent(
            original_query=original_query,
            interest=interest,
            audience=audience,
            location=location,
            target_type=target_type,
            keywords=keywords[:8],
            platform_queries=platform_queries,
        )
        if not intent.platform_queries:
            intent.platform_queries = self._default_platform_queries(intent)
        return intent

    def _heuristic_parse(self, query: str) -> QueryIntent:
        """Deterministic fallback parser when LLM credentials are unavailable."""
        # Split on "in <location>" or "from <location>" or "near <location>"
        match = re.match(
            r"^(?P<audience>.+?)\s+(?:in|from|near|around)\s+(?P<loc>.+)$",
            query,
            flags=re.IGNORECASE,
        )
        if match:
            audience_part = normalize_whitespace(match.group("audience"))
            loc_part = normalize_whitespace(match.group("loc"))
        else:
            audience_part = query
            loc_part = ""

        location = self._parse_location_text(loc_part)

        interest = re.sub(
            r"\b(lovers|fans|enthusiasts|owners|founders|creators|experts|professionals)\b",
            "",
            audience_part,
            flags=re.IGNORECASE,
        )
        interest = normalize_whitespace(interest) or audience_part

        low_aud = audience_part.lower()
        if any(w in low_aud for w in ("owner", "owners", "agency", "agencies", "restaurant", "shop", "company", "business")):
            target_type = "business"
        elif any(w in low_aud for w in ("club", "association", "community", "organization", "group")):
            target_type = "organization"
        else:
            target_type = "individual"

        keywords: List[str] = [interest]
        if audience_part.lower() != interest.lower():
            keywords.append(audience_part)

        intent = QueryIntent(
            original_query=query,
            interest=interest,
            audience=audience_part,
            location=location,
            target_type=target_type,
            keywords=keywords,
        )
        intent.platform_queries = self._default_platform_queries(intent)
        return intent

    @staticmethod
    def _parse_location_text(loc_text: str) -> LocationInfo:
        if not loc_text:
            return LocationInfo()

        low = loc_text.lower()
        for hint_key, (city, state, country) in CITY_STATE_COUNTRY_HINTS.items():
            if hint_key in low:
                return LocationInfo(city=city, state=state, country=country)

        # Handle patterns like "Washington DC in USA" or "New York, USA"
        cleaned = re.sub(r"\bin\b", ",", loc_text, flags=re.IGNORECASE)
        parts = [normalize_whitespace(p) for p in cleaned.split(",") if normalize_whitespace(p)]
        if len(parts) == 1:
            tokens = parts[0].split()
            if len(tokens) >= 2 and tokens[-1].upper() in {"USA", "UK", "UAE", "US", "INDIA"}:
                return LocationInfo(city=" ".join(tokens[:-1]), country=tokens[-1])
            return LocationInfo(city=parts[0])
        if len(parts) == 2:
            return LocationInfo(city=parts[0], country=parts[1])
        return LocationInfo(city=parts[0], state=parts[1], country=parts[2])

    @staticmethod
    def _default_platform_queries(intent: QueryIntent) -> Dict[str, str]:
        loc = " ".join(
            p for p in (intent.location.city, intent.location.country) if p
        )
        base = f"{intent.audience} {loc}".strip()
        interest_loc = f"{intent.interest} {loc}".strip()
        return {
            "Google": base,
            "LinkedIn": base,
            "X": interest_loc,
            "YouTube": interest_loc,
            "Instagram": interest_loc,
            "Facebook": base,
            "Threads": interest_loc,
            "WhatsApp": f"{interest_loc} whatsapp",
            "Email": f"{base} email contact",
        }
