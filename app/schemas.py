import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


TargetType = Literal["individual", "business", "organization", "mixed"]


class ProspectDiscoveryRequest(BaseModel):
    """Incoming request body for POST /prospect-discovery."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Natural-language prospect discovery query.",
        examples=["Football lovers in Washington DC in USA"],
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if len(cleaned) < 3:
            raise ValueError("Query must contain at least 3 meaningful characters.")
        if not re.search(r"[A-Za-z0-9]", cleaned):
            raise ValueError("Query must contain alphanumeric characters.")
        return cleaned


class LocationInfo(BaseModel):
    """Structured geographical parameters extracted from user query."""

    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    def formatted(self) -> Optional[str]:
        parts = [p for p in (self.city, self.state, self.country) if p]
        if not parts:
            return None
        # Deduplicate adjacent identical parts (e.g., Dubai, Dubai, UAE -> Dubai, UAE)
        deduped: List[str] = []
        for part in parts:
            if not deduped or deduped[-1].lower() != part.lower():
                deduped.append(part)
        return ", ".join(deduped)


class QueryIntent(BaseModel):
    """Structured representation of the user's search intent."""

    original_query: str
    interest: str
    audience: str
    location: LocationInfo = Field(default_factory=LocationInfo)
    target_type: TargetType = "individual"
    keywords: List[str] = Field(default_factory=list)
    platform_queries: Dict[str, str] = Field(default_factory=dict)


class SocialLink(BaseModel):
    """Verified public social or contact link belonging to a prospect."""

    platform: str
    link: str


class Prospect(BaseModel):
    """Final prospect output record matching the required API contract."""

    name: str
    location: Optional[str] = None
    social_link: List[SocialLink] = Field(default_factory=list)


class ProspectDiscoveryResponse(BaseModel):
    """Top-level response payload returned by POST /prospect-discovery."""

    success: bool = True
    query: str
    total_prospects: int
    prospects: List[Prospect] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error payload."""

    success: bool = False
    error: str
    detail: Optional[str] = None


# --- Internal Pipeline Models ---


class RawSearchResult(BaseModel):
    """Normalized search result item returned by a search backend or platform API."""

    title: str = ""
    url: str
    snippet: str = ""
    source_platform: str
    display_link: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateProfile(BaseModel):
    """Internal representation of a discovered prospect before consolidation/filtering."""

    name: str
    location: Optional[str] = None
    bio: str = ""
    social_links: List[SocialLink] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    whatsapp_links: List[str] = Field(default_factory=list)
    handles: Dict[str, str] = Field(default_factory=dict)
    source_urls: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    entity_type: str = "unknown"
    relevance_score: float = 0.0
    relevance_reason: str = ""
    is_relevant: bool = False
