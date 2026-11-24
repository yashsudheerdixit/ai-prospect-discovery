import logging
from typing import Dict, List

from app.config import Settings
from app.schemas import CandidateProfile, QueryIntent
from app.services.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = """You evaluate whether discovered public profiles are authentic, relevant prospects for a target search intent.
For each candidate, return an entry in JSON:
{
  "evaluations": [
    {
      "index": 0,
      "is_relevant": true,
      "score": 0.85,
      "clean_name": "Short real person, creator, club, or business name (NEVER a tweet, sentence, or headline)",
      "location": "Formatted location if evidenced, else null",
      "reason": "Short evidence-based reason"
    }
  ]
}
Strict Rejection Rules:
- Reject generic listicles, team rosters, schedules, articles, job postings, directories, or aggregator pages.
- If a specific city is requested (e.g. Washington DC, New York, London, Ahmedabad, Dubai), reject profiles clearly based in a DIFFERENT city or unrelated country (e.g. Boston, Nashville, Glasgow, Washington State) unless explicitly connected to the target city.
- Require real supporting evidence in the snippet/URL connecting the prospect to the target audience/interest.
- Do NOT invent names, URLs, or locations not supported by the provided snippet/URL."""

NON_PROSPECT_MARKERS = (
    "top 10",
    "top 20",
    "best restaurants",
    "job openings",
    "hiring ",
    "wikipedia",
    "reddit",
    "football roster",
    "official roster",
    "game schedule",
    "head of team business operations",
)


class RelevanceService:
    """Evaluates and filters candidate prospects using deterministic evidence + batched LLM scoring."""

    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self.settings = settings
        self.llm_service = llm_service

    async def evaluate_and_filter(
        self,
        candidates: List[CandidateProfile],
        intent: QueryIntent,
        min_score: float = 0.45,
    ) -> List[CandidateProfile]:
        """Score candidates for interest, location, and authenticity; return verified matches."""
        if not candidates:
            return []

        # Stage 1: Deterministic scoring & fast rejection of obvious non-prospects
        pre_filtered: List[CandidateProfile] = []
        for cand in candidates:
            score, reason = self._deterministic_score(cand, intent)
            cand.relevance_score = score
            cand.relevance_reason = reason
            if not self._is_obvious_noise(cand) and score >= 0.30:
                pre_filtered.append(cand)

        if not pre_filtered:
            return []

        # Stage 2: Cost-efficient single batched LLM classification over top candidates
        batch_slice = pre_filtered[: min(len(pre_filtered), 25)]
        if self.llm_service.has_configured_provider():
            try:
                await self._llm_batch_evaluate(batch_slice, intent)
            except LLMServiceError as exc:
                logger.warning("LLM relevance batch evaluation failed; using deterministic scores: %s", exc)
                for cand in batch_slice:
                    cand.is_relevant = cand.relevance_score >= min_score
        else:
            for cand in batch_slice:
                cand.is_relevant = cand.relevance_score >= min_score

        relevant = [c for c in batch_slice if c.is_relevant and c.relevance_score >= min_score]
        relevant.sort(key=lambda c: c.relevance_score, reverse=True)
        return relevant[: self.settings.max_prospects_returned]

    def _deterministic_score(self, cand: CandidateProfile, intent: QueryIntent) -> tuple[float, str]:
        text = f"{cand.name} {cand.bio} {' '.join(cand.source_urls)}".lower()
        score = 0.0
        reasons: List[str] = []

        # Interest & keyword signals
        keywords = [k.lower() for k in intent.keywords if k]
        interest_hits = sum(1 for kw in keywords if kw in text)
        if interest_hits > 0:
            score += min(0.45, 0.25 + 0.10 * interest_hits)
            reasons.append(f"interest_hits={interest_hits}")

        # City / State / Country signals
        city = (intent.location.city or "").lower()
        state = (intent.location.state or "").lower()
        country = (intent.location.country or "").lower()

        city_hit = bool(
            city
            and (
                city in text
                or (cand.location and city in cand.location.lower())
                or ("washington" in city and ("washington dc" in text or "washington, dc" in text or " d.c." in text or " dc " in text))
            )
        )
        state_hit = bool(state and (state in text or (cand.location and state in cand.location.lower())))
        country_hit = bool(country and (country in text or (cand.location and country in cand.location.lower())))

        if city:
            if city_hit or state_hit:
                score += 0.35
                reasons.append("city_match")
            elif country_hit:
                # Country-only match when a specific city was requested is weak
                score += 0.05
                reasons.append("country_only")
        elif country:
            if country_hit:
                score += 0.30
                reasons.append("country_match")
        else:
            score += 0.20

        # Multi-platform / contact richness bonus
        if len(cand.social_links) >= 2:
            score += 0.10
            reasons.append("multi_link")
        if cand.emails or cand.phones or cand.whatsapp_links:
            score += 0.10
            reasons.append("public_contact")

        return round(min(score, 1.0), 3), ", ".join(reasons) or "low_evidence"

    @staticmethod
    def _is_obvious_noise(cand: CandidateProfile) -> bool:
        combined = f"{cand.name} {cand.bio}".lower()
        return any(marker in combined for marker in NON_PROSPECT_MARKERS)

    async def _llm_batch_evaluate(
        self,
        candidates: List[CandidateProfile],
        intent: QueryIntent,
    ) -> None:
        compact_items = [
            {
                "index": idx,
                "name": cand.name,
                "location": cand.location,
                "platforms": [s.platform for s in cand.social_links],
                "urls": [s.link for s in cand.social_links[:4]],
                "evidence": cand.bio[:280],
            }
            for idx, cand in enumerate(candidates)
        ]

        user_prompt = (
            f"Target Audience: {intent.audience}\n"
            f"Interest: {intent.interest}\n"
            f"Location: {intent.location.formatted() or 'Any'}\n"
            f"Target Type: {intent.target_type}\n"
            f"Candidates: {compact_items}"
        )

        raw = await self.llm_service.generate_json(
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1200,
        )

        eval_map: Dict[int, dict] = {
            int(item["index"]): item
            for item in (raw.get("evaluations") or [])
            if isinstance(item, dict) and "index" in item
        }

        for idx, cand in enumerate(candidates):
            ev = eval_map.get(idx)
            if not ev:
                cand.is_relevant = cand.relevance_score >= 0.45
                continue
            llm_score = float(ev.get("score", cand.relevance_score))
            cand.relevance_score = round((0.4 * cand.relevance_score) + (0.6 * llm_score), 3)
            cand.is_relevant = bool(ev.get("is_relevant", False))
            if ev.get("clean_name"):
                cand.name = str(ev["clean_name"]).strip()
            if ev.get("location"):
                cand.location = str(ev["location"]).strip()
            if ev.get("reason"):
                cand.relevance_reason = str(ev["reason"]).strip()
