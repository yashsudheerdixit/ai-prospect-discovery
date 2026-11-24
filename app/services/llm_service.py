import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Raised when all configured LLM providers fail to produce valid JSON."""


class LLMService:
    """Async multi-provider LLM client (Gemini, OpenAI, Anthropic) with fallback & retries."""

    def __init__(self, settings: Settings, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.settings = settings
        self._external_client = http_client

    def has_configured_provider(self) -> bool:
        """Return True if at least one LLM provider API key is configured."""
        return bool(
            self.settings.gemini_api_key
            or self.settings.openai_api_key
            or self.settings.anthropic_api_key
        )

    def _resolve_provider_chain(self) -> list[Tuple[str, str]]:
        """Build ordered list of (provider, model) pairs based on settings and available keys."""
        key_map = {
            "gemini": self.settings.gemini_api_key,
            "openai": self.settings.openai_api_key,
            "anthropic": self.settings.anthropic_api_key,
        }
        default_models = {
            "gemini": "gemini-2.5-flash-lite",
            "openai": "gpt-4.1-nano",
            "anthropic": "claude-haiku-5-5",
        }

        chain: list[Tuple[str, str]] = []
        primary = self.settings.llm_provider

        if primary == "auto":
            for prov in ("gemini", "openai", "anthropic"):
                if key_map.get(prov):
                    chain.append((prov, default_models[prov]))
            return chain

        if key_map.get(primary):
            chain.append((primary, self.settings.llm_model))

        fallback = self.settings.llm_fallback_provider
        if fallback and fallback != primary and key_map.get(fallback):
            chain.append((fallback, self.settings.llm_fallback_model))

        for prov, key in key_map.items():
            if key and not any(p == prov for p, _ in chain):
                chain.append((prov, default_models[prov]))

        return chain

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Call configured LLM provider(s) and return parsed JSON dict."""
        chain = self._resolve_provider_chain()
        if not chain:
            raise LLMServiceError("No LLM API keys configured.")

        errors: list[str] = []
        for provider, model in chain:
            try:
                raw_text = await self._call_with_retry(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                )
                return self._parse_json_response(raw_text)
            except Exception as exc:
                logger.warning("LLM provider %s (%s) failed: %s", provider, model, exc)
                errors.append(f"{provider}/{model}: {exc}")

        raise LLMServiceError("; ".join(errors))

    async def _call_with_retry(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        max_attempts: int = 2,
    ) -> str:
        """Retry transient HTTP errors (429 / 5xx / timeout) with short exponential backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._dispatch(provider, model, system_prompt, user_prompt, max_tokens)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                if status is not None and status not in {429, 500, 502, 503, 504}:
                    raise
                if attempt < max_attempts:
                    await asyncio.sleep(0.8 * attempt)
            except Exception as exc:
                last_exc = exc
                break
        assert last_exc is not None
        raise last_exc

    async def _dispatch(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        if self._external_client is not None:
            return await self._send_request(
                self._external_client, provider, model, system_prompt, user_prompt, max_tokens
            )
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            return await self._send_request(
                client, provider, model, system_prompt, user_prompt, max_tokens
            )

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        if provider == "gemini":
            return await self._call_gemini(client, model, system_prompt, user_prompt, max_tokens)
        if provider == "openai":
            return await self._call_openai(client, model, system_prompt, user_prompt, max_tokens)
        if provider == "anthropic":
            return await self._call_anthropic(client, model, system_prompt, user_prompt, max_tokens)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    async def _call_gemini(
        self,
        client: httpx.AsyncClient,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.settings.llm_temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        headers = {"x-goog-api-key": self.settings.gemini_api_key or ""}
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMServiceError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text:
            raise LLMServiceError("Gemini returned empty content.")
        return text

    async def _call_openai(
        self,
        client: httpx.AsyncClient,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMServiceError("OpenAI returned no choices.")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise LLMServiceError("OpenAI returned empty content.")
        return content

    async def _call_anthropic(
        self,
        client: httpx.AsyncClient,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.settings.anthropic_api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": self.settings.llm_temperature,
            "system": f"{system_prompt}\nRespond strictly with valid JSON and no markdown fences.",
            "messages": [{"role": "user", "content": user_prompt}],
        }
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text:
            raise LLMServiceError("Anthropic returned empty content.")
        return text

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise LLMServiceError("LLM response did not contain valid JSON.")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise LLMServiceError("LLM JSON response must be an object.")
        return parsed
