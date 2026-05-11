"""
LLM Engine -- multi-backend abstraction for Qwen and GPT.

Supports synchronous and batch completion with retry, rate limiting,
and optional A/B routing for model comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False

    def parse_json(self) -> Any:
        """Extract JSON from the response, handling markdown fences."""
        text = self.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1
            end_idx = len(lines)
            for i in range(1, len(lines)):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            text = "\n".join(lines[start:end_idx])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return None


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    model: str
    api_key_env: str = ""
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60

    def resolve_api_key(self) -> str:
        """Resolve API key: direct config > env var > empty."""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


@dataclass
class AIConfig:
    """Full AI module configuration."""
    enabled: bool = True
    ab_test_enabled: bool = False
    default_provider: str = "qwen"
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    cache_enabled: bool = True
    cache_db_path: str = ".ai_cache.db"
    cache_ttl_days: int = 30
    max_concurrent: int = 5
    retry_attempts: int = 3
    retry_delay: float = 2.0
    deidentify_enabled: bool = True
    deidentify_pattern: str = r"S\d{5}"
    deidentify_replacement: str = "[SUBJ]"

    @classmethod
    def from_yaml(cls, path: str | Path) -> AIConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        ai = raw.get("ai", raw)

        providers = {}
        for name, pcfg in ai.get("providers", {}).items():
            providers[name] = ProviderConfig(
                name=name,
                model=pcfg.get("model", ""),
                api_key_env=pcfg.get("api_key_env", ""),
                api_key=pcfg.get("api_key", ""),
                base_url=pcfg.get("base_url"),
                max_tokens=pcfg.get("max_tokens", 2000),
                temperature=pcfg.get("temperature", 0.1),
                timeout=pcfg.get("timeout", 60),
            )

        cache = ai.get("cache", {})
        batch = ai.get("batch", {})
        deid = ai.get("deidentify", {})

        return cls(
            enabled=ai.get("enabled", True),
            ab_test_enabled=ai.get("ab_test", False),
            default_provider=ai.get("default_provider", "qwen"),
            providers=providers,
            cache_enabled=cache.get("enabled", True),
            cache_db_path=cache.get("db_path", ".ai_cache.db"),
            cache_ttl_days=cache.get("ttl_days", 30),
            max_concurrent=batch.get("max_concurrent", 5),
            retry_attempts=batch.get("retry_attempts", 3),
            retry_delay=batch.get("retry_delay", 2.0),
            deidentify_enabled=deid.get("strip_subject_id", True),
            deidentify_pattern=deid.get("id_pattern", r"S\d{5}"),
            deidentify_replacement=deid.get("replacement", "[SUBJ]"),
        )

    @classmethod
    def default(cls) -> AIConfig:
        default_path = Path(__file__).parent.parent / "config" / "ai_config.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()


class _QwenProvider:
    """DashScope / Qwen API provider (OpenAI-compatible mode)."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._api_key = config.resolve_api_key()

    def complete(self, system: str, user: str) -> LLMResponse:
        if self.config.base_url:
            return self._complete_openai_compat(system, user)
        return self._complete_dashscope_native(system, user)

    def _complete_openai_compat(self, system: str, user: str) -> LLMResponse:
        """Use OpenAI-compatible endpoint (DashScope compatible-mode)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = OpenAI(
            api_key=self._api_key or "EMPTY",
            base_url=self.config.base_url,
        )

        t0 = time.time()
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
        latency = (time.time() - t0) * 1000

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider="qwen",
            usage=usage,
            latency_ms=latency,
        )

    def _complete_dashscope_native(self, system: str, user: str) -> LLMResponse:
        """Fallback: native DashScope SDK."""
        try:
            import dashscope
            from dashscope import Generation
        except ImportError:
            raise ImportError("dashscope package required: pip install dashscope")

        if self._api_key:
            dashscope.api_key = self._api_key

        t0 = time.time()
        response = Generation.call(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            result_format="message",
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        latency = (time.time() - t0) * 1000

        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API error {response.status_code}: {response.message}"
            )

        content = response.output.choices[0].message.content
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider="qwen",
            usage=usage,
            latency_ms=latency,
        )


class _OpenAIProvider:
    """OpenAI / GPT API provider (supports custom base_url like rightcode)."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._api_key = config.resolve_api_key()

    def complete(self, system: str, user: str) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = OpenAI(
            api_key=self._api_key or "EMPTY",
            base_url=self.config.base_url,
        )

        t0 = time.time()
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
        latency = (time.time() - t0) * 1000

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider="gpt",
            usage=usage,
            latency_ms=latency,
        )


_PROVIDER_MAP = {
    "qwen": _QwenProvider,
    "gpt": _OpenAIProvider,
}


class LLMEngine:
    """
    Multi-backend LLM engine with caching and retry.

    Usage:
        engine = LLMEngine(AIConfig.default())
        resp = engine.complete("You are a medical coder.", "Normalize: 轻度贫血")
    """

    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig.default()
        self._providers: dict[str, Any] = {}
        self._cache: Any = None

        for name, pcfg in self.config.providers.items():
            provider_cls = _PROVIDER_MAP.get(name)
            if provider_cls:
                self._providers[name] = provider_cls(pcfg)

        if self.config.cache_enabled:
            from .cache import SemanticCache
            self._cache = SemanticCache(self.config.cache_db_path, self.config.cache_ttl_days)

    def complete(
        self,
        system: str,
        user: str,
        task_type: str = "general",
        provider: str | None = None,
    ) -> LLMResponse:
        """Send a completion request with caching and retry."""
        provider_name = provider or self.config.default_provider

        if self._cache:
            cache_key = self._make_cache_key(task_type, system, user, provider_name)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return LLMResponse(
                    content=cached["content"],
                    model=cached.get("model", ""),
                    provider=cached.get("provider", provider_name),
                    cached=True,
                )

        prov = self._providers.get(provider_name)
        if prov is None:
            available = list(self._providers.keys())
            if not available:
                raise RuntimeError("No LLM providers configured. Set API key env vars.")
            prov = self._providers[available[0]]
            provider_name = available[0]

        last_err = None
        for attempt in range(self.config.retry_attempts):
            try:
                resp = prov.complete(system, user)
                if self._cache:
                    self._cache.put(cache_key, {
                        "content": resp.content,
                        "model": resp.model,
                        "provider": resp.provider,
                    })
                return resp
            except Exception as e:
                last_err = e
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {self.config.retry_attempts} attempts: {last_err}")

    def batch_complete(
        self,
        items: list[dict[str, str]],
        task_type: str = "general",
        provider: str | None = None,
    ) -> list[LLMResponse]:
        """
        Process a batch of (system, user) pairs sequentially.

        Each item: {"system": ..., "user": ...}
        """
        results = []
        for item in items:
            resp = self.complete(
                system=item["system"],
                user=item["user"],
                task_type=task_type,
                provider=provider,
            )
            results.append(resp)
        return results

    @staticmethod
    def _make_cache_key(task_type: str, system: str, user: str, provider: str) -> str:
        raw = f"{task_type}|{provider}|{system}|{user}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
