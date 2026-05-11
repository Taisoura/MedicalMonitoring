"""
De-identification utilities for protecting subject identity before LLM calls.

Subject IDs are stripped/replaced before text is sent to cloud APIs.
A reversible mapping is maintained in-memory so results can be re-identified
after the LLM response returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DeidentifyConfig:
    enabled: bool = True
    pattern: str = r"S\d{5}"
    replacement: str = "[SUBJ]"


class Deidentifier:
    """
    Bidirectional de-identification for subject IDs.

    Usage:
        deid = Deidentifier(DeidentifyConfig())
        safe_text, mapping = deid.deidentify("Patient S00123 had AE")
        original = deid.reidentify(safe_text, mapping)
    """

    def __init__(self, config: DeidentifyConfig | None = None):
        self.config = config or DeidentifyConfig()
        self._pattern = re.compile(self.config.pattern)

    def deidentify(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace subject IDs with placeholders.

        Returns:
            (sanitized_text, {placeholder: original_id})
        """
        if not self.config.enabled or not text:
            return text, {}

        mapping: dict[str, str] = {}
        seen: dict[str, str] = {}
        counter = 0

        def _replace(match: re.Match) -> str:
            nonlocal counter
            original = match.group()
            if original in seen:
                return seen[original]
            placeholder = f"{self.config.replacement.rstrip(']')}_{counter}]"
            seen[original] = placeholder
            mapping[placeholder] = original
            counter += 1
            return placeholder

        sanitized = self._pattern.sub(_replace, text)
        return sanitized, mapping

    def reidentify(self, text: str, mapping: dict[str, str]) -> str:
        """Restore original subject IDs from placeholders."""
        if not mapping:
            return text
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result

    def deidentify_batch(self, texts: list[str]) -> tuple[list[str], list[dict[str, str]]]:
        """De-identify a batch of texts, returning parallel mappings."""
        results = []
        mappings = []
        for t in texts:
            safe, m = self.deidentify(t)
            results.append(safe)
            mappings.append(m)
        return results, mappings
