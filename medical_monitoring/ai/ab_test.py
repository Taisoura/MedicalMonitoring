"""
A/B Testing Framework for Qwen vs GPT comparison.

Routes the same input to both providers, records results and agreement,
and maintains a human annotation queue for ground truth establishment.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import AIConfig, LLMEngine, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""
    task_type: str
    input_text: str
    qwen_result: str = ""
    gpt_result: str = ""
    qwen_latency_ms: float = 0.0
    gpt_latency_ms: float = 0.0
    qwen_tokens: int = 0
    gpt_tokens: int = 0
    agreement: bool = False
    agreement_score: float = 0.0
    needs_human_review: bool = False
    human_label: str = ""
    winner: str = ""


class ABTestLog:
    """SQLite-backed A/B test result logger with human annotation queue."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS ab_results (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        task_type     TEXT NOT NULL,
        input_hash    TEXT NOT NULL,
        input_text    TEXT,
        qwen_result   TEXT,
        gpt_result    TEXT,
        qwen_latency  REAL,
        gpt_latency   REAL,
        qwen_tokens   INTEGER,
        gpt_tokens    INTEGER,
        agreement     INTEGER,
        agreement_score REAL,
        human_label   TEXT DEFAULT '',
        winner        TEXT DEFAULT '',
        created       REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ab_task ON ab_results(task_type);
    CREATE INDEX IF NOT EXISTS idx_ab_review ON ab_results(agreement, human_label);
    """

    def __init__(self, db_path: str | Path = ".ab_test.db"):
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(self._DDL)
        self._conn.commit()

    def log(self, result: ABTestResult, input_hash: str) -> None:
        """Record an A/B test result."""
        if self._conn is None:
            return
        self._conn.execute(
            """INSERT INTO ab_results
            (task_type, input_hash, input_text, qwen_result, gpt_result,
             qwen_latency, gpt_latency, qwen_tokens, gpt_tokens,
             agreement, agreement_score, human_label, winner, created)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.task_type, input_hash,
                result.input_text[:500],
                result.qwen_result[:2000],
                result.gpt_result[:2000],
                result.qwen_latency_ms, result.gpt_latency_ms,
                result.qwen_tokens, result.gpt_tokens,
                1 if result.agreement else 0,
                result.agreement_score,
                result.human_label, result.winner,
                time.time(),
            ),
        )
        self._conn.commit()

    def get_review_queue(self, limit: int = 50) -> list[dict]:
        """Get disagreement cases needing human review."""
        if self._conn is None:
            return []
        cursor = self._conn.execute(
            """SELECT id, task_type, input_text, qwen_result, gpt_result, agreement_score
            FROM ab_results
            WHERE agreement = 0 AND human_label = ''
            ORDER BY created DESC LIMIT ?""",
            (limit,),
        )
        return [
            {"id": r[0], "task_type": r[1], "input": r[2],
             "qwen": r[3], "gpt": r[4], "score": r[5]}
            for r in cursor.fetchall()
        ]

    def set_human_label(self, record_id: int, label: str, winner: str = "") -> None:
        """Record human annotation for a test case."""
        if self._conn is None:
            return
        self._conn.execute(
            "UPDATE ab_results SET human_label = ?, winner = ? WHERE id = ?",
            (label, winner, record_id),
        )
        self._conn.commit()

    def get_stats(self, task_type: str | None = None) -> dict[str, Any]:
        """Compute A/B test statistics."""
        if self._conn is None:
            return {}

        where = "WHERE task_type = ?" if task_type else ""
        params: tuple = (task_type,) if task_type else ()

        row = self._conn.execute(
            f"""SELECT
                COUNT(*) as total,
                SUM(agreement) as agreed,
                AVG(agreement_score) as avg_score,
                AVG(qwen_latency) as avg_qwen_lat,
                AVG(gpt_latency) as avg_gpt_lat,
                SUM(qwen_tokens) as total_qwen_tok,
                SUM(gpt_tokens) as total_gpt_tok,
                SUM(CASE WHEN winner = 'qwen' THEN 1 ELSE 0 END) as qwen_wins,
                SUM(CASE WHEN winner = 'gpt' THEN 1 ELSE 0 END) as gpt_wins,
                SUM(CASE WHEN human_label != '' THEN 1 ELSE 0 END) as annotated
            FROM ab_results {where}""",
            params,
        ).fetchone()

        total = row[0] or 0
        return {
            "total_tests": total,
            "agreement_rate": (row[1] or 0) / total if total > 0 else 0,
            "avg_agreement_score": row[2] or 0,
            "avg_qwen_latency_ms": row[3] or 0,
            "avg_gpt_latency_ms": row[4] or 0,
            "total_qwen_tokens": row[5] or 0,
            "total_gpt_tokens": row[6] or 0,
            "qwen_wins": row[7] or 0,
            "gpt_wins": row[8] or 0,
            "annotated": row[9] or 0,
            "pending_review": total - (row[1] or 0) - (row[9] or 0),
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


class ABTestRouter:
    """
    Dual-route A/B test engine.

    Sends the same prompt to both Qwen and GPT, compares results,
    and logs disagreements for human review.
    """

    def __init__(self, config: AIConfig, log_db: str | Path = ".ab_test.db"):
        self.config = config
        self._qwen_engine = LLMEngine(config)
        self._gpt_engine = LLMEngine(config)
        self._log = ABTestLog(log_db)

    def dual_complete(
        self,
        system: str,
        user: str,
        task_type: str = "general",
    ) -> tuple[LLMResponse, LLMResponse, ABTestResult]:
        """
        Send the same prompt to both providers and compare results.

        Returns:
            (qwen_response, gpt_response, comparison_result)
        """
        import hashlib
        input_hash = hashlib.sha256(f"{task_type}|{system}|{user}".encode()).hexdigest()[:16]

        qwen_resp = self._safe_complete(system, user, task_type, "qwen")
        gpt_resp = self._safe_complete(system, user, task_type, "gpt")

        agreement_score = self._compute_agreement(qwen_resp, gpt_resp, task_type)
        agreement = agreement_score >= 0.8

        ab_result = ABTestResult(
            task_type=task_type,
            input_text=user[:500],
            qwen_result=qwen_resp.content[:2000] if qwen_resp else "",
            gpt_result=gpt_resp.content[:2000] if gpt_resp else "",
            qwen_latency_ms=qwen_resp.latency_ms if qwen_resp else 0,
            gpt_latency_ms=gpt_resp.latency_ms if gpt_resp else 0,
            qwen_tokens=qwen_resp.usage.get("output_tokens", 0) if qwen_resp else 0,
            gpt_tokens=gpt_resp.usage.get("output_tokens", 0) if gpt_resp else 0,
            agreement=agreement,
            agreement_score=agreement_score,
            needs_human_review=not agreement,
        )

        self._log.log(ab_result, input_hash)
        return qwen_resp, gpt_resp, ab_result

    def get_preferred_response(
        self,
        system: str,
        user: str,
        task_type: str = "general",
    ) -> LLMResponse:
        """
        In A/B mode, call both and return the preferred result.

        Preference logic:
        1. If both agree, return the faster one.
        2. If they disagree, prefer Qwen for Chinese-language tasks,
           GPT for English-language tasks.
        3. Fall back to default provider.
        """
        qwen_resp, gpt_resp, ab = self.dual_complete(system, user, task_type)

        if ab.agreement:
            return qwen_resp if qwen_resp.latency_ms <= gpt_resp.latency_ms else gpt_resp

        is_chinese = any("\u4e00" <= c <= "\u9fff" for c in user[:100])
        if is_chinese:
            return qwen_resp if qwen_resp.content else gpt_resp
        return gpt_resp if gpt_resp.content else qwen_resp

    def _safe_complete(
        self,
        system: str,
        user: str,
        task_type: str,
        provider: str,
    ) -> LLMResponse:
        """Complete with error handling, returning empty response on failure."""
        try:
            engine = self._qwen_engine if provider == "qwen" else self._gpt_engine
            return engine.complete(system, user, task_type, provider=provider)
        except Exception as e:
            logger.warning("A/B test %s call failed: %s", provider, e)
            return LLMResponse(content="", model="", provider=provider)

    @staticmethod
    def _compute_agreement(
        resp_a: LLMResponse,
        resp_b: LLMResponse,
        task_type: str,
    ) -> float:
        """
        Compute agreement score between two responses.

        For structured JSON outputs, compares key fields.
        For text outputs, uses simple overlap heuristic.
        """
        if not resp_a.content or not resp_b.content:
            return 0.0

        json_a = resp_a.parse_json()
        json_b = resp_b.parse_json()

        if json_a is not None and json_b is not None:
            return _json_agreement(json_a, json_b, task_type)

        words_a = set(resp_a.content.lower().split())
        words_b = set(resp_b.content.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    @property
    def log(self) -> ABTestLog:
        return self._log


def _json_agreement(a: Any, b: Any, task_type: str) -> float:
    """Compare two JSON structures for semantic agreement."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.5
        if not a:
            return 1.0
        scores = []
        for item_a, item_b in zip(a, b):
            scores.append(_json_agreement(item_a, item_b, task_type))
        return sum(scores) / len(scores)

    if isinstance(a, dict) and isinstance(b, dict):
        key_fields = _get_key_fields(task_type)
        if key_fields:
            scores = []
            for kf in key_fields:
                va = a.get(kf)
                vb = b.get(kf)
                if va is None and vb is None:
                    scores.append(1.0)
                elif va is None or vb is None:
                    scores.append(0.0)
                elif isinstance(va, bool) and isinstance(vb, bool):
                    scores.append(1.0 if va == vb else 0.0)
                elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    scores.append(1.0 if abs(va - vb) < 0.15 else 0.0)
                elif str(va).lower() == str(vb).lower():
                    scores.append(1.0)
                else:
                    scores.append(0.3)
            return sum(scores) / len(scores) if scores else 0.5

        all_keys = set(a.keys()) | set(b.keys())
        if not all_keys:
            return 1.0
        match = sum(1 for k in all_keys if str(a.get(k, "")).lower() == str(b.get(k, "")).lower())
        return match / len(all_keys)

    return 1.0 if str(a).lower() == str(b).lower() else 0.0


def _get_key_fields(task_type: str) -> list[str]:
    """Return the critical comparison fields for each task type."""
    task_keys = {
        "normalize_ae": ["pt_en", "confidence"],
        "match_cm_ae": ["treats_ae", "confidence"],
        "match_cm_ae_batch": ["treats_ae", "confidence"],
        "match_lb_ae": ["matches", "confidence"],
        "match_lb_ae_batch": ["matches", "confidence"],
        "match_ctcae_ae": ["matches", "confidence"],
        "causality_assess": ["suggested_category", "confidence"],
        "sae_screen": ["should_be_sae", "confidence"],
        "severity_assess": ["suggested_grade", "discrepancy"],
        "semantic_match": ["matches", "confidence"],
        "field_annotate": [],
    }
    return task_keys.get(task_type, [])
