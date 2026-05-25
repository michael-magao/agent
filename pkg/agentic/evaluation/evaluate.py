from __future__ import annotations

from dataclasses import dataclass
import re

try:
    from ai_oncall_bot.models import DimensionScore
    from ai_oncall_bot.config import ScoringWeights
except ModuleNotFoundError:
    @dataclass
    class DimensionScore:
        relevance: float
        faithfulness: float
        completeness: float
        conciseness: float
        actionability: float

    @dataclass
    class ScoringWeights:
        relevance: float = 0.25
        faithfulness: float = 0.25
        completeness: float = 0.2
        conciseness: float = 0.1
        actionability: float = 0.2


class RuleScorer:
    def __init__(self, weights: ScoringWeights):
        self._weights = weights

    def score_relevance(self, query: str, answer: str) -> float:
        q_words = set(self._tokenize(query))
        a_words = set(self._tokenize(answer))
        if not q_words:
            return 5.0
        overlap = len(q_words & a_words) / len(q_words)
        # Penalize very short answers that just say "no info"
        raw_len = len(answer.split())
        if raw_len < 10:
            return min(10.0, round(overlap * 6, 1))
        return min(10.0, round(overlap * 8 + 2, 1))

    def score_faithfulness(self, context: str, answer: str) -> float:
        if not context.strip():
            return 5.0
        c_words = set(self._tokenize(context))
        a_words = set(self._tokenize(answer))
        if not a_words:
            return 5.0
        # Measure how much of the answer's content is grounded in context
        grounded_words = len(a_words & c_words)
        # Use sqrt-scaled ratio to avoid over-penalizing longer answers
        # that naturally introduce connecting/explanatory words
        ratio = grounded_words / (len(a_words) ** 0.7)
        return min(10.0, max(1.0, round(ratio * 5 + 1, 1)))

    def score_completeness(self, context: str, answer: str) -> float:
        c_words = set(self._tokenize(context))
        a_words = set(self._tokenize(answer))
        if not c_words:
            return 5.0
        # How many key context terms does the answer reference
        coverage = len(c_words & a_words) / len(c_words)
        # Also consider absolute answer length as a signal of thoroughness
        length_bonus = min(2.0, len(a_words) / 100)
        return min(10.0, max(1.0, round(coverage * 8 + length_bonus, 1)))

    def score_conciseness(self, answer: str) -> float:
        length = len(answer.split())
        if length == 0:
            return 5.0
        # Very short "no info" answers should not score high on conciseness
        if length < 10:
            return 4.0
        elif length < 50:
            return 6.0
        elif length < 150:
            return 8.0
        elif length < 400:
            return 7.0
        else:
            return 5.0

    def score_actionability(self, answer: str) -> float:
        indicators = [
            r"step\s*\d",
            r"\d+[\.\)]\s*\S",
            r"`[^`]+`",
            r"run\b",
            r"execute\b",
            r"ssh\b",
            r"curl\b",
            r"command",
            r"https?://",
            # Chinese actionability indicators
            r"[\u6b65\u9aa4]",   # step
            r"[\u64cd\u4f5c]",   # operation
            r"[\u6267\u884c]",   # execute
            r"[\u8fd0\u884c]",   # run
            r"[\u914d\u7f6e]",   # configure
            r"[\u68c0\u67e5]",   # check
            r"[\u4fee\u590d]",   # fix/repair
            r"[\u91cd\u542f]",   # restart
            r"[\u53c2\u8003]",   # reference
        ]
        matches = sum(1 for p in indicators if re.search(p, answer, re.IGNORECASE))
        # Penalize empty/no-info answers
        if len(answer.split()) < 10:
            return 2.0
        return min(10.0, round(matches * 1.0 + 2, 1))

    def evaluate(self, query: str, context: str, answer: str) -> DimensionScore:
        return DimensionScore(
            relevance=self.score_relevance(query, answer),
            faithfulness=self.score_faithfulness(context, answer),
            completeness=self.score_completeness(context, answer),
            conciseness=self.score_conciseness(answer),
            actionability=self.score_actionability(answer),
        )

    def composite(self, scores: DimensionScore) -> float:
        w = self._weights
        return round(
            scores.relevance * w.relevance
            + scores.faithfulness * w.faithfulness
            + scores.completeness * w.completeness
            + scores.conciseness * w.conciseness
            + scores.actionability * w.actionability,
            2,
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Support both CJK characters and latin words
        tokens = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]+|\w+", text)
        result = []
        for t in tokens:
            if re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", t):
                # Split CJK into individual characters as tokens
                result.extend(list(t))
            elif len(t) > 2:
                result.append(t.lower())
        return result

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?\n\u3002\uff01\uff1f]", text) if s.strip()]
