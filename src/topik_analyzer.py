"""TOPIK vocabulary lookup and coverage analysis."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.vocabulary_analyzer import VocabularyItem


@dataclass(frozen=True)
class TopikVocabularyResult:
    """TOPIK classification result for one vocabulary item."""

    lemma: str
    part_of_speech: str
    frequency: int
    is_topik_i: bool


@dataclass(frozen=True)
class TopikCoverageReport:
    """Summary of TOPIK I vocabulary coverage."""

    total_unique_items: int
    topik_unique_items: int
    non_topik_unique_items: int
    unique_coverage_percentage: float
    total_tokens: int
    topik_tokens: int
    non_topik_tokens: int
    token_coverage_percentage: float
    topik_words: tuple[str, ...]
    non_topik_words: tuple[str, ...]


class TopikVocabularyAnalyzer:
    """Compare extracted vocabulary against a TOPIK I word list."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.topik_words = self._load_topik_words()

    def _load_topik_words(self) -> set[str]:
        """Load Korean vocabulary from the CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"TOPIK vocabulary file not found: {self.csv_path}"
            )

        dataframe = pd.read_csv(self.csv_path)

        if "korean" not in dataframe.columns:
            raise ValueError(
                "The TOPIK CSV file must contain a 'korean' column."
            )

        return {
            str(word).strip()
            for word in dataframe["korean"].dropna()
            if str(word).strip()
        }

    def classify(
        self,
        vocabulary: list[VocabularyItem],
    ) -> list[TopikVocabularyResult]:
        """Classify vocabulary as TOPIK I or not found."""
        return [
            TopikVocabularyResult(
                lemma=item.lemma,
                part_of_speech=item.part_of_speech,
                frequency=item.frequency,
                is_topik_i=item.lemma in self.topik_words,
            )
            for item in vocabulary
        ]

    def create_coverage_report(
        self,
        results: list[TopikVocabularyResult],
    ) -> TopikCoverageReport:
        """Calculate TOPIK I coverage statistics."""
        total_unique_items = len(results)

        topik_results = [
            result
            for result in results
            if result.is_topik_i
        ]

        non_topik_results = [
            result
            for result in results
            if not result.is_topik_i
        ]

        topik_unique_items = len(topik_results)
        non_topik_unique_items = len(non_topik_results)

        total_tokens = sum(
            result.frequency
            for result in results
        )

        topik_tokens = sum(
            result.frequency
            for result in topik_results
        )

        non_topik_tokens = sum(
            result.frequency
            for result in non_topik_results
        )

        unique_coverage_percentage = self._calculate_percentage(
            topik_unique_items,
            total_unique_items,
        )

        token_coverage_percentage = self._calculate_percentage(
            topik_tokens,
            total_tokens,
        )

        topik_words = tuple(
            sorted(result.lemma for result in topik_results)
        )

        non_topik_words = tuple(
            sorted(result.lemma for result in non_topik_results)
        )

        return TopikCoverageReport(
            total_unique_items=total_unique_items,
            topik_unique_items=topik_unique_items,
            non_topik_unique_items=non_topik_unique_items,
            unique_coverage_percentage=unique_coverage_percentage,
            total_tokens=total_tokens,
            topik_tokens=topik_tokens,
            non_topik_tokens=non_topik_tokens,
            token_coverage_percentage=token_coverage_percentage,
            topik_words=topik_words,
            non_topik_words=non_topik_words,
        )

    @staticmethod
    def _calculate_percentage(
        part: int,
        total: int,
    ) -> float:
        """Return a percentage without dividing by zero."""
        if total == 0:
            return 0.0

        return part / total * 100