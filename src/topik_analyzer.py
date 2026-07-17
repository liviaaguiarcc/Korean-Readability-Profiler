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
    category: str

    @property
    def is_topik_i(self) -> bool:
        """Return whether the item belongs to the TOPIK I list."""
        return self.category == "topik_i"

    @property
    def is_possible_proper_noun(self) -> bool:
        """Return whether the item is an unlisted proper noun."""
        return self.category == "possible_proper_noun"

    @property
    def is_unlisted(self) -> bool:
        """Return whether the item was not found in the database."""
        return self.category == "unlisted"


@dataclass(frozen=True)
class TopikCoverageReport:
    """Summary of TOPIK I vocabulary coverage."""

    total_unique_items: int
    topik_unique_items: int
    unlisted_unique_items: int
    unique_coverage_percentage: float

    total_tokens: int
    topik_tokens: int
    unlisted_tokens: int
    token_coverage_percentage: float

    ignored_proper_noun_items: int
    ignored_proper_noun_tokens: int

    topik_words: tuple[str, ...]
    unlisted_words: tuple[str, ...]
    proper_nouns: tuple[str, ...]


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
        """Classify each vocabulary item."""
        results: list[TopikVocabularyResult] = []

        for item in vocabulary:
            category = self._classify_item(item)

            results.append(
                TopikVocabularyResult(
                    lemma=item.lemma,
                    part_of_speech=item.part_of_speech,
                    frequency=item.frequency,
                    category=category,
                )
            )

        return results

    def _classify_item(
        self,
        item: VocabularyItem,
    ) -> str:
        """Assign a transparent category to one vocabulary item."""
        if item.lemma in self.topik_words:
            return "topik_i"

        if item.part_of_speech == "proper noun":
            return "possible_proper_noun"

        return "unlisted"

    def create_coverage_report(
        self,
        results: list[TopikVocabularyResult],
    ) -> TopikCoverageReport:
        """Calculate TOPIK I coverage while ignoring proper nouns."""
        proper_noun_results = [
            result
            for result in results
            if result.is_possible_proper_noun
        ]

        evaluated_results = [
            result
            for result in results
            if not result.is_possible_proper_noun
        ]

        topik_results = [
            result
            for result in evaluated_results
            if result.is_topik_i
        ]

        unlisted_results = [
            result
            for result in evaluated_results
            if result.is_unlisted
        ]

        total_unique_items = len(evaluated_results)
        topik_unique_items = len(topik_results)
        unlisted_unique_items = len(unlisted_results)

        total_tokens = sum(
            result.frequency
            for result in evaluated_results
        )

        topik_tokens = sum(
            result.frequency
            for result in topik_results
        )

        unlisted_tokens = sum(
            result.frequency
            for result in unlisted_results
        )

        ignored_proper_noun_items = len(proper_noun_results)

        ignored_proper_noun_tokens = sum(
            result.frequency
            for result in proper_noun_results
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

        unlisted_words = tuple(
            sorted(result.lemma for result in unlisted_results)
        )

        proper_nouns = tuple(
            sorted(result.lemma for result in proper_noun_results)
        )

        return TopikCoverageReport(
            total_unique_items=total_unique_items,
            topik_unique_items=topik_unique_items,
            unlisted_unique_items=unlisted_unique_items,
            unique_coverage_percentage=unique_coverage_percentage,
            total_tokens=total_tokens,
            topik_tokens=topik_tokens,
            unlisted_tokens=unlisted_tokens,
            token_coverage_percentage=token_coverage_percentage,
            ignored_proper_noun_items=ignored_proper_noun_items,
            ignored_proper_noun_tokens=ignored_proper_noun_tokens,
            topik_words=topik_words,
            unlisted_words=unlisted_words,
            proper_nouns=proper_nouns,
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