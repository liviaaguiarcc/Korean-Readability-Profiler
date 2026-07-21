"""King Sejong vocabulary lookup and coverage analysis."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.vocabulary_analyzer import VocabularyItem
from src.vocabulary_aliases import VocabularyAliasResolver


@dataclass(frozen=True)
class SejongVocabularyResult:
    """Sejong classification result for one vocabulary item."""

    lemma: str
    part_of_speech: str
    frequency: int
    sejong_level: str | None
    unit: int | None
    page: int | None

    @property
    def is_listed(self) -> bool:
        """Return whether the item was found in the Sejong database."""
        return self.sejong_level is not None

    @property
    def is_possible_proper_noun(self) -> bool:
        """Return whether the item is an unlisted proper noun."""
        return (
            self.part_of_speech == "proper noun"
            and not self.is_listed
        )


@dataclass(frozen=True)
class SejongCoverageReport:
    """Summary of Sejong vocabulary coverage."""

    total_unique_items: int
    listed_unique_items: int
    unlisted_unique_items: int
    unique_coverage_percentage: float

    total_tokens: int
    listed_tokens: int
    unlisted_tokens: int
    token_coverage_percentage: float

    ignored_proper_noun_items: int
    ignored_proper_noun_tokens: int

    listed_words: tuple[str, ...]
    unlisted_words: tuple[str, ...]
    proper_nouns: tuple[str, ...]


class SejongVocabularyAnalyzer:
    """Compare vocabulary against the King Sejong word database."""

    REQUIRED_COLUMNS = {
        "korean",
        "sejong_level",
        "unit",
        "page",
    }

    def __init__(
        self,
        csv_path: str | Path,
        alias_resolver: VocabularyAliasResolver | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.alias_resolver = alias_resolver
        self.entries = self._load_entries()

    def _load_entries(
        self,
    ) -> dict[str, tuple[str, int | None, int | None]]:
        """Load Sejong vocabulary and its first appearance."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Sejong vocabulary file not found: {self.csv_path}"
            )

        dataframe = pd.read_csv(self.csv_path)

        missing_columns = (
            self.REQUIRED_COLUMNS - set(dataframe.columns)
        )

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))

            raise ValueError(
                "The Sejong CSV file is missing required columns: "
                f"{missing_text}"
            )

        entries: dict[
            str,
            tuple[str, int | None, int | None],
        ] = {}

        for row in dataframe.itertuples(index=False):
            if pd.isna(row.korean) or pd.isna(row.sejong_level):
                continue

            korean = str(row.korean).strip()
            sejong_level = str(row.sejong_level).strip()

            if not korean or not sejong_level:
                continue

            unit = (
                int(row.unit)
                if pd.notna(row.unit)
                else None
            )

            page = (
                int(row.page)
                if pd.notna(row.page)
                else None
            )

            # Keep the first occurrence if a duplicate exists.
            if korean not in entries:
                entries[korean] = (
                    sejong_level,
                    unit,
                    page,
                )

        return entries

    def _find_entry(
        self,
        lemma: str,
    ) -> tuple[str, int | None, int | None] | None:
        """Find a Sejong entry using a le  mma or known variant."""
        if self.alias_resolver is None:
          candidates = (lemma,)
        else:
            candidates = self.alias_resolver.candidates(lemma)
        for candidate in candidates:
         entry = self.entries.get(candidate)

        if entry is not None:
            return entry

        return None

    def classify(
        self,
        vocabulary: list[VocabularyItem],
    ) -> list[SejongVocabularyResult]:
        """Classify extracted vocabulary using the Sejong database."""
        results: list[SejongVocabularyResult] = []

        for item in vocabulary:
            entry = self._find_entry(item.lemma)

            if entry is None:
                sejong_level = None
                unit = None
                page = None
            else:
                sejong_level, unit, page = entry

            results.append(
                SejongVocabularyResult(
                    lemma=item.lemma,
                    part_of_speech=item.part_of_speech,
                    frequency=item.frequency,
                    sejong_level=sejong_level,
                    unit=unit,
                    page=page,
                )
            )

        return results

    def create_coverage_report(
        self,
        results: list[SejongVocabularyResult],
    ) -> SejongCoverageReport:
        """Calculate Sejong coverage while ignoring proper nouns."""
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

        listed_results = [
            result
            for result in evaluated_results
            if result.is_listed
        ]

        unlisted_results = [
            result
            for result in evaluated_results
            if not result.is_listed
        ]

        total_unique_items = len(evaluated_results)
        listed_unique_items = len(listed_results)
        unlisted_unique_items = len(unlisted_results)

        total_tokens = sum(
            result.frequency
            for result in evaluated_results
        )

        listed_tokens = sum(
            result.frequency
            for result in listed_results
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
            listed_unique_items,
            total_unique_items,
        )

        token_coverage_percentage = self._calculate_percentage(
            listed_tokens,
            total_tokens,
        )

        listed_words = tuple(
            sorted(result.lemma for result in listed_results)
        )

        unlisted_words = tuple(
            sorted(result.lemma for result in unlisted_results)
        )

        proper_nouns = tuple(
            sorted(result.lemma for result in proper_noun_results)
        )

        return SejongCoverageReport(
            total_unique_items=total_unique_items,
            listed_unique_items=listed_unique_items,
            unlisted_unique_items=unlisted_unique_items,
            unique_coverage_percentage=unique_coverage_percentage,
            total_tokens=total_tokens,
            listed_tokens=listed_tokens,
            unlisted_tokens=unlisted_tokens,
            token_coverage_percentage=token_coverage_percentage,
            ignored_proper_noun_items=ignored_proper_noun_items,
            ignored_proper_noun_tokens=ignored_proper_noun_tokens,
            listed_words=listed_words,
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