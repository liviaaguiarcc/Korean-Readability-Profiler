"""TOPIK vocabulary lookup for Korean texts."""

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