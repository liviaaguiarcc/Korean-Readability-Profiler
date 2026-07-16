"""Vocabulary extraction and frequency analysis for Korean texts."""

from collections import Counter
from dataclasses import dataclass

from src.morphological_analyzer import KoreanToken


@dataclass(frozen=True)
class VocabularyItem:
    """A vocabulary item extracted from a Korean text."""

    lemma: str
    part_of_speech: str
    frequency: int


class VocabularyAnalyzer:
    """Extract useful vocabulary from morphological-analysis results."""

    CONTENT_TAGS = {
        "NNG": "noun",
        "NNP": "proper noun",
        "VV": "verb",
        "VA": "adjective",
        "MAG": "adverb",
    }

    def extract_vocabulary(
        self,
        tokens: list[KoreanToken],
    ) -> list[VocabularyItem]:
        """Extract and count content words.

        Args:
            tokens: Tokens produced by KoreanMorphologicalAnalyzer.

        Returns:
            A list of vocabulary items ordered by frequency.
        """
        vocabulary_entries: list[tuple[str, str]] = []

        for token in tokens:
            if token.tag not in self.CONTENT_TAGS:
                continue

            lemma = self._normalize_lemma(token)
            part_of_speech = self.CONTENT_TAGS[token.tag]

            vocabulary_entries.append((lemma, part_of_speech))

        counts = Counter(vocabulary_entries)

        items = [
            VocabularyItem(
                lemma=lemma,
                part_of_speech=part_of_speech,
                frequency=frequency,
            )
            for (lemma, part_of_speech), frequency in counts.items()
        ]

        return sorted(
            items,
            key=lambda item: (-item.frequency, item.lemma),
        )

    @staticmethod
    def _normalize_lemma(token: KoreanToken) -> str:
        """Convert verbs and adjectives to dictionary form."""
        if token.tag in {"VV", "VA"}:
            return f"{token.form}다"

        return token.form