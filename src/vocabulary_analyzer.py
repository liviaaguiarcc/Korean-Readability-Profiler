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
        """Extract and count normalized vocabulary items."""
        vocabulary_entries: list[tuple[str, str]] = []

        index = 0

        while index < len(tokens):
            token = tokens[index]

            combined_hada = self._combine_hada_construction(
                tokens=tokens,
                index=index,
            )

            if combined_hada is not None:
                vocabulary_entries.append(combined_hada)
                index += 2
                continue

            base_tag = self._get_base_tag(token.tag)

            if base_tag in self.CONTENT_TAGS:
                lemma = self._normalize_lemma(token)
                part_of_speech = self.CONTENT_TAGS[base_tag]

                vocabulary_entries.append(
                    (lemma, part_of_speech)
                )

            index += 1

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

    def _combine_hada_construction(
        self,
        tokens: list[KoreanToken],
        index: int,
    ) -> tuple[str, str] | None:
        """Combine noun and 하 suffixes into one dictionary form."""
        if index + 1 >= len(tokens):
            return None

        current_token = tokens[index]
        next_token = tokens[index + 1]

        current_base_tag = self._get_base_tag(current_token.tag)
        next_base_tag = self._get_base_tag(next_token.tag)

        is_noun = current_base_tag in {"NNG", "NNP"}

        is_hada_suffix = (
            next_token.form == "하"
            and next_base_tag in {"XSV", "XSA"}
        )

        if not is_noun or not is_hada_suffix:
            return None

        lemma = f"{current_token.form}하다"

        if next_base_tag == "XSA":
            part_of_speech = "adjective"
        else:
            part_of_speech = "verb"

        return lemma, part_of_speech

    def _normalize_lemma(self, token: KoreanToken) -> str:
        """Convert verbs and adjectives to dictionary form."""
        base_tag = self._get_base_tag(token.tag)

        if base_tag in {"VV", "VA"}:
            return f"{token.form}다"

        return token.form

    @staticmethod
    def _get_base_tag(tag: str) -> str:
        """Remove additional information from a Kiwi tag.

        Examples:
            VA-I becomes VA.
            VV-R becomes VV.
        """
        return tag.split("-")[0]