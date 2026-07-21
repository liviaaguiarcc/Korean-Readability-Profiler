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

            # Kiwi may analyze 계세요 incorrectly as:
            # 계/NNG + 이/VCP + 세요/EF
            combined_gyesida = self._combine_gyesida_construction(
                tokens=tokens,
                index=index,
            )

            if combined_gyesida is not None:
                lemma, part_of_speech, consumed_tokens = (
                    combined_gyesida
                )

                vocabulary_entries.append(
                    (lemma, part_of_speech)
                )

                index += consumed_tokens
                continue

            # Combine words such as 선생 + 님.
            combined_nim = self._combine_nim_construction(
                tokens=tokens,
                index=index,
            )

            if combined_nim is not None:
                vocabulary_entries.append(combined_nim)
                index += 2
                continue

            # Combine 공부 + 하, 운동 + 하, etc.
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

    def _combine_gyesida_construction(
        self,
        tokens: list[KoreanToken],
        index: int,
    ) -> tuple[str, str, int] | None:
        """Normalize Kiwi segmentations of 계세요 to 계시다."""
        if index + 2 < len(tokens):
            current_token = tokens[index]
            next_token = tokens[index + 1]
            final_token = tokens[index + 2]

            standard_pattern = (
                current_token.form == "계"
                and self._get_base_tag(current_token.tag) == "NNG"
                and next_token.form == "이"
                and self._get_base_tag(next_token.tag) == "VCP"
                and final_token.form == "세요"
                and self._get_base_tag(final_token.tag) == "EF"
            )

            if standard_pattern:
                return "계시다", "verb", 3

        # In some contexts, Kiwi may produce 계세/NNG + 요/JX.
        if index + 1 < len(tokens):
            current_token = tokens[index]
            next_token = tokens[index + 1]

            compact_pattern = (
                current_token.form == "계세"
                and self._get_base_tag(current_token.tag) == "NNG"
                and next_token.form == "요"
                and self._get_base_tag(next_token.tag) == "JX"
            )

            if compact_pattern:
                return "계시다", "verb", 2

        return None

    def _combine_nim_construction(
        self,
        tokens: list[KoreanToken],
        index: int,
    ) -> tuple[str, str] | None:
        """Combine a noun and the honorific suffix 님."""
        if index + 1 >= len(tokens):
            return None

        current_token = tokens[index]
        next_token = tokens[index + 1]

        current_base_tag = self._get_base_tag(
            current_token.tag
        )

        next_base_tag = self._get_base_tag(
            next_token.tag
        )

        is_noun = current_base_tag in {"NNG", "NNP"}

        is_nim_suffix = (
            next_token.form == "님"
            and next_base_tag == "XSN"
        )

        if not is_noun or not is_nim_suffix:
            return None

        lemma = f"{current_token.form}님"

        if current_base_tag == "NNP":
            part_of_speech = "proper noun"
        else:
            part_of_speech = "noun"

        return lemma, part_of_speech

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

        current_base_tag = self._get_base_tag(
            current_token.tag
        )

        next_base_tag = self._get_base_tag(
            next_token.tag
        )

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

    def _normalize_lemma(
        self,
        token: KoreanToken,
    ) -> str:
        """Convert verbs and adjectives to dictionary form."""
        base_tag = self._get_base_tag(token.tag)

        if base_tag in {"VV", "VA"}:
            return f"{token.form}다"

        return token.form

    @staticmethod
    def _get_base_tag(tag: str) -> str:
        """Remove additional information from a Kiwi tag."""
        return tag.split("-")[0]