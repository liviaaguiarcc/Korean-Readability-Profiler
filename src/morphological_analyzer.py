"""Korean morphological analysis using Kiwi."""

from dataclasses import dataclass

from kiwipiepy import Kiwi


@dataclass(frozen=True)
class KoreanToken:
    """A single token returned by the Korean morphological analyzer."""

    form: str
    tag: str
    start: int
    length: int


class KoreanMorphologicalAnalyzer:
    """Analyze Korean text and return normalized token information."""

    def __init__(self) -> None:
        self.kiwi = Kiwi()

    def analyze(self, text: str) -> list[KoreanToken]:
        """Analyze a Korean text.

        Args:
            text: Korean text to analyze.

        Returns:
            A list of KoreanToken objects.

        Raises:
            ValueError: If the supplied text is empty.
        """
        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("The text cannot be empty.")

        tokens = self.kiwi.tokenize(cleaned_text)

        return [
            KoreanToken(
                form=token.form,
                tag=token.tag,
                start=token.start,
                length=token.len,
            )
            for token in tokens
        ]