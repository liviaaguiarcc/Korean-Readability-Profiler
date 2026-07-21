"""Tests for Korean vocabulary extraction and normalization."""

from src.morphological_analyzer import KoreanMorphologicalAnalyzer
from src.vocabulary_analyzer import VocabularyAnalyzer


def analyze_vocabulary(text: str) -> dict[str, int]:
    """Analyze text and return lemmas mapped to their frequencies."""
    morphological_analyzer = KoreanMorphologicalAnalyzer()
    vocabulary_analyzer = VocabularyAnalyzer()

    tokens = morphological_analyzer.analyze(text)
    vocabulary = vocabulary_analyzer.extract_vocabulary(tokens)

    return {
        item.lemma: item.frequency
        for item in vocabulary
    }


def test_extracts_basic_nouns() -> None:
    """It should extract common Korean nouns."""
    vocabulary = analyze_vocabulary(
        "학생은 한국어 책을 읽습니다."
    )

    assert "학생" in vocabulary
    assert "한국어" in vocabulary
    assert "책" in vocabulary


def test_normalizes_verbs_to_dictionary_form() -> None:
    """It should convert conjugated verbs to dictionary form."""
    vocabulary = analyze_vocabulary(
        "학생이 책을 읽습니다."
    )

    assert "읽다" in vocabulary


def test_normalizes_adjectives_to_dictionary_form() -> None:
    """It should convert adjectives to dictionary form."""
    vocabulary = analyze_vocabulary(
        "이 책은 어렵습니다."
    )

    assert "어렵다" in vocabulary


def test_combines_hada_verbs() -> None:
    """It should combine noun and 하 suffixes into 하다 verbs."""
    vocabulary = analyze_vocabulary(
        "학생들은 한국어를 공부합니다."
    )

    assert "공부하다" in vocabulary
    assert "공부" not in vocabulary


def test_counts_repeated_vocabulary() -> None:
    """It should count repeated normalized vocabulary items."""
    vocabulary = analyze_vocabulary(
        "학생이 공부합니다. 학생은 매일 공부합니다."
    )

    assert vocabulary["학생"] == 2
    assert vocabulary["공부하다"] == 2

def test_combines_honorific_nim_suffix() -> None:
    """It should preserve nouns containing the honorific 님 suffix."""
    vocabulary = analyze_vocabulary(
        "선생님이 학교에 있습니다."
    )

    assert "선생님" in vocabulary
    assert "선생" not in vocabulary


def test_normalizes_gyeseyo_to_gyesida() -> None:
    """It should normalize 계세요 to the dictionary form 계시다."""
    vocabulary = analyze_vocabulary(
        "선생님, 안녕히 계세요."
    )

    assert "계시다" in vocabulary
    assert "계" not in vocabulary