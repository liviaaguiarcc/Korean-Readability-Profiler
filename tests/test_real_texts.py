"""Regression tests based on real Korean reading texts."""

from pathlib import Path

from src.morphological_analyzer import KoreanMorphologicalAnalyzer
from src.sejong_analyzer import SejongVocabularyAnalyzer
from src.topik_analyzer import TopikVocabularyAnalyzer
from src.vocabulary_aliases import VocabularyAliasResolver
from src.vocabulary_analyzer import VocabularyAnalyzer


PROJECT_ROOT = Path(__file__).parent.parent


def test_korean_reading_with_culture_text_01() -> None:
    """The first real text should keep its verified classifications."""
    text_path = (
        PROJECT_ROOT
        / "examples"
        / "korean_reading_with_culture_1_text_01.txt"
    )

    alias_resolver = VocabularyAliasResolver(
        PROJECT_ROOT / "data" / "vocabulary_aliases.csv"
    )

    morphological_analyzer = KoreanMorphologicalAnalyzer()
    vocabulary_analyzer = VocabularyAnalyzer()

    topik_analyzer = TopikVocabularyAnalyzer(
        PROJECT_ROOT / "data" / "topik_i_number_korean.csv",
        alias_resolver=alias_resolver,
    )

    sejong_analyzer = SejongVocabularyAnalyzer(
        PROJECT_ROOT / "data" / "sejong_vocabulary.csv",
        alias_resolver=alias_resolver,
    )

    text = text_path.read_text(encoding="utf-8")
    tokens = morphological_analyzer.analyze(text)
    vocabulary = vocabulary_analyzer.extract_vocabulary(tokens)

    topik_results = {
        result.lemma: result
        for result in topik_analyzer.classify(vocabulary)
    }

    sejong_results = {
        result.lemma: result
        for result in sejong_analyzer.classify(vocabulary)
    }

    assert topik_results["안녕하다"].is_topik_i is True
    assert sejong_results["안녕하다"].is_listed is True
    assert sejong_results["안녕하다"].sejong_level == "1A"

    assert topik_results["기자"].is_topik_i is True
    assert sejong_results["기자"].is_listed is False

    assert topik_results["김민지"].is_possible_proper_noun is True