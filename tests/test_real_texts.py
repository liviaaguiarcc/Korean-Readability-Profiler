"""Regression tests based on real Korean reading texts."""

import pytest

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

def test_korean_reading_with_culture_text_02() -> None:
    """The second real text should preserve verified normalization."""
    text_path = (
        PROJECT_ROOT
        / "examples"
        / "korean_reading_with_culture_1_text_02.txt"
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

    lemmas = {
        item.lemma: item
        for item in vocabulary
    }

    topik_results = {
        result.lemma: result
        for result in topik_analyzer.classify(vocabulary)
    }

    sejong_results_list = sejong_analyzer.classify(vocabulary)

    sejong_results = {
        result.lemma: result
        for result in sejong_results_list
    }

    cumulative_reports = {
        report.sejong_level: report
        for report in sejong_analyzer.create_cumulative_level_reports(
            sejong_results_list
        )
    }

    # Verified normalization.
    assert "선생님" in lemmas
    assert "선생" not in lemmas

    assert "계시다" in lemmas
    assert "계" not in lemmas

    # Verified TOPIK classification.
    assert topik_results["선생님"].is_topik_i is True
    assert topik_results["계시다"].is_topik_i is True
    assert topik_results["안녕히"].is_unlisted is True

    # Verified Sejong classification.
    assert sejong_results["가다"].sejong_level == "1A"
    assert sejong_results["인사하다"].sejong_level == "1A"
    assert sejong_results["선생님"].sejong_level == "1A"

    assert sejong_results["계시다"].is_listed is False
    assert sejong_results["안녕히"].is_listed is False

    # Verified cumulative coverage.
    level_1a = cumulative_reports["1A"]
    level_1b = cumulative_reports["1B"]

    assert level_1a.total_unique_items == 9
    assert level_1a.covered_unique_items == 7
    assert level_1a.unique_coverage_percentage == pytest.approx(
        77.777,
        rel=0.01,
    )
    assert level_1a.token_coverage_percentage == 80.0

    assert level_1b.covered_unique_items == 7
    assert level_1b.token_coverage_percentage == 80.0