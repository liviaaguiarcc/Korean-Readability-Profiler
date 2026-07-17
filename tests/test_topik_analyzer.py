"""Tests for TOPIK vocabulary lookup."""

from pathlib import Path

import pandas as pd
import pytest

from src.topik_analyzer import TopikVocabularyAnalyzer
from src.vocabulary_analyzer import VocabularyItem


@pytest.fixture
def sample_topik_csv(tmp_path: Path) -> Path:
    """Create a temporary TOPIK vocabulary CSV for testing."""
    csv_path = tmp_path / "topik_test.csv"

    dataframe = pd.DataFrame(
        {
            "number": [1, 2, 3],
            "korean": ["학생", "책", "읽다"],
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    return csv_path


def test_loads_topik_vocabulary(
    sample_topik_csv: Path,
) -> None:
    """It should load Korean words from the CSV."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    assert "학생" in analyzer.topik_words
    assert "책" in analyzer.topik_words
    assert "읽다" in analyzer.topik_words


def test_classifies_topik_i_word(
    sample_topik_csv: Path,
) -> None:
    """It should classify a listed word as TOPIK I."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem(
            lemma="학생",
            part_of_speech="noun",
            frequency=1,
        )
    ]

    results = analyzer.classify(vocabulary)

    assert len(results) == 1
    assert results[0].lemma == "학생"
    assert results[0].is_topik_i is True


def test_classifies_unlisted_word(
    sample_topik_csv: Path,
) -> None:
    """It should classify an unlisted word as not TOPIK I."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem(
            lemma="언어학",
            part_of_speech="noun",
            frequency=1,
        )
    ]

    results = analyzer.classify(vocabulary)

    assert len(results) == 1
    assert results[0].lemma == "언어학"
    assert results[0].is_topik_i is False


def test_preserves_frequency_and_part_of_speech(
    sample_topik_csv: Path,
) -> None:
    """It should preserve vocabulary metadata."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem(
            lemma="책",
            part_of_speech="noun",
            frequency=3,
        )
    ]

    result = analyzer.classify(vocabulary)[0]

    assert result.part_of_speech == "noun"
    assert result.frequency == 3


def test_raises_error_when_csv_does_not_exist(
    tmp_path: Path,
) -> None:
    """It should raise an error for a missing CSV file."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        TopikVocabularyAnalyzer(missing_path)


def test_raises_error_without_korean_column(
    tmp_path: Path,
) -> None:
    """It should reject CSV files without a korean column."""
    csv_path = tmp_path / "invalid.csv"

    dataframe = pd.DataFrame(
        {
            "number": [1],
            "word": ["학생"],
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError):
        TopikVocabularyAnalyzer(csv_path)

def test_calculates_unique_vocabulary_coverage(
    sample_topik_csv: Path,
) -> None:
    """It should calculate coverage using unique vocabulary items."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("학생", "noun", 2),
        VocabularyItem("책", "noun", 1),
        VocabularyItem("언어학", "noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_unique_items == 3
    assert report.topik_unique_items == 2
    assert report.non_topik_unique_items == 1
    assert report.unique_coverage_percentage == pytest.approx(
        66.666,
        rel=0.01,
    )


def test_calculates_token_coverage(
    sample_topik_csv: Path,
) -> None:
    """It should calculate coverage using word occurrences."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("학생", "noun", 3),
        VocabularyItem("책", "noun", 1),
        VocabularyItem("언어학", "noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_tokens == 5
    assert report.topik_tokens == 4
    assert report.non_topik_tokens == 1
    assert report.token_coverage_percentage == 80.0


def test_reports_topik_and_non_topik_words(
    sample_topik_csv: Path,
) -> None:
    """It should separate found and unlisted vocabulary."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("학생", "noun", 1),
        VocabularyItem("읽다", "verb", 1),
        VocabularyItem("언어학", "noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.topik_words == ("읽다", "학생")
    assert report.non_topik_words == ("언어학",)


def test_handles_empty_vocabulary_report(
    sample_topik_csv: Path,
) -> None:
    """It should safely handle an empty vocabulary list."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    report = analyzer.create_coverage_report([])

    assert report.total_unique_items == 0
    assert report.total_tokens == 0
    assert report.unique_coverage_percentage == 0.0
    assert report.token_coverage_percentage == 0.0

def test_ignores_proper_nouns_in_unique_coverage(
    sample_topik_csv: Path,
) -> None:
    """Proper nouns should not affect unique vocabulary coverage."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("학생", "noun", 1),
        VocabularyItem("언어학", "noun", 1),
        VocabularyItem("민지", "proper noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_unique_items == 2
    assert report.topik_unique_items == 1
    assert report.non_topik_unique_items == 1
    assert report.unique_coverage_percentage == 50.0

    assert report.ignored_proper_noun_items == 1
    assert report.proper_nouns == ("민지",)


def test_ignores_proper_nouns_in_token_coverage(
    sample_topik_csv: Path,
) -> None:
    """Repeated proper nouns should not affect token coverage."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("학생", "noun", 2),
        VocabularyItem("언어학", "noun", 1),
        VocabularyItem("민지", "proper noun", 5),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_tokens == 3
    assert report.topik_tokens == 2
    assert report.non_topik_tokens == 1

    assert report.token_coverage_percentage == pytest.approx(
        66.666,
        rel=0.01,
    )

    assert report.ignored_proper_noun_tokens == 5


def test_reports_multiple_proper_nouns_alphabetically(
    sample_topik_csv: Path,
) -> None:
    """It should report ignored proper nouns alphabetically."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("서울", "proper noun", 1),
        VocabularyItem("민지", "proper noun", 1),
        VocabularyItem("학생", "noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.proper_nouns == ("민지", "서울")


def test_handles_report_containing_only_proper_nouns(
    sample_topik_csv: Path,
) -> None:
    """It should safely handle a text containing only proper nouns."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("민지", "proper noun", 2),
        VocabularyItem("서울", "proper noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_unique_items == 0
    assert report.total_tokens == 0
    assert report.unique_coverage_percentage == 0.0
    assert report.token_coverage_percentage == 0.0

    assert report.ignored_proper_noun_items == 2
    assert report.ignored_proper_noun_tokens == 3

def test_evaluates_listed_proper_nouns(
    sample_topik_csv: Path,
) -> None:
    """A listed proper noun should remain in coverage calculations."""
    analyzer = TopikVocabularyAnalyzer(sample_topik_csv)

    vocabulary = [
        VocabularyItem("학생", "proper noun", 1),
        VocabularyItem("민지", "proper noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_unique_items == 1
    assert report.topik_unique_items == 1
    assert report.topik_words == ("학생",)
    assert report.proper_nouns == ("민지",)