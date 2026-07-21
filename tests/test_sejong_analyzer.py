"""Tests for King Sejong vocabulary analysis."""

from pathlib import Path

import pandas as pd
import pytest

from src.sejong_analyzer import SejongVocabularyAnalyzer
from src.vocabulary_analyzer import VocabularyItem


@pytest.fixture
def sample_sejong_csv(tmp_path: Path) -> Path:
    """Create a temporary Sejong vocabulary database."""
    csv_path = tmp_path / "sejong_test.csv"

    dataframe = pd.DataFrame(
        {
            "korean": ["학생", "책", "읽다"],
            "sejong_level": ["1A", "1A", "1A"],
            "unit": [1, 3, 4],
            "page": [7, 11, 13],
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    return csv_path


def test_loads_sejong_entries(
    sample_sejong_csv: Path,
) -> None:
    """It should load Sejong vocabulary metadata."""
    analyzer = SejongVocabularyAnalyzer(sample_sejong_csv)

    assert analyzer.entries["학생"] == ("1A", 1, 7)
    assert analyzer.entries["읽다"] == ("1A", 4, 13)


def test_classifies_listed_word(
    sample_sejong_csv: Path,
) -> None:
    """It should return the level, unit and page."""
    analyzer = SejongVocabularyAnalyzer(sample_sejong_csv)

    vocabulary = [
        VocabularyItem("읽다", "verb", 2),
    ]

    result = analyzer.classify(vocabulary)[0]

    assert result.is_listed is True
    assert result.sejong_level == "1A"
    assert result.unit == 4
    assert result.page == 13
    assert result.frequency == 2


def test_classifies_unlisted_word(
    sample_sejong_csv: Path,
) -> None:
    """It should identify words absent from the database."""
    analyzer = SejongVocabularyAnalyzer(sample_sejong_csv)

    vocabulary = [
        VocabularyItem("언어학", "noun", 1),
    ]

    result = analyzer.classify(vocabulary)[0]

    assert result.is_listed is False
    assert result.sejong_level is None
    assert result.unit is None
    assert result.page is None


def test_calculates_sejong_coverage(
    sample_sejong_csv: Path,
) -> None:
    """It should calculate unique and token coverage."""
    analyzer = SejongVocabularyAnalyzer(sample_sejong_csv)

    vocabulary = [
        VocabularyItem("학생", "noun", 2),
        VocabularyItem("책", "noun", 1),
        VocabularyItem("언어학", "noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_unique_items == 3
    assert report.listed_unique_items == 2
    assert report.unlisted_unique_items == 1

    assert report.unique_coverage_percentage == pytest.approx(
        66.666,
        rel=0.01,
    )

    assert report.total_tokens == 4
    assert report.listed_tokens == 3
    assert report.unlisted_tokens == 1
    assert report.token_coverage_percentage == 75.0


def test_ignores_unlisted_proper_nouns(
    sample_sejong_csv: Path,
) -> None:
    """It should ignore possible proper nouns in coverage."""
    analyzer = SejongVocabularyAnalyzer(sample_sejong_csv)

    vocabulary = [
        VocabularyItem("학생", "noun", 1),
        VocabularyItem("민지", "proper noun", 3),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_unique_items == 1
    assert report.listed_unique_items == 1
    assert report.total_tokens == 1

    assert report.ignored_proper_noun_items == 1
    assert report.ignored_proper_noun_tokens == 3
    assert report.proper_nouns == ("민지",)


def test_evaluates_listed_proper_nouns(
    sample_sejong_csv: Path,
) -> None:
    """A listed proper noun should remain in coverage."""
    analyzer = SejongVocabularyAnalyzer(sample_sejong_csv)

    vocabulary = [
        VocabularyItem("학생", "proper noun", 1),
        VocabularyItem("민지", "proper noun", 1),
    ]

    results = analyzer.classify(vocabulary)
    report = analyzer.create_coverage_report(results)

    assert report.total_unique_items == 1
    assert report.listed_unique_items == 1
    assert report.listed_words == ("학생",)
    assert report.proper_nouns == ("민지",)


def test_raises_error_for_missing_csv(
    tmp_path: Path,
) -> None:
    """It should reject a nonexistent database file."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        SejongVocabularyAnalyzer(missing_path)


def test_raises_error_for_missing_columns(
    tmp_path: Path,
) -> None:
    """It should reject an incomplete Sejong CSV."""
    csv_path = tmp_path / "invalid.csv"

    dataframe = pd.DataFrame(
        {
            "korean": ["학생"],
            "sejong_level": ["1A"],
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError):
        SejongVocabularyAnalyzer(csv_path)