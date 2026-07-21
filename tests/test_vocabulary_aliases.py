"""Tests for vocabulary aliases and variant lookup."""

from pathlib import Path

import pandas as pd

from src.sejong_analyzer import SejongVocabularyAnalyzer
from src.topik_analyzer import TopikVocabularyAnalyzer
from src.vocabulary_aliases import VocabularyAliasResolver
from src.vocabulary_analyzer import VocabularyItem


def create_alias_csv(tmp_path: Path) -> Path:
    """Create a temporary vocabulary-alias CSV."""
    csv_path = tmp_path / "aliases.csv"

    dataframe = pd.DataFrame(
        {
            "lemma": ["안녕하다"],
            "variant": ["안녕하세요"],
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    return csv_path


def test_returns_lemma_and_known_variants(
    tmp_path: Path,
) -> None:
    """It should return the original lemma and its variants."""
    alias_path = create_alias_csv(tmp_path)
    resolver = VocabularyAliasResolver(alias_path)

    candidates = resolver.candidates("안녕하다")

    assert candidates == (
        "안녕하다",
        "안녕하세요",
    )


def test_topik_analyzer_recognizes_alias(
    tmp_path: Path,
) -> None:
    """TOPIK lookup should recognize a normalized lemma by its alias."""
    alias_path = create_alias_csv(tmp_path)

    topik_path = tmp_path / "topik.csv"

    pd.DataFrame(
        {
            "number": [1],
            "korean": ["안녕하세요"],
        }
    ).to_csv(
        topik_path,
        index=False,
        encoding="utf-8-sig",
    )

    resolver = VocabularyAliasResolver(alias_path)

    analyzer = TopikVocabularyAnalyzer(
        topik_path,
        alias_resolver=resolver,
    )

    vocabulary = [
        VocabularyItem(
            lemma="안녕하다",
            part_of_speech="adjective",
            frequency=1,
        )
    ]

    result = analyzer.classify(vocabulary)[0]

    assert result.is_topik_i is True
    assert result.category == "topik_i"


def test_sejong_analyzer_recognizes_alias(
    tmp_path: Path,
) -> None:
    """Sejong lookup should return metadata through an alias."""
    alias_path = create_alias_csv(tmp_path)

    sejong_path = tmp_path / "sejong.csv"

    pd.DataFrame(
        {
            "korean": ["안녕하세요"],
            "sejong_level": ["1A"],
            "unit": [1],
            "page": [35],
        }
    ).to_csv(
        sejong_path,
        index=False,
        encoding="utf-8-sig",
    )

    resolver = VocabularyAliasResolver(alias_path)

    analyzer = SejongVocabularyAnalyzer(
        sejong_path,
        alias_resolver=resolver,
    )

    vocabulary = [
        VocabularyItem(
            lemma="안녕하다",
            part_of_speech="adjective",
            frequency=1,
        )
    ]

    result = analyzer.classify(vocabulary)[0]

    assert result.is_listed is True
    assert result.sejong_level == "1A"
    assert result.unit == 1
    assert result.page == 35