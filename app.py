"""Command-line entry point for the Korean Readability Profiler."""

from pathlib import Path

from src.morphological_analyzer import KoreanMorphologicalAnalyzer
from src.topik_analyzer import (
    TopikCoverageReport,
    TopikVocabularyAnalyzer,
)
from src.vocabulary_analyzer import VocabularyAnalyzer


TOPIK_CSV_PATH = Path("data/topik_i_number_korean.csv")


def print_vocabulary_table(
    topik_results: list,
) -> None:
    """Display vocabulary classification results."""
    print("\nVOCABULARY AND TOPIK I ANALYSIS")
    print("-" * 75)
    print(
        f"{'LEMMA':<25}"
        f"{'PART OF SPEECH':<20}"
        f"{'FREQUENCY':<12}"
        f"{'CATEGORY':<18}"
    )
    print("-" * 75)

    for result in topik_results:
        category_labels = {
            "topik_i": "TOPIK I",
            "possible_proper_noun": "Proper Noun",
            "unlisted": "Unlisted",
        }

        topik_label = category_labels.get(result.category, "Unknown")

        print(
            f"{result.lemma:<25}"
            f"{result.part_of_speech:<20}"
            f"{result.frequency:<12}"
            f"{topik_label:<18}"
        )

    print("-" * 75)


def print_coverage_report(
    report: TopikCoverageReport,
) -> None:
    """Display TOPIK I coverage statistics."""
    print("\nTOPIK I COVERAGE REPORT")
    print("-" * 50)

    print(
        "Unique vocabulary coverage: "
        f"{report.unique_coverage_percentage:.1f}%"
    )
    print(
        "Evaluated unique words: "
        f"{report.total_unique_items}"
    )
    print(
        "TOPIK I unique words: "
        f"{report.topik_unique_items}"
    )
    print(
        "Non-TOPIK I unique words: "
        f"{report.unlisted_unique_items}"
    )

    print()

    print(
        "Token coverage: "
        f"{report.token_coverage_percentage:.1f}%"
    )
    print(
        "Evaluated word occurrences: "
        f"{report.total_tokens}"
    )
    print(
        "TOPIK I word occurrences: "
        f"{report.topik_tokens}"
    )
    print(
        "Non-TOPIK I word occurrences: "
        f"{report.unlisted_tokens}"
    )

    print("\nTOPIK I words found:")

    if report.topik_words:
        print(", ".join(report.topik_words))
    else:
        print("None")

    print("\nWords not found in the TOPIK I list:")

    if report.unlisted_tokens:
        print(", ".join(report.unlisted_words))
    else:
        print("None")

    print("\nProper nouns ignored in coverage calculations:")

    if report.proper_nouns:
        print(", ".join(report.proper_nouns))
        print(
            "Ignored proper-noun occurrences: "
            f"{report.ignored_proper_noun_tokens}"
        )
    else:
        print("None")


def print_analysis(text: str) -> None:
    """Analyze text and display TOPIK vocabulary coverage."""
    morphological_analyzer = KoreanMorphologicalAnalyzer()
    vocabulary_analyzer = VocabularyAnalyzer()
    topik_analyzer = TopikVocabularyAnalyzer(TOPIK_CSV_PATH)

    tokens = morphological_analyzer.analyze(text)
    vocabulary = vocabulary_analyzer.extract_vocabulary(tokens)

    topik_results = topik_analyzer.classify(vocabulary)
    coverage_report = topik_analyzer.create_coverage_report(
        topik_results
    )

    print_vocabulary_table(topik_results)
    print_coverage_report(coverage_report)


def main() -> None:
    """Run the command-line application."""
    print("Korean Readability Profiler")
    print("============================")

    text = input("\nPaste a Korean text:\n> ").strip()

    if not text:
        print("\nNo text was provided.")
        return

    try:
        print_analysis(text)
    except (ValueError, FileNotFoundError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()