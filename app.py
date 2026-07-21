"""Command-line entry point for the Korean Readability Profiler."""

from pathlib import Path

from src.morphological_analyzer import KoreanMorphologicalAnalyzer
from src.sejong_analyzer import (
    SejongCoverageReport,
    SejongVocabularyAnalyzer,
    SejongVocabularyResult,
)
from src.topik_analyzer import (
    TopikCoverageReport,
    TopikVocabularyAnalyzer,
    TopikVocabularyResult,
)
from src.vocabulary_analyzer import VocabularyAnalyzer


TOPIK_CSV_PATH = Path("data/topik_i_number_korean.csv")
SEJONG_CSV_PATH = Path("data/sejong_1a_vocabulary.csv")


def print_vocabulary_table(
    topik_results: list[TopikVocabularyResult],
    sejong_results: list[SejongVocabularyResult],
) -> None:
    """Display TOPIK and Sejong classification results."""
    print("\nVOCABULARY ANALYSIS")
    print("-" * 110)

    print(
        f"{'LEMMA':<22}"
        f"{'PART OF SPEECH':<18}"
        f"{'FREQ.':<8}"
        f"{'TOPIK':<18}"
        f"{'SEJONG':<12}"
        f"{'UNIT':<8}"
        f"{'PAGE':<8}"
    )

    print("-" * 110)

    topik_labels = {
        "topik_i": "TOPIK I",
        "possible_proper_noun": "Proper noun?",
        "unlisted": "Unlisted",
    }

    for topik_result, sejong_result in zip(
        topik_results,
        sejong_results,
    ):
        topik_label = topik_labels[topik_result.category]

        if sejong_result.is_listed:
            sejong_label = sejong_result.sejong_level or "-"
            unit_label = str(sejong_result.unit or "-")
            page_label = str(sejong_result.page or "-")
        elif sejong_result.is_possible_proper_noun:
            sejong_label = "Proper noun?"
            unit_label = "-"
            page_label = "-"
        else:
            sejong_label = "Unlisted"
            unit_label = "-"
            page_label = "-"

        print(
            f"{topik_result.lemma:<22}"
            f"{topik_result.part_of_speech:<18}"
            f"{topik_result.frequency:<8}"
            f"{topik_label:<18}"
            f"{sejong_label:<12}"
            f"{unit_label:<8}"
            f"{page_label:<8}"
        )

    print("-" * 110)


def print_topik_coverage_report(
    report: TopikCoverageReport,
) -> None:
    """Display TOPIK I coverage statistics."""
    print("\nTOPIK I COVERAGE REPORT")
    print("-" * 55)

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
        "Unlisted unique words: "
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
        "Unlisted word occurrences: "
        f"{report.unlisted_tokens}"
    )

    print("\nWords not found in the current TOPIK I database:")

    if report.unlisted_words:
        print(", ".join(report.unlisted_words))
    else:
        print("None")

    print("\nPossible proper nouns ignored:")

    if report.proper_nouns:
        print(", ".join(report.proper_nouns))
    else:
        print("None")


def print_sejong_coverage_report(
    report: SejongCoverageReport,
) -> None:
    """Display Sejong 1A vocabulary coverage statistics."""
    print("\nSEJONG 1A COVERAGE REPORT")
    print("-" * 55)

    print(
        "Unique vocabulary coverage: "
        f"{report.unique_coverage_percentage:.1f}%"
    )
    print(
        "Evaluated unique words: "
        f"{report.total_unique_items}"
    )
    print(
        "Words found in Sejong 1A: "
        f"{report.listed_unique_items}"
    )
    print(
        "Words not found in the current Sejong database: "
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
        "Sejong 1A word occurrences: "
        f"{report.listed_tokens}"
    )
    print(
        "Unlisted word occurrences: "
        f"{report.unlisted_tokens}"
    )

    print("\nWords found in Sejong 1A:")

    if report.listed_words:
        print(", ".join(report.listed_words))
    else:
        print("None")

    print("\nWords not found in the current Sejong database:")

    if report.unlisted_words:
        print(", ".join(report.unlisted_words))
    else:
        print("None")

    print("\nPossible proper nouns ignored:")

    if report.proper_nouns:
        print(", ".join(report.proper_nouns))
    else:
        print("None")


def print_analysis(text: str) -> None:
    """Analyze text using the TOPIK and Sejong databases."""
    morphological_analyzer = KoreanMorphologicalAnalyzer()
    vocabulary_analyzer = VocabularyAnalyzer()

    topik_analyzer = TopikVocabularyAnalyzer(
        TOPIK_CSV_PATH
    )

    sejong_analyzer = SejongVocabularyAnalyzer(
        SEJONG_CSV_PATH
    )

    tokens = morphological_analyzer.analyze(text)
    vocabulary = vocabulary_analyzer.extract_vocabulary(tokens)

    topik_results = topik_analyzer.classify(vocabulary)
    sejong_results = sejong_analyzer.classify(vocabulary)

    topik_report = topik_analyzer.create_coverage_report(
        topik_results
    )

    sejong_report = sejong_analyzer.create_coverage_report(
        sejong_results
    )

    print_vocabulary_table(
        topik_results,
        sejong_results,
    )

    print_topik_coverage_report(topik_report)
    print_sejong_coverage_report(sejong_report)


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