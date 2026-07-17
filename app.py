"""Command-line entry point for the Korean Readability Profiler."""

from pathlib import Path

from src.morphological_analyzer import KoreanMorphologicalAnalyzer
from src.topik_analyzer import TopikVocabularyAnalyzer
from src.vocabulary_analyzer import VocabularyAnalyzer


TOPIK_CSV_PATH = Path("data/topik_i_number_korean.csv")


def print_analysis(text: str) -> None:
    """Analyze text and display vocabulary and TOPIK results."""
    morphological_analyzer = KoreanMorphologicalAnalyzer()
    vocabulary_analyzer = VocabularyAnalyzer()
    topik_analyzer = TopikVocabularyAnalyzer(TOPIK_CSV_PATH)

    tokens = morphological_analyzer.analyze(text)
    vocabulary = vocabulary_analyzer.extract_vocabulary(tokens)
    topik_results = topik_analyzer.classify(vocabulary)

    print("\nVOCABULARY AND TOPIK I ANALYSIS")
    print("-" * 75)
    print(
        f"{'LEMMA':<25}"
        f"{'PART OF SPEECH':<20}"
        f"{'FREQUENCY':<12}"
        f"{'TOPIK I':<10}"
    )
    print("-" * 75)

    for result in topik_results:
        topik_label = "Yes" if result.is_topik_i else "No"

        print(
            f"{result.lemma:<25}"
            f"{result.part_of_speech:<20}"
            f"{result.frequency:<12}"
            f"{topik_label:<10}"
        )

    print("-" * 75)

    total_frequency = sum(
        result.frequency
        for result in topik_results
    )

    topik_frequency = sum(
        result.frequency
        for result in topik_results
        if result.is_topik_i
    )

    if total_frequency == 0:
        coverage = 0.0
    else:
        coverage = topik_frequency / total_frequency * 100

    print(f"Unique vocabulary items: {len(topik_results)}")
    print(f"TOPIK I token coverage: {coverage:.1f}%")


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