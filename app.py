"""Command-line entry point for the Korean Readability Profiler."""

from src.morphological_analyzer import KoreanMorphologicalAnalyzer
from src.vocabulary_analyzer import VocabularyAnalyzer


def print_morphological_analysis(text: str) -> None:
    """Analyze text and display morphemes and vocabulary."""
    morphological_analyzer = KoreanMorphologicalAnalyzer()
    vocabulary_analyzer = VocabularyAnalyzer()

    tokens = morphological_analyzer.analyze(text)
    vocabulary = vocabulary_analyzer.extract_vocabulary(tokens)

    print("\nMORPHOLOGICAL ANALYSIS")
    print("-" * 55)
    print(f"{'FORM':<20} {'TAG':<10} {'POSITION':<10}")
    print("-" * 55)

    for token in tokens:
        print(f"{token.form:<20} {token.tag:<10} {token.start:<10}")

    print("-" * 55)
    print(f"Total morphemes: {len(tokens)}")

    print("\nVOCABULARY")
    print("-" * 60)
    print(f"{'LEMMA':<25} {'PART OF SPEECH':<20} {'FREQUENCY':<10}")
    print("-" * 60)

    for item in vocabulary:
        print(
            f"{item.lemma:<25} "
            f"{item.part_of_speech:<20} "
            f"{item.frequency:<10}"
        )

    print("-" * 60)
    print(f"Unique vocabulary items: {len(vocabulary)}")


def main() -> None:
    """Run the command-line application."""
    print("Korean Readability Profiler")
    print("============================")

    text = input("\nPaste a Korean text:\n> ").strip()

    if not text:
        print("\nNo text was provided.")
        return

    try:
        print_morphological_analysis(text)
    except ValueError as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()