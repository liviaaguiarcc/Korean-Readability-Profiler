"""Command-line entry point for the Korean Readability Profiler."""

from src.morphological_analyzer import KoreanMorphologicalAnalyzer


def print_analysis(text: str) -> None:
    """Analyze text and display its morphemes in the terminal."""
    analyzer = KoreanMorphologicalAnalyzer()
    tokens = analyzer.analyze(text)

    print("\nKOREAN MORPHOLOGICAL ANALYSIS")
    print("-" * 55)
    print(f"{'FORM':<20} {'TAG':<10} {'POSITION':<10}")
    print("-" * 55)

    for token in tokens:
        print(f"{token.form:<20} {token.tag:<10} {token.start:<10}")

    print("-" * 55)
    print(f"Total morphemes: {len(tokens)}")


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
    except ValueError as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()