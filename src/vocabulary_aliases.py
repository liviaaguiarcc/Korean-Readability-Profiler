"""Vocabulary aliases for matching normalized and surface forms."""

from pathlib import Path

import pandas as pd


class VocabularyAliasResolver:
    """Resolve normalized lemmas to forms used in vocabulary sources."""

    REQUIRED_COLUMNS = {
        "lemma",
        "variant",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.aliases = self._load_aliases()

    def _load_aliases(self) -> dict[str, tuple[str, ...]]:
        """Load lemma-to-variant mappings from a CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Vocabulary alias file not found: {self.csv_path}"
            )

        dataframe = pd.read_csv(self.csv_path)

        missing_columns = (
            self.REQUIRED_COLUMNS - set(dataframe.columns)
        )

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))

            raise ValueError(
                "The alias CSV is missing required columns: "
                f"{missing_text}"
            )

        aliases: dict[str, list[str]] = {}

        for row in dataframe.itertuples(index=False):
            if pd.isna(row.lemma) or pd.isna(row.variant):
                continue

            lemma = str(row.lemma).strip()
            variant = str(row.variant).strip()

            if not lemma or not variant:
                continue

            aliases.setdefault(lemma, []).append(variant)

        return {
            lemma: tuple(dict.fromkeys(variants))
            for lemma, variants in aliases.items()
        }

    def candidates(self, lemma: str) -> tuple[str, ...]:
        """Return the lemma and all known surface variants."""
        variants = self.aliases.get(lemma, ())

        return (lemma, *variants)