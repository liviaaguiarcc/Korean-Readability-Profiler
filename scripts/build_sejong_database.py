"""Build the consolidated King Sejong vocabulary database."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEJONG_SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "sejong"
CONSOLIDATED_DATABASE_PATH = (
    PROJECT_ROOT / "data" / "sejong_vocabulary.csv"
)

REQUIRED_COLUMNS = [
    "korean",
    "sejong_level",
    "unit",
    "page",
]

LEVEL_ORDER = {
    "1A": 1,
    "1B": 2,
    "2A": 3,
    "2B": 4,
    "3A": 5,
    "3B": 6,
    "4A": 7,
    "4B": 8,
    "5A": 9,
    "5B": 10,
    "6A": 11,
    "6B": 12,
}


def load_level_database(file_path: Path) -> pd.DataFrame:
    """Load and validate one Sejong-level vocabulary file."""
    dataframe = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{missing_text}"
        )

    dataframe = dataframe[REQUIRED_COLUMNS].copy()

    dataframe["korean"] = (
        dataframe["korean"]
        .astype("string")
        .str.strip()
    )

    dataframe["sejong_level"] = (
        dataframe["sejong_level"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    dataframe["unit"] = pd.to_numeric(
        dataframe["unit"],
        errors="coerce",
    ).astype("Int64")

    dataframe["page"] = pd.to_numeric(
        dataframe["page"],
        errors="coerce",
    ).astype("Int64")

    dataframe = dataframe.dropna(
        subset=["korean", "sejong_level"]
    )

    dataframe = dataframe[
        (dataframe["korean"] != "")
        & (dataframe["sejong_level"] != "")
    ]

    unknown_levels = sorted(
        set(dataframe["sejong_level"]) - set(LEVEL_ORDER)
    )

    if unknown_levels:
        unknown_text = ", ".join(unknown_levels)

        raise ValueError(
            f"{file_path.name} contains unknown Sejong levels: "
            f"{unknown_text}"
        )

    return dataframe


def build_sejong_database(
    source_directory: Path = SEJONG_SOURCE_DIRECTORY,
    output_path: Path = CONSOLIDATED_DATABASE_PATH,
) -> pd.DataFrame:
    """Combine all Sejong level files and preserve first appearance."""
    source_files = sorted(
        source_directory.glob("sejong_*_vocabulary.csv")
    )

    if not source_files:
        raise FileNotFoundError(
            "No Sejong vocabulary files were found in: "
            f"{source_directory}"
        )

    dataframes = [
        load_level_database(file_path)
        for file_path in source_files
    ]

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined["_level_order"] = combined[
        "sejong_level"
    ].map(LEVEL_ORDER)

    combined["_unit_order"] = (
        combined["unit"]
        .fillna(999)
        .astype(int)
    )

    combined["_page_order"] = (
        combined["page"]
        .fillna(9999)
        .astype(int)
    )

    combined = combined.sort_values(
        by=[
            "_level_order",
            "_unit_order",
            "_page_order",
            "korean",
        ],
        kind="stable",
    )

    total_before_deduplication = len(combined)

    # A word may appear in several levels. The consolidated database
    # keeps its earliest pedagogical appearance.
    consolidated = combined.drop_duplicates(
        subset=["korean"],
        keep="first",
    )

    consolidated = consolidated[
        REQUIRED_COLUMNS
    ].reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    consolidated.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    removed_duplicates = (
        total_before_deduplication - len(consolidated)
    )

    print("Sejong database built successfully.")
    print(f"Source files: {len(source_files)}")
    print(
        "Entries before deduplication: "
        f"{total_before_deduplication}"
    )
    print(
        "Duplicate later appearances removed: "
        f"{removed_duplicates}"
    )
    print(
        "Unique vocabulary entries: "
        f"{len(consolidated)}"
    )
    print(f"Output: {output_path}")

    return consolidated


def main() -> None:
    """Build the consolidated database."""
    build_sejong_database()


if __name__ == "__main__":
    main()