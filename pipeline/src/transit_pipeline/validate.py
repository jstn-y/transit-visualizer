import numpy as np
import pandas as pd

REQUIRED_TABLES = {
    "routes": [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
    ],
    "trips": [
        "trip_id",
        "route_id",
        "direction_id",
        "shape_id",
    ],
    "shapes": [
        "shape_id",
        "shape_pt_sequence",
        "shape_pt_lon",
        "shape_pt_lat",
    ],
    "stops": [
        "stop_id",
        "stop_name",
        "stop_lon",
        "stop_lat",
    ],
    "stop_times": [
        "trip_id",
        "stop_id",
        "stop_sequence",
    ],
}

REQUIRED_IDS = {
    "routes": ["route_id"],
    "trips": ["trip_id", "route_id"],
    "shapes": ["shape_id"],
    "stops": ["stop_id"],
    "stop_times": ["trip_id", "stop_id"],
}

UNIQUE_IDS = {
    "routes": "route_id",
    "trips": "trip_id",
    "stops": "stop_id",
}

RELATIONSHIPS = [
    ("trips", "route_id", "routes", "route_id"),
    ("stop_times", "trip_id", "trips", "trip_id"),
    ("stop_times", "stop_id", "stops", "stop_id"),
]

COORDINATE_RANGES = {
    ("shapes", "shape_pt_lat"): (-90, 90),
    ("shapes", "shape_pt_lon"): (-180, 180),
    ("stops", "stop_lat"): (-90, 90),
    ("stops", "stop_lon"): (-180, 180),
}

SEQUENCES = {
    "shapes": ("shape_id", "shape_pt_sequence"),
    "stop_times": ("trip_id", "stop_sequence"),
}


def validate_required_tables(gtfs):
    """Check required GTFS tables exist and aren't empty."""
    missing = REQUIRED_TABLES.keys() - gtfs.keys()

    if missing:
        raise ValueError(f"Missing required GTFS tables: {sorted(missing)}")

    for table in REQUIRED_TABLES:
        if gtfs[table].empty:
            raise ValueError(f"Required GTFS table '{table}' is empty")


def validate_required_columns(gtfs):
    """Check every table contains the expected columns."""
    for table, columns in REQUIRED_TABLES.items():
        missing = set(columns) - set(gtfs[table].columns)

        if missing:
            raise ValueError(
                f"Missing required columns in '{table}': {sorted(missing)}"
            )


def validate_identifiers(gtfs):
    """Check required IDs are populated and unique where needed."""

    for table, columns in REQUIRED_IDS.items():
        for col in columns:
            values = gtfs[table][col]
            missing_count = (
                values.isna() | values.astype("string").str.strip().eq("").fillna(False)
            ).sum()

            if missing_count > 0:
                raise ValueError(f"'{table}.{col}' has {missing_count} missing values")

    for table, col in UNIQUE_IDS.items():
        duplicate_count = gtfs[table][col].duplicated().sum()

        if duplicate_count > 0:
            raise ValueError(f"'{table}.{col}' has {duplicate_count} duplicate values")


def find_invalid_references(child_df, child_column, parent_df, parent_column):
    """Return child rows whose foreign keys don't exist in the parent."""
    invalid = child_df[~child_df[child_column].isin(parent_df[parent_column])]

    return invalid


def validate_relationships(gtfs):
    """Check foreign keys and report missing shape references."""
    for child, child_col, parent, parent_col in RELATIONSHIPS:
        invalid = find_invalid_references(
            gtfs[child], child_col, gtfs[parent], parent_col
        )

        if not invalid.empty:
            raise ValueError(
                f"{child}.{child_col} contains "
                f"{len(invalid)} references not found in "
                f"{parent}.{parent_col}"
            )

    trips = gtfs["trips"]
    shapes = gtfs["shapes"]

    missing_count = trips["shape_id"].isna().sum()

    with_shapes = trips[trips["shape_id"].notna()]

    invalid = with_shapes[~with_shapes["shape_id"].isin(shapes["shape_id"])]

    return {
        "trips_missing_shape_id": int(missing_count),
        "trips_invalid_shape_id": len(invalid),
        "invalid_shape_trip_ids": invalid["trip_id"].tolist(),
    }


def validate_coordinates(gtfs):
    """Validate latitude and longitude values in shapes and stops."""
    for (table, col), (lo, hi) in COORDINATE_RANGES.items():
        values = pd.to_numeric(gtfs[table][col], errors="coerce")
        invalid_count = int((~values.between(lo, hi)).sum())

        if invalid_count:
            raise ValueError(
                f"'{table}.{col}' contains {invalid_count} invalid coordinates. "
                f"Expected finite numeric values between {lo} and {hi}."
            )


def validate_sequence(gtfs):
    """Validate GTFS shape and stop-time sequence values."""
    for table, (group_col, seq_col) in SEQUENCES.items():
        df = gtfs[table]

        values = pd.to_numeric(df[seq_col], errors="coerce")

        invalid = (
            values.isna() | ~np.isfinite(values) | (values < 0) | (values % 1 != 0)
        )

        invalid_count = int(invalid.sum())

        if invalid_count > 0:
            raise ValueError(
                f"'{table}.{seq_col}' contains {invalid_count} "
                f"invalid sequence values. Expected non-negative integers."
            )

        duplicates = df.assign(_numeric_sequence=values).duplicated(
            subset=[group_col, "_numeric_sequence"]
        )

        duplicate_count = int(duplicates.sum())

        if duplicate_count > 0:
            raise ValueError(
                f"'{table}' contains {duplicate_count} duplicate "
                f"'{seq_col}' values within the same '{group_col}'."
            )


def validate_gtfs_input(gtfs):
    """Run input validation before transforming data."""
    validate_required_tables(gtfs)
    validate_required_columns(gtfs)
    validate_identifiers(gtfs)
    report = validate_relationships(gtfs)
    validate_coordinates(gtfs)
    validate_sequence(gtfs)

    return report
