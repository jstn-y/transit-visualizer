"""Unit tests for the GTFS input validators.

Run from pipeline/: python -m pytest tests/test_validate.py -v
"""

import numpy as np
import pandas as pd
import pytest
from transit_pipeline.validate import (
    find_invalid_references,
    validate_coordinates,
    validate_gtfs_input,
    validate_identifiers,
    validate_relationships,
    validate_required_columns,
    validate_required_tables,
    validate_sequence,
)


@pytest.fixture
def valid_gtfs():
    """Minimal, internally consistent GTFS tables for testing."""
    return {
        "routes": pd.DataFrame(
            {
                "route_id": ["R01", "R02"],
                "route_short_name": ["1", "2"],
                "route_long_name": ["Route One", "Route Two"],
                "route_type": ["3", "2"],
            }
        ),
        "trips": pd.DataFrame(
            {
                "trip_id": ["T01", "T02"],
                "route_id": ["R01", "R02"],
                "direction_id": ["0", "1"],
                "shape_id": ["S01", "S02"],
            }
        ),
        "shapes": pd.DataFrame(
            {
                "shape_id": ["S01", "S01", "S02", "S02"],
                "shape_pt_sequence": ["1", "2", "1", "2"],
                "shape_pt_lon": ["-79.38", "-79.39", "-80.48", "-80.49"],
                "shape_pt_lat": ["43.65", "43.66", "43.45", "43.46"],
            }
        ),
        "stops": pd.DataFrame(
            {
                "stop_id": ["ST01", "ST02"],
                "stop_name": ["Stop One", "Stop Two"],
                "stop_lon": ["-79.38", "-80.48"],
                "stop_lat": ["43.65", "43.45"],
            }
        ),
        "stop_times": pd.DataFrame(
            {
                "trip_id": ["T01", "T01", "T02"],
                "stop_id": ["ST01", "ST02", "ST02"],
                "stop_sequence": ["1", "2", "1"],
            }
        ),
    }


def test_valid_gtfs_passes_all_input_checks(valid_gtfs):
    report = validate_gtfs_input(valid_gtfs)
    assert report == {
        "trips_missing_shape_id": 0,
        "trips_invalid_shape_id": 0,
        "invalid_shape_trip_ids": [],
    }


def test_missing_required_table(valid_gtfs):
    del valid_gtfs["stops"]
    with pytest.raises(ValueError, match="Missing required GTFS tables"):
        validate_required_tables(valid_gtfs)


def test_empty_required_table(valid_gtfs):
    valid_gtfs["routes"] = valid_gtfs["routes"].iloc[0:0]
    with pytest.raises(ValueError, match="table 'routes' is empty"):
        validate_required_tables(valid_gtfs)


def test_missing_required_column(valid_gtfs):
    valid_gtfs["trips"] = valid_gtfs["trips"].drop(columns=["route_id"])
    with pytest.raises(ValueError, match="Missing required columns in 'trips'"):
        validate_required_columns(valid_gtfs)


@pytest.mark.parametrize("bad_id", [None, np.nan, "", "   "])
def test_missing_or_blank_identifier(valid_gtfs, bad_id):
    valid_gtfs["routes"].loc[0, "route_id"] = bad_id
    with pytest.raises(ValueError, match=r"'routes\.route_id'.*missing values"):
        validate_identifiers(valid_gtfs)


@pytest.mark.parametrize(
    "table,col",
    [
        ("routes", "route_id"),
        ("trips", "trip_id"),
        ("stops", "stop_id"),
    ],
)
def test_duplicate_primary_identifier(valid_gtfs, table, col):
    valid_gtfs[table].loc[1, col] = valid_gtfs[table].loc[0, col]
    with pytest.raises(ValueError, match="duplicate values"):
        validate_identifiers(valid_gtfs)


def test_repeated_shape_ids_are_allowed(valid_gtfs):
    # Each shape has multiple points. Likewise, stop_times repeats trip IDs.
    validate_identifiers(valid_gtfs)


@pytest.mark.parametrize(
    "child,col,bad_value",
    [
        ("trips", "route_id", "R999"),
        ("stop_times", "trip_id", "T999"),
        ("stop_times", "stop_id", "ST999"),
    ],
)
def test_invalid_foreign_key(valid_gtfs, child, col, bad_value):
    valid_gtfs[child].loc[0, col] = bad_value
    with pytest.raises(ValueError, match=rf"{child}\.{col} contains 1 references"):
        validate_relationships(valid_gtfs)


def test_find_invalid_references_returns_orphaned_rows(valid_gtfs):
    valid_gtfs["trips"].loc[0, "route_id"] = "R999"
    invalid = find_invalid_references(
        valid_gtfs["trips"], "route_id", valid_gtfs["routes"], "route_id"
    )
    assert invalid["trip_id"].tolist() == ["T01"]


def test_missing_trip_shape_is_reported_not_rejected(valid_gtfs):
    valid_gtfs["trips"].loc[0, "shape_id"] = np.nan
    report = validate_gtfs_input(valid_gtfs)
    assert report["trips_missing_shape_id"] == 1
    assert report["trips_invalid_shape_id"] == 0
    assert report["invalid_shape_trip_ids"] == []


def test_nonexistent_trip_shape_is_reported_not_rejected(valid_gtfs):
    valid_gtfs["trips"].loc[0, "shape_id"] = "S999"
    report = validate_gtfs_input(valid_gtfs)
    assert report["trips_missing_shape_id"] == 0
    assert report["trips_invalid_shape_id"] == 1
    assert report["invalid_shape_trip_ids"] == ["T01"]


@pytest.mark.parametrize(
    "table,col,bad_value",
    [
        ("shapes", "shape_pt_lat", 91),
        ("shapes", "shape_pt_lon", -181),
        ("stops", "stop_lat", -91),
        ("stops", "stop_lon", 181),
        ("stops", "stop_lat", "not-a-number"),
        ("shapes", "shape_pt_lat", np.nan),
        ("shapes", "shape_pt_lon", np.inf),
    ],
)
def test_invalid_coordinate(valid_gtfs, table, col, bad_value):
    valid_gtfs[table].loc[0, col] = str(bad_value)
    with pytest.raises(ValueError, match=rf"'{table}\.{col}'.*invalid coordinates"):
        validate_coordinates(valid_gtfs)


def test_valid_coordinate_bounds_are_accepted(valid_gtfs):
    valid_gtfs["stops"].loc[0, "stop_lat"] = "-90"
    valid_gtfs["stops"].loc[0, "stop_lon"] = "180"
    validate_coordinates(valid_gtfs)


@pytest.mark.parametrize(
    "table,col,bad_value",
    [
        ("shapes", "shape_pt_sequence", None),
        ("shapes", "shape_pt_sequence", ""),
        ("shapes", "shape_pt_sequence", "abc"),
        ("shapes", "shape_pt_sequence", "-1"),
        ("shapes", "shape_pt_sequence", "1.5"),
        ("shapes", "shape_pt_sequence", "inf"),
        ("stop_times", "stop_sequence", None),
        ("stop_times", "stop_sequence", ""),
        ("stop_times", "stop_sequence", "abc"),
        ("stop_times", "stop_sequence", "-1"),
        ("stop_times", "stop_sequence", "2.5"),
        ("stop_times", "stop_sequence", "inf"),
    ],
)
def test_invalid_sequence(valid_gtfs, table, col, bad_value):
    valid_gtfs[table].loc[0, col] = bad_value

    with pytest.raises(ValueError, match="invalid sequence values"):
        validate_sequence(valid_gtfs)


@pytest.mark.parametrize(
    "table,group_col,seq_col",
    [
        ("shapes", "shape_id", "shape_pt_sequence"),
        ("stop_times", "trip_id", "stop_sequence"),
    ],
)
def test_duplicate_sequence(valid_gtfs, table, group_col, seq_col):
    # Duplicate an existing row to create a repeated sequence
    duplicate = valid_gtfs[table].iloc[[0]].copy()

    valid_gtfs[table] = pd.concat(
        [valid_gtfs[table], duplicate],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_sequence(valid_gtfs)


def test_valid_sequence_passes(valid_gtfs):
    # The original fixture should contain valid sequences
    validate_sequence(valid_gtfs)


def test_nonconsecutive_sequences_are_allowed(valid_gtfs):
    # Replace existing sequence values with nonconsecutive ones
    for table, group_col, seq_col in [
        ("shapes", "shape_id", "shape_pt_sequence"),
        ("stop_times", "trip_id", "stop_sequence"),
    ]:
        df = valid_gtfs[table]

        # Assign 10, 20, 30... separately within each group
        valid_gtfs[table][seq_col] = (
            df.groupby(group_col).cumcount().add(1).mul(10).astype(str)
        )

    validate_sequence(valid_gtfs)
