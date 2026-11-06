from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
from src.transit_pipeline.ingest import extract_gtfs

GTFS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "GO-GTFS.zip"
)

EXPECTED_COLUMNS = {
    "routes": {"route_id", "route_type"},
    "trips": {"route_id", "service_id", "trip_id", "shape_id"},
    "shapes": {
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
    },
    "stops": {"stop_id", "stop_lat", "stop_lon"},
}


@pytest.fixture
def gtfs_data():
    """Load the GO Transit GTFS feed for a test."""
    assert GTFS_PATH.is_file(), f"Missing GTFS file: {GTFS_PATH}"
    return extract_gtfs(GTFS_PATH)


def test_all_dataframes_load(gtfs_data):
    """Verify that all four GTFS files load successfully."""
    assert set(gtfs_data.keys()) == set(EXPECTED_COLUMNS)

    for name, df in gtfs_data.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, f"{name} DataFrame is empty"


def test_expected_columns_exist(gtfs_data):
    """Verify that each DataFrame has the expected columns."""
    for name, required_columns in EXPECTED_COLUMNS.items():
        actual_columns = set(gtfs_data[name].columns)

        assert required_columns.issubset(actual_columns), (
            f"{name} is missing columns: "
            f"{required_columns - actual_columns}"
        )


def test_identifiers_preserved(gtfs_data):
    """Compare loaded IDs against the original ZIP contents."""
    id_columns = {
        "routes": ["route_id"],
        "trips": ["trip_id", "route_id", "service_id", "shape_id"],
        "shapes": ["shape_id"],
        "stops": ["stop_id"],
    }

    with ZipFile(GTFS_PATH) as archive:
        for name, columns in id_columns.items():
            with archive.open(f"{name}.txt") as file:
                original = pd.read_csv(file, dtype=str)

            for column in columns:
                pd.testing.assert_series_equal(
                    gtfs_data[name][column],
                    original[column],
                )


def test_repeated_calls_are_independent():
    """Verify that calls return separate, consistent DataFrames."""
    first = extract_gtfs(GTFS_PATH)
    second = extract_gtfs(GTFS_PATH)

    for name in EXPECTED_COLUMNS:
        assert first[name] is not second[name]

        pd.testing.assert_frame_equal(
            first[name],
            second[name],
        )

    # Modify the first result.
    original_id = second["routes"].loc[0, "route_id"]
    first["routes"].loc[0, "route_id"] = "TEST_CHANGED"

    # The second result should be unaffected.
    assert second["routes"].loc[0, "route_id"] == original_id
