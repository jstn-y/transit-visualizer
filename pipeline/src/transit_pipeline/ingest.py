from zipfile import ZipFile

import pandas as pd


def extract_gtfs(gtfs_zip):

    gtfs = {}

    file_names = ["routes", "trips", "shapes", "stops", "stop_times", "calendar_dates"]

    with ZipFile(gtfs_zip, "r") as z:
        for key in file_names:
            with z.open(f"{key}.txt") as f:
                df = pd.read_csv(f, dtype=str)
                gtfs[key] = df

    return gtfs
