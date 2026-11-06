import pandas as pd


def json_value(value):
    return None if pd.isna(value) else value


def prepare_routes(gtfs):
    routes = gtfs["routes"][
        ["route_id", "route_short_name", "route_long_name", "route_type"]
    ]

    trips = gtfs["trips"][["route_id", "direction_id", "shape_id"]]

    shapes = gtfs["shapes"][
        ["shape_id", "shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]
    ].copy()

    for column in ["shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]:
        shapes[column] = pd.to_numeric(shapes[column])

    route_trips = pd.merge(
        routes,
        trips,
        on="route_id",
        how="inner",
        validate="one_to_many",
    )

    route_trips = route_trips.drop_duplicates(
        subset=["route_id", "direction_id", "shape_id"]
    )

    route_shapes = pd.merge(
        route_trips,
        shapes,
        on="shape_id",
        how="inner",
    )

    route_shapes = route_shapes.sort_values(
        by=["route_id", "direction_id", "shape_id", "shape_pt_sequence"]
    )

    return route_shapes


def build_routes_geojson(route_shapes):
    features = []

    grouped_routes = route_shapes.groupby(
        ["route_id", "direction_id", "shape_id"],
        dropna=False,
        sort=False,
    )

    for (route_id, direction_id, shape_id), group_df in grouped_routes:
        coordinates = group_df[["shape_pt_lon", "shape_pt_lat"]].to_numpy().tolist()

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "route_id": route_id,
                "route_short_name": group_df["route_short_name"].iloc[0],
                "route_long_name": group_df["route_long_name"].iloc[0],
                "route_type": group_df["route_type"].iloc[0],
                "direction_id": json_value(direction_id),
                "shape_id": shape_id,
            },
        }

        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def prepare_stops(gtfs):
    trips = gtfs["trips"][["trip_id", "route_id", "direction_id", "shape_id"]]

    stop_times = gtfs["stop_times"][["trip_id", "stop_id", "stop_sequence"]].copy()

    stops = gtfs["stops"][["stop_id", "stop_name", "stop_lon", "stop_lat"]].copy()

    for column in ["stop_lon", "stop_lat"]:
        stops[column] = pd.to_numeric(stops[column])

    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"])

    trip_stops = pd.merge(
        trips,
        stop_times,
        on="trip_id",
        how="inner",
        validate="one_to_many",
    )

    route_stops = pd.merge(
        trip_stops,
        stops,
        on="stop_id",
        how="inner",
        validate="many_to_one",
    )

    return route_stops


def build_stops_geojson(route_stops):
    features = []

    grouped_routes = route_stops.groupby("stop_id", sort=False)

    for stop_id, group_df in grouped_routes:
        coordinates = group_df[["stop_lon", "stop_lat"]].iloc[0].tolist()

        route_ids = sorted(group_df["route_id"].unique().tolist())

        unique_services = group_df.drop_duplicates(
            subset=["route_id", "direction_id", "shape_id"]
        )

        services = []

        for row in unique_services.itertuples():
            services.append(
                {
                    "route_id": row.route_id,
                    "direction_id": json_value(row.direction_id),
                    "shape_id": json_value(row.shape_id),
                }
            )

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": coordinates,
            },
            "properties": {
                "stop_id": stop_id,
                "stop_name": group_df["stop_name"].iloc[0],
                "route_ids": route_ids,
                "services": services,
            },
        }

        features.append(feature)

    return {"type": "FeatureCollection", "features": features}
