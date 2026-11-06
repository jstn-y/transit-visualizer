from transit_pipeline.ingest import extract_gtfs
from transit_pipeline.publish import publish_geojson
from transit_pipeline.transform import (
    build_routes_geojson,
    build_stops_geojson,
    prepare_routes,
    prepare_stops,
)
from transit_pipeline.validate import validate_gtfs_input

gtfs = extract_gtfs("data/raw/GO-GTFS.zip")

report = validate_gtfs_input(gtfs)

route_shapes = prepare_routes(gtfs)
routes_geojson = build_routes_geojson(route_shapes)

route_stops = prepare_stops(gtfs)
stops_geojson = build_stops_geojson(route_stops)

publish_geojson(routes_geojson, "data/processed/routes.geojson")
publish_geojson(stops_geojson, "data/processed/stops.geojson")
