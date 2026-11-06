import json
from pathlib import Path


def publish_geojson(geojson, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, allow_nan=False)
