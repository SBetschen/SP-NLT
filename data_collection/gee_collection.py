#!/usr/bin/env python3
"""
Convert a sampled cities CSV to a Google Earth Engine FeatureCollection JS snippet.

Usage:
  python gee_collection.py --input samples.csv --output cities_fc.txt

The output file will contain something like:
  var cities = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([lon, lat]), {...}),
    ...
  ]);
"""

from __future__ import annotations

import argparse
import json
import math
import pandas as pd


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def clean_value(v):
    """Make values JSON-serializable and avoid NaN/inf."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to sampled CSV (e.g., samples.csv)")
    ap.add_argument("--output", required=True, help="Path to output text file (e.g., cities_fc.txt)")
    ap.add_argument("--varname", default="cities", help="GEE variable name to use (default: cities)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Required: lat & lon
    lat_col = find_col(df, ["lat", "latitude"])
    lon_col = find_col(df, ["lng", "lon", "long", "longitude"])
    if lat_col is None or lon_col is None:
        raise ValueError("Could not find lat/lon columns. Need 'lat' and one of [lng, lon, long, longitude].")

    # Prefer city_ascii but fall back to city
    name_col = find_col(df, ["city_ascii", "city", "name"])
    if name_col is None:
        raise ValueError("Could not find a city name column. Need one of [city_ascii, city, name].")

    # Make sure numerics are numeric
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    lines = []
    lines.append(f"var {args.varname} = ee.FeatureCollection([")

    for i, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        # Properties: include everything except lat/lon columns 
        props = {}
        for col in df.columns:
            if col in (lat_col, lon_col):
                continue
            props[col] = clean_value(row[col])

        # Ensure there's a clean 'name' property for convenience in GEE
        props.setdefault("name", clean_value(row[name_col]))

        # JSON for properties (valid JS object literal too)
        props_json = json.dumps(props, ensure_ascii=False)

        # Build the feature line
        # NOTE: Point expects [lon, lat]
        lines.append(f"  ee.Feature(ee.Geometry.Point([{lon:.8f}, {lat:.8f}]), {props_json}),")

    # Remove last trailing comma
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")

    lines.append("]);")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote GEE FeatureCollection snippet with {len(df)} features to: {args.output}")


if __name__ == "__main__":
    main()