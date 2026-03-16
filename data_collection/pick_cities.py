#!/usr/bin/env python3
"""
Sample cities from a CSV with population-diverse stratification and a minimum spacing constraint.

Input CSV expected columns (at least):
- city / city_ascii (one of them)
- lat, lng (or long / longitude)
- population (or pop)

Example:
  python pick_cities.py --input worldcities.csv --output sampled.csv --n 500 --min_km 62 --seed 42 --bins 6
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import numpy as np
import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth (km)."""
    R = 6371.0088  # mean Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def far_enough(candidate: Tuple[float, float], chosen: List[Tuple[float, float]], min_km: float) -> bool:
    clat, clon = candidate
    for lat, lon in chosen:
        if haversine_km(clat, clon, lat, lon) < min_km:
            return False
    return True


def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    cols_lower = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # Latitude
    if "lat" not in df.columns:
        raise ValueError("Missing required column: lat")

    # Longitude might be named in various ways
    lon_col = None
    for c in ["lng", "lon", "long", "longitude"]:
        if c in df.columns:
            lon_col = c
            break
    if lon_col is None:
        raise ValueError("Missing required longitude column: expected one of [lng, lon, long, longitude]")
    if lon_col != "lng":
        df = df.rename(columns={lon_col: "lng"})

    # Population might be named in various ways
    pop_col = None
    for c in ["population", "pop"]:
        if c in df.columns:
            pop_col = c
            break
    if pop_col is None:
        raise ValueError("Missing population column: expected 'population' or 'pop'")
    if pop_col != "population":
        df = df.rename(columns={pop_col: "population"})

    # City name
    if "city" not in df.columns and "city_ascii" not in df.columns:
        raise ValueError("Missing city name column: expected 'city' or 'city_ascii'")
    if "city_ascii" not in df.columns:
        df["city_ascii"] = df["city"]
    if "city" not in df.columns:
        df["city"] = df["city_ascii"]

    # Ensure numeric types and drop bad rows
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["lat", "lng", "population"])
    df = df[df["population"] > 0]

    # Keep reasonable coordinate bounds
    df = df[(df["lat"].between(-90, 90)) & (df["lng"].between(-180, 180))]

    return df


def make_population_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Create population bins for diversity.
    Uses log10(pop) quantiles so the bins behave well across huge pop ranges.
    """
    logp = np.log10(df["population"].astype(float))
    # qcut can fail if many duplicates; use rank to break ties a bit
    try:
        df["pop_bin"] = pd.qcut(logp, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Fallback: bin on ranked values
        df["pop_bin"] = pd.qcut(logp.rank(method="average"), q=n_bins, labels=False, duplicates="drop")
    df["pop_bin"] = df["pop_bin"].astype(int)
    return df


def spaced_stratified_sample(
    df: pd.DataFrame,
    n: int,
    min_km: float,
    seed: int,
    n_bins: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = make_population_bins(df.copy(), n_bins=n_bins)

    # Build candidate lists per bin (shuffled)
    bins = sorted(df["pop_bin"].unique().tolist())
    bin_to_candidates = {}
    for b in bins:
        cand = df[df["pop_bin"] == b].copy()
        # shuffle rows
        cand = cand.sample(frac=1.0, random_state=seed + int(b)).reset_index(drop=True)
        bin_to_candidates[b] = cand

    chosen_rows = []
    chosen_points: List[Tuple[float, float]] = []

    # Round-robin across bins to enforce diversity
    bin_cycle = bins.copy()
    idx_in_bin = {b: 0 for b in bins}

    # Safety: avoid infinite loops if spacing is too strict for the requested n
    stalls = 0
    max_stalls = 10_000

    while len(chosen_rows) < n and stalls < max_stalls and len(bin_cycle) > 0:
        progressed = False

        for b in list(bin_cycle):
            cand_df = bin_to_candidates[b]
            i = idx_in_bin[b]

            # Advance within bin until we find a valid spaced candidate or exhaust bin
            while i < len(cand_df):
                row = cand_df.iloc[i]
                pt = (float(row["lat"]), float(row["lng"]))
                i += 1
                if far_enough(pt, chosen_points, min_km=min_km):
                    chosen_rows.append(row)
                    chosen_points.append(pt)
                    progressed = True
                    break

            idx_in_bin[b] = i

            # Remove bin from cycle if exhausted
            if idx_in_bin[b] >= len(cand_df):
                bin_cycle.remove(b)

            if len(chosen_rows) >= n:
                break

        if progressed:
            stalls = 0
        else:
            stalls += 1

    out = pd.DataFrame(chosen_rows).reset_index(drop=True)

    # Add a nearest-neighbor distance diagnostic
    if len(out) > 1:
        nn = []
        pts = out[["lat", "lng"]].to_numpy(dtype=float)
        for i in range(len(pts)):
            dmin = float("inf")
            for j in range(len(pts)):
                if i == j:
                    continue
                d = haversine_km(pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1])
                if d < dmin:
                    dmin = d
            nn.append(dmin)
        out["nearest_neighbor_km"] = nn
    else:
        out["nearest_neighbor_km"] = np.nan

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument("--n", type=int, default=500, help="Number of cities to sample")
    ap.add_argument("--min_km", type=float, default=64.0, help="Minimum distance between any two sampled cities (km)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--bins", type=int, default=6, help="Number of population bins (diversity strata)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = coerce_columns(df)

    sampled = spaced_stratified_sample(
        df=df,
        n=args.n,
        min_km=args.min_km,
        seed=args.seed,
        n_bins=args.bins,
    )

    if len(sampled) < args.n:
        print(
            f"Warning: only sampled {len(sampled)} cities (requested {args.n}). "
            f"Try lowering --min_km or --n, or increasing input coverage."
        )

    sampled.to_csv(args.output, index=False)
    print(f"Wrote {len(sampled)} cities to {args.output}")


if __name__ == "__main__":
    main()  