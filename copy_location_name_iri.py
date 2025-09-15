#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Copy road names/locations and segment iri/location from one JSON to another")
    p.add_argument("--src", default="poc-data-0828.json", help="Source JSON with filled names/locations/iri")
    p.add_argument("--dst", default="scripts/all_segments3.updated.json", help="Destination JSON to update in place")
    p.add_argument("--out", default=None, help="Optional output path (default: overwrite dst)")
    args = p.parse_args()

    src_path = Path(args.src).resolve()
    dst_path = Path(args.dst).resolve()
    out_path = Path(args.out).resolve() if args.out else dst_path

    with open(src_path, "r") as f:
        src = json.load(f)
    with open(dst_path, "r") as f:
        dst = json.load(f)

    src_roads_by_id = {r.get("id"): r for r in src.get("roads", [])}

    for d_road in dst.get("roads", []):
        s_road = src_roads_by_id.get(d_road.get("id"))
        if not s_road:
            continue
        # Copy road-level fields
        if "name" in s_road:
            d_road["name"] = s_road.get("name", d_road.get("name", ""))
        if "location" in s_road:
            d_road["location"] = s_road.get("location", d_road.get("location", ""))

        # Copy segment-level fields by id
        s_segments_by_id = {s.get("id"): s for s in s_road.get("road_segments", [])}
        for d_seg in d_road.get("road_segments", []):
            s_seg = s_segments_by_id.get(d_seg.get("id"))
            if not s_seg:
                continue
            if "iri" in s_seg:
                d_seg["iri"] = s_seg.get("iri", d_seg.get("iri"))
            if "location" in s_seg:
                d_seg["location"] = s_seg.get("location", d_seg.get("location", ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dst, f, indent=2)


if __name__ == "__main__":
    main()