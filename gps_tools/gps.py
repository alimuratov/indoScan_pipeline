from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class GpsPoint:
    timestamp: float
    lat: float
    lng: float
    alt: Optional[float]


def _read_text_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def parse_gps_file(gps_path: str) -> List[GpsPoint]:
    """Parse gps.txt with format: "#timestamp latitude longitude altitude" per line."""
    points: List[GpsPoint] = []
    try:
        for line in _read_text_lines(gps_path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            ts = _parse_float(parts[0])
            lat = _parse_float(parts[1])
            lng = _parse_float(parts[2])
            alt = _parse_float(parts[3]) if len(parts) >= 4 else None
            if ts is None or lat is None or lng is None:
                continue
            points.append(GpsPoint(timestamp=ts, lat=lat, lng=lng, alt=alt))
    except FileNotFoundError:
        logger.debug("gps.txt not found: %s", gps_path)
    return points


def pick_nearest_gps(points: List[GpsPoint], target_ts: float) -> Optional[GpsPoint]:
    """Pick GPS point whose timestamp is closest to the target timestamp."""
    if not points:
        return None
    return min(points, key=lambda p: abs(p.timestamp - target_ts))


def collect_segment_gps(segment_dir: str) -> List[GpsPoint]:
    """Aggregate GPS points from all gps.txt files within the segment."""
    logger.debug("Collecting GPS points for segment: %s", segment_dir)
    all_points: List[GpsPoint] = []
    for dirpath, _, filenames in os.walk(segment_dir):
        for fname in filenames:
            if fname.lower() == "gps.txt":
                path = os.path.join(dirpath, fname)
                logger.debug("Parsing GPS file: %s", path)
                all_points.extend(parse_gps_file(path))
    return sorted(all_points, key=lambda p: p.timestamp)


def compute_segment_start_end_loc(segment_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Compute segment start/end lat,lng from aggregated GPS points."""
    points = collect_segment_gps(segment_dir)
    if not points:
        logger.warning("No GPS points found for segment: %s", segment_dir)
        return None, None
    start = points[0]
    end = points[-1]
    logger.debug(
        "Segment %s start/end loc computed: (%s, %s) -> (%s, %s)",
        segment_dir,
        start.lat,
        start.lng,
        end.lat,
        end.lng,
    )
    return f"{start.lat}, {start.lng}", f"{end.lat}, {end.lng}"


