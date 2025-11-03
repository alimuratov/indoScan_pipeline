"""retrieve_depth_timestamps

Aggregate pothole depth measurements and their relative video timestamps into a
single JSON file.

This script scans a segment directory (``--root-dir``) that contains multiple
pothole folders and a reference images folder (default: ``raw_images``). Each
pot-hole subfolder is expected to contain:
- At least one image file named as a floating-point timestamp, e.g.
  ``2430.799631730.jpg`` (or ``.png``). The numeric portion encodes the
  capture time in seconds.
- An ``output.txt`` file that contains a line like
  ``Surface-based max depth: 0.0259 m`` from which the pothole's depth (in
  meters) is parsed.

The earliest timestamp in the segment's raw images folder is treated as
the video's start time. For every pothole subfolder discovered under the
segment directory, the script computes the pothole's relative timestamp
(seconds since the first image) and emits a compact JSON list with objects of
the form:

{"pothole_depth": <float>, "video_timestamp": "<seconds_as_string>"}

Typical usage:
    python scripts/sensors/segment_depth_timestamps.py \
        --root-dir /path/to/road/<road_id>/<segment_id> \
        --images-folder-name raw_images \
        --output segment_depth_timestamps.json

Directory structure (example):

    /path/to/road/<road_id>/<segment_id>/raw_images
    ├── 2430.799631730.jpg
    ├── 2431.012345678.jpg
    └── ...

    /path/to/road/<road_id>/<segment_id>
    ├── pothole-0001
    │   ├── 2431.100000000.jpg
    │   ├── 2431.100000000.png     # optional, .jpg or .png accepted
    │   └── output.txt             # contains "Surface-based max depth: <value> m"
    └── pothole-0002
        ├── 2432.250000000.jpg
        └── output.txt

"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from exceptions.exceptions import StepPreconditionError
from common.discovery import discover_pothole_dirs

def process_pothole_folder(pothole_folder: str) -> Tuple[Optional[str], Optional[float]]:
    """Extract the first image timestamp and parsed depth for a pothole folder.

    The function looks for any ``.jpg`` or ``.png`` file inside
    ``pothole_folder`` and derives the timestamp from the filename by removing
    the extension. It then parses the depth value from ``output.txt`` if a line
    containing ``"Surface-based max depth:"`` exists.

    Args:
        pothole_folder (str): Path to a pothole subfolder containing timestamped
            images and an ``output.txt`` file with a depth line.

    Returns:
        tuple[str | None, float | None]: A tuple of ``(timestamp_str, depth)``.
            - ``timestamp_str`` is the string before the image extension, or
              ``None`` if no image is found.
            - ``depth`` is a float parsed from ``output.txt`` (meters), or
              ``None`` if the file/line is missing or invalid.

    Notes:
        - Only the first image found is used for the timestamp.
        - If a timestamp is found but depth cannot be parsed, the tuple
          ``(timestamp_str, None)`` is returned.
    """
    timestamp = None
    # find an image with .jpg/.png extension, parse its stem (e.g., 2430.799631730)
    for filename in os.listdir(pothole_folder):
        name, ext = os.path.splitext(filename)
        if ext.lower() in (".jpg", ".png"):
            timestamp = name
            break  # Only need the first image
    
    if not timestamp:
        logging.warning("No image found in pothole folder: %s", pothole_folder)
        return None, None
    
    # retrieve the depth value specified like this: "Surface-based max depth: 0.0259 m" from the output.txt file
    output_file = os.path.join(pothole_folder, "output.txt")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if "Surface-based max depth:" in line:
                    depth_str = line.split(":")[1].strip().split(" ")[0]
                    try:
                        depth = float(depth_str)
                    except ValueError:
                        logging.warning("Invalid depth value '%s' in %s", depth_str, output_file)
                        return timestamp, None
                    return timestamp, depth

    logging.warning("No depth found for timestamp %s in pothole folder: %s", timestamp, pothole_folder)
    return timestamp, None

def process_segment_folder(segment_folder: str, images_folder_name: str = "raw_images") -> List[Dict]:
    """Collect depth/timestamp entries for all potholes within a segment folder.

    Determines the earliest reference timestamp by scanning
    ``segment_folder/images_folder_name``. Then, for each subdirectory of
    ``segment_folder``, attempts to extract the pothole's image timestamp and
    depth via ``process_pothole_folder`` and compute the relative timestamp
    using that earliest reference.

    Args:
        segment_folder (str): Path to a segment folder that contains pothole subfolders.
        images_folder_name (str): Name of the reference images folder under the
            segment folder (default: "raw_images").

    Returns:
        list[dict]: A list of entries, where each entry has:
            - ``"pothole_depth"`` (float): depth in meters.
            - ``"video_timestamp"`` (str): seconds since the first timestamp,
              rounded to 6 decimals and serialized as a string.

    Skips entries missing timestamps or depths, and continues if timestamps are
    not parseable as floats.
    """
    # Determine earliest reference timestamp from the images folder
    images_dir = os.path.join(segment_folder, images_folder_name)
    if not os.path.isdir(images_dir):
        raise StepPreconditionError("IMAGE_FOLDER_NOT_FOUND", f"Image folder not found: {images_dir}", context="segment_depth_timestamps")

    timestamps = []
    for filename in os.listdir(images_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() == ".jpg":
            timestamps.append(name)
    if not timestamps:
        logging.warning("No images found in the image folder: %s", images_dir)
        return []

    timestamps.sort()
    first_timestamp = timestamps[0]

    # Collect entries for this segment folder
    data_list = []
    # Go over each pothole folder in the segment folder
    for pothole_folder in discover_pothole_dirs(Path(segment_folder)):
        timestamp, depth = process_pothole_folder(pothole_folder)
        if timestamp is None or depth is None:
            continue
        try:
            # Calculate relative video timestamp in seconds and stringify
            # round 6 digits
            rel_seconds = round((float(timestamp) - float(first_timestamp)), 6)
        except ValueError:
            logging.warning("Invalid timestamp(s): %s or %s", timestamp, first_timestamp)
            continue
        json_data = {
            "pothole_depth": depth,
            "video_timestamp": str(rel_seconds),
        }
        data_list.append(json_data)
    try:
        data_list.sort(key=lambda x: float(x["video_timestamp"]))
    except Exception:
        from exceptions.exceptions import StepPreconditionError
        raise StepPreconditionError(
            "SEGMENT_DEPTH_SORT_FAILED",
            f"Failed to sort data list for segment: {segment_folder}",
            context="segment_depth_timestamps",
        )
    return data_list

def write_depth_timestamps(entries: List[Dict], output_path: str) -> None:
    """Write the aggregated entries to JSON.

    Creates parent directories if missing.
    """
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as json_file:
        json.dump(entries, json_file, indent=4)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate pothole depths and relative video timestamps into JSON for a single segment.")
    parser.add_argument(
        "-r", "--root-dir",
        default=".",
        help="Segment directory containing pothole subfolders."
    )
    parser.add_argument(
        "--images-folder-name",
        default="raw_images",
        help="Name of the reference images folder under the segment directory (default: raw_images)."
    )
    parser.add_argument(
        "-o", "--output",
        default="segment_depth_timestamps.json",
        help="Output JSON file path for aggregated results."
    )
    return parser

def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for aggregating pothole depths and relative timestamps for a segment.

    This function determines the earliest reference timestamp from the
    ``--image-folder-name`` directory, traverses all set folders under
    ``--root-dir``, aggregates valid pothole entries, sorts them by relative
    time, and writes the result as JSON to ``--output``.

    Args:
        args (argparse.Namespace): Parsed command-line arguments with fields:
            - ``image_folder`` (str): Reference images directory.
            - ``root_dir`` (str): Root directory containing set folders.
            - ``output`` (str): Output JSON path.

    Returns:
        None

    Side Effects:
        Writes a JSON file to the path specified by ``--output``.
    """
    p = build_parser()
    args = p.parse_args(argv)

    # Process the provided root_dir as a single segment folder
    root_dir = args.root_dir
    output_path = args.output
    images_folder_name = args.images_folder_name

    aggregated_list = process_segment_folder(root_dir, images_folder_name)
    write_depth_timestamps(aggregated_list, output_path)

    logging.info("Data written to %s with %d entries", output_path, len(aggregated_list))
    return 0
    
if __name__ == "__main__":
    import sys
    sys.exit(main())