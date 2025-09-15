"""retrieve_depth_timestamps

Aggregate pothole depth measurements and their relative video timestamps into a
single JSON file.

This script scans a segment directory (``--root-dir``) that contains multiple
pothole folders. Each pothole subfolder is expected to contain:
- At least one image file named as a floating-point timestamp, e.g.
  ``2430.799631730.jpg`` (or ``.png``). The numeric portion encodes the
  capture time in seconds.
- An ``output.txt`` file that contains a line like
  ``Surface-based max depth: 0.0259 m`` from which the pothole's depth (in
  meters) is parsed.

Given a separate image folder that also contains timestamped reference images,
the earliest timestamp from that folder is treated as the video's start time.
For every pothole subfolder discovered under the segment directory, the script
computes the pothole's relative timestamp (seconds since the first image) and
emits a compact JSON list with objects of the form:

{"pothole_depth": <float>, "video_timestamp": "<seconds_as_string>"}

Typical usage:
    python scripts/retrieve_depth_timestamps.py \
        --image-folder /path/to/reference_images \
        --root-dir /path/to/road/<road_id>/<segment_id> \
        --output aggregated_depth_data.json

Directory structure (example):

    /path/to/reference_images
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

def process_pothole_folder(pothole_folder):
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
    # find an image with .jpg extension, parse its name (2430.799631730.jpg) to get the timestamp
    for filename in os.listdir(pothole_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            timestamp = filename[:-4]  # Remove .jpg extension
            break  # Only need the first image
    
    if not timestamp:
        print(f"No image found in {pothole_folder}")
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
                        print(f"Invalid depth value '{depth_str}' in {output_file}")
                        return timestamp, None
                    return timestamp, depth

    print(f"No depth found for timestamp {timestamp} in {pothole_folder}")
    return timestamp, None

def process_segment_folder(segment_folder, first_timestamp):
    """Collect depth/timestamp entries for all potholes within a segment folder.

    For each subdirectory of ``segment_folder``, this function attempts to
    extract the pothole's image timestamp and depth via
    ``process_pothole_folder`` and compute the relative timestamp using
    ``first_timestamp``.

    Args:
        segment_folder (str): Path to a segment folder that contains pothole subfolders.
        first_timestamp (str): The earliest absolute timestamp (seconds as
            string) from the reference image folder, used as the zero point for
            relative timing.

    Returns:
        list[dict]: A list of entries, where each entry has:
            - ``"pothole_depth"`` (float): depth in meters.
            - ``"video_timestamp"`` (str): seconds since the first timestamp,
              rounded to 6 decimals and serialized as a string.

    Skips entries missing timestamps or depths, and continues if timestamps are
    not parseable as floats.
    """
    # Collect entries for this segment folder
    data_list = []
    # Go over each pothole folder in the segment folder
    for pothole_folder in os.listdir(segment_folder):
        pothole_path = os.path.join(segment_folder, pothole_folder)
        if os.path.isdir(pothole_path):
            timestamp, depth = process_pothole_folder(pothole_path)
            if timestamp is None or depth is None:
                continue
            try:
                # Calculate relative video timestamp in seconds and stringify
                # round 6 digits
                rel_seconds = round((float(timestamp) - float(first_timestamp)), 6)
            except ValueError:
                print(f"Invalid timestamp(s): {timestamp} or {first_timestamp}")
                continue
            json_data = {
                "pothole_depth": depth,
                "video_timestamp": str(rel_seconds),
            }
            data_list.append(json_data)
    return data_list

def main(args):
    """Entry point for aggregating pothole depths and relative timestamps for a segment.

    This function determines the earliest reference timestamp from the
    ``--image-folder`` directory, traverses all set folders under
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
    # Sort images by their names (timestamps) in the given folder and retrieve the first timestamp
    image_folder = args.image_folder
    root_dir = args.root_dir
    output_path = args.output
    timestamps = []
    
    if not os.path.exists(image_folder):
        print(f"Image folder not found: {image_folder}")
        return
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            timestamps.append(filename[:-4])  # Remove .jpg extension
    
    if not timestamps:
        print("No images found in the image folder")
        return
        
    timestamps.sort()
    first_timestamp = timestamps[0]

    # Process the provided root_dir as a single segment folder
    aggregated_list = process_segment_folder(root_dir, first_timestamp)

    # sort aggregated_list entries by the timestamp field
    aggregated_list.sort(key=lambda x: float(x["video_timestamp"]))

    # Write the aggregated json data to a file
    with open(output_path, "w") as json_file:
        json.dump(aggregated_list, json_file, indent=4)
    
    print(f"Data written to {output_path} with {len(aggregated_list)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate pothole depths and relative video timestamps into JSON for a single segment.")
    parser.add_argument(
        "-i", "--image-folder",
        default="/home/kodifly/Downloads/segmented_images_001/segmented_images",
        help="Folder containing reference images named as <timestamp>.jpg to determine the first timestamp."
    )
    parser.add_argument(
        "-r", "--root-dir",
        default=".",
        help="Segment directory containing pothole subfolders."
    )
    parser.add_argument(
        "-o", "--output",
        default="aggregated_depth_data.json",
        help="Output JSON file path for aggregated results."
    )
    args = parser.parse_args()
    main(args)