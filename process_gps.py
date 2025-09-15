"""process_imu

Extract vertical displacement samples from an IMU text log at fixed intervals
(every 30 seconds by default) and write them to a JSON file.

What it does:
- Reads a whitespace-delimited IMU text file.
- Ignores comment lines that begin with '#'.
- Interprets the first column as an absolute timestamp in seconds (float).
- Interprets the 4th column (index 3) as vertical displacement ("z").
- Samples the vertical displacement every 30 seconds relative to the first
  timestamp encountered, producing entries like:
  {"vertical_displacement": <float>, "video_timestamp": "<int_seconds>"}.

How to use:
    python scripts/process_imu.py /path/to/imu_file.txt [output.json]

Input expectations:
- Text file with lines like:
    1712345678.000  ax  ay  z  ...
  where the first token is the timestamp (seconds) and the 4th token is the
  vertical displacement value. Columns beyond the 4th are ignored.
- Lines starting with '#' are treated as comments and skipped.

Directory structure (example; no strict layout required):

    /path/to/imu_data
    └── imu_readings.txt

Output:
- JSON file (default: output.json) containing a list of sampled objects with
  timestamps at 0, 30, 60, ... seconds relative to the file's first timestamp.

"""

import json
import sys

def process_imu_file(filename, output_filename='output.json'):
    """Process an IMU text file and export vertical displacement samples.

    Reads ``filename`` as a whitespace-delimited IMU log, skipping comment
    lines that start with '#'. Treats column 0 as an absolute timestamp in
    seconds and column 3 as the vertical displacement ("z"). Samples the
    vertical displacement at fixed 30-second intervals relative to the first
    timestamp encountered, then writes the samples to ``output_filename`` in
    JSON format.

    Args:
        filename (str): Path to the IMU text file to parse.
        output_filename (str): Path to the JSON file to write (default
            'output.json').

    Returns:
        list[dict] | None: A list of sampled data points, each containing
            ``{"vertical_displacement": float, "video_timestamp": str}``,
            or ``None`` if no valid data is found.

    Notes:
        - Sampling interval is hard-coded to 30 seconds; change ``interval``
          below to modify.
        - The script prints each line used for a sample to stdout for traceability.
    """
    data_points = []
    
    # Read all lines from the IMU text file
    with open(filename, 'r') as f:
        lines = f.readlines()

    interval = 30.0  # seconds
    next_time = 0
    start_timestamp = None

    for line in lines:
        # Skip comment lines
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        # Expect at least 4 columns (timestamp + ... + z)
        if len(parts) >= 4:
            timestamp = float(parts[0])

            if start_timestamp is None:
                start_timestamp = timestamp

            relative_time = timestamp - start_timestamp

            if (relative_time >= next_time):
                z_displacement = float(parts[3])  # z column (vertical displacement)
                data_point = {
                    "vertical_displacement": round(z_displacement, 6),
                    "video_timestamp": str(int(next_time))
                }
                data_points.append(data_point)
                next_time += interval

                # Log the original line used for this sample (optional)
                print(line)

    if not data_points:
        print("No valid data found in file")
        return
    
    # Write to JSON file
    with open(output_filename, 'w') as f:
        json.dump(data_points, f, indent=2)
    
    print(f"JSON file created: {output_filename}")
    print(f"Total data points: {len(data_points)}")

    return data_points

def printTotalDuration(filename):
    """Print total duration covered by the IMU log in seconds.

    Determines the duration as ``last_timestamp - first_timestamp`` where the
    first timestamp is the first non-comment line's first token and the last
    timestamp is taken from the final line in the file (regardless of comment).

    Args:
        filename (str): Path to the IMU text file.

    Returns:
        None

    Notes:
        - Assumes the file is time-ordered. If not, the calculation may be
          inaccurate.
    """
    # Open and read all lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2: 
        print("File is too short to calculate duration.")
        return

    # Locate first non-comment line to get the starting timestamp
    for line in lines:
        if line.startswith('#'):
            continue
        first_timestamp = line.strip().split()[0]
        break
    
    # Take the final line's first token as the ending timestamp
    last_timestamp = lines[-1].strip().split()[0]
    total_duration = float(last_timestamp) - float(first_timestamp)

    print(f"Total duration: {total_duration} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/process_imu.py <imu_file.txt> [output.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output.json'
    
    process_imu_file(input_file, output_file)
    printTotalDuration(input_file)