import numpy as np
import argparse
import os

from common.config import load_config

def calculate_route_length(filename):
    """
    Calculates the total length of a 3D route from an odometry file.

    Args:
        filename (str): Path to the odometry file.

    Returns:
        float: Total length of the route in meters.
    """
    positions = []

    # Read the file and extract X, Y, Z coordinates
    with open(filename, 'r') as file:
        for line in file:
            # Split the line by whitespace
            parts = line.strip().split()
            # Ensure the line has enough data (at least 4 values: timestamp, x, y, z)
            if len(parts) >= 4:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    positions.append([x, y, z])
                except ValueError:
                    # Skip lines with invalid data
                    print(f"Warning: Could not parse line: {line}")
                    continue

    # Convert to numpy array for easier calculations
    positions = np.array(positions)

    if len(positions) < 2:
        print("Not enough valid points to calculate a distance.")
        return 0.0

    # Calculate the cumulative sum of distances between consecutive points
    segment_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_length_meters = np.sum(segment_distances)

    return total_length_meters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate 3D route length from odometry text file")
    parser.add_argument("filename", help="Path to odometry file (columns: timestamp x y z [...])")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = os.path.abspath(args.filename)
    total_distance_meters = calculate_route_length(input_path)
    total_distance_km = total_distance_meters / 1000.0
    print(f"Total route length: {total_distance_meters:.2f} meters ({total_distance_km:.5f} km)")