"""segment_processing bounded context.

Compute segment-level artifacts and metadata:
- Survey video (from raw_images/)
- Vertical displacement series (from imu.txt)
- Route length (from odometry log)
- Pothole depth timeline aligned to video timestamps
"""
