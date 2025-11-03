import cv2
import os
import numpy as np
from pathlib import Path
import re
import logging

def get_timestamp_from_filename(filename):
    match = re.match(r'(\d+\.\d+)\.jpg', filename)
    if match:
        return float(match.group(1))
    return 0

def create_video_from_images(input_folder, output_path='output_video.mp4', fps=10):
    """
    Create a video from timestamped images
    
    Args:
        input_folder: Path to folder containing images
        output_path: Path for output video file
        fps: Frames per second for the output video
    """
    # Get all jpg files
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    if not image_files:
        logging.debug("No .jpg files found in the specified folder")
        return
    
    # Sort files by timestamp
    image_files.sort(key=get_timestamp_from_filename)
    
    logging.debug(f"Found {len(image_files)} images")
    logging.debug(f"Creating video at {fps} fps...")
    
    # Read first image to get dimensions
    first_image_path = os.path.join(input_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        logging.debug(f"Error reading image: {first_image_path}")
        return
    
    height, width, layers = frame.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each image
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            logging.debug(f"Warning: Could not read image {image_file}, skipping...")
            continue
        
        # Resize if necessary (all frames must have same dimensions)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
        
        # Show progress
        if (idx + 1) % 10 == 0:
            logging.debug(f"Processed {idx + 1}/{len(image_files)} images")
    
    # Release everything
    video_writer.release()
    cv2.destroyAllWindows()
    
    logging.debug(f"Video created successfully: {output_path}")
    logging.debug(f"Total frames: {len(image_files)}")
    logging.debug(f"Video duration: {len(image_files)/fps:.2f} seconds")

if __name__ == "__main__":
    # Configure these parameters
    INPUT_FOLDER = "."  # Current directory, change to your image folder path
    OUTPUT_VIDEO = "output_10fps.mp4"
    FPS = 10
    
    # Create video
    create_video_from_images(INPUT_FOLDER, OUTPUT_VIDEO, FPS)