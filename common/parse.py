from __future__ import annotations

import logging
import re
from typing import Optional, Tuple
from .text import read_text_lines
from .text import parse_float


def parse_depth_and_volume_from_output(output_txt_path: str) -> Tuple[Optional[float], Optional[float]]:
    depth: Optional[float] = None
    volume: Optional[float] = None
    try:
        for line in read_text_lines(output_txt_path):
            low = line.lower()
            if depth is None and "surface-based max depth" in low:
                nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                if nums:
                    depth = parse_float(nums[0])
            if volume is None and "surface-based volume" in low:
                nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                if nums:
                    volume = parse_float(nums[0])
        if not depth or not volume:
            logging.error("Failed to parse depth and volume from %s", output_txt_path)
    except FileNotFoundError:
        logging.debug("output.txt not found: %s", output_txt_path)
    return depth, volume


