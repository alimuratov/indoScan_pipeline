from __future__ import annotations

import logging
import re
from typing import Optional, Tuple
from .text import read_text_lines
from .text import parse_float


class ParseError(Exception):
    pass


def parse_pothole_parameters(output_txt_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    depth: Optional[float] = None
    volume: Optional[float] = None
    area: Optional[float] = None
    try:
        lines = read_text_lines(output_txt_path)
    except FileNotFoundError as e:
        raise ParseError(f"output.txt not found: {output_txt_path}") from e

    for line in lines:
        low = line.lower()
        if depth is None and ("average of mean depths" in low or "mean depth:" in low or "max depth:" in low):
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            if nums:
                depth = parse_float(nums[0])
        if volume is None and "sum of volumes" in low:
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            if nums:
                volume = parse_float(nums[0])
        if area is None and "sum of areas" in low:
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            if nums:
                area = parse_float(nums[0])

    if depth is None or volume is None or area is None:
        raise ParseError(f"no pothole parameters parsed from: {output_txt_path}")
    return depth, volume, area


