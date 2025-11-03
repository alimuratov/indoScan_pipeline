from pathlib import Path
from typing import List
from validation.validation_helpers import index_by_stem, count_stems, ValidationIssue

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PCD_EXTS = {".pcd"}

def validate_input_pairing(images_dir: Path, pcds_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    image_index = index_by_stem(images_dir, IMAGE_EXTS)
    pcd_index = index_by_stem(pcds_dir, PCD_EXTS)
    
    image_stems = set(image_index.keys())
    pcd_stems = set(pcd_index.keys())

    missing_images = sorted(pcd_stems - image_stems)
    missing_pcds = sorted(image_stems - pcd_stems)

    for s in missing_images:
        issues.append(ValidationIssue(str(pcd_index.get(s, pcds_dir)), "MISSING_IMAGE", "error", f"Missing image: {s}"))
    for s in missing_pcds:
        issues.append(ValidationIssue(str(image_index.get(s, images_dir)), "MISSING_PCD", "error", f"Missing PCD: {s}"))

    return issues

def validate_no_duplicates(images_dir: Path, pcds_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    image_counter = count_stems(images_dir, IMAGE_EXTS)
    pcd_counter = count_stems(pcds_dir, PCD_EXTS)

    for s in image_counter.keys():
        if image_counter[s] > 1:
            issues.append(ValidationIssue(str(images_dir / s), "DUPLICATE_IMAGE", "error", f"Duplicate image: {s}"))
    for s in pcd_counter.keys():
        if pcd_counter[s] > 1:
            issues.append(ValidationIssue(str(pcds_dir / s), "DUPLICATE_PCD", "error", f"Duplicate PCD: {s}"))
    return issues

def validate_input(images_dir: Path, pcds_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    issues.extend(validate_input_pairing(images_dir, pcds_dir))
    issues.extend(validate_no_duplicates(images_dir, pcds_dir))
    return issues