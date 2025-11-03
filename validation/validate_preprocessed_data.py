from pathlib import Path
from validation.validation_helpers import count_stems, get_stems, ValidationIssue
from typing import List

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PCD_EXTS = {".pcd"}

def validate_pothole_fodler_nonempty(pothole_folders: List[Path]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for folder in pothole_folders:
        if not any(folder.iterdir()):
            issues.append(ValidationIssue(
                str(folder),
                "POTHOLE_FOLDER_EMPTY",
                "error",
                f"Pothole folder is empty: {folder}"))
    return issues

def validate_preprocessed_data(images_dir: Path, segment_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    expected_count = len(get_stems(images_dir, IMAGE_EXTS))
    pothole_folders = [f for f in segment_dir.iterdir() if f.is_dir() and f.name.lower().startswith("pothole_")]

    if expected_count != len(pothole_folders):
        issues.append(ValidationIssue(
            str(segment_dir),
            "NUMBER_OF_POTHOLES_MISMATCH",
            "error",
            f"{expected_count} != {len(pothole_folders)}"
        ))

    issues.extend(validate_pothole_fodler_nonempty(pothole_folders))
    return issues